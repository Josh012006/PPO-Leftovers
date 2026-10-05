"""Confirmatory ablation pipeline -- a separate, dedicated run from every
exploratory sweep in this project, meant to settle one question: does the
effective-sample-weighting mechanism (local count, near-exact bandwidths,
KL-anchored, k=0.10) help in realistic, from-scratch, multi-window PPO
training, and is that benefit attributable to the mechanism itself rather
than to this project's own hyperparameter tuning?

WHY THIS IS A SEPARATE SCRIPT, NOT ANOTHER MODE IN run_realistic_
multiwindow_sweep.py: that script is exploratory -- it has swept lr, beta
schedules, etc., and its own analysis has been written ad hoc, after
seeing each batch of results. A confirmatory run needs its primary metric,
its statistical test, and its configuration grid fixed BEFORE running,
so the analysis in this file is not something to edit after seeing the
output -- editing _print_report after the sweep has run defeats the
entire point of pre-registration.

PRE-REGISTERED, FIXED BEFORE RUNNING -- do not change after seeing results:

  Learning rate: 0.001 ONLY. Chosen from PRIOR exploratory sweeps (see
  the README and results/phase3/analysis/realistic_multiwindow_sweep/),
  not re-swept here -- re-optimizing lr inside the very run meant to
  confirm an effect is exactly the kind of researcher-degrees-of-freedom
  problem a confirmatory run exists to avoid.

  2x2 ablation grid (4 configurations, each run for n_seeds seeds):
    hp_profile x method
    "default" PPO hyperparameters (Schulman et al. 2017 / Stable-
      Baselines3-style defaults: clip_eps=0.2, gae_lambda=0.95,
      entropy_coef=0.0, value_coef=0.5, max_grad_norm=0.5) -- NOT this
      project's own tuned values, so the ablation can separate "does
      weighting help" from "does weighting only look good because it
      rides on hyperparameters this project already tuned".
    "tuned" PPO hyperparameters -- this project's own validated values
      (clip_eps=0.55, gae_lambda=0.90, entropy_coef=0.0, value_coef=1.0,
      max_grad_norm=0.1).
    method "none" (no weighting) vs "kl_anchored" (local count,
      near-exact bandwidths, beta=0.9995, effective_sample_kl_anchor=
      True, k=0.10) -- this project's single validated mechanism, not a
      new variant.

  Primary metric: best success_rate over the run (the checkpoint a real
  practitioner would actually deploy). Secondary metric: late_mean, the
  plateau over the last late_window_fraction of evaluation points (NOT
  the full-trajectory mean, which blends in the unavoidable random-init
  warm-up -- see run_realistic_multiwindow_sweep.py's own docstring for
  why).

  Statistical test: Wilcoxon signed-rank, PAIRED BY SEED (every
  configuration at a given seed starts from the identical theta_0 and
  uses the identical per-window collection-RNG formula, so pairing is
  legitimate, not assumed), on four pre-specified comparisons: the method
  effect at default hyperparameters, the method effect at tuned
  hyperparameters, the hyperparameter effect without the method, and the
  hyperparameter effect with the method. Each reported with a bootstrap
  95% CI on the median paired difference and a simple win-rate (fraction
  of seeds where the later configuration beats the earlier one), plus
  the interquartile mean (IQM) of each of the 4 groups alongside the
  plain mean -- RL run distributions are routinely skewed by occasional
  catastrophic-failure seeds (this project has already seen this exact
  pattern: a seed of "constant" stuck at exactly 0 for an entire run),
  and IQM is the standard robust alternative for exactly this reason
  (Agarwal, Schwarzer, Castro, Bellemare & Courville, 2021, "Deep
  Reinforcement Learning at the Edge of the Statistical Precipice",
  NeurIPS).

n_seeds: 50 by default -- chosen from a power analysis run directly on
this project's own 5-seed pilot data (late_mean, constant/kl_anchored vs
none, lr=0.001): the observed effect sizes were large (Cohen's d 1.3-2.0,
implying as few as 2-5 seeds for 80% power), but a pilot estimate from n=5
is itself noisy and likely optimistic (a "winner's curse" on a
small-sample effect-size estimate) -- even assuming the TRUE effect is
only a third of what the pilot showed, the required n is 17-40. 50 is a
deliberately generous margin over that pessimistic bound, not an
arbitrary round number.

Everything else (the multi-window training loop itself: random init,
fresh per-window collection, fresh per-window local count, carrying
weights forward) is unchanged from run_realistic_multiwindow_sweep.py --
see that file's own docstring for the full rationale of the loop itself,
not repeated here.

Reads the ablation grid and hyperparameter profiles from a YAML config
(configs/phase3/confirmatory_ablation.yaml) except --seeds, --out-dir,
--cpu-count and --force, which stay as command-line flags.

Parallel execution, resumability, per-combination logging mirror this
project's other sweep scripts (spawn context, --cpu-count clamped to
pending work and os.cpu_count(), 30 s heartbeat, one .log per
combination, skip combinations whose .csv already exists unless
--force). No checkpoints are saved.

Usage:
    python scripts/run_confirmatory_ablation.py \
        --config configs/phase3/confirmatory_ablation.yaml \
        --env-config configs/phase2/env_maze.yaml \
        --seeds $(seq 0 49) \
        --out-dir results/phase3/analysis/confirmatory_ablation \
        --cpu-count 5
"""
from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.ppo.fixed_d_trainer_local_count import LocalCountFixedDPPOTrainer
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed

_WORKER: dict = {}
_HP_PROFILES = ("default", "tuned")
_METHODS = ("none", "kl_anchored")


def _init_worker(env_cfg_dict, layout):
    collect_env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    eval_env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    _WORKER.update(collect_env=collect_env, eval_env=eval_env)


def _late_stats(df: pd.DataFrame, frac: float) -> tuple[float, float]:
    n_tail = max(1, int(round(len(df) * frac)))
    tail = df["success_rate"].iloc[-n_tail:]
    return float(tail.mean()), float(tail.std() if len(tail) > 1 else 0.0)


def _build_cfg(hp_profile: str, method: str, lr: float, task: dict, seed: int) -> PPOHyperparams:
    hp = task["hp_profiles"][hp_profile]
    common = dict(
        epochs=task["epochs_per_window"], minibatch_size=task["minibatch_size"], clip_eps=hp["clip_eps"],
        gae_lambda=hp["gae_lambda"], entropy_coef=hp["entropy_coef"], value_coef=hp["value_coef"],
        max_grad_norm=hp["max_grad_norm"], hidden_sizes=tuple(task["hidden_sizes"]), lr=lr, seed=seed,
    )
    if method == "none":
        return PPOHyperparams(**common, use_effective_sample_weighting=False)
    if method == "kl_anchored":
        return PPOHyperparams(
            **common, use_effective_sample_weighting=True, effective_sample_beta=task["beta"],
            effective_sample_kl_anchor=True, effective_sample_kl_k=task["kl_k"],
        )
    raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


def _run_one_combo(task: dict) -> dict:
    prefix = task["prefix"]
    out_dir = Path(task["out_dir"])
    log_path = out_dir / f"{prefix}.log"
    seed = task["seed"]
    hp_profile = task["hp_profile"]
    method = task["method"]
    lr = task["lr"]
    try:
        set_global_seed(seed)
        collect_env = _WORKER["collect_env"]
        eval_env = _WORKER["eval_env"]
        obs_dim = collect_env.observation_space.shape[0]
        n_actions = collect_env.n_actions

        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            net = ActorCritic(obs_dim, n_actions, hidden_sizes=tuple(task["hidden_sizes"]))
            state_dict = net.state_dict()
            print(f"hp_profile={hp_profile} method={method} lr={lr} seed={seed} -- theta_0 randomly initialized")

            results = []

            def do_eval(window):
                eval_net = ActorCritic(obs_dim, n_actions, hidden_sizes=tuple(task["hidden_sizes"]))
                eval_net.load_state_dict(state_dict)
                r = evaluate_policy(
                    eval_env, lambda o, s: eval_net.act_numpy(o, deterministic=True)[0],
                    n_episodes=task["eval_episodes"], seed=task["eval_seed"],
                )
                results.append({"window": window, **r})
                print(f"  window {window:4d}: success_rate={r['success_rate']:.4f}")

            do_eval(0)

            for window in range(1, task["num_windows"] + 1):
                collect_net = ActorCritic(obs_dim, n_actions, hidden_sizes=tuple(task["hidden_sizes"]))
                collect_net.load_state_dict(state_dict)
                dataset = collect_fixed_dataset(
                    collect_env, collect_net, n_episodes=task["episodes_per_window"],
                    seed=seed * 100_000 + window, sample_actions=True,
                )

                cfg = _build_cfg(hp_profile, method, lr, task, seed)
                if method == "none":
                    trainer = FixedDPPOTrainer(dataset, obs_dim, n_actions, cfg, prior_state_dict=state_dict)
                else:
                    trainer = LocalCountFixedDPPOTrainer(
                        dataset, obs_dim, n_actions, cfg, prior_state_dict=state_dict,
                        bandwidth_state=task["bandwidth_state"], bandwidth_action=task["bandwidth_action"],
                        kernel=task["kernel"],
                    )
                trainer.train(verbose=False)
                state_dict = trainer.net.state_dict()

                if window % task["eval_every_windows"] == 0:
                    do_eval(window)

        df = pd.DataFrame(results)
        csv_path = out_dir / f"{prefix}.csv"
        df.to_csv(csv_path, index=False)

        late_mean, late_std = _late_stats(df, task["late_window_fraction"])
        return {
            "seed": seed, "hp_profile": hp_profile, "method": method, "lr": lr,
            "best": float(df["success_rate"].max()),
            "best_window": int(df.loc[df["success_rate"].idxmax(), "window"]),
            "late_mean": late_mean, "late_std": late_std,
            "final": float(df["success_rate"].iloc[-1]),
            "csv_path": str(csv_path), "status": "ran", "log_path": str(log_path),
        }
    except Exception as e:
        with open(log_path, "a") as logf:
            logf.write(f"\n[ERROR] {e}\n{traceback.format_exc()}\n")
        return {"seed": seed, "hp_profile": hp_profile, "method": method, "lr": lr,
                "status": f"FAILED: {e}", "csv_path": None, "log_path": str(log_path)}


def _iqm(x: np.ndarray) -> float:
    """Interquartile mean: mean of the values strictly between the 25th
    and 75th percentiles -- robust to the occasional catastrophic-failure
    seed a plain mean is not. See the module docstring for the citation."""
    x = np.sort(np.asarray(x, dtype=float))
    lo, hi = np.percentile(x, 25), np.percentile(x, 75)
    mid = x[(x >= lo) & (x <= hi)]
    return float(mid.mean()) if len(mid) else float(x.mean())


def _paired_report(name: str, a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> None:
    """a, b: PAIRED (same seed order) arrays. Reports b - a: positive means
    b beats a. Pre-specified analysis -- see module docstring."""
    diff = b - a
    n = len(diff)
    win_rate = float((diff > 0).mean())
    try:
        _, wp = stats.wilcoxon(b, a)
    except ValueError:
        wp = float("nan")
    rng = np.random.default_rng(0)
    boot = [rng.choice(diff, size=n, replace=True).mean() for _ in range(10000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print(
        f"  {name}: {label_b} - {label_a}  |  win_rate={win_rate:.2f} ({int((diff>0).sum())}/{n})  "
        f"median_diff={np.median(diff):+.4f}  mean_diff={diff.mean():+.4f}  "
        f"Wilcoxon p={wp:.4f}  bootstrap 95% CI of mean diff=[{ci_lo:+.4f}, {ci_hi:+.4f}]"
    )


def _print_report(ok: pd.DataFrame, metric: str) -> None:
    print(f"\n--- Pre-registered analysis on '{metric}' ---")
    groups = {}
    for hp in _HP_PROFILES:
        for m in _METHODS:
            sub = ok[(ok.hp_profile == hp) & (ok.method == m)].sort_values("seed")
            groups[(hp, m)] = sub.set_index("seed")[metric]
    common_seeds = sorted(set.intersection(*(set(v.index) for v in groups.values())))
    if len(common_seeds) < len(next(iter(groups.values()))):
        print(f"  (note: only {len(common_seeds)} seeds completed in all 4 groups so far -- using those for pairing)")
    g = {k: v.loc[common_seeds].to_numpy() for k, v in groups.items()}

    print("\n  Group summary (plain mean, IQM, std), n = {} seeds:".format(len(common_seeds)))
    for (hp, m), vals in g.items():
        print(f"    hp_profile={hp:<8} method={m:<12} mean={vals.mean():.4f}  IQM={_iqm(vals):.4f}  std={vals.std():.4f}")

    print("\n  Four pre-specified paired comparisons:")
    _paired_report("Method effect @ default HP ", g[("default", "none")], g[("default", "kl_anchored")], "none", "kl_anchored")
    _paired_report("Method effect @ tuned HP   ", g[("tuned", "none")], g[("tuned", "kl_anchored")], "none", "kl_anchored")
    _paired_report("HP effect @ no method      ", g[("default", "none")], g[("tuned", "none")], "default", "tuned")
    _paired_report("HP effect @ with method    ", g[("default", "kl_anchored")], g[("tuned", "kl_anchored")], "default", "tuned")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--out-dir", default="results/phase3/analysis/confirmatory_ablation")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu-count", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        y = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    env_cfg_dict = dict(
        width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
        extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
        step_penalty=env_cfg.step_penalty, goal_reward=env_cfg.goal_reward, hazard_reward=env_cfg.hazard_reward,
        max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed, num_start_states=env_cfg.num_start_states,
        gamma=env_cfg.gamma,
    )
    ref_env = StochasticMazeEnv(**env_cfg_dict)
    lr = y["lr"]
    print(f"CONFIRMATORY RUN -- lr FIXED at {lr} (not swept). 2x2 grid x {len(args.seeds)} seeds.")
    print(
        f"Budget per run: {y['num_windows']} windows x {y['episodes_per_window']} episodes collected "
        f"x {y['epochs_per_window']} gradient epochs."
    )

    combos = [
        {"seed": s, "hp_profile": hp, "method": m, "lr": lr}
        for hp in _HP_PROFILES for m in _METHODS for s in args.seeds
    ]
    print(f"=== {len(combos)} combination(s): {len(args.seeds)} seed(s) x {len(_HP_PROFILES)} hp_profile(s) x {len(_METHODS)} method(s) ===")

    results, pending = [], []
    for combo in combos:
        prefix = f"seed_{combo['seed']}_{combo['hp_profile']}_{combo['method']}"
        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            df_existing = pd.read_csv(csv_path)
            late_mean, late_std = _late_stats(df_existing, y.get("late_window_fraction", 0.25))
            results.append({
                **combo, "prefix": prefix,
                "best": float(df_existing["success_rate"].max()),
                "best_window": int(df_existing.loc[df_existing["success_rate"].idxmax(), "window"]),
                "late_mean": late_mean, "late_std": late_std,
                "final": float(df_existing.iloc[-1]["success_rate"]),
                "csv_path": str(csv_path), "status": "skipped (already existed)",
            })
        else:
            pending.append({**combo, "prefix": prefix})
    if any(r["status"] == "skipped (already existed)" for r in results):
        print(f"{len(results)} combination(s) already present, skipped (use --force to re-run).")

    t_start = time.time()
    common = dict(
        out_dir=str(out_dir), epochs_per_window=y["epochs_per_window"], episodes_per_window=y["episodes_per_window"],
        num_windows=y["num_windows"], eval_every_windows=y["eval_every_windows"], minibatch_size=y["minibatch_size"],
        beta=y["beta"], kl_k=y["kl_k"], kernel=y.get("kernel", "gaussian"), bandwidth_state=y.get("bandwidth_state", 1e-6),
        bandwidth_action=y.get("bandwidth_action", 1e-6), hidden_sizes=y.get("hidden_sizes", [64, 64]),
        eval_episodes=y["eval_episodes"], eval_seed=y["eval_seed"], late_window_fraction=y.get("late_window_fraction", 0.25),
        hp_profiles=y["hp_profiles"],
    )

    if args.cpu_count <= 1 or len(pending) <= 1:
        _init_worker(env_cfg_dict, ref_env.layout)
        for i, combo in enumerate(pending, start=1):
            print(f"\n[{i}/{len(pending)} {combo['prefix']}] Starting -- {combo}")
            t0 = time.time()
            res = _run_one_combo({**combo, **common})
            dt = (time.time() - t0) / 60
            status_msg = res["status"] if str(res.get("status", "")).startswith("FAILED") else f"best={res['best']:.4f}"
            print(f"[{i}/{len(pending)} {combo['prefix']}] {'FAILED' if 'FAILED' in str(res.get('status','')) else 'Done'} in {dt:.1f} min -- {status_msg}")
            results.append(res)
    else:
        cpu_count = max(1, args.cpu_count)
        available = os.cpu_count() or cpu_count
        clamped = min(cpu_count, len(pending), available)
        if clamped != cpu_count:
            print(f"Note: --cpu-count {cpu_count} clamped to {clamped} (min of requested, {len(pending)} pending, {available} CPUs available).")
        cpu_count = clamped
        tasks = [{**combo, **common} for combo in pending]
        print(f"\nRunning {len(pending)} pending combination(s) across {cpu_count} worker process(es). "
              f"Per-window logs go to <out_dir>/<prefix>.log, not this console.\n")
        n_done = 0
        with ProcessPoolExecutor(
            max_workers=cpu_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(env_cfg_dict, ref_env.layout),
        ) as executor:
            futures = {executor.submit(_run_one_combo, t): t for t in tasks}
            pending_futures = set(futures.keys())
            while pending_futures:
                done, pending_futures = wait(pending_futures, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    print(f"... still running ({n_done}/{len(pending)} done, {(time.time() - t_start) / 60:.1f} min elapsed)", flush=True)
                    continue
                for fut in done:
                    t = futures[fut]
                    n_done += 1
                    elapsed = (time.time() - t_start) / 60
                    try:
                        res = fut.result()
                    except Exception as e:  # pragma: no cover
                        res = {"seed": t["seed"], "hp_profile": t["hp_profile"], "method": t["method"], "lr": t["lr"],
                               "status": f"FAILED: {e}", "csv_path": None, "log_path": f"{out_dir}/{t['prefix']}.log"}
                    results.append(res)
                    tag = "FAILED" if str(res.get("status", "")).startswith("FAILED") else "done"
                    extra = res["status"] if tag == "FAILED" else f"best={res['best']:.4f}"
                    print(f"[{n_done}/{len(pending)}] {t['prefix']}: {tag} -- {extra}  [{elapsed:.1f} min elapsed]")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_dir / "sweep_summary.csv", index=False)
    n_failed = int(summary_df["status"].astype(str).str.startswith("FAILED").sum()) if len(summary_df) else 0
    print(f"\n=== Sweep complete in {(time.time() - t_start) / 60:.1f} min total ({n_failed} failed) ===")

    ok = summary_df[summary_df["status"].isin(["ran", "skipped (already existed)"])] if not summary_df.empty else summary_df
    if len(ok) == len(combos):
        _print_report(ok, metric="best")
        _print_report(ok, metric="late_mean")
    else:
        print(f"\n{len(ok)}/{len(combos)} combinations completed -- re-run this script (it will skip what's done) "
              f"until all {len(combos)} are in before trusting the pre-registered report.")
    print(f"\nSaved combined summary to {out_dir / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
