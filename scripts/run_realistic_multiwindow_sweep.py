"""Full, realistic multi-window PPO training, from scratch -- not the
fixed-D single-window protocol every earlier script in this project used.

This is the actual training loop realistic PPO runs: random initialization
(no prior_checkpoint), then repeatedly (a) collect a fresh rollout under
the CURRENT policy, (b) run a short window of gradient epochs on it, (c)
carry the resulting weights forward as the start of the next window. Each
window internally IS one call to this project's already-validated fixed-D
machinery (FixedDPPOTrainer / LocalCountFixedDPPOTrainer) -- pi_prior for
window t is exactly pi_old at the start of window t, matching THEORICAL_
ANALYSIS.md Section 4.1's own framing, not an approximation of it. For
"constant" and "kl_anchored", the local count is recomputed FRESH each
window from that window's own freshly-collected data, for the same
reason: a stale count from an earlier, now-superseded policy would not be
what the theory describes.

Three modes (see run_lr_mode_isolation_sweep.py's own docstring for the
identical distinction, here applied per-window instead of once):
  none        -- no weighting: plain FixedDPPOTrainer each window.
  constant    -- LocalCountFixedDPPOTrainer each window, near-exact
                 bandwidths, beta fixed at 0.9995, never decays within a
                 window (effective_sample_kl_anchor=False).
  kl_anchored -- same, but effective_sample_kl_anchor=True, k=0.10 --
                 decaying WITHIN each window as that window's own KL
                 grows, resetting fresh at the start of the next window
                 (pi_old just changed).

MULTI-SEED VALIDATION. A first 1-seed, 50-window run at lr in {0.0003,
0.001, 0.002} found lr=0.0003 too slow for this budget to show any
learning at all, and -- at the two lr that DID show real learning -- a
ranking between modes that reversed between lr=0.001 (none finished
highest) and lr=0.002 (kl_anchored finished highest), with swings between
evaluation points (e.g. 0.80 -> 0.32 -> 0.50 across three consecutive
evaluations for kl_anchored at lr=0.001) far too large to be explained by
eval_episodes=500's own sampling noise (expected stderr ~0.02) -- i.e.
real, not measurement, volatility. This run is the multi-seed, longer-
budget follow-up needed to tell a genuine failure mode of the weighting
mechanism in this realistic regime apart from one seed's own unlucky
trajectory: lr in {0.001, 0.002} only (0.0003 already established as too
slow to be informative), 5 seeds, num_windows raised from 50 to 200.

EVALUATION METRIC: a single, plain success_rate (evaluate_policy, the
env's own default start-state reset), NOT this project's usual
weighted_success_rate -- the covered/held-out split has no referent in a
loop where every window's data is freshly collected under the env's own
default reset, not a fixed, deliberately-skewed dataset D.

DETERMINISM: evaluate_policy reseeds its own RNG fresh
(np.random.default_rng(seed)) on every call, so repeated evaluations at
different windows with the same eval_seed are fully reproducible given the
same policy weights -- verified directly against evaluate_policy's own
source. eval_seed is deliberately set to a value distinct from 999 (used
everywhere else in this project, always for the fixed-D, tiered-coverage
evaluation this experiment does not use), to mark this from-scratch
multi-window track as its own thing.

Budget per run (configs/phase3/realistic_training_sweep.yaml): 100
episodes collected per window, 200 windows, epochs=30 per window,
evaluated every 5 windows. Total over a full run: 20,000 episodes
collected, 6,000 cumulative gradient epochs.

Reads all settings from a YAML config except --seeds, --out-dir,
--cpu-count and --force, which stay as command-line flags, matching this
project's other sweep scripts.

Parallel execution, resumability, per-combination logging mirror this
project's other sweep scripts (spawn context, --cpu-count clamped to
pending work and os.cpu_count(), 30 s heartbeat, one .log per combination,
skip combinations whose .csv already exists unless --force). No
checkpoints are saved.

Usage:
    python scripts/run_realistic_multiwindow_sweep.py \
        --config configs/phase3/realistic_training_sweep.yaml \
        --env-config configs/phase2/env_maze.yaml \
        --seeds 0 1 2 3 4 \
        --out-dir results/phase3/analysis/realistic_multiwindow_sweep \
        --cpu-count 10
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

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.ppo.fixed_d_trainer_local_count import LocalCountFixedDPPOTrainer
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed

_WORKER: dict = {}
_MODES = ("none", "constant", "kl_anchored")


def _init_worker(env_cfg_dict, layout):
    # Two separate env instances (same layout): one dedicated to
    # collection, one to evaluation, so a mid-rollout collection episode
    # can never leave state behind that an evaluation episode -- run
    # inside the same eval_cb -- would otherwise observe.
    collect_env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    eval_env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    _WORKER.update(collect_env=collect_env, eval_env=eval_env)


def _first_crossing_window(df: pd.DataFrame, threshold: float) -> int | None:
    hit = df[df["success_rate"] >= threshold]
    return int(hit["window"].iloc[0]) if len(hit) else None


def _build_cfg(mode: str, lr: float, task: dict, seed: int) -> PPOHyperparams:
    common = dict(
        epochs=task["epochs_per_window"], minibatch_size=task["minibatch_size"], clip_eps=task["clip_eps"],
        gae_lambda=task["gae_lambda"], entropy_coef=0.0, value_coef=1.0, max_grad_norm=0.1,
        hidden_sizes=tuple(task["hidden_sizes"]), lr=lr, seed=seed,
    )
    if mode == "none":
        return PPOHyperparams(**common, use_effective_sample_weighting=False)
    if mode == "constant":
        return PPOHyperparams(
            **common, use_effective_sample_weighting=True, effective_sample_beta=task["beta"],
            effective_sample_kl_anchor=False,
        )
    if mode == "kl_anchored":
        return PPOHyperparams(
            **common, use_effective_sample_weighting=True, effective_sample_beta=task["beta"],
            effective_sample_kl_anchor=True, effective_sample_kl_k=task["kl_k"],
        )
    raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")


def _run_one_combo(task: dict) -> dict:
    prefix = task["prefix"]
    out_dir = Path(task["out_dir"])
    log_path = out_dir / f"{prefix}.log"
    seed = task["seed"]
    lr = task["lr"]
    mode = task["mode"]
    try:
        set_global_seed(seed)
        collect_env = _WORKER["collect_env"]
        eval_env = _WORKER["eval_env"]
        obs_dim = collect_env.observation_space.shape[0]
        n_actions = collect_env.n_actions

        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            # Window 0: theta initialized RANDOMLY -- no prior_checkpoint,
            # no prior training of any kind. This is the actual starting
            # point of a real training run.
            net = ActorCritic(obs_dim, n_actions, hidden_sizes=tuple(task["hidden_sizes"]))
            state_dict = net.state_dict()
            print(f"mode={mode} lr={lr} seed={seed} -- theta_0 randomly initialized")

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

            do_eval(0)  # theta_0, before any training -- the true starting point

            for window in range(1, task["num_windows"] + 1):
                # (a) collect a fresh rollout under the CURRENT policy --
                # pi_old for this window, exactly. A fresh ActorCritic is
                # built from state_dict rather than mutating `net` in
                # place, since the trainer below also needs its own copy
                # to start from (prior_state_dict), and the two must not
                # alias the same tensors once training starts.
                collect_net = ActorCritic(obs_dim, n_actions, hidden_sizes=tuple(task["hidden_sizes"]))
                collect_net.load_state_dict(state_dict)
                dataset = collect_fixed_dataset(
                    collect_env, collect_net, n_episodes=task["episodes_per_window"],
                    seed=seed * 100_000 + window, sample_actions=True,
                )

                # (b) one window of gradient epochs on this window's own
                # data -- pi_prior = the policy that JUST collected it, so
                # the count (for "constant"/"kl_anchored") is computed
                # fresh from this window's own rollout, never carried over
                # from an earlier, now-superseded window.
                cfg = _build_cfg(mode, lr, task, seed)
                if mode == "none":
                    trainer = FixedDPPOTrainer(dataset, obs_dim, n_actions, cfg, prior_state_dict=state_dict)
                else:
                    trainer = LocalCountFixedDPPOTrainer(
                        dataset, obs_dim, n_actions, cfg, prior_state_dict=state_dict,
                        bandwidth_state=task["bandwidth_state"], bandwidth_action=task["bandwidth_action"],
                        kernel=task["kernel"],
                    )
                trainer.train(verbose=False)

                # (c) carry the trained weights forward as the start of
                # the next window.
                state_dict = trainer.net.state_dict()

                if window % task["eval_every_windows"] == 0:
                    do_eval(window)

        df = pd.DataFrame(results)
        csv_path = out_dir / f"{prefix}.csv"
        df.to_csv(csv_path, index=False)

        crossings = {f"window_at_{t:g}": _first_crossing_window(df, t) for t in task["success_rate_thresholds"]}
        return {
            "seed": seed, "mode": mode, "lr": lr,
            "mean": float(df["success_rate"].mean()), "best": float(df["success_rate"].max()),
            "best_window": int(df.loc[df["success_rate"].idxmax(), "window"]),
            "final": float(df["success_rate"].iloc[-1]),
            **crossings, "csv_path": str(csv_path), "status": "ran", "log_path": str(log_path),
        }
    except Exception as e:
        with open(log_path, "a") as logf:
            logf.write(f"\n[ERROR] {e}\n{traceback.format_exc()}\n")
        return {"seed": seed, "mode": mode, "lr": lr, "status": f"FAILED: {e}", "csv_path": None, "log_path": str(log_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/phase3/realistic_training_sweep.yaml")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out-dir", default="results/phase3/analysis/realistic_multiwindow_sweep")
    parser.add_argument("--force", action="store_true", help="Re-run combinations even if their CSV already exists.")
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
    print(
        f"Budget per run: {y['num_windows']} windows x {y['episodes_per_window']} episodes collected "
        f"x {y['epochs_per_window']} gradient epochs = {y['num_windows'] * y['episodes_per_window']} episodes "
        f"collected, {y['num_windows'] * y['epochs_per_window']} cumulative gradient epochs total."
    )

    modes = y.get("modes", list(_MODES))
    for m in modes:
        if m not in _MODES:
            raise ValueError(f"configs/phase3/realistic_training_sweep.yaml: mode must be one of {_MODES}, got {m!r}")

    thresholds = y["success_rate_thresholds"]
    combos = [{"seed": s, "mode": m, "lr": lr} for m in modes for lr in y["learning_rates"] for s in args.seeds]
    print(f"=== {len(combos)} combination(s): {len(args.seeds)} seed(s) x {len(modes)} mode(s) x "
          f"{len(y['learning_rates'])} learning rate(s) ===")

    def fmt(x):
        return f"{x:g}".replace(".", "_")

    results, pending = [], []
    for combo in combos:
        prefix = f"seed_{combo['seed']}_{combo['mode']}_lr_{fmt(combo['lr'])}"
        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            print(f"[{prefix}] SKIPPING -- {csv_path} already exists (use --force to re-run)")
            df_existing = pd.read_csv(csv_path)
            crossings = {f"window_at_{t:g}": _first_crossing_window(df_existing, t) for t in thresholds}
            results.append(
                {
                    **combo, "prefix": prefix,
                    "mean": float(df_existing["success_rate"].mean()),
                    "best": float(df_existing["success_rate"].max()),
                    "best_window": int(df_existing.loc[df_existing["success_rate"].idxmax(), "window"]),
                    "final": float(df_existing.iloc[-1]["success_rate"]),
                    **crossings, "csv_path": str(csv_path), "status": "skipped (already existed)",
                }
            )
        else:
            pending.append({**combo, "prefix": prefix})

    t_start = time.time()
    common = dict(
        out_dir=str(out_dir), epochs_per_window=y["epochs_per_window"], episodes_per_window=y["episodes_per_window"],
        num_windows=y["num_windows"], eval_every_windows=y["eval_every_windows"], minibatch_size=y["minibatch_size"],
        clip_eps=y["clip_eps"], gae_lambda=y["gae_lambda"], beta=y["beta"], kl_k=y.get("kl_k", 0.10),
        kernel=y.get("kernel", "gaussian"), bandwidth_state=y.get("bandwidth_state", 1e-6),
        bandwidth_action=y.get("bandwidth_action", 1e-6), hidden_sizes=y.get("hidden_sizes", [64, 64]),
        eval_episodes=y["eval_episodes"], eval_seed=y["eval_seed"], success_rate_thresholds=thresholds,
    )

    if args.cpu_count <= 1 or len(pending) <= 1:
        _init_worker(env_cfg_dict, ref_env.layout)
        for i, combo in enumerate(pending, start=1):
            print(f"\n[{i}/{len(pending)} {combo['prefix']}] Starting -- {combo}")
            t0 = time.time()
            res = _run_one_combo({**combo, **common})
            dt = (time.time() - t0) / 60
            if str(res.get("status", "")).startswith("FAILED"):
                print(f"[{i}/{len(pending)} {combo['prefix']}] FAILED -- {res['status']} (see {res.get('log_path')})  [{dt:.1f} min]")
            else:
                print(f"[{i}/{len(pending)} {combo['prefix']}] Done in {dt:.1f} min -- mean={res['mean']:.4f} best={res['best']:.4f}")
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
                    except Exception as e:  # pragma: no cover -- _run_one_combo already catches its own exceptions
                        res = {"seed": t["seed"], "mode": t["mode"], "lr": t["lr"], "status": f"FAILED: {e}",
                               "csv_path": None, "log_path": f"{out_dir}/{t['prefix']}.log"}
                    results.append(res)
                    if str(res.get("status", "")).startswith("FAILED"):
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: FAILED -- {res['status']} (see {res.get('log_path')})  [{elapsed:.1f} min elapsed]")
                    else:
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: done -- mean={res['mean']:.4f} best={res['best']:.4f}  [{elapsed:.1f} min elapsed]")

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    n_failed = int(summary_df["status"].astype(str).str.startswith("FAILED").sum()) if len(summary_df) else 0
    print(f"\n=== Sweep complete in {(time.time() - t_start) / 60:.1f} min total ({n_failed} failed) ===")
    ok = summary_df[summary_df["status"].isin(["ran", "skipped (already existed)"])] if not summary_df.empty else summary_df
    if not ok.empty:
        crossing_cols = [c for c in ok.columns if c.startswith("window_at_")]
        print("\n=== Every run ===")
        print(ok.sort_values(["lr", "mode", "seed"])[["mode", "lr", "seed"] + crossing_cols + ["mean", "best", "best_window", "final"]].to_string(index=False))

        if ok["seed"].nunique() > 1:
            print("\n=== Aggregated across seeds, by mode and learning rate (mean_of_means / std_of_means / n_seeds) ===")
            agg = ok.groupby(["lr", "mode"])["mean"].agg(["mean", "std", "count"])
            agg.columns = ["mean_of_means", "std_of_means", "n_seeds"]
            print(agg.round(4).to_string())
    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
