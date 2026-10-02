"""Full, realistic multi-window PPO training, from scratch -- not the
fixed-D single-window protocol every earlier script in this project used.

Random initialization (no prior_checkpoint), then repeatedly: (a) collect
a fresh rollout under the CURRENT policy, (b) run a short window of
gradient epochs on it, (c) carry the resulting weights forward as the
start of the next window. Each window internally IS one call to this
project's already-validated fixed-D machinery (FixedDPPOTrainer /
LocalCountFixedDPPOTrainer) -- pi_prior for window t is exactly pi_old at
the start of window t. For every weighted mode, the local count is
recomputed FRESH each window from that window's own freshly-collected
data.

Five modes now (the first three validated on a 1-seed/50-window run, then
a 5-seed/200-window one -- see the README -- which found the mechanism
beats no-weighting reliably at lr=0.001 but not at lr=0.002, with real,
non-eval-noise instability/oscillation at both, not shrinking over the
back half of training (checked directly: per-tier std of success_rate
does not decrease from the first third of windows to the last)):

  none                     -- no weighting: plain FixedDPPOTrainer.
  constant                 -- LocalCountFixedDPPOTrainer, beta FIXED at
                              0.9995, never decays within a window.
  kl_anchored              -- same, but effective_sample_kl_anchor=True,
                              k=0.10 -- decaying WITHIN each window as
                              that window's own KL grows, resetting fresh
                              at the start of the next window.
  linear_beta_schedule     -- NEW. Same intra-window KL-anchor mechanism
                              as "kl_anchored" (so the only thing that
                              differs from "kl_anchored" is where that
                              anchor STARTS each window), but the
                              beta fed in as that window's own starting
                              point drifts linearly, across windows, from
                              beta_init=0.9995 at window 0 to beta_target
                              =0.99999 at the final window -- the
                              across-window schedule this project
                              discussed and never tested, isolated here
                              from the already-tested within-window
                              decay by holding the latter fixed.
  success_rate_beta_schedule -- NEW. Same structure, but the per-window
                              starting beta adapts to the TREND in a
                              success rate computed for free from each
                              window's own freshly-collected rollout (no
                              extra eval call -- see _window_success_rate):
                              only after 3 CONSECUTIVE windows all moving
                              the same direction (a guardrail against
                              adjusting on one noisy window, learned from
                              this project's own earlier F-test
                              miscalibration mistake) does beta move --
                              toward beta_target if success rate has been
                              consistently RISING (less correction
                              needed), or back toward beta_init if it has
                              been consistently FALLING OR FLAT (more
                              correction injected, as a stabilizing
                              response, not just a passive default). Each
                              move is a fixed fraction of the remaining
                              gap to whichever bound applies, so it can
                              never overshoot either bound and naturally
                              slows as it approaches one.

EVALUATION METRIC: a single, plain success_rate (evaluate_policy, the
env's own default start-state reset), NOT this project's usual
weighted_success_rate -- the covered/held-out split has no referent in a
loop where every window's data is freshly collected under the env's own
default reset, not a fixed, deliberately-skewed dataset D.

DETERMINISM: evaluate_policy reseeds its own RNG fresh on every call, so
repeated evaluations at different windows with the same eval_seed are
fully reproducible given the same policy weights. eval_seed is
deliberately distinct from 999 (used everywhere else in this project, for
the fixed-D, tiered-coverage evaluation this experiment does not use).

ANALYSIS CONVENTION: use the BEST and late-window (plateau) success_rate,
not the full-trajectory mean, which blends in the necessarily-low
random-init warm-up every run starts from and answers neither "how high"
nor "how fast" nor "how stable" on its own.

Reads all settings from a YAML config except --seeds, --out-dir,
--cpu-count and --force, which stay as command-line flags.

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
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy
from ppo_exploitation.ppo.buffer import FixedDataset
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.ppo.fixed_d_trainer_local_count import LocalCountFixedDPPOTrainer
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed

_WORKER: dict = {}
_MODES = ("none", "constant", "kl_anchored", "linear_beta_schedule", "success_rate_beta_schedule")
_SCHEDULE_MODES = ("linear_beta_schedule", "success_rate_beta_schedule")


def _init_worker(env_cfg_dict, layout):
    # Two separate env instances (same layout): one dedicated to
    # collection, one to evaluation, so a mid-rollout collection episode
    # can never leave state behind that an evaluation episode would
    # otherwise observe.
    collect_env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    eval_env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    _WORKER.update(collect_env=collect_env, eval_env=eval_env)


def _first_crossing_window(df: pd.DataFrame, threshold: float) -> int | None:
    hit = df[df["success_rate"] >= threshold]
    return int(hit["window"].iloc[0]) if len(hit) else None


def _late_stats(df: pd.DataFrame, frac: float) -> tuple[float, float, int]:
    """Mean and std of success_rate over the LAST `frac` fraction of
    evaluation points (at least 1) -- the plateau level and its
    stability, not the full-trajectory mean (see module docstring)."""
    n_tail = max(1, int(round(len(df) * frac)))
    tail = df["success_rate"].iloc[-n_tail:]
    return float(tail.mean()), float(tail.std() if len(tail) > 1 else 0.0), n_tail


def _window_success_rate(dataset: FixedDataset) -> float:
    """Success rate of the episodes THIS window just collected -- free,
    no extra eval call: an episode succeeded iff it ended in a true
    terminal (not a timeout/rollout cutoff) with a positive final reward
    (the goal, not a hazard). Feeds success_rate_beta_schedule only; never
    used as this project's own reported evaluation metric."""
    trajs = dataset.trajectories
    if not trajs:
        return 0.0
    successes = sum(1 for t in trajs if t.terminated_final and t.rewards[-1] > 0)
    return successes / len(trajs)


def _beta_linear_schedule(window: int, num_windows: int, beta_init: float, beta_target: float) -> float:
    frac = window / max(num_windows, 1)
    return beta_init + (beta_target - beta_init) * frac


class _SuccessRateScheduler:
    """Stateful, one-way-guarded beta scheduler for
    success_rate_beta_schedule -- see the module docstring for the
    guardrail rationale (3 consecutive windows required before acting) and
    why moves are a fraction of the remaining gap to whichever bound
    applies (never overshoots beta_init or beta_target)."""

    def __init__(self, beta_init: float, beta_target: float, step_frac: float = 0.1, history_len: int = 3):
        self.beta_init = beta_init
        self.beta_target = beta_target
        self.step_frac = step_frac
        self.history_len = history_len
        self.beta = beta_init
        self._history: deque[float] = deque(maxlen=history_len)

    def update(self, window_success_rate: float) -> float:
        self._history.append(window_success_rate)
        if len(self._history) == self.history_len:
            h = list(self._history)
            rising = all(h[i] < h[i + 1] for i in range(len(h) - 1))
            falling_or_flat = all(h[i] >= h[i + 1] for i in range(len(h) - 1))
            if rising:
                self.beta = self.beta + self.step_frac * (self.beta_target - self.beta)
            elif falling_or_flat:
                self.beta = self.beta - self.step_frac * (self.beta - self.beta_init)
        return self.beta


def _build_cfg(mode: str, lr: float, beta: float, task: dict, seed: int) -> PPOHyperparams:
    common = dict(
        epochs=task["epochs_per_window"], minibatch_size=task["minibatch_size"], clip_eps=task["clip_eps"],
        gae_lambda=task["gae_lambda"], entropy_coef=0.0, value_coef=1.0, max_grad_norm=0.1,
        hidden_sizes=tuple(task["hidden_sizes"]), lr=lr, seed=seed,
    )
    if mode == "none":
        return PPOHyperparams(**common, use_effective_sample_weighting=False)
    if mode == "constant":
        return PPOHyperparams(
            **common, use_effective_sample_weighting=True, effective_sample_beta=beta,
            effective_sample_kl_anchor=False,
        )
    if mode in ("kl_anchored",) + _SCHEDULE_MODES:
        return PPOHyperparams(
            **common, use_effective_sample_weighting=True, effective_sample_beta=beta,
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

        schedule_hist = []  # (window, beta_used, window_success_rate or None) -- schedule modes only
        sr_scheduler = (
            _SuccessRateScheduler(task["beta_init"], task["beta_target"], task["schedule_step_frac"], task["schedule_history_len"])
            if mode == "success_rate_beta_schedule" else None
        )

        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            # Window 0: theta initialized RANDOMLY -- no prior_checkpoint,
            # no prior training of any kind.
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
                # (a) collect a fresh rollout under the CURRENT policy.
                collect_net = ActorCritic(obs_dim, n_actions, hidden_sizes=tuple(task["hidden_sizes"]))
                collect_net.load_state_dict(state_dict)
                dataset = collect_fixed_dataset(
                    collect_env, collect_net, n_episodes=task["episodes_per_window"],
                    seed=seed * 100_000 + window, sample_actions=True,
                )

                # (b) this window's starting beta -- fixed for the first
                # three modes, scheduled for the last two.
                if mode == "linear_beta_schedule":
                    beta = _beta_linear_schedule(window, task["num_windows"], task["beta_init"], task["beta_target"])
                    schedule_hist.append((window, beta, None))
                elif mode == "success_rate_beta_schedule":
                    w_sr = _window_success_rate(dataset)
                    beta = sr_scheduler.update(w_sr)
                    schedule_hist.append((window, beta, w_sr))
                else:
                    beta = task["beta"]

                # (c) one window of gradient epochs on this window's own
                # data -- pi_prior = the policy that JUST collected it, so
                # the count (for every weighted mode) is computed fresh
                # from this window's own rollout, never carried over from
                # an earlier, now-superseded window.
                cfg = _build_cfg(mode, lr, beta, task, seed)
                if mode == "none":
                    trainer = FixedDPPOTrainer(dataset, obs_dim, n_actions, cfg, prior_state_dict=state_dict)
                else:
                    trainer = LocalCountFixedDPPOTrainer(
                        dataset, obs_dim, n_actions, cfg, prior_state_dict=state_dict,
                        bandwidth_state=task["bandwidth_state"], bandwidth_action=task["bandwidth_action"],
                        kernel=task["kernel"],
                    )
                trainer.train(verbose=False)

                # (d) carry the trained weights forward as the start of
                # the next window.
                state_dict = trainer.net.state_dict()

                if window % task["eval_every_windows"] == 0:
                    do_eval(window)

        df = pd.DataFrame(results)
        csv_path = out_dir / f"{prefix}.csv"
        df.to_csv(csv_path, index=False)
        if schedule_hist:
            pd.DataFrame(schedule_hist, columns=["window", "beta", "window_success_rate"]).to_csv(
                out_dir / f"{prefix}_schedule.csv", index=False
            )

        crossings = {f"window_at_{t:g}": _first_crossing_window(df, t) for t in task["success_rate_thresholds"]}
        late_mean, late_std, n_tail = _late_stats(df, task["late_window_fraction"])
        return {
            "seed": seed, "mode": mode, "lr": lr,
            "late_mean": late_mean, "late_std": late_std, "n_tail_points": n_tail,
            "best": float(df["success_rate"].max()),
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
            raise ValueError(f"config: mode must be one of {_MODES}, got {m!r}")

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
            late_mean, late_std, n_tail = _late_stats(df_existing, y.get("late_window_fraction", 0.25))
            results.append(
                {
                    **combo, "prefix": prefix,
                    "late_mean": late_mean, "late_std": late_std, "n_tail_points": n_tail,
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
        clip_eps=y["clip_eps"], gae_lambda=y["gae_lambda"], beta=y["beta"],
        beta_init=y.get("beta_init", y["beta"]), beta_target=y.get("beta_target", 0.99999),
        schedule_step_frac=y.get("schedule_step_frac", 0.1), schedule_history_len=y.get("schedule_history_len", 3),
        kl_k=y.get("kl_k", 0.10), kernel=y.get("kernel", "gaussian"), bandwidth_state=y.get("bandwidth_state", 1e-6),
        bandwidth_action=y.get("bandwidth_action", 1e-6), hidden_sizes=y.get("hidden_sizes", [64, 64]),
        eval_episodes=y["eval_episodes"], eval_seed=y["eval_seed"], success_rate_thresholds=thresholds,
        late_window_fraction=y.get("late_window_fraction", 0.25),
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
                print(f"[{i}/{len(pending)} {combo['prefix']}] Done in {dt:.1f} min -- late_mean={res['late_mean']:.4f} best={res['best']:.4f}")
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
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: done -- late_mean={res['late_mean']:.4f} best={res['best']:.4f}  [{elapsed:.1f} min elapsed]")

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    n_failed = int(summary_df["status"].astype(str).str.startswith("FAILED").sum()) if len(summary_df) else 0
    print(f"\n=== Sweep complete in {(time.time() - t_start) / 60:.1f} min total ({n_failed} failed) ===")
    ok = summary_df[summary_df["status"].isin(["ran", "skipped (already existed)"])] if not summary_df.empty else summary_df
    if not ok.empty:
        crossing_cols = [c for c in ok.columns if c.startswith("window_at_")]
        print("\n=== Every run ===")
        print(ok.sort_values(["lr", "mode", "seed"])[["mode", "lr", "seed"] + crossing_cols + ["late_mean", "late_std", "best", "best_window", "final"]].to_string(index=False))

        if ok["seed"].nunique() > 1:
            print("\n=== Aggregated across seeds, by mode and learning rate ===")
            agg = ok.groupby(["lr", "mode"])[["late_mean", "best"]].agg(["mean", "std", "count"])
            print(agg.round(4).to_string())
    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
