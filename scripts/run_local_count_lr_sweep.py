"""Learning-rate sweep for LocalCountFixedDPPOTrainer
(fixed_d_trainer_local_count.py), at near-exact bandwidths
(bandwidth_state, bandwidth_action ~ 0, this project's maze being discrete
with unordered actions -- see the README, "Counting locally instead of
exactly"). This is NOT a test of the local count itself: at these
bandwidths it reduces to the already-validated exact-count mechanism. The
question here is purely whether a higher learning rate reaches a good
weighted_success_rate SOONER, before running a full, expensive sweep at the
current default lr=0.0003 that all this project's other results share.

Unlike every other sweep script in this project, the metric of interest is
NOT the run's mean over the full 300 epochs: it is the EPOCH at which
weighted_success_rate first crosses each of a few thresholds (read from the
YAML config, e.g. 0.35 / 0.40 / 0.42). mean/best/final are still recorded
for reference, but the printed summary leads with the crossing epochs.

Reads all settings from a YAML config (see configs/phase3/
local_count_lr_sweep.yaml) except --seeds, --out-dir, --cpu-count and
--force, which stay as command-line flags, matching this project's other
sweep scripts.

Parallel execution, resumability, per-combination logging and worker setup
mirror run_local_count_bandwidth_sweep.py / analyze_sweep.py (spawn
context, one env/dataset/prior load per WORKER, --cpu-count clamped to
pending work and os.cpu_count(), 30 s heartbeat, one .log per combination,
skip combinations whose .csv already exists unless --force). No
checkpoints are saved.

Usage:
    python scripts/run_local_count_lr_sweep.py \
        --config configs/phase3/local_count_lr_sweep.yaml \
        --env-config configs/phase2/env_maze.yaml \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --seeds 0 \
        --out-dir results/phase3/analysis/local_count_lr_sweep \
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

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy_weighted, get_tier_start_lists
from ppo_exploitation.ppo.fixed_d_trainer_local_count import LocalCountFixedDPPOTrainer
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams, StartTierConfig
from ppo_exploitation.utils.seeding import set_global_seed

_WORKER: dict = {}


def _init_worker(env_cfg_dict, layout, dataset_path, prior_checkpoint_path, covered_starts, held_out_starts):
    env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    dataset = load_dataset(dataset_path)
    ckpt = torch.load(prior_checkpoint_path, map_location="cpu", weights_only=False)
    _WORKER.update(
        env=env, dataset=dataset, prior_state_dict=ckpt["state_dict"],
        hidden_sizes=tuple(ckpt.get("hidden_sizes", (64, 64))),
        covered_starts=covered_starts, held_out_starts=held_out_starts,
    )


def _first_crossing_epoch(df: pd.DataFrame, threshold: float) -> int | None:
    hit = df[df["weighted_success_rate"] >= threshold]
    return int(hit["epoch"].iloc[0]) if len(hit) else None


def _run_one_combo(task: dict) -> dict:
    prefix = task["prefix"]
    out_dir = Path(task["out_dir"])
    log_path = out_dir / f"{prefix}.log"
    seed = task["seed"]
    lr = task["lr"]
    try:
        set_global_seed(seed)
        env = _WORKER["env"]
        dataset = _WORKER["dataset"]

        cfg = PPOHyperparams(
            epochs=task["epochs"], minibatch_size=task["minibatch_size"], clip_eps=task["clip_eps"],
            gae_lambda=task["gae_lambda"], entropy_coef=0.0, value_coef=1.0, max_grad_norm=0.1,
            hidden_sizes=_WORKER["hidden_sizes"], lr=lr, seed=seed,
            use_effective_sample_weighting=True, effective_sample_beta=task["beta"],
            effective_sample_kl_anchor=True, effective_sample_kl_k=task["kl_k"],
        )

        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            trainer = LocalCountFixedDPPOTrainer(
                dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg,
                prior_state_dict=_WORKER["prior_state_dict"], bandwidth_state=task["bandwidth_state"],
                bandwidth_action=task["bandwidth_action"], kernel=task["kernel"],
            )
            print(f"lr={lr}")

            results = []

            def eval_cb(epoch, net, summary):
                r = evaluate_policy_weighted(
                    env, lambda o, s: net.act_numpy(o, deterministic=True)[0], n_episodes=task["eval_episodes"],
                    seed=task["eval_seed"], covered_starts=_WORKER["covered_starts"], held_out_starts=_WORKER["held_out_starts"],
                )
                results.append({"epoch": epoch, **r})
                print(f"  epoch {epoch:4d}: weighted={r['weighted_success_rate']:.4f}  approx_kl={summary['approx_kl']:.4f}")

            # theta == pi_beta exactly here, before any training step --
            # matches scripts/_analysis_lib.py's own epoch-0 evaluation, so
            # every mean in this project is computed over the same epoch
            # range (0..epochs).
            eval_cb(0, trainer.net, {"approx_kl": 0.0})

            trainer.train(verbose=True, eval_every_epochs=task["checkpoint_every"], eval_callback=eval_cb)

        df = pd.DataFrame(results)
        csv_path = out_dir / f"{prefix}.csv"
        df.to_csv(csv_path, index=False)

        crossings = {f"epoch_at_{t:g}": _first_crossing_epoch(df, t) for t in task["success_rate_thresholds"]}
        return {
            "seed": seed, "lr": lr,
            "mean": float(df["weighted_success_rate"].mean()), "best": float(df["weighted_success_rate"].max()),
            "best_epoch": int(df.loc[df["weighted_success_rate"].idxmax(), "epoch"]),
            "final": float(df["weighted_success_rate"].iloc[-1]),
            **crossings, "csv_path": str(csv_path), "status": "ran", "log_path": str(log_path),
        }
    except Exception as e:
        with open(log_path, "a") as logf:
            logf.write(f"\n[ERROR] {e}\n{traceback.format_exc()}\n")
        return {"seed": seed, "lr": lr, "status": f"FAILED: {e}", "csv_path": None, "log_path": str(log_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/phase3/local_count_lr_sweep.yaml")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--start-tiers-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--out-dir", default="results/phase3/analysis/local_count_lr_sweep")
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
    eval_env = StochasticMazeEnv(**env_cfg_dict)
    tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
    covered_starts, held_out_starts = get_tier_start_lists(eval_env, tier_cfg)
    print(f"Loaded start tiers: {len(covered_starts)} covered, {len(held_out_starts)} held-out.")

    thresholds = y["success_rate_thresholds"]
    combos = [{"seed": s, "lr": lr} for lr in y["learning_rates"] for s in args.seeds]
    print(f"=== {len(combos)} combination(s): {len(args.seeds)} seed(s) x {len(y['learning_rates'])} learning rate(s) "
          f"(bandwidth_state={y['bandwidth_state']}, bandwidth_action={y['bandwidth_action']}) ===")

    def fmt(x):
        return f"{x:g}".replace(".", "_")

    results, pending = [], []
    for combo in combos:
        prefix = f"seed_{combo['seed']}_lr_{fmt(combo['lr'])}"
        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            print(f"[{prefix}] SKIPPING -- {csv_path} already exists (use --force to re-run)")
            df_existing = pd.read_csv(csv_path)
            crossings = {f"epoch_at_{t:g}": _first_crossing_epoch(df_existing, t) for t in thresholds}
            results.append(
                {
                    **combo, "prefix": prefix,
                    "mean": float(df_existing["weighted_success_rate"].mean()),
                    "best": float(df_existing["weighted_success_rate"].max()),
                    "best_epoch": int(df_existing.loc[df_existing["weighted_success_rate"].idxmax(), "epoch"]),
                    "final": float(df_existing.iloc[-1]["weighted_success_rate"]),
                    **crossings, "csv_path": str(csv_path), "status": "skipped (already existed)",
                }
            )
        else:
            pending.append({**combo, "prefix": prefix})

    t_start = time.time()
    common = dict(
        out_dir=str(out_dir), epochs=y["epochs"], minibatch_size=args.minibatch_size, clip_eps=y["clip_eps"],
        gae_lambda=y["gae_lambda"], beta=y["beta"], kl_k=y["kl_k"], kernel=y["kernel"],
        bandwidth_state=y["bandwidth_state"], bandwidth_action=y["bandwidth_action"],
        checkpoint_every=y["checkpoint_every"], eval_episodes=y["eval_episodes"], eval_seed=y["eval_seed"],
        success_rate_thresholds=thresholds,
    )

    if args.cpu_count <= 1 or len(pending) <= 1:
        _init_worker(env_cfg_dict, eval_env.layout, args.dataset, args.prior_checkpoint, covered_starts, held_out_starts)
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
              f"Per-epoch logs go to <out_dir>/<prefix>.log, not this console.\n")
        n_done = 0
        with ProcessPoolExecutor(
            max_workers=cpu_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(env_cfg_dict, eval_env.layout, args.dataset, args.prior_checkpoint, covered_starts, held_out_starts),
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
                        res = {"seed": t["seed"], "lr": t["lr"], "status": f"FAILED: {e}", "csv_path": None, "log_path": f"{out_dir}/{t['prefix']}.log"}
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
        crossing_cols = [c for c in ok.columns if c.startswith("epoch_at_")]
        print("\n=== Epoch of first crossing each threshold, by learning rate (lower = faster) ===")
        print(ok.sort_values("lr")[["lr", "seed"] + crossing_cols + ["mean", "best", "best_epoch", "final"]].to_string(index=False))
    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
