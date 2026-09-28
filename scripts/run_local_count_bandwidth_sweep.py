"""Bandwidth sweep for LocalCountFixedDPPOTrainer
(fixed_d_trainer_local_count.py): does replacing the exact (state, action)
count with a kernel-smoothed LOCAL count -- the natural continuous analogue
of counting -- keep (or improve on) the validated count-based mechanism, and
how does that depend on the kernel bandwidth h?

Everything except the count is held at this project's practical
configuration (clip_eps=0.55, gae_lambda=0.90, entropy_coef=0.0,
value_coef=1.0, max_grad_norm=0.1; effective-sample weighting with
beta=0.9995, KL-anchored, k=0.10): only the number handed to
w(n)/n changes, so any difference from the exact-count result is
attributable to the local count alone.

h is in STANDARD DEVIATIONS of the state features (standardized over the
distinct observed states). The script prints the dataset's nearest-neighbor
scale at startup -- the natural yardstick: h far below it pools nothing,
h around it merges immediate neighbors, h far above it merges regions.

SANITY CHECK BUILT INTO THE GRID: as h -> 0 the local count IS the exact
count (verified bit-for-bit on this project's maze), so a tiny bandwidth
(default grid includes 1e-6) must reproduce the exact-count run exactly --
mean=0.4289 at seed 0 for this configuration. If it does not, something in
the pipeline is wrong and the other bandwidths should not be trusted.

Two-phase use, same convention as run_variance_weighting_tau_sweep.py:
  1. Coarse scan, ONE seed, several bandwidths -- find a promising region
     (default grid below).
  2. Multi-seed validation of the promising bandwidth(s): --seeds 0 1 2 3 4
     --bandwidths <winner>. NOTE the local count itself is deterministic, so
     seeds here only change minibatch shuffling, unlike the ensemble
     variants where they also changed the signal.

Per-combination diagnostics are recorded next to each score, because they
say WHAT the bandwidth did: median/mean/max of n_local / n_exact and the
fraction of transitions whose count changed at all.

Parallel execution, resumability, per-combination logging and worker setup
mirror run_variance_weighting_tau_sweep.py / analyze_sweep.py (spawn
context, one env/dataset/prior load per WORKER, --cpu-count clamped to
pending work and os.cpu_count(), 30 s heartbeat, one .log per combination,
skip combinations whose .csv already exists unless --force). No
checkpoints are saved.

Usage:
    python scripts/run_local_count_bandwidth_sweep.py \
        --env-config configs/phase2/env_maze.yaml \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --seeds 0 --bandwidths 1e-6 0.05 0.1 0.2 0.4 0.8 \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase3/analysis/local_count_bandwidth_sweep \
        --cpu-count 4
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

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy_weighted, get_tier_start_lists
from ppo_exploitation.ppo.fixed_d_trainer_local_count import LocalCountFixedDPPOTrainer, nearest_neighbor_scale
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


def _run_one_combo(task: dict) -> dict:
    prefix = task["prefix"]
    out_dir = Path(task["out_dir"])
    log_path = out_dir / f"{prefix}.log"
    seed = task["seed"]
    bandwidth = task["bandwidth"]
    try:
        set_global_seed(seed)
        env = _WORKER["env"]
        dataset = _WORKER["dataset"]

        cfg = PPOHyperparams(
            epochs=task["epochs"], minibatch_size=task["minibatch_size"], clip_eps=task["clip_eps"],
            gae_lambda=task["gae_lambda"], entropy_coef=0.0, value_coef=1.0, max_grad_norm=0.1,
            hidden_sizes=_WORKER["hidden_sizes"], lr=task["lr"], seed=seed,
            use_effective_sample_weighting=True, effective_sample_beta=task["beta"],
            effective_sample_kl_anchor=True, effective_sample_kl_k=task["kl_k"],
        )

        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            trainer = LocalCountFixedDPPOTrainer(
                dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg,
                prior_state_dict=_WORKER["prior_state_dict"], bandwidth=bandwidth, kernel=task["kernel"],
            )
            ratio = trainer._n_per_transition / trainer.exact_n_per_transition
            diag = {
                "median_ratio": float(np.median(ratio)), "mean_ratio": float(ratio.mean()),
                "max_ratio": float(ratio.max()), "frac_changed": float((ratio > 1 + 1e-9).mean()),
            }
            print(f"bandwidth={bandwidth}  n_local/n_exact: " + "  ".join(f"{k}={v:.4g}" for k, v in diag.items()))

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

        return {
            "seed": seed, "bandwidth": bandwidth, "kernel": task["kernel"],
            "mean": float(df["weighted_success_rate"].mean()), "best": float(df["weighted_success_rate"].max()),
            "best_epoch": int(df.loc[df["weighted_success_rate"].idxmax(), "epoch"]),
            "final": float(df["weighted_success_rate"].iloc[-1]), "std": float(df["weighted_success_rate"].std()),
            **diag, "csv_path": str(csv_path), "status": "ran", "log_path": str(log_path),
        }
    except Exception as e:
        with open(log_path, "a") as logf:
            logf.write(f"\n[ERROR] {e}\n{traceback.format_exc()}\n")
        return {"seed": seed, "bandwidth": bandwidth, "kernel": task["kernel"], "status": f"FAILED: {e}",
                "csv_path": None, "log_path": str(log_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--start-tiers-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--bandwidths", type=float, nargs="+", default=[1e-6, 0.05, 0.1, 0.2, 0.4, 0.8],
        help="Kernel bandwidths h, in standard deviations of the state features. Include a tiny value "
        "(default 1e-6) as the built-in sanity check: it must reproduce the exact-count run.",
    )
    parser.add_argument("--kernel", choices=["gaussian", "ball"], default="gaussian")
    parser.add_argument("--beta", type=float, default=0.9995, help="effective_sample_beta (this project's validated value).")
    parser.add_argument("--kl-k", type=float, default=0.1, help="effective_sample_kl_k (k=0.10, this project's practical choice).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-eps", type=float, default=0.55)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--out-dir", default="results/phase3/analysis/local_count_bandwidth_sweep")
    parser.add_argument("--force", action="store_true", help="Re-run combinations even if their CSV already exists.")
    parser.add_argument("--cpu-count", type=int, default=1)
    args = parser.parse_args()

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

    scale_dataset = load_dataset(args.dataset)
    obs_all = np.concatenate([tr.obs for tr in scale_dataset.trajectories])
    print(
        f"Bandwidth yardstick: median nearest-neighbor distance between distinct states = "
        f"{nearest_neighbor_scale(obs_all):.4f} (standardized units). h far below it pools nothing; "
        f"h around it merges immediate neighbors; h far above it merges whole regions."
    )
    del scale_dataset, obs_all

    combos = [{"seed": s, "bandwidth": h} for h in args.bandwidths for s in args.seeds]
    print(f"=== {len(combos)} combination(s): {len(args.seeds)} seed(s) x {len(args.bandwidths)} bandwidth(s) "
          f"(kernel={args.kernel}, beta={args.beta}, k={args.kl_k}) ===")

    def fmt(x):
        return f"{x:g}".replace(".", "_").replace("-", "m").replace("+", "")

    results, pending = [], []
    for combo in combos:
        prefix = f"seed_{combo['seed']}_h_{fmt(combo['bandwidth'])}"
        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            print(f"[{prefix}] SKIPPING -- {csv_path} already exists (use --force to re-run)")
            df_existing = pd.read_csv(csv_path)
            results.append(
                {
                    **combo, "kernel": args.kernel, "prefix": prefix,
                    "mean": float(df_existing["weighted_success_rate"].mean()),
                    "best": float(df_existing["weighted_success_rate"].max()),
                    "best_epoch": int(df_existing.loc[df_existing["weighted_success_rate"].idxmax(), "epoch"]),
                    "final": float(df_existing.iloc[-1]["weighted_success_rate"]),
                    "std": float(df_existing["weighted_success_rate"].std()),
                    "csv_path": str(csv_path), "status": "skipped (already existed)",
                }
            )
        else:
            pending.append({**combo, "prefix": prefix})

    t_start = time.time()
    common = dict(
        out_dir=str(out_dir), epochs=args.epochs, minibatch_size=args.minibatch_size, clip_eps=args.clip_eps,
        gae_lambda=args.gae_lambda, lr=args.lr, beta=args.beta, kl_k=args.kl_k, kernel=args.kernel,
        checkpoint_every=args.checkpoint_every, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed,
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
                print(f"[{i}/{len(pending)} {combo['prefix']}] Done in {dt:.1f} min -- mean={res['mean']:.4f} best={res['best']:.4f} final={res['final']:.4f}")
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
                        res = {"seed": t["seed"], "bandwidth": t["bandwidth"], "kernel": t["kernel"], "status": f"FAILED: {e}",
                               "csv_path": None, "log_path": f"{out_dir}/{t['prefix']}.log"}
                    results.append(res)
                    if str(res.get("status", "")).startswith("FAILED"):
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: FAILED -- {res['status']} (see {res.get('log_path')})  [{elapsed:.1f} min elapsed]")
                    else:
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: done -- mean={res['mean']:.4f} best={res['best']:.4f} "
                              f"final={res['final']:.4f}  [{elapsed:.1f} min elapsed]")

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    n_failed = int(summary_df["status"].astype(str).str.startswith("FAILED").sum()) if len(summary_df) else 0
    print(f"\n=== Sweep complete in {(time.time() - t_start) / 60:.1f} min total ({n_failed} failed) ===")
    ok = summary_df[summary_df["status"].isin(["ran", "skipped (already existed)"])] if not summary_df.empty else summary_df
    if not ok.empty:
        print("\n=== Per-bandwidth mean weighted_success_rate, averaged over seeds ===")
        grouped = ok.groupby("bandwidth")[["mean", "best", "final"]].mean()
        grouped["n_seeds"] = ok.groupby("bandwidth")["mean"].count()
        print(grouped.sort_index().to_string())
    print(f"\nSaved combined summary to {summary_path}")
    print("REFERENCE (baseline, no weighting): mean=0.4127")
    print("REFERENCE (exact count, KL-anchored, k=0.10, seed 0 -- what bandwidth->0 must reproduce): mean=0.4289 best=0.4845 final=0.4705")


if __name__ == "__main__":
    main()
