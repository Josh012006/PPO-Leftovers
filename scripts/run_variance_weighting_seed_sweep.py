"""Seed x ensemble-size robustness sweep for DirectVarianceWeightedTrainer
(fixed_d_trainer_variance_weighted.py), on the real project maze --
answers "does this method converge to good values reliably, or did we get
lucky/unlucky on one seed?" before proposing it as a real contribution.

Motivated directly by run_direct_variance_comparison.py's own two
single-seed runs giving very different results (mean=0.4306 then
mean=0.3779) BEFORE that script's missing set_global_seed call was fixed
-- i.e. before this sweep even existed, the method had already shown it
can land on either side of the established baseline (mean=0.4127)
depending on the ensemble's random initialization alone. A quick check
found more ensemble heads reduces that seed-to-seed noise (median
coefficient of variation in per-transition variance across seeds: 0.449
at 5 heads, 0.295 at 15 heads, on a 20k-transition subsample) -- this
sweep tests that properly, at full scale, across ensemble_n_heads in
{5, 10, 15} x 5 seeds = 15 runs, and reports the full mean/best/final
distribution per head-count so a real decision (is K=15 enough? do we
need more?) can be made from data, not a single anecdote.

Everything else is held fixed at this project's established best
configuration for the underlying PPO hyperparameters (clip_eps=0.55,
gae_lambda=0.90, entropy_coef=0.0, value_coef=1.0, max_grad_norm=0.1) and
for the direct-variance mechanism itself (tau_kl_anchor=True,
tau_kl_k=0.1, tau_quantile=0.5) -- only `seed` (which also seeds the
ensemble's own random initialization, via set_global_seed) and
ensemble_n_heads vary.

NO CHECKPOINTS ARE SAVED (per explicit request -- this sweep is purely
about characterizing the mean/best/final DISTRIBUTION across seeds; if a
head-count/seed combination turns out worth inspecting further, re-run it
individually with scripts/run_direct_variance_comparison.py
--save-best-checkpoint).

Parallel execution, resumability, per-combination logging, and worker
setup all mirror scripts/analyze_sweep.py exactly (see that file's own
module docstring for the full rationale of each piece) -- spawn context,
one env/dataset/prior-checkpoint load per WORKER (not per combination),
--cpu-count clamped to pending combinations and os.cpu_count(), a
30-second heartbeat, per-combination .log files so concurrent workers
never interleave console output, resuming by skipping any combination
whose <prefix>.csv already exists (--force to override).

Usage:
    python scripts/run_variance_weighting_seed_sweep.py \
        --env-config configs/phase2/env_maze.yaml \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --seeds 0 1 2 3 4 --ensemble-n-heads-list 5 10 15 \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase2/analysis/variance_weighting_seed_sweep \
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

import pandas as pd
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy_weighted, get_tier_start_lists
from ppo_exploitation.ppo.fixed_d_trainer_variance_weighted import DirectVarianceWeightedTrainer
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
    ensemble_n_heads = task["ensemble_n_heads"]
    try:
        set_global_seed(seed)
        env = _WORKER["env"]
        dataset = _WORKER["dataset"]

        cfg = PPOHyperparams(
            epochs=task["epochs"], minibatch_size=task["minibatch_size"], clip_eps=task["clip_eps"],
            gae_lambda=task["gae_lambda"], entropy_coef=0.0, value_coef=1.0, max_grad_norm=0.1,
            hidden_sizes=_WORKER["hidden_sizes"], lr=task["lr"], seed=seed,
        )

        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            trainer = DirectVarianceWeightedTrainer(
                dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg,
                prior_state_dict=_WORKER["prior_state_dict"], ensemble_n_heads=ensemble_n_heads,
                ensemble_hidden_sizes=tuple(task["ensemble_hidden_sizes"]), ensemble_epochs=task["ensemble_epochs"],
                ensemble_lr=task["ensemble_lr"], tau_quantile=task["tau_quantile"],
                tau_kl_anchor=task["tau_kl_anchor"], tau_kl_k=task["tau_kl_k"],
            )
            print(
                f"tau={trainer.tau:.6f}  weights: min={trainer.policy_loss_weights.min():.4f} "
                f"max={trainer.policy_loss_weights.max():.4f} mean={trainer.policy_loss_weights.mean():.4f}"
            )

            results = []

            def eval_cb(epoch, net, summary):
                r = evaluate_policy_weighted(
                    env, lambda o, s: net.act_numpy(o, deterministic=True)[0], n_episodes=task["eval_episodes"],
                    seed=task["eval_seed"], covered_starts=_WORKER["covered_starts"], held_out_starts=_WORKER["held_out_starts"],
                )
                results.append({"epoch": epoch, **r})
                print(f"  epoch {epoch:4d}: weighted={r['weighted_success_rate']:.4f}")

            trainer.train(verbose=True, eval_every_epochs=task["checkpoint_every"], eval_callback=eval_cb)

        df = pd.DataFrame(results)
        csv_path = out_dir / f"{prefix}.csv"
        df.to_csv(csv_path, index=False)

        summary = {
            "seed": seed, "ensemble_n_heads": ensemble_n_heads,
            "mean": float(df["weighted_success_rate"].mean()), "best": float(df["weighted_success_rate"].max()),
            "best_epoch": int(df.loc[df["weighted_success_rate"].idxmax(), "epoch"]),
            "final": float(df["weighted_success_rate"].iloc[-1]), "std": float(df["weighted_success_rate"].std()),
            "csv_path": str(csv_path), "status": "ran", "log_path": str(log_path),
        }
        return summary
    except Exception as e:
        with open(log_path, "a") as logf:
            logf.write(f"\n[ERROR] {e}\n{traceback.format_exc()}\n")
        return {
            "seed": seed, "ensemble_n_heads": ensemble_n_heads, "status": f"FAILED: {e}",
            "csv_path": None, "log_path": str(log_path),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--start-tiers-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ensemble-n-heads-list", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-eps", type=float, default=0.55)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--ensemble-hidden-sizes", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--ensemble-epochs", type=int, default=30)
    parser.add_argument("--ensemble-lr", type=float, default=1e-3)
    parser.add_argument("--tau-quantile", type=float, default=0.5)
    parser.add_argument("--tau-kl-k", type=float, default=0.1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--out-dir", default="results/phase2/analysis/variance_weighting_seed_sweep")
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

    combos = [
        {"seed": s, "ensemble_n_heads": k}
        for k in args.ensemble_n_heads_list
        for s in args.seeds
    ]
    print(f"=== {len(combos)} combination(s): {len(args.seeds)} seeds x {len(args.ensemble_n_heads_list)} ensemble sizes ===")

    results = []
    pending = []
    for combo in combos:
        prefix = f"seed_{combo['seed']}_heads_{combo['ensemble_n_heads']}"
        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            print(f"[{prefix}] SKIPPING -- {csv_path} already exists (use --force to re-run)")
            df_existing = pd.read_csv(csv_path)
            results.append(
                {
                    **combo, "prefix": prefix,
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

    if args.cpu_count <= 1 or len(pending) <= 1:
        _init_worker(env_cfg_dict, eval_env.layout, args.dataset, args.prior_checkpoint, covered_starts, held_out_starts)
        for i, combo in enumerate(pending, start=1):
            print(f"\n[{i}/{len(pending)} {combo['prefix']}] Starting -- seed={combo['seed']} ensemble_n_heads={combo['ensemble_n_heads']}")
            t0 = time.time()
            res = _run_one_combo(
                {
                    **combo, "out_dir": str(out_dir), "epochs": args.epochs, "minibatch_size": args.minibatch_size,
                    "clip_eps": args.clip_eps, "gae_lambda": args.gae_lambda, "lr": args.lr,
                    "ensemble_hidden_sizes": args.ensemble_hidden_sizes, "ensemble_epochs": args.ensemble_epochs,
                    "ensemble_lr": args.ensemble_lr, "tau_quantile": args.tau_quantile, "tau_kl_anchor": True,
                    "tau_kl_k": args.tau_kl_k, "checkpoint_every": args.checkpoint_every,
                    "eval_episodes": args.eval_episodes, "eval_seed": args.eval_seed,
                }
            )
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

        tasks = [
            {
                **combo, "out_dir": str(out_dir), "epochs": args.epochs, "minibatch_size": args.minibatch_size,
                "clip_eps": args.clip_eps, "gae_lambda": args.gae_lambda, "lr": args.lr,
                "ensemble_hidden_sizes": args.ensemble_hidden_sizes, "ensemble_epochs": args.ensemble_epochs,
                "ensemble_lr": args.ensemble_lr, "tau_quantile": args.tau_quantile, "tau_kl_anchor": True,
                "tau_kl_k": args.tau_kl_k, "checkpoint_every": args.checkpoint_every,
                "eval_episodes": args.eval_episodes, "eval_seed": args.eval_seed,
            }
            for combo in pending
        ]

        print(
            f"\nRunning {len(pending)} pending combination(s) across {cpu_count} worker process(es). "
            f"Per-epoch logs go to <out_dir>/<prefix>.log, not this console.\n"
        )
        n_done = 0
        with ProcessPoolExecutor(
            max_workers=cpu_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(env_cfg_dict, eval_env.layout, args.dataset, args.prior_checkpoint, covered_starts, held_out_starts),
        ) as executor:
            futures = {executor.submit(_run_one_combo, t): t for t in tasks}
            pending_futures = set(futures.keys())
            heartbeat_seconds = 30
            while pending_futures:
                done, pending_futures = wait(pending_futures, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
                if not done:
                    elapsed = (time.time() - t_start) / 60
                    print(f"... still running ({n_done}/{len(pending)} done, {elapsed:.1f} min elapsed)", flush=True)
                    continue
                for fut in done:
                    t = futures[fut]
                    n_done += 1
                    elapsed = (time.time() - t_start) / 60
                    try:
                        res = fut.result()
                    except Exception as e:  # pragma: no cover -- _run_one_combo already catches its own exceptions
                        res = {"seed": t["seed"], "ensemble_n_heads": t["ensemble_n_heads"], "status": f"FAILED: {e}", "csv_path": None, "log_path": f"{out_dir}/{t['prefix']}.log"}
                    results.append(res)
                    if str(res.get("status", "")).startswith("FAILED"):
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: FAILED -- {res['status']} (see {res.get('log_path')})  [{elapsed:.1f} min elapsed]")
                    else:
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: done -- mean={res['mean']:.4f} best={res['best']:.4f} final={res['final']:.4f}  [{elapsed:.1f} min elapsed]")

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    total_min = (time.time() - t_start) / 60
    n_failed = int(summary_df["status"].astype(str).str.startswith("FAILED").sum()) if len(summary_df) else 0
    print(f"\n=== Sweep complete in {total_min:.1f} min total ({n_failed} failed) ===")
    if not summary_df.empty and "ensemble_n_heads" in summary_df.columns:
        ok = summary_df[summary_df["status"] == "ran"]
        if not ok.empty:
            print("\n=== Per-ensemble-size distribution across seeds ===")
            grouped = ok.groupby("ensemble_n_heads")["mean"].agg(["mean", "std", "min", "max"])
            grouped.columns = ["mean_of_means", "std_of_means", "min_mean", "max_mean"]
            print(grouped.to_string())
    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()
