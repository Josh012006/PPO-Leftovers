"""Cost/performance tradeoff for effective-sample-count weighting of the
policy loss (see README, "An attempt to close the gap" -- the idea
motivated by state 717, where the sparser of two competing actions is
genuinely better but gets outvoted by raw sample volume).

Same YAML-driven design as scripts/analyze_sweep.py, reusing its
combination-parsing helpers directly: ONE config where every
PPOHyperparams field is fixed at the best checkpoint's values EXCEPT
`effective_sample_beta`, swept as a list (see configs/phase2/
ppo_fixed_d_effective_sample_sweep.yaml). `use_effective_sample_
weighting` should be fixed to `true` in that file -- the standard-PPO
BASELINE (weighting off) is added automatically by this script as one
extra combination, not part of the YAML grid, so the same config never
needs a redundant "off" variant hand-added to the beta list.

Per-epoch UPDATE cost is measured in its own dedicated pass for every
combination (baseline included): one FixedDPPOTrainer.train() call with
NO eval_callback, timed end to end and divided by cfg.epochs -- this
isolates the actual overhead the new weighting adds to a gradient step
from live-rollout evaluation cost, which is timed and reported
separately. Performance (mean/best/std/final weighted_success_rate)
comes from a SEPARATE run through scripts/_analysis_lib.py:
run_single_analysis, exactly as scripts/analyze_sweep.py reports it --
training runs twice per combination as a result (once for clean timing,
once for the standard eval curve), which this project's fast fixed-D
training (no env.step() calls at all -- D is read-only, see
fixed_d_trainer.py) makes cheap enough to be worth the clarity of not
conflating the two measurements.

Parallel execution (--cpu-count) and resumability (--force) work
identically to scripts/analyze_sweep.py -- same spawn-context
ProcessPoolExecutor, same per-combination log files, same heartbeat,
same clamping. See that script's module docstring for the full
rationale; not repeated here.

Outputs, under --out-dir (default results/phase2/analysis/effective_
sample_weighting/):
  one <prefix>.csv/.log and matching _success_return/_weighted .svg/.png
  PER COMBINATION (baseline included), plus:
  effective_sample_weighting_summary.csv -- one row per combination:
    beta (empty for the baseline), seconds_per_epoch,
    best/mean/std/final weighted_success_rate, csv path

Usage:
    python scripts/analyze_effective_sample_weighting.py \
        --env-config configs/phase2/env_maze.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --pi-d-star-empirical results/phase2/pi_d_star_empirical.pkl \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --sweep-config configs/phase2/ppo_fixed_d_effective_sample_sweep.yaml \
        --base-prefix eff_sample \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase2/analysis/effective_sample_weighting \
        --cpu-count 4
"""
from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
import yaml

from analyze_sweep import generate_combinations, make_combo_prefix, parse_sweep_yaml
from _analysis_lib import compute_ceiling_success_rates, run_single_analysis
from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import DEFAULT_EVAL_WEIGHTS, get_tier_start_lists
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams, StartTierConfig
from ppo_exploitation.utils.seeding import set_global_seed


def time_pure_training(dataset, obs_dim, n_actions, cfg, prior_state_dict) -> float:
    """Seconds per epoch of the training loop ALONE -- no evaluation, no
    plotting, nothing but trainer.train(). A fresh trainer every call
    (theta must start at pi_beta's weights either way, so this matches
    exactly what a real run does)."""
    set_global_seed(cfg.seed)
    trainer = FixedDPPOTrainer(dataset, obs_dim=obs_dim, n_actions=n_actions, cfg=cfg, prior_state_dict=prior_state_dict)
    t0 = time.time()
    trainer.train(verbose=False, eval_every_epochs=None, eval_callback=None)
    dt = time.time() - t0
    return dt / cfg.epochs


# --------------------------------------------------------------------------
# Parallel worker machinery -- same shape as scripts/analyze_sweep.py's,
# extended to also run the dedicated timing pass per combination.
# --------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(env_cfg_dict, layout, dataset_path, prior_state_dict, ceiling_success_rates, covered_starts, held_out_starts):
    env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    dataset = load_dataset(dataset_path)
    _WORKER.update(
        env=env, dataset=dataset, prior_state_dict=prior_state_dict,
        ceiling_success_rates=ceiling_success_rates, covered_starts=covered_starts, held_out_starts=held_out_starts,
    )


def _run_one_combo(task: dict) -> dict:
    prefix = task["prefix"]
    out_dir = Path(task["out_dir"])
    log_path = out_dir / f"{prefix}.log"
    beta = task["combo"].get("effective_sample_beta") if task["combo"].get("use_effective_sample_weighting") else None
    base = {"beta": beta}
    try:
        cfg = PPOHyperparams(**task["combo"])
        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            print(f"Timing pure training (no eval) for {prefix}...")
            seconds_per_epoch = time_pure_training(
                _WORKER["dataset"], _WORKER["dataset"].obs_dim, _WORKER["dataset"].n_actions, cfg, _WORKER["prior_state_dict"]
            )
            print(f"  {seconds_per_epoch:.4f} s/epoch")
            set_global_seed(cfg.seed)
            summary = run_single_analysis(
                eval_env=_WORKER["env"],
                dataset=_WORKER["dataset"],
                prior_state_dict=_WORKER["prior_state_dict"],
                ceiling_success_rates=_WORKER["ceiling_success_rates"],
                cfg=cfg,
                checkpoint_every=task["checkpoint_every"],
                eval_episodes=task["eval_episodes"],
                eval_seed=task["eval_seed"],
                out_dir=out_dir,
                prefix=prefix,
                covered_starts=_WORKER["covered_starts"],
                held_out_starts=_WORKER["held_out_starts"],
                weights=task["weights"],
                title_suffix=task["combo_desc"],
                verbose=True,
                log_prefix=task["log_tag"],
            )
        return {**base, "seconds_per_epoch": seconds_per_epoch, **summary, "status": "ran", "log_path": str(log_path)}
    except Exception as e:
        with open(log_path, "a") as logf:
            logf.write(f"\n[ERROR] {e}\n{traceback.format_exc()}\n")
        return {**base, "prefix": prefix, "status": f"FAILED: {e}", "csv_path": None, "log_path": str(log_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument("--dataset", default="results/phase2/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/phase2/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star-empirical", default="results/phase2/pi_d_star_empirical.pkl")
    parser.add_argument("--start-tiers-config", default="configs/phase2/start_tiers.yaml")
    parser.add_argument(
        "--eval-weights", type=float, nargs=3, default=list(DEFAULT_EVAL_WEIGHTS),
        metavar=("OVERALL", "COVERED", "HELD_OUT"),
    )
    parser.add_argument(
        "--sweep-config",
        default="configs/phase2/ppo_fixed_d_effective_sample_sweep.yaml",
        help="Full PPOHyperparams YAML, effective_sample_beta as the swept list, "
        "use_effective_sample_weighting fixed to true (see that file's own comments).",
    )
    parser.add_argument("--base-prefix", default="eff_sample")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--out-dir", default="results/phase2/analysis/effective_sample_weighting")
    parser.add_argument("--force", action="store_true", help="Re-run combinations even if their CSV already exists.")
    parser.add_argument(
        "--cpu-count", type=int, default=1,
        help="Number of combinations to train concurrently, in separate worker processes (default 1, "
        "sequential). See scripts/analyze_sweep.py's module docstring, 'Parallel execution'.",
    )
    args = parser.parse_args()
    weights = tuple(args.eval_weights)

    with open(args.sweep_config, "r") as f:
        raw = yaml.safe_load(f)
    fixed, swept = parse_sweep_yaml(raw)
    if "effective_sample_beta" not in swept:
        raise SystemExit(
            "--sweep-config must have effective_sample_beta as a YAML list -- see "
            "configs/phase2/ppo_fixed_d_effective_sample_sweep.yaml for the expected shape."
        )
    combos = generate_combinations(fixed, swept)
    swept_keys = list(swept.keys())

    # The standard-PPO baseline: same fixed hyperparameters, weighting off,
    # added once, never part of the YAML's beta grid.
    baseline_combo = dict(fixed)
    baseline_combo["use_effective_sample_weighting"] = False
    baseline_combo.pop("effective_sample_beta", None)
    combos = [baseline_combo] + combos

    print(f"=== Effective-sample-weighting sweep: {len(combos)} combination(s) (1 baseline + {len(combos) - 1} beta values) ===")
    print(f"Swept field: effective_sample_beta = {swept['effective_sample_beta']}")
    print(f"Fixed fields: {fixed}\n")

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)

    def make_env():
        return StochasticMazeEnv(
            width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
            extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
            step_penalty=env_cfg.step_penalty, goal_reward=env_cfg.goal_reward, hazard_reward=env_cfg.hazard_reward,
            max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed, num_start_states=env_cfg.num_start_states,
            gamma=env_cfg.gamma,
        )

    eval_env = make_env()
    tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
    covered_starts, held_out_starts = get_tier_start_lists(eval_env, tier_cfg)
    print(f"Loaded start tiers: {len(covered_starts)} covered, {len(held_out_starts)} held-out.")

    dataset = load_dataset(args.dataset)
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes.")
    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_state_dict = ckpt["state_dict"]

    ceiling_success_rates = compute_ceiling_success_rates(
        eval_env, args.pi_d_star_empirical, args.eval_episodes, args.eval_seed, covered_starts, held_out_starts, weights=weights
    )
    print(
        f"pi_D* (empirical) ceiling (seed={args.eval_seed}): overall={ceiling_success_rates['overall']:.3f}, "
        f"covered={ceiling_success_rates['covered']:.3f}, held_out={ceiling_success_rates['held_out']:.3f}, "
        f"weighted={ceiling_success_rates['weighted']:.3f}\n"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def combo_prefix(combo: dict) -> str:
        if not combo.get("use_effective_sample_weighting", False):
            return f"{args.base_prefix}_baseline"
        return make_combo_prefix(args.base_prefix, ["effective_sample_beta"], combo)

    def combo_desc(combo: dict) -> str:
        if not combo.get("use_effective_sample_weighting", False):
            return "baseline (standard PPO)"
        return f"beta={combo['effective_sample_beta']}"

    results = []
    pending = []
    for i, combo in enumerate(combos, start=1):
        prefix = combo_prefix(combo)
        desc = combo_desc(combo)
        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            print(f"[{i}/{len(combos)} {prefix}] SKIPPING -- {csv_path} already exists (use --force to re-run)")
            df_existing = pd.read_csv(csv_path)
            results.append(
                {
                    "beta": combo.get("effective_sample_beta") if combo.get("use_effective_sample_weighting") else None,
                    "prefix": prefix, "seconds_per_epoch": None,
                    "best": float(df_existing["weighted_success_rate"].max()),
                    "mean": float(df_existing["weighted_success_rate"].mean()),
                    "std": float(df_existing["weighted_success_rate"].std()),
                    "final": float(df_existing.iloc[-1]["weighted_success_rate"]),
                    "csv_path": str(csv_path), "status": "skipped (already existed)",
                }
            )
        else:
            pending.append((i, combo, prefix, desc))

    t_start = time.time()

    if args.cpu_count <= 1 or len(pending) <= 1:
        for i, combo, prefix, desc in pending:
            log_tag = f"[{i}/{len(combos)} {prefix}] "
            cfg = PPOHyperparams(**combo)
            print(f"\n{log_tag}Starting -- {desc}")
            t0 = time.time()
            print(f"{log_tag}Timing pure training (no eval)...")
            seconds_per_epoch = time_pure_training(dataset, dataset.obs_dim, dataset.n_actions, cfg, prior_state_dict)
            print(f"{log_tag}  {seconds_per_epoch:.4f} s/epoch")
            set_global_seed(cfg.seed)
            summary = run_single_analysis(
                eval_env=eval_env, dataset=dataset, prior_state_dict=prior_state_dict,
                ceiling_success_rates=ceiling_success_rates, cfg=cfg,
                checkpoint_every=args.checkpoint_every, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed,
                out_dir=out_dir, prefix=prefix, covered_starts=covered_starts, held_out_starts=held_out_starts,
                weights=weights, title_suffix=desc, verbose=True, log_prefix=log_tag,
            )
            dt = time.time() - t0
            print(f"{log_tag}Done in {dt / 60:.1f} min -- best={summary['best']:.3f} mean={summary['mean']:.3f} std={summary['std']:.3f}")
            results.append(
                {"beta": combo.get("effective_sample_beta") if combo.get("use_effective_sample_weighting") else None,
                 "seconds_per_epoch": seconds_per_epoch, **summary, "status": "ran"}
            )
    else:
        cpu_count = max(1, args.cpu_count)
        available = os.cpu_count() or cpu_count
        clamped = min(cpu_count, len(pending), available)
        if clamped != cpu_count:
            print(f"Note: --cpu-count {cpu_count} clamped to {clamped} (min of requested, {len(pending)} pending, {available} CPUs available).")
        cpu_count = clamped

        env_cfg_dict = dict(
            width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
            extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
            step_penalty=env_cfg.step_penalty, goal_reward=env_cfg.goal_reward, hazard_reward=env_cfg.hazard_reward,
            max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed, num_start_states=env_cfg.num_start_states,
            gamma=env_cfg.gamma,
        )
        tasks = []
        for i, combo, prefix, desc in pending:
            tasks.append(
                {"combo": combo, "prefix": prefix, "combo_desc": desc, "checkpoint_every": args.checkpoint_every,
                 "eval_episodes": args.eval_episodes, "eval_seed": args.eval_seed, "out_dir": str(out_dir),
                 "weights": weights, "log_tag": f"[{i}/{len(combos)} {prefix}] "}
            )

        print(
            f"\nRunning {len(pending)} pending combination(s) across {cpu_count} worker process(es). "
            f"Per-combination logs go to <out_dir>/<prefix>.log.\n"
        )
        n_done = 0
        with ProcessPoolExecutor(
            max_workers=cpu_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(env_cfg_dict, eval_env.layout, args.dataset, prior_state_dict, ceiling_success_rates, covered_starts, held_out_starts),
        ) as executor:
            futures = {executor.submit(_run_one_combo, t): t for t in tasks}
            pending_futures = set(futures.keys())
            while pending_futures:
                done, pending_futures = wait(pending_futures, timeout=30, return_when=FIRST_COMPLETED)
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
                    except Exception as e:  # pragma: no cover
                        res = {"beta": None, "prefix": t["prefix"], "status": f"FAILED: {e}", "csv_path": None, "log_path": f"{out_dir}/{t['prefix']}.log"}
                    results.append(res)
                    if str(res.get("status", "")).startswith("FAILED"):
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: FAILED -- {res['status']} (see {res.get('log_path')})  [{elapsed:.1f} min elapsed]")
                    else:
                        print(
                            f"[{n_done}/{len(pending)}] {t['prefix']}: done -- {res['seconds_per_epoch']:.4f} s/epoch, "
                            f"best={res['best']:.3f} mean={res['mean']:.3f}  [{elapsed:.1f} min elapsed]"
                        )

    summary_df = pd.DataFrame(results).sort_values(by="beta", na_position="first")
    summary_path = out_dir / "effective_sample_weighting_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    total_min = (time.time() - t_start) / 60
    print(f"\n=== Complete in {total_min:.1f} min total ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()