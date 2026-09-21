"""Cost/performance tradeoff for effective-sample-count weighting of the
policy loss (see README, "An attempt to close the gap" -- the idea
motivated by state 717, where the sparser of two competing actions is
genuinely better but gets outvoted by raw sample volume).

Trains the project's best configuration
(configs/phase2/ppo_fixed_d_best_config.yaml) as the standard-PPO
baseline (use_effective_sample_weighting=False -- byte-identical to
every other run in this project) and once per swept beta value
(use_effective_sample_weighting=True), all other hyperparameters held
at the best-checkpoint values.

Per-epoch UPDATE cost is measured in its own dedicated pass: one
FixedDPPOTrainer.train() call with NO eval_callback, timed end to end
and divided by cfg.epochs -- this isolates the actual overhead the new
weighting adds to a gradient step from live-rollout evaluation cost,
which is timed and reported separately. Performance (mean/best/std/final
weighted_success_rate) comes from a SEPARATE run through this project's
standard scripts/_analysis_lib.py:run_single_analysis, exactly as every
other script here reports it -- training runs twice per config as a
result (once for clean timing, once for the standard eval curve), which
this project's fast fixed-D training (no env.step() calls at all -- D is
read-only, see fixed_d_trainer.py) makes cheap enough to be worth the
clarity of not conflating the two measurements.

Outputs, under --out-dir (default results/phase2/analysis/effective_
sample_weighting/):
  effective_sample_weighting_summary.csv -- one row per config: beta
                                             (None for the baseline),
                                             seconds_per_epoch,
                                             best/mean/std/final
                                             weighted_success_rate
  plus everything scripts/_analysis_lib.py:run_single_analysis normally
  produces per config (<prefix>.csv, _success_return/_weighted .svg/.png)

Usage:
    python scripts/analyze_effective_sample_weighting.py \
        --env-config configs/phase2/env_maze.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --pi-d-star-empirical results/phase2/pi_d_star_empirical.pkl \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --base-config configs/phase2/ppo_fixed_d_best_config.yaml \
        --betas 0.7 0.85 0.9 0.92 0.95 0.99 \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase2/analysis/effective_sample_weighting
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument("--dataset", default="results/phase2/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/phase2/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star-empirical", default="results/phase2/pi_d_star_empirical.pkl")
    parser.add_argument("--start-tiers-config", default="configs/phase2/start_tiers.yaml")
    parser.add_argument(
        "--base-config",
        default="configs/phase2/ppo_fixed_d_best_config.yaml",
        help="Every other hyperparameter is held at this config's values -- only "
        "use_effective_sample_weighting/effective_sample_beta are varied.",
    )
    parser.add_argument("--betas", type=float, nargs="+", default=[0.7, 0.85, 0.9, 0.92, 0.95, 0.99])
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument(
        "--eval-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_EVAL_WEIGHTS),
        metavar=("OVERALL", "COVERED", "HELD_OUT"),
    )
    parser.add_argument("--out-dir", default="results/phase2/analysis/effective_sample_weighting")
    args = parser.parse_args()

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    env = StochasticMazeEnv(
        width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
        extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
        step_penalty=env_cfg.step_penalty, goal_reward=env_cfg.goal_reward, hazard_reward=env_cfg.hazard_reward,
        max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed, num_start_states=env_cfg.num_start_states,
        gamma=env_cfg.gamma,
    )
    tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
    covered_starts, held_out_starts = get_tier_start_lists(env, tier_cfg)
    weights = tuple(args.eval_weights)
    print(f"Loaded start tiers: {len(covered_starts)} covered, {len(held_out_starts)} held-out.")

    dataset = load_dataset(args.dataset)
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes.")
    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_state_dict = ckpt["state_dict"]

    ceiling_success_rates = compute_ceiling_success_rates(
        env, args.pi_d_star_empirical, args.eval_episodes, args.eval_seed, covered_starts, held_out_starts, weights=weights
    )
    print(
        f"pi_D* (empirical) ceiling (seed={args.eval_seed}): overall={ceiling_success_rates['overall']:.3f}, "
        f"covered={ceiling_success_rates['covered']:.3f}, held_out={ceiling_success_rates['held_out']:.3f}, "
        f"weighted={ceiling_success_rates['weighted']:.3f}\n"
    )

    base_cfg = PPOHyperparams.from_yaml(args.base_config)
    configs = [("baseline (standard PPO)", None)] + [(f"beta={b}", b) for b in args.betas]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for label, beta in configs:
        cfg = copy.deepcopy(base_cfg)
        if beta is None:
            cfg.use_effective_sample_weighting = False
            prefix = "baseline"
        else:
            cfg.use_effective_sample_weighting = True
            cfg.effective_sample_beta = beta
            prefix = f"beta_{str(beta).replace('.', '_')}"

        print(f"=== {label} ===")
        print("Timing pure training (no eval)...")
        seconds_per_epoch = time_pure_training(dataset, dataset.obs_dim, dataset.n_actions, cfg, prior_state_dict)
        print(f"  {seconds_per_epoch:.4f} s/epoch")

        print("Running the standard eval curve (separate training pass)...")
        summary = run_single_analysis(
            eval_env=env, dataset=dataset, prior_state_dict=prior_state_dict,
            ceiling_success_rates=ceiling_success_rates, cfg=cfg,
            checkpoint_every=args.checkpoint_every, eval_episodes=args.eval_episodes, eval_seed=args.eval_seed,
            out_dir=out_dir, prefix=prefix, covered_starts=covered_starts, held_out_starts=held_out_starts,
            weights=weights, title_suffix=label, verbose=False,
        )
        print(f"  mean={summary['mean']:.4f} best={summary['best']:.4f} std={summary['std']:.4f} final={summary['final']:.4f}\n")

        results.append(
            {
                "label": label, "beta": beta, "seconds_per_epoch": seconds_per_epoch,
                "mean_weighted": summary["mean"], "best_weighted": summary["best"],
                "std_weighted": summary["std"], "final_weighted": summary["final"], "csv_path": summary["csv_path"],
            }
        )

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "effective_sample_weighting_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("=== Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
