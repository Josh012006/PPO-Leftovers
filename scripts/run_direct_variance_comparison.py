"""Compare DirectVarianceWeightedTrainer (fixed_d_trainer_variance_weighted.py)
against the established baseline (no weighting) on the real project maze,
using the SAME evaluation protocol as every other result in this project
(eval_seed=999, evaluate_policy_weighted, mean/best/final over the run --
see README, "Our new starting point").

Usage:
    python scripts/run_direct_variance_comparison.py \
        --env-config configs/phase2/env_maze.yaml \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --tau-quantile 0.5 --ensemble-epochs 30 \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase2/analysis/direct_variance_weighted
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy_weighted, get_tier_start_lists
from ppo_exploitation.ppo.fixed_d_trainer_variance_weighted import DirectVarianceWeightedTrainer
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams, StartTierConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--start-tiers-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prior-checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-eps", type=float, default=0.55)
    parser.add_argument("--gae-lambda", type=float, default=0.90)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--ensemble-n-heads", type=int, default=5)
    parser.add_argument("--ensemble-hidden-sizes", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--ensemble-epochs", type=int, default=30)
    parser.add_argument("--ensemble-lr", type=float, default=1e-3)
    parser.add_argument("--tau-quantile", type=float, default=0.5, help="See calibrate_tau in fixed_d_trainer_variance_weighted.py.")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Evaluate every N epochs.")
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true", help="Suppress the per-epoch training log (policy/value loss, entropy, KL) between evaluation checkpoints.")
    parser.add_argument("--out-dir", default="results/phase2/analysis/direct_variance_weighted")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    env = StochasticMazeEnv(
        width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
        extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
        step_penalty=env_cfg.step_penalty, max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed,
        num_start_states=env_cfg.num_start_states, gamma=env_cfg.gamma,
    )
    tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
    covered_starts, held_out_starts = get_tier_start_lists(env, tier_cfg)

    dataset = load_dataset(args.dataset)
    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    hidden_sizes = tuple(ckpt.get("hidden_sizes", (64, 64)))

    cfg = PPOHyperparams(
        epochs=args.epochs, minibatch_size=args.minibatch_size, clip_eps=args.clip_eps,
        gae_lambda=args.gae_lambda, entropy_coef=0.0, value_coef=1.0, max_grad_norm=0.1,
        hidden_sizes=hidden_sizes, lr=args.lr, seed=args.seed,
    )
    trainer = DirectVarianceWeightedTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=ckpt["state_dict"],
        ensemble_n_heads=args.ensemble_n_heads, ensemble_hidden_sizes=tuple(args.ensemble_hidden_sizes),
        ensemble_epochs=args.ensemble_epochs, ensemble_lr=args.ensemble_lr, tau_quantile=args.tau_quantile,
    )
    print(
        f"tau={trainer.tau:.6f}  weights: min={trainer.policy_loss_weights.min():.4f} "
        f"max={trainer.policy_loss_weights.max():.4f} mean={trainer.policy_loss_weights.mean():.4f}"
    )

    results = []

    def eval_cb(epoch, net, summary):
        r = evaluate_policy_weighted(
            env, lambda o, s: net.act_numpy(o, deterministic=True)[0], n_episodes=args.eval_episodes,
            seed=args.eval_seed, covered_starts=covered_starts, held_out_starts=held_out_starts,
        )
        results.append({"epoch": epoch, **r})
        print(f"  epoch {epoch:4d}: weighted={r['weighted_success_rate']:.4f}")

    trainer.train(verbose=not args.quiet, eval_every_epochs=args.checkpoint_every, eval_callback=eval_cb)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "direct_variance_weighted.csv", index=False)
    print(
        f"\nmean={df['weighted_success_rate'].mean():.4f}  best={df['weighted_success_rate'].max():.4f} "
        f"(epoch {df.loc[df['weighted_success_rate'].idxmax(), 'epoch']})  "
        f"final={df['weighted_success_rate'].iloc[-1]:.4f}  std={df['weighted_success_rate'].std():.4f}"
    )
    print("REFERENCE (baseline, no weighting): mean=0.4127 best=0.4720(180) final=0.4160 std=0.0495")
    print("REFERENCE (KL-anchored, k=0.1, current best): mean=0.4289 best=0.4845(140) final=0.4705 std=0.0679")


if __name__ == "__main__":
    main()
