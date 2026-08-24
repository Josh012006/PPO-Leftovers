"""Collect the fixed dataset D from the frozen prior checkpoint (pi_beta).
D is collected once and never touched again -- every later stage reads this
exact file.

Usage:
    python scripts/02_collect_dataset.py \
        --env-config configs/env_maze.yaml \
        --checkpoint results/prior_checkpoint.pt \
        --n-episodes 4000 \
        --seed 1 \
        --out results/dataset_D.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ppo_exploitation.data.collect import collect_fixed_dataset, save_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig
from ppo_exploitation.utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--n-episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Collect using the checkpoint's argmax action instead of sampling "
        "(narrower, more repetitive D -- off by default, see data/collect.py docstring).",
    )
    parser.add_argument("--out", default="results/dataset_D.pkl")
    args = parser.parse_args()

    set_global_seed(args.seed)
    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    env = StochasticMazeEnv(
        width=env_cfg.width,
        height=env_cfg.height,
        slip_prob=env_cfg.slip_prob,
        extra_connection_prob=env_cfg.extra_connection_prob,
        num_hazards=env_cfg.num_hazards,
        step_penalty=env_cfg.step_penalty,
        goal_reward=env_cfg.goal_reward,
        hazard_reward=env_cfg.hazard_reward,
        max_steps=env_cfg.max_steps,
        layout_seed=env_cfg.layout_seed,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden_sizes"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"Loaded prior checkpoint (final eval: {ckpt['final_eval']})")

    dataset = collect_fixed_dataset(
        env, net, n_episodes=args.n_episodes, seed=args.seed, sample_actions=not args.deterministic
    )
    print(
        f"Collected D: {len(dataset)} transitions across {dataset.n_episodes} episodes, "
        f"state-action coverage = {dataset.coverage():.1%}"
    )
    save_dataset(dataset, args.out)
    print(f"Saved dataset to {args.out}")


if __name__ == "__main__":
    main()
