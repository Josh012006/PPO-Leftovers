"""Evaluate the prior, both pi_D* variants, and an arbitrary number of
fixed-D PPO checkpoints (standard, modified, any H1-H7 ablation) under the
IDENTICAL live-rollout protocol (same env, same seeds, same episode count),
then print/save the exploitation-gap table.

Usage:
    python scripts/05_evaluate_all.py \
        --env-config configs/env_maze.yaml \
        --reference-config configs/reference.yaml \
        --prior-checkpoint results/prior_checkpoint.pt \
        --pi-d-star-empirical results/pi_d_star_empirical.pkl \
        --pi-d-star-true-restricted results/pi_d_star_true_restricted.pkl \
        --ppo-checkpoints standard=results/ppo_standard_on_D.pt modified=results/ppo_modified_on_D.pt \
        --n-episodes 500 \
        --eval-seed 999 \
        --out results/gap_report.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import build_gap_report, evaluate_policy, make_neural_act_fn, make_tabular_act_fn
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, ReferenceConfig


def load_ppo_checkpoint(path: str) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden_sizes"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--reference-config", default="configs/reference.yaml")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star-empirical", default="results/pi_d_star_empirical.pkl")
    parser.add_argument("--pi-d-star-true-restricted", default="results/pi_d_star_true_restricted.pkl")
    parser.add_argument(
        "--ppo-checkpoints",
        nargs="+",
        required=True,
        help="name=path pairs, e.g. standard=results/ppo_standard_on_D.pt modified=results/ppo_modified_on_D.pt",
    )
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Evaluate neural PPO policies by sampling actions instead of taking the deterministic "
        "(argmax) action. Off by default -- see README on exploitation frequency vs quality.",
    )
    parser.add_argument("--out", default="results/gap_report.csv")
    args = parser.parse_args()

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    ref_cfg = ReferenceConfig.from_yaml(args.reference_config)
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

    results = {}

    # --- prior checkpoint ---
    prior_net = load_ppo_checkpoint(args.prior_checkpoint)
    results["prior_pi_beta"] = evaluate_policy(
        env, make_neural_act_fn(prior_net, deterministic=not args.stochastic_eval), args.n_episodes, args.eval_seed
    )

    # --- both pi_D* variants ---
    with open(args.pi_d_star_empirical, "rb") as f:
        ref_empirical = pickle.load(f)
    with open(args.pi_d_star_true_restricted, "rb") as f:
        ref_true = pickle.load(f)
    results["pi_D*_empirical"] = evaluate_policy(
        env, make_tabular_act_fn(ref_empirical), args.n_episodes, args.eval_seed, unseen_penalty=ref_cfg.unseen_penalty
    )
    results["pi_D*_true_restricted"] = evaluate_policy(
        env, make_tabular_act_fn(ref_true), args.n_episodes, args.eval_seed, unseen_penalty=ref_cfg.unseen_penalty
    )

    # --- fixed-D PPO checkpoints (standard, modified, any ablations) ---
    for pair in args.ppo_checkpoints:
        name, path = pair.split("=", 1)
        net = load_ppo_checkpoint(path)
        results[name] = evaluate_policy(
            env, make_neural_act_fn(net, deterministic=not args.stochastic_eval), args.n_episodes, args.eval_seed
        )

    report = build_gap_report(results, reference_key="pi_D*_empirical")
    pd_options = None
    try:
        import pandas as pd

        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    except ImportError:
        pass
    print("\n=== Exploitation gap report (same live-rollout protocol for every policy) ===")
    print(report.to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
