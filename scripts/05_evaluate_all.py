"""Evaluate the prior, both pi_D* variants, and an arbitrary number of
fixed-D PPO checkpoints (standard, modified, any H1-H7 ablation) under the
IDENTICAL live-rollout protocol (same env, same seeds, same episode count),
then print/save the exploitation-gap table.

Usage (phase 1, unaffected -- no --start-tiers-config, single number per policy):
    python scripts/05_evaluate_all.py \
        --env-config configs/phase1/env_maze.yaml \
        --prior-checkpoint results/phase1/prior_checkpoint.pt \
        --pi-d-star-empirical results/phase1/pi_d_star_empirical.pkl \
        --pi-d-star-true-restricted results/phase1/pi_d_star_true_restricted.pkl \
        --ppo-checkpoints standard=results/phase1/ppo_standard_on_D.pt modified=results/phase1/ppo_modified_on_D.pt \
        --n-episodes 500 \
        --eval-seed 999 \
        --out results/phase1/gap_report.csv

Usage (phase 2 -- see README, "Our new starting point"): every policy is
evaluated THREE ways (overall/covered/held-out) and combined into one
weighted_success_rate (0.25/0.25/0.50 by default -- held-out weighted
most heavily to protect the SELECTION of "which policy is better" against
overfitting, not just the policies themselves). The weighted number, not
any of the three raw ones, is what gap_vs_reference_success_rate is
computed against:
    python scripts/05_evaluate_all.py \
        --env-config configs/phase2/env_maze.yaml \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --pi-d-star-empirical results/phase2/pi_d_star_empirical.pkl \
        --pi-d-star-true-restricted results/phase2/pi_d_star_true_restricted.pkl \
        --ppo-checkpoints standard=results/phase2/ppo_standard_on_D.pt \
        --n-episodes 500 \
        --eval-seed 999 \
        --out results/phase2/gap_report.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import (
    DEFAULT_EVAL_WEIGHTS,
    build_gap_report,
    evaluate_policy,
    evaluate_policy_weighted,
    get_tier_start_lists,
    make_neural_act_fn,
    make_tabular_act_fn,
)
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, StartTierConfig


def load_ppo_checkpoint(path: str) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden_sizes"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase1/env_maze.yaml")
    parser.add_argument("--prior-checkpoint", default="results/phase1/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star-empirical", default="results/phase1/pi_d_star_empirical.pkl")
    parser.add_argument("--pi-d-star-true-restricted", default="results/phase1/pi_d_star_true_restricted.pkl")
    parser.add_argument(
        "--ppo-checkpoints",
        nargs="+",
        required=True,
        help="name=path pairs, e.g. standard=results/phase1/ppo_standard_on_D.pt",
    )
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Evaluate neural PPO policies by sampling actions instead of taking the deterministic "
        "(argmax) action. Off by default -- see README on exploitation frequency vs quality.",
    )
    parser.add_argument(
        "--start-tiers-config",
        default=None,
        help="Phase 2 only. If given, EVERY policy is evaluated three ways -- 'overall' (the env's "
        "own default reset, uniform across all starts), 'covered' (well- + moderately-covered "
        "tier starts), 'held_out' (held-out tier starts, never seen during D collection) -- and "
        "combined into one weighted_success_rate (see --eval-weights). If omitted (phase 1's "
        "default), only the single 'overall' number is produced, unchanged from before.",
    )
    parser.add_argument(
        "--eval-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_EVAL_WEIGHTS),
        metavar=("OVERALL", "COVERED", "HELD_OUT"),
        help=f"Weights for the three eval modes, must sum to 1.0 (default {DEFAULT_EVAL_WEIGHTS}).",
    )
    parser.add_argument("--out", default="results/phase1/gap_report.csv")
    args = parser.parse_args()

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
        num_start_states=env_cfg.num_start_states,
        gamma=env_cfg.gamma,
    )

    covered_starts = held_out_starts = None
    weights = tuple(args.eval_weights)
    if args.start_tiers_config:
        tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
        covered_starts, held_out_starts = get_tier_start_lists(env, tier_cfg)
        print(
            f"Loaded start tiers from {args.start_tiers_config}: "
            f"{len(covered_starts)} covered starts, {len(held_out_starts)} held-out starts. "
            f"Weights (overall/covered/held_out): {weights}."
        )

    def eval_all_ways(name: str, act_fn, results: dict, **eval_kwargs):
        if covered_starts is None:
            results[name] = evaluate_policy(env, act_fn, args.n_episodes, args.eval_seed, **eval_kwargs)
            return
        w = evaluate_policy_weighted(
            env, act_fn, args.n_episodes, args.eval_seed, covered_starts, held_out_starts,
            weights=weights, **eval_kwargs,
        )
        results[f"{name}_overall"] = w["overall"]
        results[f"{name}_covered"] = w["covered"]
        results[f"{name}_held_out"] = w["held_out"]
        # A synthetic row for the combined number: only success_rate (and,
        # for a readable table, mean_return/mean_length under the same
        # weights) are meaningful here -- stderr isn't a simple weighted
        # combination of the three, so it's left as NaN rather than
        # approximated.
        results[f"{name}_weighted"] = {
            "mean_return": (
                weights[0] * w["overall"]["mean_return"]
                + weights[1] * w["covered"]["mean_return"]
                + weights[2] * w["held_out"]["mean_return"]
            ),
            "stderr_return": float("nan"),
            "success_rate": w["weighted_success_rate"],
            "success_rate_stderr": float("nan"),
            "mean_length": (
                weights[0] * w["overall"]["mean_length"]
                + weights[1] * w["covered"]["mean_length"]
                + weights[2] * w["held_out"]["mean_length"]
            ),
        }

    results: dict = {}

    # --- prior checkpoint ---
    prior_net = load_ppo_checkpoint(args.prior_checkpoint)
    eval_all_ways("prior_pi_beta", make_neural_act_fn(prior_net, deterministic=not args.stochastic_eval), results)

    # --- both pi_D* variants. covered_states enables the uncovered-state
    # frequency diagnostic (see eval/evaluate.py) -- act() itself never
    # forces termination or applies a penalty at this stage anymore, it
    # always returns a real, if sometimes uninformed, action. ---
    with open(args.pi_d_star_empirical, "rb") as f:
        ref_empirical = pickle.load(f)
    with open(args.pi_d_star_true_restricted, "rb") as f:
        ref_true = pickle.load(f)
    eval_all_ways(
        "pi_D*_empirical", make_tabular_act_fn(ref_empirical), results, covered_states=ref_empirical.covered_states
    )
    eval_all_ways(
        "pi_D*_true_restricted", make_tabular_act_fn(ref_true), results, covered_states=ref_true.covered_states
    )

    # --- fixed-D PPO checkpoints (standard, modified, any ablations) ---
    for pair in args.ppo_checkpoints:
        name, path = pair.split("=", 1)
        net = load_ppo_checkpoint(path)
        eval_all_ways(name, make_neural_act_fn(net, deterministic=not args.stochastic_eval), results)

    reference_key = "pi_D*_empirical_weighted" if covered_starts is not None else "pi_D*_empirical"
    report = build_gap_report(results, reference_key=reference_key)
    try:
        import pandas as pd

        pd.set_option("display.width", 120)
        pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    except ImportError:
        pass
    print("\n=== Exploitation gap report (same live-rollout protocol for every policy) ===")
    print(report.to_string())
    if covered_starts is not None:
        print(
            f"\n(rows suffixed _overall / _covered / _held_out are the same policy evaluated from that "
            f"population only; _weighted combines them {weights} -- see README, 'Our new starting "
            f"point'. gap_vs_reference_* is computed against {reference_key}.)"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()