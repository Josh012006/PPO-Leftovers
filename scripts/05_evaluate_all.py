"""Evaluate the prior, both pi_D* variants, and an arbitrary number of
fixed-D PPO checkpoints (standard, modified, any H1-H7 ablation) under the
IDENTICAL live-rollout protocol (same env, same seeds, same episode count),
then print/save the exploitation-gap table.

Usage (phase 1, unaffected):
    python scripts/05_evaluate_all.py \
        --env-config configs/phase1/env_maze.yaml \
        --prior-checkpoint results/phase1/prior_checkpoint.pt \
        --pi-d-star-empirical results/phase1/pi_d_star_empirical.pkl \
        --pi-d-star-true-restricted results/phase1/pi_d_star_true_restricted.pkl \
        --ppo-checkpoints standard=results/phase1/ppo_standard_on_D.pt modified=results/phase1/ppo_modified_on_D.pt \
        --n-episodes 500 \
        --eval-seed 999 \
        --out results/phase1/gap_report.csv

Usage (phase 2 -- adds a covered-tier and a held-out-tier success_rate for
EVERY policy, never blended into one number; see README, "Phase 2"):
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
from ppo_exploitation.eval.evaluate import build_gap_report, evaluate_policy, make_neural_act_fn, make_tabular_act_fn
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
        help="Phase 2 only. If given, EVERY policy below is evaluated three ways: 'overall' (the "
        "env's own default reset, uniform across all starts), '_covered' (well- + moderately-"
        "covered tier starts only), and '_held_out' (held-out tier starts only, never seen "
        "during D collection) -- these two are never blended into one number. If omitted (phase "
        "1's default), only the single 'overall' number is produced, unchanged from before.",
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
    if args.start_tiers_config:
        tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
        covered_starts = [
            env.layout.state_id(*env.layout.starts[i])
            for i in (*tier_cfg.well_covered_indices, *tier_cfg.moderately_covered_indices)
        ]
        held_out_starts = [env.layout.state_id(*env.layout.starts[i]) for i in tier_cfg.held_out_indices]
        print(
            f"Loaded start tiers from {args.start_tiers_config}: "
            f"{len(covered_starts)} covered starts, {len(held_out_starts)} held-out starts."
        )

    def eval_all_ways(name: str, act_fn, results: dict, **eval_kwargs):
        results[name] = evaluate_policy(env, act_fn, args.n_episodes, args.eval_seed, **eval_kwargs)
        if covered_starts is not None:
            results[f"{name}_covered"] = evaluate_policy(
                env, act_fn, args.n_episodes, args.eval_seed, eval_start_states=covered_starts, **eval_kwargs
            )
            results[f"{name}_held_out"] = evaluate_policy(
                env, act_fn, args.n_episodes, args.eval_seed, eval_start_states=held_out_starts, **eval_kwargs
            )

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
    if covered_starts is not None:
        print(
            "\n(rows suffixed _covered / _held_out are the same policy evaluated only from that "
            "tier's starts -- never blended into the unsuffixed 'overall' row; see README, 'Phase 2'.)"
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out)
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()