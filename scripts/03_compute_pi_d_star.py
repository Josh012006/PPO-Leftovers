"""Compute pi_D*, both the empirical/MLE definition (i) and the
true-dynamics support-restricted definition (ii), via exact tabular value
iteration on D. Saves both ReferenceSolution objects.

Usage:
    python scripts/03_compute_pi_d_star.py \
        --env-config configs/env_maze.yaml \
        --reference-config configs/reference.yaml \
        --dataset results/dataset_D.pkl \
        --out-empirical results/pi_d_star_empirical.pkl \
        --out-true-restricted results/pi_d_star_true_restricted.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.reference.experience_optimal import (
    compute_pi_d_star_empirical,
    compute_pi_d_star_true_restricted,
)
from ppo_exploitation.utils.config import MazeEnvConfig, ReferenceConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--reference-config", default="configs/reference.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--out-empirical", default="results/pi_d_star_empirical.pkl")
    parser.add_argument("--out-true-restricted", default="results/pi_d_star_true_restricted.pkl")
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
    dataset = load_dataset(args.dataset)
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes, coverage={dataset.coverage():.1%}")

    print("Solving empirical (MLE) pi_D* ...")
    ref_empirical = compute_pi_d_star_empirical(
        dataset, gamma=ref_cfg.gamma, unseen_penalty=ref_cfg.unseen_penalty,
        theta=ref_cfg.vi_theta, max_iter=ref_cfg.vi_max_iter,
    )
    start_state = env.get_state()
    print(
        f"  empirical pi_D*: V(s0)={ref_empirical.V[start_state]:.4f}  "
        f"covered_states={len(ref_empirical.covered_states)}/{ref_empirical.n_states}  "
        f"visited_sa_pairs={ref_empirical.visited_sa_count}/{ref_empirical.n_states * ref_empirical.n_actions}"
    )

    print("Solving true-dynamics, support-restricted pi_D* ...")
    ref_true = compute_pi_d_star_true_restricted(
        env, dataset, gamma=ref_cfg.gamma, unseen_penalty=ref_cfg.unseen_penalty,
        theta=ref_cfg.vi_theta, max_iter=ref_cfg.vi_max_iter,
    )
    print(
        f"  true-restricted pi_D*: V(s0)={ref_true.V[start_state]:.4f}  "
        f"covered_states={len(ref_true.covered_states)}/{ref_true.n_states}"
    )
    print(
        f"  V(s0) empirical vs true-restricted gap: "
        f"{ref_true.V[start_state] - ref_empirical.V[start_state]:+.4f} "
        f"(this is the DP-value estimate of pure sampling-noise-in-the-ceiling; "
        f"the number that matters for the actual exploitation gap is the LIVE-ROLLOUT "
        f"score computed in scripts/05_evaluate_all.py, not this closed-form value)"
    )

    for path, ref in [(args.out_empirical, ref_empirical), (args.out_true_restricted, ref_true)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(ref, f)
        print(f"Saved {ref.kind} pi_D* to {path}")


if __name__ == "__main__":
    main()
