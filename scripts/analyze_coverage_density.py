"""Check #1 for the "why does disagreement concentrate on near-death
states" question (see README, "Policy agreement"): among states D
actually covers, are near-death states covered more SPARSELY than
non-near-death ones -- fewer visited actions, fewer samples per visited
action -- not just covered-vs-uncovered, which the policy-agreement
diagnostic already restricted to?

If density is systematically lower on near-death states, that directly
supports "weak-but-present D signal is handled worse by a generalizing
neural critic than by exact counting (value iteration)" as the mechanism
behind the disagreement -- rather than the disagreement being about
something else PPO's own optimization mechanism does.

No training involved -- this is pure data analysis over D and pi_beta's
checkpoint (needed only to compute the exact V^pi_beta(s) that defines
"near-death", exactly as in scripts/analyze_critic_accuracy.py and
scripts/analyze_policy_agreement.py). Cheap and fast.

Outputs, under --out-dir (default results/analysis/coverage_density/):
  coverage_density.csv                    -- per-state: covered,
                                              near_death, true_value_pi_beta,
                                              n_actions_visited (0-4),
                                              total_samples,
                                              min/mean/max samples per
                                              visited action
  coverage_density_vs_true_value.svg/png  -- scatter: true V^pi_beta (x)
                                              vs. total_samples (y),
                                              covered states only
  coverage_density_strip.svg/png          -- jittered strip plot:
                                              total_samples per covered
                                              state, split by near_death

Usage:
    python scripts/analyze_coverage_density.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --out-dir results/analysis/coverage_density
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.reference.experience_optimal import compute_true_value_of_policy
from ppo_exploitation.utils.config import MazeEnvConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--out-dir", default="results/analysis/coverage_density")
    parser.add_argument(
        "--near-death-threshold",
        type=float,
        default=-0.9,
        help="Same convention as scripts/analyze_policy_agreement.py -- states with true "
        "V^pi_beta below this are flagged as near-certain-death states.",
    )
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
    )

    dataset = load_dataset(args.dataset)
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes.")

    # Per-(state, action) sample counts, directly from D.
    sa_counts: Counter[tuple[int, int]] = Counter()
    for tr in dataset.trajectories:
        for s, a in zip(tr.states.tolist(), tr.actions.tolist()):
            sa_counts[(int(s), int(a))] += 1

    prior_ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(prior_ckpt["obs_dim"], prior_ckpt["n_actions"], prior_ckpt["hidden_sizes"])
    net.load_state_dict(prior_ckpt["state_dict"])
    net.eval()
    print(f"Loaded prior checkpoint (final eval: {prior_ckpt['final_eval']})")

    terminal_states = {s for s in range(env.n_states) if env.is_terminal_state(s)}
    non_terminal = [s for s in range(env.n_states) if s not in terminal_states]
    obs_batch = np.stack([env.state_to_obs(s) for s in non_terminal]).astype(np.float32)
    with torch.no_grad():
        logits, _ = net.forward(torch.as_tensor(obs_batch))
        probs = torch.softmax(logits, dim=-1).numpy()

    action_probs_full = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    for i, s in enumerate(non_terminal):
        action_probs_full[s] = probs[i]

    print(f"Computing exact V^pi_beta(s) via policy evaluation against true dynamics (gamma={env_cfg.gamma})...")
    true_value = compute_true_value_of_policy(env, action_probs_full, gamma=env_cfg.gamma)

    rows = []
    for s in non_terminal:
        visited_counts = [sa_counts.get((s, a), 0) for a in range(env.n_actions)]
        n_visited = sum(1 for c in visited_counts if c > 0)
        visited_nonzero = [c for c in visited_counts if c > 0]
        rows.append(
            {
                "state": s,
                "covered": n_visited > 0,
                "near_death": bool(true_value[s] < args.near_death_threshold),
                "true_value_pi_beta": float(true_value[s]),
                "n_actions_visited": n_visited,
                "total_samples": int(sum(visited_counts)),
                "min_samples_per_visited_action": int(min(visited_nonzero)) if visited_nonzero else 0,
                "mean_samples_per_visited_action": float(np.mean(visited_nonzero)) if visited_nonzero else 0.0,
                "max_samples_per_visited_action": int(max(visited_nonzero)) if visited_nonzero else 0,
            }
        )
    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "coverage_density.csv"
    df.to_csv(csv_path, index=False)

    covered = df[df["covered"]]

    def report(label: str, sub: pd.DataFrame):
        print(
            f"  {label:24s} n={len(sub):4d}  n_actions_visited={sub['n_actions_visited'].mean():.2f}/4  "
            f"total_samples={sub['total_samples'].mean():7.1f}  "
            f"mean_samples_per_action={sub['mean_samples_per_visited_action'].mean():6.1f}"
        )

    print(f"\n=== Coverage density (covered states only, n={len(covered)}) ===")
    report("all covered", covered)
    report("covered & near_death", covered[covered["near_death"]])
    report("covered & not near_death", covered[~covered["near_death"]])
    print(f"\nSaved {csv_path}")

    # --- Plot 1: true_value vs total_samples, covered states ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for flag, color, label in [(False, "tab:blue", "not near-death"), (True, "tab:red", "near-death")]:
        sub = covered[covered["near_death"] == flag]
        ax.scatter(sub["true_value_pi_beta"], sub["total_samples"], s=16, alpha=0.6, color=color, label=label)
    ax.axvline(args.near_death_threshold, color="0.5", linestyle="--", linewidth=1, label=f"near-death threshold ({args.near_death_threshold})")
    ax.set_xlabel("exact true V\u03c0\u03b2(s)")
    ax.set_ylabel("total samples in D for this state (all actions)")
    ax.set_title("Is data density related to how dangerous a state is?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot1 = out_dir / "coverage_density_vs_true_value"
    fig.savefig(plot1.with_suffix(".svg"))
    fig.savefig(plot1.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot1}.svg/.png")

    # --- Plot 2: jittered strip plot, total_samples by near_death group ---
    fig, ax = plt.subplots(figsize=(6, 6))
    rng = np.random.default_rng(0)
    for x0, flag, color, label in [(0, False, "tab:blue", "not near-death"), (1, True, "tab:red", "near-death")]:
        sub = covered[covered["near_death"] == flag]
        jitter = rng.uniform(-0.15, 0.15, size=len(sub))
        ax.scatter(np.full(len(sub), x0) + jitter, sub["total_samples"], s=16, alpha=0.6, color=color)
        ax.scatter([x0], [sub["total_samples"].mean()], s=140, color=color, marker="_", linewidths=3, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["not near-death", "near-death"])
    ax.set_ylabel("total samples in D for this state (all actions)")
    ax.set_title("Coverage density by near-death status (covered states only)\nthick marks = group mean")
    fig.tight_layout()
    plot2 = out_dir / "coverage_density_strip"
    fig.savefig(plot2.with_suffix(".svg"))
    fig.savefig(plot2.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot2}.svg/.png")


if __name__ == "__main__":
    main()
