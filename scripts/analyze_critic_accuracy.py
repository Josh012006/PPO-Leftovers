"""H6 pre-check: how accurate is pi_beta's own critic, right now, without
retraining anything?

Testing H6 (hidden_sizes) properly requires a new prior checkpoint (and
therefore a new D and a new pi_D*, since theta must start exactly at
pi_beta's weights) -- a much bigger, less comparable change than any other
single-field test run in this project. Before paying that cost, this
script asks a narrower, much cheaper question: is pi_beta's CURRENT critic
actually inaccurate at all?

It computes, for every ground-truth maze state, two numbers:
  1. What pi_beta's own trained critic currently predicts (one forward
     pass through value_net).
  2. The EXACT true value of pi_beta under its own policy, computed via
     exact policy evaluation against the environment's real dynamics
     (reference.experience_optimal.compute_true_value_of_policy) -- not
     an estimate, a closed-form-converged ground truth given pi_beta's
     actual action probabilities at every state.

The gap between these two is a direct, no-retraining-required measurement
of how much the critic that everything upstream of this project (D's
collection, GAE's targets, gae_lambda's whole story) has been relying on
is actually wrong by. A small gap argues against H6 (capacity was already
enough to represent an accurate critic; the earlier findings are not a
capacity problem). A large gap doesn't prove capacity is the cause
(undertraining is a separate, competing explanation this script cannot
distinguish from capacity), but it means the question is worth taking
seriously before deciding H6 isn't worth its cost.

Comparison is reported both across ALL states and restricted to the
states D actually covers (the ones that matter for GAE's Bellman
backups, since D's other states are never touched by advantage
computation regardless of critic accuracy there). Terminal states are
excluded from both -- their value is hardcoded to 0 by convention
everywhere in this project and the critic's raw prediction there is
never actually used by anything downstream.

Outputs, under --out-dir (default results/analysis/critic_accuracy/):
  critic_accuracy.csv                 -- per-state: covered, critic_pred,
                                          true_value, abs_error
  critic_accuracy_scatter.svg/png     -- critic prediction vs. true value,
                                          covered/uncovered states in
                                          different colors, y=x reference

Usage:
    python scripts/analyze_critic_accuracy.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --out-dir results/analysis/critic_accuracy
"""
from __future__ import annotations

import argparse
import sys
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
    parser.add_argument("--theta", type=float, default=1e-8)
    parser.add_argument("--max-iter", type=int, default=100_000)
    parser.add_argument("--out-dir", default="results/analysis/critic_accuracy")
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

    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden_sizes"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"Loaded prior checkpoint (final eval: {ckpt['final_eval']})")

    dataset = load_dataset(args.dataset)
    covered_states = {int(s) for tr in dataset.trajectories for s in tr.states}
    print(f"D covers {len(covered_states)}/{env.n_states} states.")

    terminal_states = {s for s in range(env.n_states) if env.is_terminal_state(s)}

    non_terminal = [s for s in range(env.n_states) if s not in terminal_states]
    obs_batch = np.stack([env.state_to_obs(s) for s in non_terminal]).astype(np.float32)
    with torch.no_grad():
        logits, values = net.forward(torch.as_tensor(obs_batch))
        probs = torch.softmax(logits, dim=-1).numpy()
        critic_pred = values.numpy()

    action_probs = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    critic_pred_full = np.zeros(env.n_states, dtype=np.float64)
    for i, s in enumerate(non_terminal):
        action_probs[s] = probs[i]
        critic_pred_full[s] = critic_pred[i]

    print(
        f"Computing exact V^pi_beta(s) via policy evaluation against true dynamics "
        f"(gamma={env_cfg.gamma}, theta={args.theta})..."
    )
    true_value = compute_true_value_of_policy(env, action_probs, gamma=env_cfg.gamma, theta=args.theta, max_iter=args.max_iter)

    rows = []
    for s in non_terminal:
        rows.append(
            {
                "state": s,
                "covered": s in covered_states,
                "critic_pred": float(critic_pred_full[s]),
                "true_value": float(true_value[s]),
                "abs_error": float(abs(critic_pred_full[s] - true_value[s])),
                "signed_error": float(critic_pred_full[s] - true_value[s]),
            }
        )
    df = pd.DataFrame(rows)

    def report(label: str, sub: pd.DataFrame):
        print(
            f"  {label:10s} n={len(sub):4d}  mean_abs_error={sub['abs_error'].mean():.4f}  "
            f"mean_signed_error={sub['signed_error'].mean():+.4f}  "
            f"corr={sub['critic_pred'].corr(sub['true_value']):.4f}"
        )

    print("\n=== Critic accuracy: predicted vs. exact true V^pi_beta ===")
    report("all", df)
    report("covered", df[df["covered"]])
    report("uncovered", df[~df["covered"]])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "critic_accuracy.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    fig, ax = plt.subplots(figsize=(6, 6))
    covered_df = df[df["covered"]]
    uncovered_df = df[~df["covered"]]
    ax.scatter(uncovered_df["critic_pred"], uncovered_df["true_value"], s=12, alpha=0.5, label="uncovered by D", color="tab:gray")
    ax.scatter(covered_df["critic_pred"], covered_df["true_value"], s=12, alpha=0.7, label="covered by D", color="tab:blue")
    lo = min(df["critic_pred"].min(), df["true_value"].min())
    hi = max(df["critic_pred"].max(), df["true_value"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect calibration (y=x)")
    ax.set_xlabel("critic's predicted V(s)")
    ax.set_ylabel("exact true V\u03c0\u03b2(s)")
    ax.set_title("pi_beta critic accuracy: predicted vs. exact true value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = out_dir / "critic_accuracy_scatter"
    fig.savefig(plot_path.with_suffix(".svg"))
    fig.savefig(plot_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}.svg/.png")


if __name__ == "__main__":
    main()
