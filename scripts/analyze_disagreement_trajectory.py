"""Is the disagreement on the 54 known problem states already present at
the earliest checkpoints, or does it develop as training goes on?

The leading mechanism so far (README, "Second hypothesis"): PPO repeats
gradient updates on the same frozen D for up to 300 epochs; a state-action
pair backed by very little data could get progressively overfit this
way -- thin, possibly-unrepresentative evidence turning into strong
conviction through repetition, something pi_D*'s exact one-shot counting
structurally cannot do. This script tests the TIME COURSE of that
directly, rather than only comparing pi_beta (epoch 0) against the final/
best checkpoint (which is all every earlier script in this project does).

Both possible outcomes are informative, not just one:
  - Disagreement already present at epoch 0 or very early, then roughly
    flat -> the bias looks baked in by the single, one-time advantage
    computation itself, not by repeated gradient descent -- overfitting-
    by-repetition is NOT the mechanism, something upstream of training is.
  - Disagreement builds up gradually as epochs increase (mean agreement
    with pi_D* trending down, or individual states' pi_theta(a*|s) drifting
    away from pi_beta's own starting point over many epochs) -> consistent
    with the repeated-updates-on-thin-data story.
  - Abrupt jumps between two states rather than a gradual trend -> more
    consistent with the oscillation already documented extensively
    elsewhere in this project (see "Epoch-count ceiling analysis") than
    with either story above.

Retrains the best configuration (same as scripts/analyze_policy_agreement.py,
deterministic given the same seed/config) but, instead of the expensive
per-state discounted rollout that script does for ALL ~887 states, this
only ever does cheap forward passes (no rollout, no environment stepping)
over the FIXED list of already-known disagreement states, at every
--checkpoint-every epochs. Fast regardless of how many episodes-per-state
the original analysis used.

CHECKPOINTS ARE SAVED (by default): one *.pt per recorded epoch, under
--checkpoints-dir. Each is small (this project's networks are tiny), and
saving them means never having to retrain from scratch just to compute a
DIFFERENT per-checkpoint quantity later (a different diagnostic, a larger
tracked-state set, etc.) -- only this one 300-epoch retrain is ever
needed. Pass --no-save-checkpoints to skip this if disk space is a
genuine concern.

Outputs, under --out-dir (default results/analysis/disagreement_trajectory/):
  checkpoints/epoch_XXXX.pt         -- (unless --no-save-checkpoints) one
                                        checkpoint per recorded epoch --
                                        state_dict, epoch, obs_dim,
                                        n_actions, hidden_sizes, same
                                        shape as every other checkpoint
                                        saved elsewhere in this project
  disagreement_trajectory.csv       -- long format, one row per
                                        (epoch, state): pi_d_star_action,
                                        argmax_action, agrees,
                                        prob_pi_d_star_action
  disagreement_trajectory_summary.csv -- one row per epoch: fraction of
                                        the tracked states that agree with
                                        pi_D*, mean probability assigned
                                        to pi_D*'s action
  disagreement_trajectory_aggregate.svg/png -- fraction agreeing (and mean
                                        prob_pi_d_star_action) vs. epoch,
                                        one line each
  disagreement_trajectory_spaghetti.svg/png -- every tracked state's own
                                        prob_pi_d_star_action(epoch) curve
                                        overlaid, colored by whether it
                                        ends up agreeing or not

Usage:
    python scripts/analyze_disagreement_trajectory.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --pi-d-star results/pi_d_star_empirical.pkl \
        --best-config configs/ppo_fixed_d_best_config.yaml \
        --policy-agreement-csv results/analysis/policy_agreement/policy_agreement.csv \
        --checkpoint-every 5 \
        --out-dir results/analysis/disagreement_trajectory
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star", default="results/pi_d_star_empirical.pkl")
    parser.add_argument("--best-config", default="configs/ppo_fixed_d_best_config.yaml")
    parser.add_argument(
        "--policy-agreement-csv",
        default="results/analysis/policy_agreement/policy_agreement.csv",
        help="Used only to pick which states to track: every state with is_disagreement==1 "
        "there. The final-checkpoint numbers from that CSV are not otherwise reused here -- "
        "this script retrains from scratch (deterministically) to get every intermediate "
        "checkpoint, which no earlier script saved.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--out-dir", default="results/analysis/disagreement_trajectory")
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one *.pt per recorded epoch under --checkpoints-dir (default: on). Each is "
        "small; this avoids ever needing to retrain from scratch again to compute a different "
        "per-checkpoint quantity later. Use --no-save-checkpoints to skip.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=None,
        help="Where to save per-epoch checkpoints. Defaults to <out-dir>/checkpoints.",
    )
    parser.add_argument(
        "--spaghetti-sample",
        type=int,
        default=0,
        help="If >0, only this many randomly-chosen tracked states are drawn as individual "
        "lines in the spaghetti plot (still all of them contribute to the aggregate CSV/plot). "
        "0 (default) draws all of them -- fine up to the ~54 states this project has seen so "
        "far, but set this if a future disagreement set is much larger.",
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

    pa = pd.read_csv(args.policy_agreement_csv)
    tracked = pa[pa["is_disagreement"] == 1][["state", "pi_d_star_action"]].reset_index(drop=True)
    tracked_states = tracked["state"].to_numpy(dtype=int)
    a_star_by_state = dict(zip(tracked["state"].tolist(), tracked["pi_d_star_action"].tolist()))
    print(f"Tracking {len(tracked_states)} known disagreement states from {args.policy_agreement_csv}.")
    if len(tracked_states) == 0:
        print("No disagreement states found in that CSV -- nothing to track. Exiting.")
        return

    dataset = load_dataset(args.dataset)
    prior_ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_state_dict = prior_ckpt["state_dict"]
    print(f"Loaded prior checkpoint (final eval: {prior_ckpt['final_eval']})")

    with open(args.pi_d_star, "rb") as f:
        ref = pickle.load(f)
    print(f"Loaded pi_D* ({ref.kind}) from {args.pi_d_star}")

    cfg = PPOHyperparams.from_yaml(args.best_config)
    set_global_seed(cfg.seed)

    trainer = FixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=prior_state_dict
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else out_dir / "checkpoints"
    if args.save_checkpoints:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    obs_batch = np.stack([env.state_to_obs(s) for s in tracked_states]).astype(np.float32)
    obs_t = torch.as_tensor(obs_batch)
    a_star_arr = np.array([a_star_by_state[s] for s in tracked_states], dtype=int)

    rows: list[dict] = []

    def save_checkpoint(epoch: int, net) -> None:
        if not args.save_checkpoints:
            return
        torch.save(
            {
                "state_dict": net.state_dict(),
                "epoch": epoch,
                "obs_dim": dataset.obs_dim,
                "n_actions": dataset.n_actions,
                "hidden_sizes": cfg.hidden_sizes,
            },
            checkpoints_dir / f"epoch_{epoch:04d}.pt",
        )

    def record(epoch: int, net) -> float:
        with torch.no_grad():
            logits, _ = net.forward(obs_t)
            probs = torch.softmax(logits, dim=-1).numpy()
        argmax_actions = probs.argmax(axis=1)
        agrees = (argmax_actions == a_star_arr).astype(int)
        prob_a_star = probs[np.arange(len(tracked_states)), a_star_arr]
        for i, s in enumerate(tracked_states):
            rows.append(
                {
                    "epoch": epoch,
                    "state": int(s),
                    "pi_d_star_action": int(a_star_arr[i]),
                    "argmax_action": int(argmax_actions[i]),
                    "agrees": int(agrees[i]),
                    "prob_pi_d_star_action": float(prob_a_star[i]),
                }
            )
        save_checkpoint(epoch, net)
        return float(agrees.mean())

    frac0 = record(0, trainer.net)  # theta == pi_beta exactly here, before .train() runs
    print(f"[epoch    0] {frac0:.3f} of tracked states already agree with pi_D* (pi_beta itself)")

    def eval_callback(epoch: int, net, summary: dict):
        frac = record(epoch, net)
        print(f"[epoch {epoch:4d}] {frac:.3f} of tracked states agree with pi_D*")

    print(f"\nTraining {cfg.epochs} epochs, recording every {args.checkpoint_every}...\n")
    trainer.train(verbose=False, eval_every_epochs=args.checkpoint_every, eval_callback=eval_callback)

    if args.save_checkpoints:
        n_saved = len(list(checkpoints_dir.glob("epoch_*.pt")))
        print(f"\nSaved {n_saved} checkpoints under {checkpoints_dir}/ -- reusable for any later "
              f"per-checkpoint analysis without retraining.")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "disagreement_trajectory.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    summary = df.groupby("epoch").agg(
        frac_agree=("agrees", "mean"), mean_prob_pi_d_star_action=("prob_pi_d_star_action", "mean")
    ).reset_index()
    summary_path = out_dir / "disagreement_trajectory_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")

    final_epoch = summary["epoch"].max()
    frac_start = summary.loc[summary["epoch"] == 0, "frac_agree"].iloc[0]
    frac_end = summary.loc[summary["epoch"] == final_epoch, "frac_agree"].iloc[0]
    frac_peak = summary["frac_agree"].max()
    print(
        f"\nfrac_agree: epoch 0 = {frac_start:.3f}, epoch {final_epoch} = {frac_end:.3f}, "
        f"peak over the run = {frac_peak:.3f}"
    )
    if frac_start >= frac_peak - 1e-9:
        print(
            "  -> already at (or above) its best right from epoch 0: consistent with the bias "
            "being present from the single, one-time advantage computation, not built up by "
            "repeated updates."
        )
    else:
        print(
            "  -> improves after epoch 0 before whatever it ends at: at least some of this is "
            "NOT present from the start. Check the aggregate/spaghetti plots for whether that "
            "improvement holds, oscillates, or reverses."
        )

    # --- Plot 1: aggregate trajectory ---
    fig, ax1 = plt.subplots(figsize=(9, 5))
    c1, c2 = "tab:blue", "tab:orange"
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("fraction of tracked states agreeing with \u03c0D*", color=c1)
    ax1.plot(summary["epoch"], summary["frac_agree"], color=c1, marker="o", markersize=3)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(-0.02, 1.02)
    ax2 = ax1.twinx()
    ax2.set_ylabel("mean probability assigned to \u03c0D*'s action", color=c2)
    ax2.plot(summary["epoch"], summary["mean_prob_pi_d_star_action"], color=c2, marker="s", markersize=3)
    ax2.tick_params(axis="y", labelcolor=c2)
    plt.title(f"The {len(tracked_states)} known disagreement states: agreement with \u03c0D* over training")
    fig.tight_layout()
    plot1 = out_dir / "disagreement_trajectory_aggregate"
    fig.savefig(plot1.with_suffix(".svg"))
    fig.savefig(plot1.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot1}.svg/.png")

    # --- Plot 2: spaghetti -- every tracked state's own curve ---
    rng = np.random.default_rng(0)
    plot_states = tracked_states
    if args.spaghetti_sample > 0 and args.spaghetti_sample < len(tracked_states):
        plot_states = rng.choice(tracked_states, size=args.spaghetti_sample, replace=False)
    final_agree_by_state = df[df["epoch"] == final_epoch].set_index("state")["agrees"].to_dict()
    fig, ax = plt.subplots(figsize=(9, 6))
    for s in plot_states:
        sub = df[df["state"] == s].sort_values("epoch")
        ends_agreeing = final_agree_by_state.get(int(s), 0)
        color = "tab:green" if ends_agreeing else "tab:red"
        ax.plot(sub["epoch"], sub["prob_pi_d_star_action"], color=color, alpha=0.35, linewidth=1)
    ax.axhline(0.25, color="0.5", linestyle=":", linewidth=1, label="uniform (0.25, 4 actions)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("probability assigned to \u03c0D*'s action")
    ax.set_title(
        f"Individual trajectories ({len(plot_states)} states) -- "
        f"green = agrees with \u03c0D* at epoch {final_epoch}, red = does not"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot2 = out_dir / "disagreement_trajectory_spaghetti"
    fig.savefig(plot2.with_suffix(".svg"))
    fig.savefig(plot2.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot2}.svg/.png")


if __name__ == "__main__":
    main()
