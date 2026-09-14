"""Does the sudden success_rate_held_out collapse (see README, H4;
scripts/analyze_heldout_drift.py) correspond to a genuine ARGMAX FLIP at
one or more held-out states -- a close top-2 action probability crossing
over -- even though the underlying KL/entropy drift (aggregated across
all held-out states) is smooth and gradual throughout, with no
discontinuity at the collapse epoch itself?

This tracks the FULL action-probability vector at EACH held-out state
INDIVIDUALLY (not aggregated, unlike analyze_heldout_drift.py) at every
checkpoint of a single fixed-D training run -- cheap forward passes only,
no rollout, no gradient ever touches these states directly. If one or
more states show their top-2 probabilities crossing right around the
epoch where success_rate_held_out is already known to collapse, that
confirms the mechanism: a smooth, continuous probability drift producing
a discontinuous DETERMINISTIC-argmax behavioral change, not a
discontinuity in the policy itself.

This does NOT explain WHY this specific pattern (drift, partial recovery,
renewed drift crossing a boundary) appears at clip_eps=0.2 specifically,
reproducing at nearly the same epoch under a different training seed,
while 0.1 shows none and 0.3/0.4 show a delayed/absent version -- that is
a separate question about what's special about that particular clip
width's optimization dynamics on this D, not something a single run's
probability trace can answer on its own.

Outputs, under --out-dir (default results/phase2/analysis/heldout_argmax_flip/):
  heldout_argmax_flip.csv       -- epoch, state, action_0..action_3 probs,
                                    argmax_action
  heldout_argmax_flip.svg/png   -- one panel per held-out state, all 4
                                    action probabilities vs. epoch

Usage:
    python scripts/analyze_heldout_argmax_flip.py \
        --env-config configs/phase2/env_maze.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --ppo-config configs/phase2/ppo_fixed_d_epochs_analysis.yaml \
        --checkpoint-every 5 \
        --out-dir results/phase2/analysis/heldout_argmax_flip
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
from ppo_exploitation.eval.evaluate import get_tier_start_lists
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams, StartTierConfig
from ppo_exploitation.utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument("--dataset", default="results/phase2/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/phase2/prior_checkpoint.pt")
    parser.add_argument("--start-tiers-config", required=True)
    parser.add_argument("--ppo-config", default="configs/phase2/ppo_fixed_d_epochs_analysis.yaml")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--out-dir", default="results/phase2/analysis/heldout_argmax_flip")
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
    _, held_out_starts = get_tier_start_lists(env, tier_cfg)
    print(f"Tracking {len(held_out_starts)} held-out states individually: {held_out_starts}")

    dataset = load_dataset(args.dataset)
    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    cfg = PPOHyperparams.from_yaml(args.ppo_config)
    set_global_seed(cfg.seed)
    trainer = FixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=ckpt["state_dict"]
    )
    print(f"clip_eps={cfg.clip_eps}, epochs={cfg.epochs}, seed={cfg.seed}")

    held_obs = torch.as_tensor(np.stack([env.state_to_obs(s) for s in held_out_starts]).astype(np.float32))
    n_actions = dataset.n_actions

    rows: list[dict] = []

    def probe(epoch: int, net):
        with torch.no_grad():
            logits, _ = net.forward(held_obs)
            probs = torch.softmax(logits, dim=-1).numpy()
        for i, s in enumerate(held_out_starts):
            row = {"epoch": epoch, "state": s, "argmax_action": int(np.argmax(probs[i]))}
            for a in range(n_actions):
                row[f"action_{a}"] = float(probs[i, a])
            rows.append(row)

    probe(0, trainer.net)

    def cb(epoch: int, net, summary: dict):
        probe(epoch, net)
        top2 = np.sort(np.stack([np.array([r[f"action_{a}"] for a in range(n_actions)]) for r in rows if r["epoch"] == epoch]), axis=-1)[:, -2:]
        gaps = top2[:, 1] - top2[:, 0]
        print(f"[epoch {epoch:4d}] min top-2 gap across held-out states: {gaps.min():.4f}")

    trainer.train(verbose=False, eval_every_epochs=args.checkpoint_every, eval_callback=cb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "heldout_argmax_flip.csv", index=False)

    n_states = len(held_out_starts)
    fig, axes = plt.subplots(1, n_states, figsize=(5 * n_states, 4.5), sharey=True)
    if n_states == 1:
        axes = [axes]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, s in enumerate(held_out_starts):
        sub = df[df["state"] == s].sort_values("epoch")
        ax = axes[i]
        for a in range(n_actions):
            ax.plot(sub["epoch"], sub[f"action_{a}"], color=colors[a % len(colors)], label=f"action {a}")
        flips = sub["argmax_action"].to_numpy()
        flip_epochs = sub["epoch"].to_numpy()[1:][flips[1:] != flips[:-1]]
        for fe in flip_epochs:
            ax.axvline(fe, color="black", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_title(f"held-out state {s}" + (f"\nargmax flips at epoch(s) {list(flip_epochs)}" if len(flip_epochs) else "\nno argmax flip"))
        ax.set_xlabel("epoch")
        ax.set_ylim(-0.02, 1.02)
        if i == 0:
            ax.set_ylabel("action probability")
        ax.legend(fontsize=7)
    fig.suptitle(f"Per-state held-out action probabilities (clip_eps={cfg.clip_eps}, seed={cfg.seed})")
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_argmax_flip.svg")
    fig.savefig(out_dir / "heldout_argmax_flip.png", dpi=150)
    print(f"\nSaved {out_dir}/heldout_argmax_flip.csv and .svg/.png")


if __name__ == "__main__":
    main()
