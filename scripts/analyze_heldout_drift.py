"""Why does the held-out collapse (see README, H4) happen specifically at
clip_eps=0.2, reproducing at nearly the same epoch under a different
training seed, while 0.1 shows none and 0.3/0.4 show a delayed/absent
version of it? clip_frac and entropy (already tracked in every epochs/h7
run) are computed OVER D -- which contains zero held-out transitions --
so they are blind to whatever is actually happening at held-out states by
construction. This script tracks what those diagnostics cannot:

  - entropy/KL(pi_beta || pi_theta) evaluated DIRECTLY at the held-out
    states themselves, via cheap forward passes (no rollout, no gradient
    ever touches these states -- this is pure generalization drift)
  - the same two quantities at the covered states, for contrast
  - the network's total weight L2 norm, a generic "sharpening" signal

at every checkpoint of a single fixed-D training run, so the moment
(if any) where held-out-specific drift accelerates or jumps can be lined
up directly against the epoch where success_rate_held_out actually
collapses.

No dataset D changes, no new training method -- this is pure
instrumentation around the exact same FixedDPPOTrainer every other
script in this project already uses.

Outputs, under --out-dir (default results/phase2/analysis/heldout_drift/):
  heldout_drift.csv       -- epoch, entropy/kl at held-out and covered,
                              weight_norm
  heldout_drift.svg/png   -- three panels: entropy (both populations),
                              KL from pi_beta (both populations), weight
                              norm -- all vs. epoch

Usage:
    python scripts/analyze_heldout_drift.py \
        --env-config configs/phase2/env_maze.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --ppo-config configs/phase2/ppo_fixed_d_epochs_analysis.yaml \
        --checkpoint-every 5 \
        --out-dir results/phase2/analysis/heldout_drift
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
    parser.add_argument("--out-dir", default="results/phase2/analysis/heldout_drift")
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
    covered_starts, held_out_starts = get_tier_start_lists(env, tier_cfg)

    dataset = load_dataset(args.dataset)
    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    cfg = PPOHyperparams.from_yaml(args.ppo_config)
    set_global_seed(cfg.seed)
    trainer = FixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=ckpt["state_dict"]
    )
    print(f"clip_eps={cfg.clip_eps}, epochs={cfg.epochs}, seed={cfg.seed}")

    held_obs = torch.as_tensor(np.stack([env.state_to_obs(s) for s in held_out_starts]).astype(np.float32))
    covered_obs = torch.as_tensor(np.stack([env.state_to_obs(s) for s in covered_starts]).astype(np.float32))

    with torch.no_grad():
        logits_beta_h, _ = trainer.net.forward(held_obs)
        probs_beta_h = torch.softmax(logits_beta_h, dim=-1)
        logits_beta_c, _ = trainer.net.forward(covered_obs)
        probs_beta_c = torch.softmax(logits_beta_c, dim=-1)

    rows = []

    def probe(epoch: int, net):
        with torch.no_grad():
            logits_h, _ = net.forward(held_obs)
            probs_h = torch.softmax(logits_h, dim=-1)
            ent_h = -(probs_h * torch.log(probs_h + 1e-12)).sum(dim=-1).mean().item()
            kl_h = (probs_beta_h * torch.log((probs_beta_h + 1e-12) / (probs_h + 1e-12))).sum(dim=-1).mean().item()

            logits_c, _ = net.forward(covered_obs)
            probs_c = torch.softmax(logits_c, dim=-1)
            ent_c = -(probs_c * torch.log(probs_c + 1e-12)).sum(dim=-1).mean().item()
            kl_c = (probs_beta_c * torch.log((probs_beta_c + 1e-12) / (probs_c + 1e-12))).sum(dim=-1).mean().item()

            wnorm = sum(p.norm().item() ** 2 for p in net.parameters()) ** 0.5
        rows.append(
            {"epoch": epoch, "entropy_heldout": ent_h, "kl_heldout": kl_h, "entropy_covered": ent_c, "kl_covered": kl_c, "weight_norm": wnorm}
        )
        print(f"[epoch {epoch:4d}] entropy_heldout={ent_h:.4f} kl_heldout={kl_h:.4f} weight_norm={wnorm:.3f}")

    probe(0, trainer.net)

    def cb(epoch: int, net, summary: dict):
        probe(epoch, net)

    trainer.train(verbose=False, eval_every_epochs=args.checkpoint_every, eval_callback=cb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "heldout_drift.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    ax.plot(df["epoch"], df["entropy_heldout"], label="held-out", color="tab:red")
    ax.plot(df["epoch"], df["entropy_covered"], label="covered", color="tab:green")
    ax.set_xlabel("epoch"); ax.set_ylabel("mean entropy of pi_theta"); ax.legend(fontsize=8); ax.set_title("Policy entropy")

    ax = axes[1]
    ax.plot(df["epoch"], df["kl_heldout"], label="held-out", color="tab:red")
    ax.plot(df["epoch"], df["kl_covered"], label="covered", color="tab:green")
    ax.set_xlabel("epoch"); ax.set_ylabel("KL(pi_beta || pi_theta)"); ax.legend(fontsize=8); ax.set_title("Drift from pi_beta")

    ax = axes[2]
    ax.plot(df["epoch"], df["weight_norm"], color="tab:purple")
    ax.set_xlabel("epoch"); ax.set_ylabel("||theta||_2"); ax.set_title("Weight norm")

    fig.suptitle(f"Held-out-specific drift (clip_eps={cfg.clip_eps}, seed={cfg.seed}) -- diagnostics blind in D itself")
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_drift.svg")
    fig.savefig(out_dir / "heldout_drift.png", dpi=150)
    print(f"\nSaved {out_dir}/heldout_drift.csv and .svg/.png")


if __name__ == "__main__":
    main()
