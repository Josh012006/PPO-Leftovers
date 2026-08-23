"""Train PPO on the fixed dataset D only -- no environment interaction.
Pass --ppo-config configs/ppo_fixed_d_standard.yaml for the baseline run, or
any other ppo_fixed_d_*.yaml for a modified/ablation run. This script's code
path never changes between the two; only the YAML does.

Usage:
    python scripts/04_train_fixed_d_ppo.py \
        --dataset results/dataset_D.pkl \
        --ppo-config configs/ppo_fixed_d_standard.yaml \
        --prior-checkpoint results/prior_checkpoint.pt \
        --out results/ppo_standard_on_D.pt \
        --history-out results/ppo_standard_on_D_history.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.utils.config import PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--ppo-config", required=True)
    parser.add_argument(
        "--prior-checkpoint",
        default=None,
        help="Required only if the config's init_from_checkpoint=true.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--history-out", default=None)
    args = parser.parse_args()

    cfg = PPOHyperparams.from_yaml(args.ppo_config)
    set_global_seed(cfg.seed)
    dataset = load_dataset(args.dataset)
    print(
        f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes "
        f"(this run must be trained on the SAME D as every other comparison run)."
    )

    init_state_dict = None
    if cfg.init_from_checkpoint:
        if args.prior_checkpoint is None:
            raise ValueError("Config has init_from_checkpoint=true; pass --prior-checkpoint.")
        ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
        init_state_dict = ckpt["state_dict"]
        print(f"Initializing from prior checkpoint (final eval: {ckpt['final_eval']})")
    else:
        print("Initializing from a fresh random network (init_from_checkpoint=false).")

    trainer = FixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, init_state_dict=init_state_dict
    )
    history = trainer.train(verbose=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": trainer.state_dict(),
            "obs_dim": dataset.obs_dim,
            "n_actions": dataset.n_actions,
            "hidden_sizes": cfg.hidden_sizes,
            "ppo_config_path": args.ppo_config,
            "dataset_path": args.dataset,
        },
        out_path,
    )
    print(f"Saved trained fixed-D PPO policy to {out_path}")

    if args.history_out:
        Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(args.history_out, index=False)
        print(f"Saved training diagnostics history to {args.history_out}")


if __name__ == "__main__":
    main()
