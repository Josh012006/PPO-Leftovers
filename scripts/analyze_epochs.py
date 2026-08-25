"""H4 ablation: how far does the epochs hyperparameter alone move things?

Trains a SINGLE long fixed-D PPO run (default 300 epochs, `configs/
ppo_fixed_d_epochs_analysis.yaml`) on the already-frozen `D` / `pi_beta`
from the baseline run, evaluating live success_rate/mean_return on
`--eval-episodes` FIXED held-out episodes every `--checkpoint-every`
epochs (default 5), plus once before any training at all (epoch 0 =
pi_beta itself, unmodified). This directly probes whether the earlier
epochs=10-vs-30 null result (see README, "Results analysis") holds all
the way out to 300, or whether it was just an early-training artifact.

No hidden overfitting risk in the success_rate/mean_return curves: D is
fixed and these eval episodes are never part of D or of any fixed-D
training batch. The eval seed used here (`--eval-seed`, default 24680) is
also deliberately distinct from every other eval seed already in use
elsewhere in this project (tracking=12345, confirmation=54321, script 05's
final report=999) -- this is its own independent held-out set, not a reuse
of any seed a checkpoint was ever selected against.

Outputs, all under `--out-dir` (default results/analysis_epochs/):
  epochs_analysis.csv                 -- epoch, mean_return(+stderr),
                                          success_rate(+stderr), clip_frac,
                                          entropy
  epochs_analysis_success_return.png  -- success_rate & mean_return vs epoch
  epochs_analysis_clip_entropy.png    -- clip_frac & entropy vs epoch

This is a long run by design (up to `epochs` epochs of training, plus
epochs/checkpoint_every + 1 live evaluations of `eval_episodes` episodes
each) -- expect it to take a while.

Usage:
    python scripts/analyze_epochs.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --ppo-config configs/ppo_fixed_d_epochs_analysis.yaml \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 24680 \
        --out-dir results/analysis_epochs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy, make_neural_act_fn
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--ppo-config", default="configs/ppo_fixed_d_epochs_analysis.yaml")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=24680,
        help="Deliberately distinct from every other eval seed already used in this project "
        "(tracking=12345, confirmation=54321, script 05's report=999).",
    )
    parser.add_argument("--out-dir", default="results/analysis")
    args = parser.parse_args()

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    cfg = PPOHyperparams.from_yaml(args.ppo_config)
    set_global_seed(cfg.seed)

    def make_env():
        return StochasticMazeEnv(
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

    eval_env = make_env()
    dataset = load_dataset(args.dataset)
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes.")

    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_state_dict = ckpt["state_dict"]
    print(f"theta and pi_old both start from the prior checkpoint (final eval: {ckpt['final_eval']})")

    trainer = FixedDPPOTrainer(
        dataset,
        obs_dim=dataset.obs_dim,
        n_actions=dataset.n_actions,
        cfg=cfg,
        prior_state_dict=prior_state_dict,
    )

    rows: list[dict] = []

    def live_eval(net) -> dict:
        act_fn = make_neural_act_fn(net, deterministic=True)
        return evaluate_policy(eval_env, act_fn, args.eval_episodes, seed=args.eval_seed)

    print(
        f"\nEvaluating epoch 0 (pi_beta, before any fixed-D update) on {args.eval_episodes} "
        f"held-out episodes (seed={args.eval_seed})..."
    )
    res0 = live_eval(trainer.net)  # theta == pi_beta exactly at this point, before .train() runs
    entropy0 = trainer.compute_mean_entropy_over_dataset()
    rows.append(
        {
            "epoch": 0,
            "mean_return": res0["mean_return"],
            "mean_return_stderr": res0["stderr_return"],
            "success_rate": res0["success_rate"],
            "success_rate_stderr": res0["success_rate_stderr"],
            "clip_frac": 0.0,  # theta == pi_old exactly here: ratio == 1 everywhere, nothing clipped
            "entropy": entropy0,
        }
    )
    print(
        f"[epoch    0] success_rate={res0['success_rate']:.3f}\u00b1{res0['success_rate_stderr']:.3f} "
        f"mean_return={res0['mean_return']:.3f} entropy={entropy0:.4f} clip_frac=0.0000"
    )

    def eval_callback(epoch: int, net, summary: dict):
        res = live_eval(net)
        rows.append(
            {
                "epoch": epoch,
                "mean_return": res["mean_return"],
                "mean_return_stderr": res["stderr_return"],
                "success_rate": res["success_rate"],
                "success_rate_stderr": res["success_rate_stderr"],
                "clip_frac": summary["clip_frac"],
                "entropy": summary["entropy"],
            }
        )
        print(
            f"[epoch {epoch:4d}] success_rate={res['success_rate']:.3f}\u00b1{res['success_rate_stderr']:.3f} "
            f"mean_return={res['mean_return']:.3f} entropy={summary['entropy']:.4f} "
            f"clip_frac={summary['clip_frac']:.4f}"
        )

    print(
        f"\nTraining {cfg.epochs} epochs total, evaluating every {args.checkpoint_every} on "
        f"{args.eval_episodes} fixed held-out episodes...\n"
    )
    trainer.train(verbose=False, eval_every_epochs=args.checkpoint_every, eval_callback=eval_callback)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "epochs_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # --- Plot 1: success_rate & mean_return vs epoch ---
    fig, ax1 = plt.subplots(figsize=(9, 5))
    c1, c2 = "tab:blue", "tab:orange"
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("success_rate", color=c1)
    ax1.plot(df["epoch"], df["success_rate"], color=c1, marker="o", markersize=3, label="success_rate")
    ax1.fill_between(
        df["epoch"],
        df["success_rate"] - df["success_rate_stderr"],
        df["success_rate"] + df["success_rate_stderr"],
        color=c1,
        alpha=0.15,
    )
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2 = ax1.twinx()
    ax2.set_ylabel("mean_return", color=c2)
    ax2.plot(df["epoch"], df["mean_return"], color=c2, marker="s", markersize=3, label="mean_return")
    ax2.fill_between(
        df["epoch"],
        df["mean_return"] - df["mean_return_stderr"],
        df["mean_return"] + df["mean_return_stderr"],
        color=c2,
        alpha=0.15,
    )
    ax2.tick_params(axis="y", labelcolor=c2)
    plt.title(f"Held-out success_rate & mean_return vs. fixed-D PPO epochs (n={args.eval_episodes}/check)")
    fig.tight_layout()
    plot1_path = out_dir / "epochs_analysis_success_return.png"
    fig.savefig(plot1_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot1_path}")

    # --- Plot 2: clip_frac & entropy vs epoch ---
    fig, ax1 = plt.subplots(figsize=(9, 5))
    c1, c2 = "tab:green", "tab:red"
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("clip_frac", color=c1)
    ax1.plot(df["epoch"], df["clip_frac"], color=c1, marker="o", markersize=3, label="clip_frac")
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2 = ax1.twinx()
    ax2.set_ylabel("entropy", color=c2)
    ax2.plot(df["epoch"], df["entropy"], color=c2, marker="s", markersize=3, label="entropy")
    ax2.tick_params(axis="y", labelcolor=c2)
    plt.title("Training diagnostics (clip_frac & entropy) vs. fixed-D PPO epochs")
    fig.tight_layout()
    plot2_path = out_dir / "epochs_analysis_clip_entropy.png"
    fig.savefig(plot2_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot2_path}")


if __name__ == "__main__":
    main()