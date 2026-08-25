"""H4 ablation: how far does the epochs hyperparameter alone move things?

Trains a SINGLE long fixed-D PPO run (default 300 epochs, e.g. `configs/
ppo_fixed_d_epochs_analysis.yaml`) on the already-frozen `D` / `pi_beta`
from the baseline run, evaluating live success_rate/mean_return on
`--eval-episodes` FIXED held-out episodes every `--checkpoint-every`
epochs (default 5), plus once before any training at all (epoch 0 =
pi_beta itself, unmodified). This directly probes whether an epochs-count
null result at small scale (e.g. 10 vs 30) holds all the way out to 300,
or whether it was just an early-training artifact. Re-run with different
`--ppo-config` files (varying e.g. clip_eps) to probe other hyperparameters
against the same epoch-count axis -- see README, "Results analysis" for
the clip_eps=0.1/0.2/0.3/0.4 sweep this was used for.

No hidden overfitting risk in the success_rate/mean_return curves: D is
fixed and these eval episodes are never part of D or of any fixed-D
training batch. The eval seed used here (`--eval-seed`, default 24680) is
also deliberately distinct from every other eval seed already in use
elsewhere in this project (tracking=12345, confirmation=54321, script 05's
final report=999) -- this is its own independent held-out set, not a reuse
of any seed a checkpoint was ever selected against. The pi_D* ceiling
reference line plotted alongside success_rate is evaluated under this
SAME seed/episode-count, for the same reason -- comparing it against a
number computed under a different eval sample (e.g. script 05's report,
seed=999) would reintroduce exactly the kind of cross-sample noise this
project has been careful to keep separate elsewhere.

Outputs, all under `--out-dir` (default results/analysis/), each as both
.svg and .png:
  <prefix>.csv                    -- epoch, mean_return(+stderr),
                                      success_rate(+stderr), clip_frac,
                                      entropy
  <prefix>_success_return.svg/png -- success_rate & mean_return vs epoch,
                                      with reference lines for the prior,
                                      the pi_D* ceiling, and this run's
                                      best observed success_rate
  <prefix>_clip_entropy.svg/png   -- clip_frac & entropy vs epoch

This is a long run by design (up to `epochs` epochs of training, plus
epochs/checkpoint_every + 1 live evaluations of `eval_episodes` episodes
each) -- expect it to take a while.

Usage:
    python scripts/analyze_epochs.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --pi-d-star-empirical results/pi_d_star_empirical.pkl \
        --ppo-config configs/ppo_fixed_d_epochs_analysis.yaml \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 24680 \
        --out-dir results/analysis --prefix epochs_analysis
"""
from __future__ import annotations

import argparse
import pickle
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
from ppo_exploitation.eval.evaluate import evaluate_policy, make_neural_act_fn, make_tabular_act_fn
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


# --------------------------------------------------------------------------
# Plotting -- pulled out as standalone functions (rather than inlined in
# main()) so they can also be reused directly against previously-saved CSVs
# without re-running training, e.g. when regenerating plots after a styling
# change. Each function saves both an .svg and a .png.
# --------------------------------------------------------------------------
def plot_success_return(df: pd.DataFrame, out_path_stem: Path, title: str, prior_success_rate: float, ceiling_success_rate: float) -> float:
    """Returns the achieved best success_rate (also drawn as a reference
    line), so callers can report it alongside the plot."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    c1, c2 = "tab:blue", "tab:orange"
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("success_rate", color=c1)

    # mean_return (ax2) is drawn first so success_rate (ax1) ends up on top
    # wherever the two would otherwise overlap; the explicit zorder/patch
    # call below enforces this regardless of matplotlib's default twinx
    # stacking.
    ax2 = ax1.twinx()
    ax2.set_ylabel("mean_return", color=c2)
    l2 = ax2.plot(df["epoch"], df["mean_return"], color=c2, marker="s", markersize=3, label="mean_return")[0]
    ax2.fill_between(
        df["epoch"], df["mean_return"] - df["mean_return_stderr"], df["mean_return"] + df["mean_return_stderr"],
        color=c2, alpha=0.15,
    )
    ax2.tick_params(axis="y", labelcolor=c2)

    l1 = ax1.plot(df["epoch"], df["success_rate"], color=c1, marker="o", markersize=3, label="success_rate")[0]
    ax1.fill_between(
        df["epoch"], df["success_rate"] - df["success_rate_stderr"], df["success_rate"] + df["success_rate_stderr"],
        color=c1, alpha=0.15,
    )
    ax1.tick_params(axis="y", labelcolor=c1)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    achieved_best = float(df["success_rate"].max())
    h_prior = ax1.axhline(prior_success_rate, color="0.4", linestyle="--", linewidth=1.3, label=f"prior \u03c0\u03b2 ({prior_success_rate:.3f})")
    h_ceiling = ax1.axhline(ceiling_success_rate, color="black", linestyle=":", linewidth=1.3, label=f"\u03c0D* ceiling ({ceiling_success_rate:.3f})")
    h_best = ax1.axhline(achieved_best, color="tab:purple", linestyle="-.", linewidth=1.3, label=f"PPO best, this run ({achieved_best:.3f})")

    handles = [l1, l2, h_prior, h_ceiling, h_best]
    ax1.legend(handles, [h.get_label() for h in handles], loc="lower right", fontsize=8)

    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix(".svg"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return achieved_best


def plot_clip_entropy(df: pd.DataFrame, out_path_stem: Path, title: str) -> None:
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
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix(".svg"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star-empirical", default="results/pi_d_star_empirical.pkl")
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
    parser.add_argument(
        "--prefix",
        default="epochs_analysis",
        help="Filename prefix, e.g. 'epochs_analysis_clip_0_3' when sweeping clip_eps across runs "
        "into the same --out-dir.",
    )
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

    with open(args.pi_d_star_empirical, "rb") as f:
        ref_empirical = pickle.load(f)
    ceiling_res = evaluate_policy(
        eval_env, make_tabular_act_fn(ref_empirical), args.eval_episodes, seed=args.eval_seed,
        covered_states=ref_empirical.covered_states,
    )
    ceiling_success_rate = ceiling_res["success_rate"]
    print(
        f"pi_D* (empirical) ceiling under this run's own eval protocol "
        f"(seed={args.eval_seed}, n={args.eval_episodes}): success_rate={ceiling_success_rate:.3f} "
        f"(deliberately re-evaluated here rather than reused from script 05's report, which used a "
        f"different eval seed)"
    )

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
    csv_path = out_dir / f"{args.prefix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    achieved_best = plot_success_return(
        df,
        out_dir / f"{args.prefix}_success_return",
        f"Held-out success_rate & mean_return vs. fixed-D PPO epochs (n={args.eval_episodes}/check, clip_eps={cfg.clip_eps})",
        prior_success_rate=res0["success_rate"],
        ceiling_success_rate=ceiling_success_rate,
    )
    print(f"Saved {out_dir / (args.prefix + '_success_return')}.svg/.png (best observed: {achieved_best:.3f})")

    plot_clip_entropy(
        df,
        out_dir / f"{args.prefix}_clip_entropy",
        f"Training diagnostics (clip_frac & entropy) vs. fixed-D PPO epochs (clip_eps={cfg.clip_eps})",
    )
    print(f"Saved {out_dir / (args.prefix + '_clip_entropy')}.svg/.png")


if __name__ == "__main__":
    main()