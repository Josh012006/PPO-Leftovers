"""Shared logic between scripts/analyze_epochs.py and scripts/analyze_h7.py
(and any future single-window analysis script): the plotting functions and
the "train one fixed-D PPO config for up to `epochs` epochs, checkpointing
live eval every N epochs" routine. Not a standalone entrypoint -- import
from it, don't run it directly.

Kept here (scripts/) rather than under src/ppo_exploitation because this is
orchestration/reporting logic specific to the analysis scripts, not part of
the core research library.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ppo_exploitation.eval.evaluate import evaluate_policy, make_neural_act_fn, make_tabular_act_fn
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.utils.config import PPOHyperparams


# --------------------------------------------------------------------------
# Plotting -- each function saves both an .svg and a .png from the same
# figure, and is pure (only reads its DataFrame argument), so it can also be
# reused directly against previously-saved CSVs without re-running training,
# e.g. when regenerating plots after a styling change.
# --------------------------------------------------------------------------
def plot_success_return(
    df: pd.DataFrame, out_path_stem: Path, title: str, prior_success_rate: float, ceiling_success_rate: float
) -> float:
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


# --------------------------------------------------------------------------
# Ceiling computation -- re-evaluates pi_D* (empirical) under the SAME
# eval_seed/eval_episodes as everything else in a given analysis run, on
# purpose (see analyze_epochs.py's module docstring for why this must not
# be reused from a report computed under a different seed).
# --------------------------------------------------------------------------
def compute_ceiling_success_rate(eval_env, pi_d_star_empirical_path: str, eval_episodes: int, eval_seed: int) -> float:
    with open(pi_d_star_empirical_path, "rb") as f:
        ref_empirical = pickle.load(f)
    ceiling_res = evaluate_policy(
        eval_env, make_tabular_act_fn(ref_empirical), eval_episodes, seed=eval_seed,
        covered_states=ref_empirical.covered_states,
    )
    return ceiling_res["success_rate"]


# --------------------------------------------------------------------------
# The actual "train one config, checkpoint every N epochs" routine.
# --------------------------------------------------------------------------
def run_single_analysis(
    eval_env,
    dataset,
    prior_state_dict: dict,
    ceiling_success_rate: float,
    cfg: PPOHyperparams,
    checkpoint_every: int,
    eval_episodes: int,
    eval_seed: int,
    out_dir: Path,
    prefix: str,
    title_suffix: str = "",
    verbose: bool = True,
    log_prefix: str = "",
) -> dict:
    """Runs one fixed-D training + periodic-eval sweep for a single
    PPOHyperparams config. Saves `<prefix>.csv`, `<prefix>_success_return
    .svg/.png`, `<prefix>_clip_entropy.svg/.png` under `out_dir`. Returns a
    small summary dict (best/mean/std/final success_rate + the csv path)
    for cross-run comparison tables, e.g. from a grid sweep.

    `log_prefix` is prepended to every per-epoch print line -- used by
    analyze_h7.py to make clear, in a long combined log, which grid
    combination a given line belongs to.
    """
    trainer = FixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=prior_state_dict
    )
    rows: list[dict] = []

    def live_eval(net) -> dict:
        return evaluate_policy(eval_env, make_neural_act_fn(net, deterministic=True), eval_episodes, seed=eval_seed)

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
    if verbose:
        print(
            f"{log_prefix}[epoch    0] success_rate={res0['success_rate']:.3f}\u00b1{res0['success_rate_stderr']:.3f} "
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
        if verbose:
            print(
                f"{log_prefix}[epoch {epoch:4d}] success_rate={res['success_rate']:.3f}\u00b1{res['success_rate_stderr']:.3f} "
                f"mean_return={res['mean_return']:.3f} entropy={summary['entropy']:.4f} "
                f"clip_frac={summary['clip_frac']:.4f}"
            )

    trainer.train(verbose=False, eval_every_epochs=checkpoint_every, eval_callback=eval_callback)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)

    achieved_best = plot_success_return(
        df,
        out_dir / f"{prefix}_success_return",
        f"success_rate & mean_return vs. epoch ({title_suffix or prefix})",
        prior_success_rate=res0["success_rate"],
        ceiling_success_rate=ceiling_success_rate,
    )
    plot_clip_entropy(
        df,
        out_dir / f"{prefix}_clip_entropy",
        f"clip_frac & entropy vs. epoch ({title_suffix or prefix})",
    )

    return {
        "prefix": prefix,
        "best": achieved_best,
        "mean": float(df["success_rate"].mean()),
        "std": float(df["success_rate"].std()),
        "final": float(df.iloc[-1]["success_rate"]),
        "csv_path": str(csv_path),
    }