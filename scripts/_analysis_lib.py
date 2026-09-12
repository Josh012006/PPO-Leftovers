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

import copy
import pickle
from collections import Counter, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ppo_exploitation.eval.evaluate import (
    DEFAULT_EVAL_WEIGHTS,
    evaluate_policy,
    evaluate_policy_weighted,
    make_neural_act_fn,
    make_tabular_act_fn,
)
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.utils.config import PPOHyperparams


def compute_sa_counts(dataset) -> Counter:
    """Raw (state, action) -> sample count in D, as a Counter. Used by
    scripts/analyze_policy_agreement.py and
    scripts/analyze_disagreement_factors.py for PAIR-level (not summed
    over actions) coverage: a state can have high TOTAL coverage while the
    one specific action a policy ends up preferring there was sampled
    only a handful of times. (analyze_coverage_density.py and
    analyze_prior_correction.py independently re-implement the
    state-summed version of this same counting loop -- not yet
    consolidated onto this helper, since they weren't touched by this
    change.)"""
    counts: Counter = Counter()
    for tr in dataset.trajectories:
        for s, a in zip(tr.states.tolist(), tr.actions.tolist()):
            counts[(int(s), int(a))] += 1
    return counts


# --------------------------------------------------------------------------
# Maze-graph BFS distance -- shared by every script that needs a purely
# structural (policy-independent) notion of "how far is state s from X"
# (a hazard, the start, the goal), respecting the maze's actual walls via
# `env.layout.open_walls`. Previously duplicated in
# scripts/analyze_masked_pi_d_star.py; centralized here after that
# duplication became a real maintenance risk once a second script needed
# the same logic for a different source (start/goal, not just hazards).
# --------------------------------------------------------------------------
def bfs_distance_from(env, source_cells: list[tuple[int, int]]) -> dict[int, int]:
    """BFS distance (maze-graph hops) from every reachable state to the
    nearest of `source_cells` (a list of (row, col) tuples). States not
    reachable from any source are absent from the returned dict."""
    from ppo_exploitation.envs.stochastic_maze import ACTIONS, _ACTION_DELTA

    layout = env.layout
    dist: dict[int, int] = {}
    frontier: deque = deque()
    for (r, c) in source_cells:
        sid = layout.state_id(r, c)
        dist[sid] = 0
        frontier.append(sid)
    while frontier:
        s = frontier.popleft()
        r, c = layout.rc(s)
        for d in ACTIONS:
            if not layout.open_walls[r, c, d]:
                continue
            dr, dc = _ACTION_DELTA[d]
            nr, nc = r + dr, c + dc
            ns = layout.state_id(nr, nc)
            if ns not in dist:
                dist[ns] = dist[s] + 1
                frontier.append(ns)
    return dist


def hazard_bfs_distance(env) -> dict[int, int]:
    return bfs_distance_from(env, list(env.layout.hazards))


def start_bfs_distance(env) -> dict[int, int]:
    return bfs_distance_from(env, [env.layout.start])


def goal_bfs_distance(env) -> dict[int, int]:
    return bfs_distance_from(env, [env.layout.goal])


def local_connectivity(env) -> dict[int, int]:
    """Number of open walls (0-4) at each state -- a purely structural
    measure of how many alternative routes exist locally. Low connectivity
    means fewer ways to recover from a wrong turn near this state."""
    layout = env.layout
    out: dict[int, int] = {}
    for s in range(env.n_states):
        r, c = layout.rc(s)
        out[s] = int(layout.open_walls[r, c].sum())
    return out


# --------------------------------------------------------------------------
# Plotting -- each function saves both an .svg and a .png from the same
# figure, and is pure (only reads its DataFrame argument), so it can also be
# reused directly against previously-saved CSVs without re-running training,
# e.g. when regenerating plots after a styling change.
# --------------------------------------------------------------------------
def plot_success_return(
    df: pd.DataFrame,
    out_path_stem: Path,
    title: str,
    prior_success_rates: dict,
    ceiling_success_rates: dict,
) -> float:
    """Plots the THREE raw success_rate curves -- overall, covered,
    held-out -- against epoch (see README, "Our new starting point").
    mean_return is dropped from this plot (three success-rate lines are
    already a lot to read at once); it stays in the CSV for anyone who
    wants it. `prior_success_rates`/`ceiling_success_rates` are dicts with
    keys "overall"/"covered"/"held_out", each drawn as a thin reference
    line in that curve's own color. Returns the achieved best OVERALL
    success_rate (also drawn as a reference line), so callers can report
    it alongside the plot -- checkpoint SELECTION itself now uses the
    weighted metric (see plot_weighted_success / run_single_analysis),
    not this number."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"overall": "tab:blue", "covered": "tab:green", "held_out": "tab:red"}
    for key, label in [("overall", "overall"), ("covered", "covered"), ("held_out", "held-out")]:
        c = colors[key]
        ax.plot(df["epoch"], df[f"success_rate_{key}"], color=c, marker="o", markersize=3, label=f"success_rate ({label})")
        ax.fill_between(
            df["epoch"],
            df[f"success_rate_{key}"] - df[f"success_rate_{key}_stderr"],
            df[f"success_rate_{key}"] + df[f"success_rate_{key}_stderr"],
            color=c, alpha=0.12,
        )
        ax.axhline(prior_success_rates[key], color=c, linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(ceiling_success_rates[key], color=c, linestyle=":", linewidth=1, alpha=0.6)

    achieved_best = float(df["success_rate_overall"].max())
    ax.set_xlabel("epoch")
    ax.set_ylabel("success_rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="best")
    plt.title(title)
    fig.tight_layout()
    fig.savefig(out_path_stem.with_suffix(".svg"))
    fig.savefig(out_path_stem.with_suffix(".png"), dpi=150)
    plt.close(fig)
    return achieved_best


def plot_weighted_success(
    df: pd.DataFrame, out_path_stem: Path, title: str, prior_weighted: float, ceiling_weighted: float, weights: tuple
) -> float:
    """The additional plot requested alongside plot_success_return: JUST
    the weighted_success_rate curve (see README, "Our new starting
    point") -- the actual number used to decide which epoch/config is
    "better" throughout this project, not any of the three raw curves on
    their own. Returns the achieved best weighted_success_rate."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["weighted_success_rate"], color="tab:purple", marker="o", markersize=3, label="weighted_success_rate")
    achieved_best = float(df["weighted_success_rate"].max())
    ax.axhline(prior_weighted, color="0.4", linestyle="--", linewidth=1.3, label=f"prior \u03c0\u03b2 ({prior_weighted:.3f})")
    ax.axhline(ceiling_weighted, color="black", linestyle=":", linewidth=1.3, label=f"\u03c0D* ceiling ({ceiling_weighted:.3f})")
    ax.axhline(achieved_best, color="tab:orange", linestyle="-.", linewidth=1.3, label=f"PPO best, this run ({achieved_best:.3f})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"weighted_success_rate (overall={weights[0]}, covered={weights[1]}, held_out={weights[2]})")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="best")
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
def compute_ceiling_success_rates(
    eval_env, pi_d_star_empirical_path: str, eval_episodes: int, eval_seed: int,
    covered_starts: list[int], held_out_starts: list[int], weights: tuple = DEFAULT_EVAL_WEIGHTS,
) -> dict:
    """Returns {"overall": ..., "covered": ..., "held_out": ..., "weighted": ...}
    -- the pi_D* (empirical) ceiling under all three eval populations plus
    the combined weighted number, all under this run's own eval_seed/
    eval_episodes (see analyze_epochs.py's module docstring for why this
    must not be reused from a report computed under a different seed)."""
    with open(pi_d_star_empirical_path, "rb") as f:
        ref_empirical = pickle.load(f)
    w = evaluate_policy_weighted(
        eval_env, make_tabular_act_fn(ref_empirical), eval_episodes, eval_seed,
        covered_starts, held_out_starts, weights=weights, covered_states=ref_empirical.covered_states,
    )
    return {
        "overall": w["overall"]["success_rate"],
        "covered": w["covered"]["success_rate"],
        "held_out": w["held_out"]["success_rate"],
        "weighted": w["weighted_success_rate"],
    }


# --------------------------------------------------------------------------
# The actual "train one config, checkpoint every N epochs" routine.
# --------------------------------------------------------------------------
def run_single_analysis(
    eval_env,
    dataset,
    prior_state_dict: dict,
    ceiling_success_rates: dict,
    cfg: PPOHyperparams,
    checkpoint_every: int,
    eval_episodes: int,
    eval_seed: int,
    out_dir: Path,
    prefix: str,
    covered_starts: list[int],
    held_out_starts: list[int],
    weights: tuple = DEFAULT_EVAL_WEIGHTS,
    title_suffix: str = "",
    verbose: bool = True,
    log_prefix: str = "",
    save_best_checkpoint_path: Path | None = None,
) -> dict:
    """Runs one fixed-D training + periodic-eval sweep for a single
    PPOHyperparams config. Saves `<prefix>.csv`, `<prefix>_success_return
    .svg/.png`, `<prefix>_weighted.svg/.png`, `<prefix>_clip_entropy
    .svg/.png` under `out_dir`. Returns a small summary dict (best/mean/
    std/final WEIGHTED success_rate + the csv path) for cross-run
    comparison tables, e.g. from a grid sweep.

    Every checkpoint is now evaluated three ways -- overall/covered/
    held-out -- combined into weighted_success_rate (see README, "Our new
    starting point"). This is the number `save_best_checkpoint_path`
    selects on, not the raw overall success_rate: checkpoint SELECTION
    itself needs the same protection against overfitting as any policy
    comparison does, or picking "the best epoch" could just as easily
    reward memorization as the plain PPO-vs-pi_D* comparison this
    project started from could.

    `log_prefix` is prepended to every per-epoch print line -- used by
    analyze_h7.py to make clear, in a long combined log, which grid
    combination a given line belongs to.

    `save_best_checkpoint_path`, if given, tracks the network state at
    whichever epoch achieved the HIGHEST weighted_success_rate during
    training -- not just the final epoch -- and saves it (plus which
    epoch/weighted_success_rate it came from) to that path. This matters
    because fixed-D training here is not monotonic (see README, "Epoch-
    count ceiling analysis"): the final epoch is not reliably the best
    one, and no earlier script in this project persists a trained network
    at all (only the CSV/plots of its trajectory) -- this is the first
    place that gap is closed, needed by scripts/analyze_policy_agreement.py.
    """
    trainer = FixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=prior_state_dict
    )
    rows: list[dict] = []
    best_tracker = {"weighted_success_rate": -1.0, "epoch": None, "state_dict": None}

    def maybe_track_best(epoch: int, net, weighted_success_rate: float):
        if save_best_checkpoint_path is not None and weighted_success_rate > best_tracker["weighted_success_rate"]:
            best_tracker["weighted_success_rate"] = weighted_success_rate
            best_tracker["epoch"] = epoch
            best_tracker["state_dict"] = copy.deepcopy(net.state_dict())

    def live_eval(net) -> dict:
        return evaluate_policy_weighted(
            eval_env, make_neural_act_fn(net, deterministic=True), eval_episodes, eval_seed,
            covered_starts, held_out_starts, weights=weights,
        )

    def to_row(epoch: int, w: dict, clip_frac: float, entropy: float) -> dict:
        return {
            "epoch": epoch,
            "mean_return_overall": w["overall"]["mean_return"],
            "mean_return_overall_stderr": w["overall"]["stderr_return"],
            "success_rate_overall": w["overall"]["success_rate"],
            "success_rate_overall_stderr": w["overall"]["success_rate_stderr"],
            "success_rate_covered": w["covered"]["success_rate"],
            "success_rate_covered_stderr": w["covered"]["success_rate_stderr"],
            "success_rate_held_out": w["held_out"]["success_rate"],
            "success_rate_held_out_stderr": w["held_out"]["success_rate_stderr"],
            "weighted_success_rate": w["weighted_success_rate"],
            "clip_frac": clip_frac,
            "entropy": entropy,
        }

    w0 = live_eval(trainer.net)  # theta == pi_beta exactly at this point, before .train() runs
    entropy0 = trainer.compute_mean_entropy_over_dataset()
    rows.append(to_row(0, w0, clip_frac=0.0, entropy=entropy0))  # theta == pi_old exactly here: nothing clipped
    maybe_track_best(0, trainer.net, w0["weighted_success_rate"])
    if verbose:
        print(
            f"{log_prefix}[epoch    0] weighted_sr={w0['weighted_success_rate']:.3f} "
            f"(overall={w0['overall']['success_rate']:.3f}, covered={w0['covered']['success_rate']:.3f}, "
            f"held_out={w0['held_out']['success_rate']:.3f}) entropy={entropy0:.4f} clip_frac=0.0000"
        )

    def eval_callback(epoch: int, net, summary: dict):
        w = live_eval(net)
        rows.append(to_row(epoch, w, clip_frac=summary["clip_frac"], entropy=summary["entropy"]))
        maybe_track_best(epoch, net, w["weighted_success_rate"])
        if verbose:
            print(
                f"{log_prefix}[epoch {epoch:4d}] weighted_sr={w['weighted_success_rate']:.3f} "
                f"(overall={w['overall']['success_rate']:.3f}, covered={w['covered']['success_rate']:.3f}, "
                f"held_out={w['held_out']['success_rate']:.3f}) entropy={summary['entropy']:.4f} "
                f"clip_frac={summary['clip_frac']:.4f}"
            )

    trainer.train(verbose=False, eval_every_epochs=checkpoint_every, eval_callback=eval_callback)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)

    achieved_best_overall = plot_success_return(
        df,
        out_dir / f"{prefix}_success_return",
        f"success_rate (overall/covered/held-out) vs. epoch ({title_suffix or prefix})",
        prior_success_rates={"overall": w0["overall"]["success_rate"], "covered": w0["covered"]["success_rate"], "held_out": w0["held_out"]["success_rate"]},
        ceiling_success_rates=ceiling_success_rates,
    )
    achieved_best_weighted = plot_weighted_success(
        df,
        out_dir / f"{prefix}_weighted",
        f"weighted_success_rate vs. epoch ({title_suffix or prefix})",
        prior_weighted=w0["weighted_success_rate"],
        ceiling_weighted=ceiling_success_rates["weighted"],
        weights=weights,
    )
    plot_clip_entropy(
        df,
        out_dir / f"{prefix}_clip_entropy",
        f"clip_frac & entropy vs. epoch ({title_suffix or prefix})",
    )

    if save_best_checkpoint_path is not None:
        torch.save(
            {
                "state_dict": best_tracker["state_dict"],
                "epoch": best_tracker["epoch"],
                "weighted_success_rate": best_tracker["weighted_success_rate"],
                "obs_dim": dataset.obs_dim,
                "n_actions": dataset.n_actions,
                "hidden_sizes": cfg.hidden_sizes,
            },
            save_best_checkpoint_path,
        )
        if verbose:
            print(
                f"{log_prefix}Saved best-observed checkpoint (epoch {best_tracker['epoch']}, "
                f"weighted_success_rate={best_tracker['weighted_success_rate']:.3f}) to {save_best_checkpoint_path}"
            )

    return {
        "prefix": prefix,
        "best": achieved_best_weighted,
        "best_overall": achieved_best_overall,
        "mean": float(df["weighted_success_rate"].mean()),
        "std": float(df["weighted_success_rate"].std()),
        "final": float(df.iloc[-1]["weighted_success_rate"]),
        "csv_path": str(csv_path),
        "best_checkpoint_epoch": best_tracker["epoch"],
    }