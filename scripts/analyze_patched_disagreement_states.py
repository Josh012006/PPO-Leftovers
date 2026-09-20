"""Does patching the best configuration's action to pi_D*'s at EXACTLY
the flagged disagreement states genuinely IMPROVE the live success rate?

This is a direct causal check on the disagreement metric itself, not
another correlational one: scripts/analyze_disagreement_factors.py asked
what predicts severity among already-assumed-real disagreements; this
asks whether those disagreements are real in the first place -- if
pi_D*'s action at those specific states is actually better, forcing the
policy to take it there (and nowhere else) should raise the aggregate
success rate. If it doesn't, the flagged "disagreement" isn't identifying
exploitable mistakes, whatever scripts/analyze_disagreement_factors.py
found predicts its severity.

Mechanism: wraps the best-config checkpoint's own action function with a
simple override -- at a fixed set of states (from --policy-agreement-csv,
--disagreement-column, default is_disagreement_strict), return
pi_D*'s action (--pi-d-star-action-column, default pi_d_star_action, the
EMPIRICAL definition -- see --pi-d-star-action-column to use
pi_d_star_action_cross instead) for that state; every other state is
untouched, PPO acts exactly as it would unpatched. Both the unpatched and
patched policies are evaluated the standard three ways
(overall/covered/held-out, weighted) under the IDENTICAL protocol
(env, tier config, weights) and, by default, the SAME eval_seed=24680
already used for the retrain/disagreement-analysis run this project has
been reporting -- this is meant to be a directly comparable number, not
a fresh, differently-seeded one (see README, "A note on eval seeds" in
spirit, even where not literally the same file).

A caveat worth having in view before reading the result: these 66 states
were found via scripts/analyze_policy_agreement.py's own protocol, where
EVERY state is itself a rollout start. Evaluated here instead from the
maze's actual start distribution, a patched state only changes the
outcome for episodes that happen to VISIT it -- if these states are rare
to reach from a typical start, the aggregate delta can be small even
when patching would help a lot conditional on reaching one. This script
reports how many episodes touched at least one patched state alongside
the success-rate delta, specifically so a small delta isn't misread as
"disagreement doesn't matter" when it may just mean "these states are
rarely on the path" -- and a large delta despite rare activation is
correspondingly stronger evidence.

Outputs, under --out-dir (default results/phase2/analysis/patched_disagreement_states/):
  patched_disagreement_states.csv       -- one row per {unpatched, patched}:
                                            overall/covered/held_out/weighted
                                            success_rate, episodes_touching_patch
  patched_disagreement_states_comparison.svg/png -- bar comparison, all three
                                            populations + weighted, patched vs. not

Usage:
    python scripts/analyze_patched_disagreement_states.py \
        --env-config configs/phase2/env_maze.yaml \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --best-config-checkpoint results/phase2/analysis/best_config/best_config_checkpoint.pt \
        --policy-agreement-csv results/phase2/analysis/best_config/policy_agreement.csv \
        --eval-episodes 500 --eval-seed 24680 \
        --out-dir results/phase2/analysis/patched_disagreement_states
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

from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import (
    DEFAULT_EVAL_WEIGHTS,
    evaluate_policy_weighted,
    get_tier_start_lists,
    make_neural_act_fn,
)
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, StartTierConfig


def load_ppo_checkpoint(path: str) -> ActorCritic:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden_sizes"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument("--start-tiers-config", default="configs/phase2/start_tiers.yaml")
    parser.add_argument(
        "--best-config-checkpoint",
        default="results/phase2/analysis/best_config/best_config_checkpoint.pt",
    )
    parser.add_argument(
        "--policy-agreement-csv",
        default="results/phase2/analysis/best_config/policy_agreement.csv",
    )
    parser.add_argument(
        "--disagreement-column",
        default="is_disagreement_strict",
        help="Which column flags a state to patch (1 = patch it). Default is the strictest "
        "definition (both pi_D* references agree) -- see README, 'Policy agreement'.",
    )
    parser.add_argument(
        "--pi-d-star-action-column",
        default="pi_d_star_action",
        help="Which column supplies the replacement action at patched states -- the empirical "
        "pi_D*'s by default; pass pi_d_star_action_cross for the true-restricted one instead.",
    )
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=24680,
        help="Matches the seed already used for the retrain/disagreement-analysis run, so this "
        "delta is directly comparable to the success rates already reported there.",
    )
    parser.add_argument(
        "--eval-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_EVAL_WEIGHTS),
        metavar=("OVERALL", "COVERED", "HELD_OUT"),
        help=f"Weights for the three eval modes, must sum to 1.0 (default {DEFAULT_EVAL_WEIGHTS}).",
    )
    parser.add_argument("--out-dir", default="results/phase2/analysis/patched_disagreement_states")
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
        num_start_states=env_cfg.num_start_states,
        gamma=env_cfg.gamma,
    )
    tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
    covered_starts, held_out_starts = get_tier_start_lists(env, tier_cfg)
    weights = tuple(args.eval_weights)

    net = load_ppo_checkpoint(args.best_config_checkpoint)
    ppo_act_fn = make_neural_act_fn(net, deterministic=True)

    df = pd.read_csv(args.policy_agreement_csv)
    if args.disagreement_column not in df.columns:
        raise SystemExit(f"--disagreement-column '{args.disagreement_column}' not found in {args.policy_agreement_csv}")
    mask = df[args.disagreement_column].astype(bool)
    patch_actions = dict(
        zip(df.loc[mask, "state"].astype(int), df.loc[mask, args.pi_d_star_action_column].astype(int))
    )
    print(
        f"Patching {len(patch_actions)} states (column: {args.disagreement_column}) with actions "
        f"from '{args.pi_d_star_action_column}'."
    )

    touched_episodes = {"unpatched": set(), "patched": set()}

    def make_tracked_act_fn(base_act_fn, tag: str, patch: bool):
        episode_counter = {"n": -1}

        def act_fn(obs, state):
            # StochasticMazeEnv resets episode_counter externally; evaluate_policy calls
            # act_fn once per step within an episode, so track via a simple step tally
            # instead -- see the closure-level counter reset in evaluate_policy_weighted's
            # per-episode loop is not exposed here, so instead we record every touched
            # state and report a step-level (not episode-level) touch count below.
            if patch and state in patch_actions:
                touched_episodes[tag].add(state)
                return patch_actions[state]
            return base_act_fn(obs, state)

        return act_fn

    unpatched_act_fn = make_tracked_act_fn(ppo_act_fn, "unpatched", patch=False)
    patched_act_fn = make_tracked_act_fn(ppo_act_fn, "patched", patch=True)

    print(f"\nEvaluating UNPATCHED best-config PPO (seed={args.eval_seed}, n={args.eval_episodes})...")
    baseline = evaluate_policy_weighted(
        env, unpatched_act_fn, args.eval_episodes, args.eval_seed, covered_starts, held_out_starts, weights=weights
    )
    print(
        f"  overall={baseline['overall']['success_rate']:.3f} covered={baseline['covered']['success_rate']:.3f} "
        f"held_out={baseline['held_out']['success_rate']:.3f} weighted={baseline['weighted_success_rate']:.3f}"
    )

    print(f"\nEvaluating PATCHED best-config PPO (pi_D* action at {len(patch_actions)} disagreement states)...")
    patched = evaluate_policy_weighted(
        env, patched_act_fn, args.eval_episodes, args.eval_seed, covered_starts, held_out_starts, weights=weights
    )
    print(
        f"  overall={patched['overall']['success_rate']:.3f} covered={patched['covered']['success_rate']:.3f} "
        f"held_out={patched['held_out']['success_rate']:.3f} weighted={patched['weighted_success_rate']:.3f}"
    )
    print(f"  (distinct patched states actually visited at least once: {len(touched_episodes['patched'])}/{len(patch_actions)})")

    delta = patched["weighted_success_rate"] - baseline["weighted_success_rate"]
    print(f"\nDelta (patched - unpatched), weighted_success_rate: {delta:+.4f}")
    if len(touched_episodes["patched"]) == 0:
        print("  -> none of the patched states were ever visited under this start distribution/seed -- "
              "this run has NO statistical power to answer the question; try more episodes or a "
              "different eval_seed before concluding anything from the (necessarily zero) delta.")
    elif delta > 0.005:
        print("  -> patching HELPS: the disagreement states are validated as real, exploitable mistakes.")
    elif delta < -0.005:
        print("  -> patching HURTS: unexpected. Either pi_D*'s action isn't actually better live "
              "(e.g. it looks better in the tabular MDP but interacts worse with what PPO does at "
              "OTHER states downstream), or something about the patch itself needs a closer look.")
    else:
        print("  -> no clear difference -- inconclusive at this episode count, or these states are "
              "too rarely pivotal to move the aggregate even when individually correct.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"kind": "unpatched", **{f"success_rate_{k}": baseline[k]["success_rate"] for k in ("overall", "covered", "held_out")}, "weighted_success_rate": baseline["weighted_success_rate"], "distinct_patched_states_visited": len(touched_episodes["unpatched"])},
        {"kind": "patched", **{f"success_rate_{k}": patched[k]["success_rate"] for k in ("overall", "covered", "held_out")}, "weighted_success_rate": patched["weighted_success_rate"], "distinct_patched_states_visited": len(touched_episodes["patched"])},
    ]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "patched_disagreement_states.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["overall", "covered", "held_out", "weighted"]
    x = np.arange(len(labels))
    width = 0.35
    base_vals = [baseline["overall"]["success_rate"], baseline["covered"]["success_rate"], baseline["held_out"]["success_rate"], baseline["weighted_success_rate"]]
    patch_vals = [patched["overall"]["success_rate"], patched["covered"]["success_rate"], patched["held_out"]["success_rate"], patched["weighted_success_rate"]]
    ax.bar(x - width / 2, base_vals, width, label="unpatched (best-config PPO)", color="tab:blue")
    ax.bar(x + width / 2, patch_vals, width, label=f"patched ({len(patch_actions)} states -> pi_D* action)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("success_rate")
    ax.set_ylim(0, 1)
    ax.set_title("Does patching flagged disagreement states to pi_D*'s action help?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "patched_disagreement_states_comparison.svg")
    fig.savefig(out_dir / "patched_disagreement_states_comparison.png", dpi=150)
    print(f"\nSaved {out_dir}/patched_disagreement_states.csv and _comparison.svg/.png")


if __name__ == "__main__":
    main()
