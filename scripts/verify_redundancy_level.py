"""Phase 2 design requirement 2 (README, "Phase 2"): a limited, deliberately
small number of good decisions -- not one (a perfect maze), not many
(phase 1's original redundancy). This script checks that DIRECTLY, rather
than trusting `extra_connection_prob` blindly (the relationship between
that raw probability and how many genuinely good routes exist is not
obvious).

Solves the environment's TRUE dynamics exactly and completely (every
(state, action) pair, no dataset D or coverage restriction at all -- this
is the full-information optimal policy pi*, using the same
_value_iteration routine reference/experience_optimal.py's two pi_D*
definitions already build on). For every non-terminal state, counts how
many of the 4 actions are "near-optimal": Q(s,a) within
--near-optimal-tolerance of the best action's Q(s, a*). A state with
exactly 1 near-optimal action has no real redundancy; one with all 4 is
indifferent to the agent's choice entirely.

No training, no dataset needed -- this only requires the environment
itself. Cheap and fast.

Outputs, under --out-dir (default results/phase2/analysis/redundancy_level/):
  redundancy_level.csv           -- per non-terminal state: n_near_optimal
                                     (1-4), best_action, Q values, whether
                                     the state lies on some shortest
                                     start-to-goal path
  redundancy_level_histogram.svg/png -- distribution of n_near_optimal,
                                     all states vs. on-shortest-path states
  redundancy_level_map.svg/png   -- maze map colored by n_near_optimal,
                                     starts/goal/hazards marked

Usage:
    python scripts/verify_redundancy_level.py \
        --env-config configs/phase2/env_maze.yaml \
        --near-optimal-tolerance 0.05 \
        --out-dir results/phase2/analysis/redundancy_level
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppo_exploitation.envs.stochastic_maze import ACTIONS, StochasticMazeEnv
from ppo_exploitation.eval.evaluate import DEFAULT_EVAL_WEIGHTS, evaluate_policy_weighted, get_tier_start_lists
from ppo_exploitation.reference.experience_optimal import _value_iteration
from ppo_exploitation.utils.config import MazeEnvConfig, StartTierConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument(
        "--near-optimal-tolerance",
        type=float,
        default=0.05,
        help="An action counts as near-optimal at a state if its Q is within this many return "
        "units of the best action's Q there. Default 0.05 is 2.5% of the goal-to-hazard reward "
        "range (1.0 - (-1.0) = 2.0), a fairly strict cutoff.",
    )
    parser.add_argument("--gamma", type=float, default=None, help="Defaults to --env-config's gamma.")
    parser.add_argument("--vi-theta", type=float, default=1e-8)
    parser.add_argument("--vi-max-iter", type=int, default=100_000)
    parser.add_argument(
        "--start-tiers-config",
        required=True,
        help="Required for the corruption-sensitivity check below, scored by weighted_success_rate "
        "(see README, 'Our new starting point').",
    )
    parser.add_argument(
        "--eval-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_EVAL_WEIGHTS),
        metavar=("OVERALL", "COVERED", "HELD_OUT"),
        help=f"Weights for the three eval modes, must sum to 1.0 (default {DEFAULT_EVAL_WEIGHTS}).",
    )
    parser.add_argument(
        "--corruption-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 1.0],
        help="Fractions of on-shortest-path states to deliberately corrupt (replace pi*'s action "
        "with the worst one) when checking that disagreement is actually felt, not just nominal.",
    )
    parser.add_argument("--corruption-episodes", type=int, default=300)
    parser.add_argument("--corruption-seed", type=int, default=999)
    parser.add_argument("--out-dir", default="results/phase2/analysis/redundancy_level")
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
    gamma = args.gamma if args.gamma is not None else env_cfg.gamma
    print(f"Maze: {env_cfg.width}x{env_cfg.height}, extra_connection_prob={env_cfg.extra_connection_prob}, "
          f"{len(env.layout.starts)} starts, gamma={gamma}")

    # --- Full-information exact solve: every (s,a) pair, true dynamics,
    # no coverage restriction at all -- this is pi*, not pi_D*. ---
    n_states, n_actions = env.n_states, env.n_actions
    terminal_states = {s for s in range(n_states) if env.is_terminal_state(s)}
    absorbing_id = n_states
    transition_model = {}
    for s in range(n_states):
        if s in terminal_states:
            continue
        for a in range(n_actions):
            probs = env.true_transition_probs(s, a)
            transition_model[(s, a)] = [(sp, p, env.true_reward(s, a, sp)) for sp, p in probs.items()]
    print("Solving pi* exactly (full coverage, true dynamics)...")
    V, Q, policy = _value_iteration(
        transition_model, terminal_states, absorbing_id, n_states, n_actions, gamma, args.vi_theta, args.vi_max_iter
    )

    # --- Which states lie on SOME shortest path from any start to goal
    # (BFS on the maze graph, ignoring stochasticity) -- the states where
    # redundancy actually matters most for navigation. ---
    def bfs_dist_from(sources):
        dist = {}
        frontier = deque()
        for s in sources:
            dist[s] = 0
            frontier.append(s)
        while frontier:
            s = frontier.popleft()
            r, c = env.layout.rc(s)
            for d in ACTIONS:
                if not env.layout.open_walls[r, c, d]:
                    continue
                from ppo_exploitation.envs.stochastic_maze import _ACTION_DELTA

                dr, dc = _ACTION_DELTA[d]
                ns = env.layout.state_id(r + dr, c + dc)
                if ns not in dist:
                    dist[ns] = dist[s] + 1
                    frontier.append(ns)
        return dist

    goal_state = env.layout.state_id(*env.layout.goal)
    dist_from_goal = bfs_dist_from([goal_state])
    on_shortest_path = set()
    for start_rc in env.layout.starts:
        start_state = env.layout.state_id(*start_rc)
        dist_from_start = bfs_dist_from([start_state])
        total = dist_from_start.get(goal_state, None)
        if total is None:
            continue
        for s in range(n_states):
            if s in dist_from_start and s in dist_from_goal and dist_from_start[s] + dist_from_goal[s] == total:
                on_shortest_path.add(s)

    rows = []
    for s in range(n_states):
        if s in terminal_states:
            continue
        q = Q[s]
        best_q = q.max()
        n_near_optimal = int((q >= best_q - args.near_optimal_tolerance).sum())
        r, c = env.layout.rc(s)
        rows.append(
            {
                "state": s,
                "row": r,
                "col": c,
                "best_action": int(np.argmax(q)),
                "best_q": float(best_q),
                "n_near_optimal": n_near_optimal,
                "on_shortest_path": s in on_shortest_path,
            }
        )
    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "redundancy_level.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n=== n_near_optimal distribution (tolerance={args.near_optimal_tolerance}) ===")
    print("-- all non-terminal states --")
    print(df["n_near_optimal"].value_counts().sort_index().to_string())
    on_path = df[df["on_shortest_path"]]
    print(f"\n-- states on some shortest start-to-goal path (n={len(on_path)}) --")
    print(on_path["n_near_optimal"].value_counts().sort_index().to_string())
    mean_on_path = on_path["n_near_optimal"].mean()
    print(f"\nMean near-optimal actions per on-path state: {mean_on_path:.2f}")
    if mean_on_path < 1.3:
        print("  -> close to 1: redundancy may be too LOW (near a perfect maze) -- consider raising "
              "extra_connection_prob.")
    elif mean_on_path > 2.5:
        print("  -> above 2.5: redundancy may be too HIGH (phase 1's problem) -- consider lowering "
              "extra_connection_prob.")
    else:
        print("  -> in the targeted 'a handful, not one, not many' range.")
    print(f"\nSaved {csv_path}")

    # --- Corruption-sensitivity check: does disagreement with pi* actually
    # cost something felt, or is n_near_optimal just a nominal number?
    # Deliberately replaces pi*'s action with the WORST one at a controlled
    # fraction of on-shortest-path states, then evaluates live -- three
    # ways (overall/covered/held-out), combined into weighted_success_rate
    # (see README, "Our new starting point"), same as every other check. ---
    tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
    covered_starts, held_out_starts = get_tier_start_lists(env, tier_cfg)
    weights = tuple(args.eval_weights)
    on_path_states = on_path["state"].to_numpy(dtype=int)
    rng = np.random.default_rng(args.corruption_seed)

    print(
        f"\n=== Corruption sensitivity: replacing pi*'s action with the worst one at a fraction of "
        f"the {len(on_path_states)} on-path states ==="
    )
    corruption_rows = []
    for frac in args.corruption_fractions:
        corrupted = policy.copy()
        n_corrupt = int(round(frac * len(on_path_states)))
        to_corrupt = rng.choice(on_path_states, size=n_corrupt, replace=False) if n_corrupt > 0 else []
        for s in to_corrupt:
            corrupted[s] = int(np.argmin(Q[s]))

        def corrupted_act_fn(obs, state, pol=corrupted):
            return int(pol[state])

        w = evaluate_policy_weighted(
            env, corrupted_act_fn, args.corruption_episodes, args.corruption_seed,
            covered_starts, held_out_starts, weights=weights,
        )
        corruption_rows.append(
            {
                "corrupted_fraction": frac,
                "n_states_corrupted": n_corrupt,
                "success_rate_overall": w["overall"]["success_rate"],
                "success_rate_covered": w["covered"]["success_rate"],
                "success_rate_held_out": w["held_out"]["success_rate"],
                "weighted_success_rate": w["weighted_success_rate"],
            }
        )
        print(
            f"  frac={frac:.2f} ({n_corrupt:3d} states): weighted={w['weighted_success_rate']:.3f} "
            f"(overall={w['overall']['success_rate']:.3f}, covered={w['covered']['success_rate']:.3f}, "
            f"held_out={w['held_out']['success_rate']:.3f})"
        )

    corruption_df = pd.DataFrame(corruption_rows)
    corruption_csv = out_dir / "redundancy_corruption_sensitivity.csv"
    corruption_df.to_csv(corruption_csv, index=False)
    print(f"Saved {corruption_csv}")

    zero_row = corruption_df[corruption_df["corrupted_fraction"] == 0.0]
    if len(zero_row) > 0 and len(corruption_df) > 1:
        drop = zero_row.iloc[0]["weighted_success_rate"] - corruption_df.iloc[1]["weighted_success_rate"]
        if drop < 0.05:
            print(
                f"  -> WARNING: weighted_success_rate barely moves from 0% to "
                f"{corruption_df.iloc[1]['corrupted_fraction']:.0%} corruption (drop={drop:.3f}) -- "
                f"disagreement with pi* may not be clearly felt here; reconsider redundancy parameters."
            )
        else:
            print(f"  -> disagreement is clearly felt: a {drop:.3f} drop in weighted_success_rate from the first corruption step alone.")

    # --- Plot 1: histogram, all states vs. on-shortest-path states ---
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(0.5, 5.5, 1)
    ax.hist(df["n_near_optimal"], bins=bins, alpha=0.5, label=f"all states (n={len(df)})", density=True)
    ax.hist(on_path["n_near_optimal"], bins=bins, alpha=0.5, label=f"on-shortest-path states (n={len(on_path)})", density=True)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel(f"number of near-optimal actions (within {args.near_optimal_tolerance} of best)")
    ax.set_ylabel("fraction of states")
    ax.set_title("How many good decisions does a typical state actually offer?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    plot1 = out_dir / "redundancy_level_histogram"
    fig.savefig(plot1.with_suffix(".svg"))
    fig.savefig(plot1.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot1}.svg/.png")

    # --- Plot 2: maze map colored by n_near_optimal ---
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        df["col"], df["row"], c=df["n_near_optimal"], cmap="RdYlGn_r", vmin=1, vmax=4, s=28, marker="s",
    )
    hz_rows = [r for (r, c) in env.layout.hazards]
    hz_cols = [c for (r, c) in env.layout.hazards]
    ax.scatter(hz_cols, hz_rows, marker="x", s=80, color="black", linewidths=1.5, label="hazard")
    start_rows = [r for (r, c) in env.layout.starts]
    start_cols = [c for (r, c) in env.layout.starts]
    ax.scatter(start_cols, start_rows, marker="*", s=140, color="blue", label="start", edgecolors="black")
    ax.scatter([env.layout.goal[1]], [env.layout.goal[0]], marker="*", s=180, color="gold", label="goal", edgecolors="black")
    ax.invert_yaxis()
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(f"Near-optimal action count by cell (extra_connection_prob={env_cfg.extra_connection_prob})")
    fig.colorbar(sc, ax=ax, label="number of near-optimal actions (1=none redundant, 4=fully indifferent)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.25, 1.0))
    fig.tight_layout()
    plot2 = out_dir / "redundancy_level_map"
    fig.savefig(plot2.with_suffix(".svg"))
    fig.savefig(plot2.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot2}.svg/.png")

    # --- Plot 3: corruption sensitivity, three raw curves + weighted ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colors = {"overall": "tab:blue", "covered": "tab:green", "held_out": "tab:red"}
    for key, label in [("overall", "overall"), ("covered", "covered"), ("held_out", "held-out")]:
        ax.plot(
            corruption_df["corrupted_fraction"], corruption_df[f"success_rate_{key}"],
            color=colors[key], marker="o", markersize=4, label=label,
        )
    ax.set_xlabel("fraction of on-path states corrupted")
    ax.set_ylabel("success_rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Raw success_rate vs. corruption")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        corruption_df["corrupted_fraction"], corruption_df["weighted_success_rate"],
        color="tab:purple", marker="o", markersize=4, label="weighted_success_rate",
    )
    ax.set_xlabel("fraction of on-path states corrupted")
    ax.set_ylabel(f"weighted_success_rate (overall={weights[0]}, covered={weights[1]}, held_out={weights[2]})")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Is disagreement with pi* clearly felt?")
    ax.legend(fontsize=8)

    fig.suptitle("Corruption-sensitivity check (requirement 2)")
    fig.tight_layout()
    plot3 = out_dir / "redundancy_corruption_sensitivity"
    fig.savefig(plot3.with_suffix(".svg"))
    fig.savefig(plot3.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot3}.svg/.png")


if __name__ == "__main__":
    main()
