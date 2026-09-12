"""Phase 2 design requirement 3 (README, "Phase 2"): overfitting to D must
be directly observable, not merely assumed absent.

An earlier version of this script relied on a specifically-tuned PPO
config (many epochs, entropy_coef=0) to induce overfitting, then compared
its covered/held-out gap against pi_beta's. That approach confounded two
different things: training LONGER (which improves performance broadly,
covered and held-out alike, since pi_beta itself was trained with a
uniform start distribution) with actually OVERFITTING to D's specific
skewed coverage. A version trained for far more epochs than the baseline
ends up with a SMALLER gap than pi_beta itself, not a larger one --
because "more epochs" alone is not what overfitting means here.

This version builds a policy that is overfit BY CONSTRUCTION instead:
for every state, memorize whichever action D shows MOST OFTEN -- pure
behavioral-cloning-style memorization, no value estimation, no
bootstrapping, no PPO training at all. This policy should:
  - match pi_beta's own success rate on covered starts (it deterministically
    replays pi_beta's single most common local choice there), and
  - have LITERALLY ZERO information at held-out states (never appear in D
    at all), falling back to an arbitrary default action there.

If the covered/held-out infrastructure actually detects overfitting, this
memorization policy's gap should be clearly LARGER than pi_beta's own
baseline gap (pi_beta was trained with a uniform start distribution --
see README, "Phase 2", requirement 3 -- so its own gap should be small).
Checked directly on this project's real phase-2 data: memorization gives
covered=0.464 / held_out=0.000 (gap +0.464) against pi_beta's covered=0.462
/ held_out=0.194 (gap +0.268) -- matching covered performance almost
exactly while collapsing to zero on held-out, roughly double pi_beta's
own gap.

No training needed at all -- just building a lookup table directly from
D (a single pass over its trajectories) and live-rollout evaluation of
that table plus the already-existing prior checkpoint. Cheap and fast;
does not depend on scripts/04_train_fixed_d_ppo.py or
scripts/05_evaluate_all.py having been run at all.

Outputs, under --out-dir (default results/phase2/analysis/overfitting_detectable/):
  overfitting_detectable.csv     -- covered/held-out success_rate for
                                     both policies (D-mode memorization,
                                     pi_beta)
  overfitting_detectable.svg/png -- bar chart comparison

Usage:
    python scripts/verify_overfitting_detectable.py \
        --env-config configs/phase2/env_maze.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase2/analysis/overfitting_detectable
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import DEFAULT_EVAL_WEIGHTS, evaluate_policy_weighted, get_tier_start_lists, make_neural_act_fn
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, StartTierConfig


def build_d_mode_policy(dataset, default_action: int = 0):
    """State -> the single action D shows most often there. Pure
    memorization: no returns, no value estimation, no propagation through
    the MDP at all -- just "what did pi_beta usually do here". States D
    never covers fall back to `default_action`, same uninformed-default
    convention used throughout this project for genuinely-unseen states."""
    state_action_counts: dict[int, Counter] = defaultdict(Counter)
    for tr in dataset.trajectories:
        for s, a in zip(tr.states.tolist(), tr.actions.tolist()):
            state_action_counts[int(s)][int(a)] += 1
    table = {s: counts.most_common(1)[0][0] for s, counts in state_action_counts.items()}

    def act_fn(obs, state):
        return table.get(int(state), default_action)

    return act_fn, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument("--dataset", default="results/phase2/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/phase2/prior_checkpoint.pt")
    parser.add_argument("--start-tiers-config", default="configs/phase2/start_tiers.yaml")
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.15,
        help="The memorization policy's covered-minus-held-out success_rate gap must clear this "
        "for the check to pass -- i.e. how large a gap counts as clearly detectable, not noise.",
    )
    parser.add_argument("--out-dir", default="results/phase2/analysis/overfitting_detectable")
    parser.add_argument(
        "--eval-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_EVAL_WEIGHTS),
        metavar=("OVERALL", "COVERED", "HELD_OUT"),
        help=f"Weights for the three eval modes, must sum to 1.0 (default {DEFAULT_EVAL_WEIGHTS}).",
    )
    args = parser.parse_args()
    weights = tuple(args.eval_weights)

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

    dataset = load_dataset(args.dataset)
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes.")
    d_mode_act_fn, table = build_d_mode_policy(dataset)
    print(f"D-mode memorization policy defined for {len(table)}/{env.n_states} states.")

    prior_ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_net = ActorCritic(prior_ckpt["obs_dim"], prior_ckpt["n_actions"], prior_ckpt["hidden_sizes"])
    prior_net.load_state_dict(prior_ckpt["state_dict"])
    prior_net.eval()
    prior_act_fn = make_neural_act_fn(prior_net, deterministic=True)

    tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
    covered_starts, held_out_starts = get_tier_start_lists(env, tier_cfg)
    print(f"{len(covered_starts)} covered starts, {len(held_out_starts)} held-out starts.\n")

    rows = []
    for name, act_fn in [("d_mode_memorization", d_mode_act_fn), ("prior_pi_beta", prior_act_fn)]:
        w = evaluate_policy_weighted(
            env, act_fn, args.eval_episodes, args.eval_seed, covered_starts, held_out_starts, weights=weights
        )
        gap = w["covered"]["success_rate"] - w["held_out"]["success_rate"]
        rows.append(
            {
                "policy": name,
                "overall": w["overall"]["success_rate"],
                "covered": w["covered"]["success_rate"],
                "held_out": w["held_out"]["success_rate"],
                "weighted": w["weighted_success_rate"],
                "gap": gap,
            }
        )
        print(
            f"{name:22s} overall={w['overall']['success_rate']:.3f}  covered={w['covered']['success_rate']:.3f}  "
            f"held_out={w['held_out']['success_rate']:.3f}  weighted={w['weighted_success_rate']:.3f}  gap={gap:+.3f}"
        )

    df = pd.DataFrame(rows).set_index("policy")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "overfitting_detectable.csv"
    df.to_csv(csv_path)

    mem_gap = df.loc["d_mode_memorization", "gap"]
    beta_gap = df.loc["prior_pi_beta", "gap"]
    print()
    if mem_gap >= args.min_gap and mem_gap > beta_gap:
        print(
            f"PASS: the memorization policy's gap ({mem_gap:+.3f}) clears --min-gap ({args.min_gap}) "
            f"and exceeds pi_beta's own gap ({beta_gap:+.3f}) -- the covered/held-out split detects "
            f"overfitting when a genuinely overfit policy exists, and pi_beta (uniform starts during "
            f"its own training) serves as the small-gap baseline it should be."
        )
    else:
        print(
            f"FAIL: either the gap is under --min-gap or pi_beta's own gap is not clearly smaller. "
            f"Reconsider the start tiers or held-out tier size before trusting this metric."
        )
    print(f"\nSaved {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = [0, 1, 2]
    width = 0.35
    beta_vals = [df.loc["prior_pi_beta", "overall"], df.loc["prior_pi_beta", "covered"], df.loc["prior_pi_beta", "held_out"]]
    mem_vals = [
        df.loc["d_mode_memorization", "overall"], df.loc["d_mode_memorization", "covered"], df.loc["d_mode_memorization", "held_out"]
    ]
    ax.bar([xi - width / 2 for xi in x], beta_vals, width, label="\u03c0\u03b2 (prior)", color="tab:gray")
    ax.bar([xi + width / 2 for xi in x], mem_vals, width, label="D-mode memorization", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(["overall", "covered tier", "held-out tier"])
    ax.set_ylabel("success_rate")
    ax.set_ylim(0, 1)
    ax.set_title("Is a policy that's overfit BY CONSTRUCTION detectable?")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "overfitting_detectable"
    fig.savefig(plot_path.with_suffix(".svg"))
    fig.savefig(plot_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}.svg/.png")


if __name__ == "__main__":
    main()