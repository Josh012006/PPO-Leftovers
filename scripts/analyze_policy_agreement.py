"""Where, exactly, does the best fixed-D PPO configuration's policy
underperform pi_D*'s -- not just pick a different action, but actually
get a worse expected return?

The heatmap separates TWO notions:

1. POLICY DISAGREEMENT:
   Does the best-config PPO policy choose a different greedy action
   than pi_D*?

2. DISAGREEMENT SEVERITY:
   When the actions differ, how much worse is PPO's expected return
   than pi_D*'s expected value?

A state therefore gets a positive severity if and only if BOTH hold:

1. pi_D*'s greedy action differs from the best-config policy's
   greedy action.
2. PPO's expected discounted return is lower than pi_D*'s value.

The raw severity is:

    severity_raw(s) = max(0, V_pi_D*(s) - V_PPO(s))

when the greedy actions differ, and zero otherwise.

Importantly, statistical significance does NOT determine whether a state
contributes to severity. The z-threshold is kept as a separate diagnostic
(`statistically_significant`) so that the heatmap represents the actual
magnitude of the disagreement cost rather than suppressing small but
positive gaps.

This gives the following interpretation:

- same action -> green
- different action but PPO is equally good or better -> green
- different action and PPO is slightly worse -> green/yellow
- different action and PPO is substantially worse -> orange/red

IMPORTANT correctness point: pi_D*'s V(s) is DISCOUNTED (value iteration
uses reference.yaml's gamma=0.99). Comparing it against an UNDISCOUNTED
empirical return -- which is what evaluate_policy computes everywhere
else in this project -- would silently reproduce exactly the discounted-
vs-undiscounted mismatch this project has been careful to avoid since its
first "Baseline run" (see README). So the best-config policy's expected
return here is estimated as a DISCOUNTED return-to-go (same gamma), via
direct Monte Carlo rollout starting from each state -- not the project's
usual undiscounted mean_return metric. This is a deliberate, narrow
exception for this one apples-to-apples comparison, not a change to how
success/return is reported anywhere else.

How the rollout-from-an-arbitrary-state works: `env.set_state(s)` already
exists (used by the reference solver's own machinery) -- call
`env.reset()` (clears the step counter), then `env.set_state(s)` to
override the position, then `env.state_to_obs(s)` for the correct initial
observation, then step normally. `--episodes-per-state` independent
rollouts per state (freshly reseeded each time, so slip randomness
differs) give both a mean discounted return and its standard error.

COST WARNING: this evaluates every one of ~887 non-terminal states with
`--episodes-per-state` rollouts each (887 * episodes-per-state episodes
total) -- no network training involved, but still a lot of environment
steps. Lower --episodes-per-state for a faster, noisier pass; the
per-state standard error in the output CSV tells you how much that
costs. This is why this script is meant to be run locally, not repeatedly
in a sandbox.

Also computes (cheaply, no rollout needed) pi_beta's own critic accuracy
at each state -- reusing reference.experience_optimal.compute_true_value_of_policy,
same as scripts/analyze_critic_accuracy.py -- and saves it in the output
CSV for a later diagnostic (critic error vs. disagreement severity),
deliberately NOT plotted yet: see README, "Policy agreement" -- the
mechanism connecting disagreement to any specific state property is
being established first (scripts/analyze_disagreement_factors.py), before
revisiting that specific plot.

## Severity normalization

The heatmap does NOT use the raw minimum/maximum over all states.

States outside the dataset can have pi_D* values such as -50. These values
represent states that are outside the support/coverage of dataset D and
must NOT dominate the color scale.

For each state:

    severity_raw(s) =
        max(0, V_pi_D*(s) - V_PPO(s))    if actions differ
        0                                otherwise

Only positive raw severities are then used to determine the heatmap scale.
The scale is the 95th percentile:

    scale = P95(severity_raw | severity_raw > 0)

and the displayed severity is:

    severity(s) = clip(severity_raw(s) / scale, 0, 1)

Thus states with pi_D*(s) = -50 do not automatically become extreme
red cells. If PPO has a better value there, the gap is negative and the
severity is exactly zero.

## Dataset coverage

`covered` indicates whether dataset D contains the state at all;
`n_actions_covered` (0-4) is how many distinct actions D actually has at
least one sample for at that state -- a finer-grained signal than the
boolean, and the one used to size the marker (see below).

States NOT covered by D (`covered=False`, `n_actions_covered=0`) are
rendered blank (white, no border) in the maze map: they carry no severity
signal of their own (pi_D*'s tie-break there is uninformed), so leaving
them blank avoids implying a real disagreement measurement exists there.
They remain in the CSV for completeness.

Marker size scales with `n_actions_covered` (more actions sampled by D at
a state -> a visibly larger square), with a floor so no square ever
becomes too small to see, even at low coverage.

## CSV outputs

policy_agreement.csv contains:

state
row
col
covered
n_actions_covered
pi_d_star_V
pi_d_star_action
best_config_action
argmax_disagree
rank_correlation
ppo_discounted_return
ppo_discounted_return_stderr
value_gap
is_disagreement
severity_raw
severity
statistically_significant
true_value_pi_beta
critic_pred_pi_beta
critic_abs_error
n_pi_d_star_action
n_best_config_action
pair_min_samples

Here:

is_disagreement:
1 if actions differ AND PPO has a positive value gap.

statistically_significant:
1 if actions differ AND value_gap > z_threshold * stderr.

The latter is a statistical diagnostic and does not suppress severity.

n_pi_d_star_action / n_best_config_action / pair_min_samples:
D's raw (state, action) sample count for pi_D*'s preferred action, for
whichever action the best-config policy actually chose, and the smaller
of the two -- distinct from n_actions_covered, which only counts how
many of the 4 actions were sampled AT ALL. A state can look
well-covered overall while the one action a policy ends up preferring
there was seen only a handful of times. Feeds the second plot below and
scripts/analyze_disagreement_factors.py's pair-level factors.

Usage (retrain, first run):

python scripts/analyze_policy_agreement.py \
    --env-config configs/env_maze.yaml \
    --dataset results/dataset_D.pkl \
    --prior-checkpoint results/prior_checkpoint.pt \
    --pi-d-star results/pi_d_star_empirical.pkl \
    --reference-config configs/reference.yaml \
    --best-config configs/ppo_fixed_d_best_config.yaml \
    --episodes-per-state 20 \
    --eval-seed 24680 \
    --out-dir results/analysis/policy_agreement

Usage (reuse an already-trained checkpoint, e.g. against true-restricted pi_D*):

python scripts/analyze_policy_agreement.py \
    --env-config configs/env_maze.yaml \
    --dataset results/dataset_D.pkl \
    --prior-checkpoint results/prior_checkpoint.pt \
    --pi-d-star results/pi_d_star_true_restricted.pkl \
    --reference-config configs/reference.yaml \
    --reuse-checkpoint results/analysis/policy_agreement/best_config_checkpoint.pt \
    --episodes-per-state 20 \
    --eval-seed 24680 \
    --out-dir results/analysis/policy_agreement_true_restricted
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from _analysis_lib import compute_ceiling_success_rate, compute_sa_counts, run_single_analysis
from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import make_neural_act_fn
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.reference.experience_optimal import (
    compute_true_value_of_policy,
)
from ppo_exploitation.utils.config import (
    MazeEnvConfig,
    PPOHyperparams,
    ReferenceConfig,
)
from ppo_exploitation.utils.seeding import set_global_seed


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, ties handled by average rank.

    Uses plain NumPy, with no scipy dependency for this one function.
    """

    def rankdata_avg(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)

        sorted_a = a[order]

        i = 0

        while i < len(a):
            j = i

            while j < len(a) - 1 and sorted_a[j + 1] == sorted_a[i]:
                j += 1

            if j > i:
                ranks[order[i : j + 1]] = ranks[
                    order[i : j + 1]
                ].mean()

            i = j + 1

        return ranks

    rx = rankdata_avg(np.asarray(x, dtype=float))
    ry = rankdata_avg(np.asarray(y, dtype=float))

    if rx.std() == 0 or ry.std() == 0:
        return float("nan")

    return float(np.corrcoef(rx, ry)[0, 1])


def discounted_rollout_from_state(
    env,
    act_fn,
    state: int,
    n_episodes: int,
    gamma: float,
    seed: int,
) -> tuple[float, float]:
    """Mean and stderr of the DISCOUNTED return.

    Runs n_episodes independent rollouts starting at `state`, via
    env.reset() followed by env.set_state(state), under `act_fn`.

    Each episode starts from the same requested state. The environment is
    freshly seeded for every episode so stochastic transitions differ
    across rollouts.
    """

    returns = []

    for ep in range(n_episodes):
        start_state = state

        env.reset(seed=seed + ep)
        env.set_state(start_state)

        obs = env.state_to_obs(start_state)
        current_state = start_state
        done = False

        g = 0.0
        discount = 1.0

        while not done:
            action = act_fn(obs, current_state)

            obs, reward, terminated, truncated, info = env.step(action)

            current_state = info["state"]

            g += discount * reward
            discount *= gamma

            done = terminated or truncated

        returns.append(g)

    arr = np.asarray(returns, dtype=np.float64)

    mean = float(arr.mean())

    stderr = (
        float(arr.std(ddof=1) / np.sqrt(len(arr)))
        if len(arr) > 1
        else 0.0
    )

    return mean, stderr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env-config",
        default="configs/env_maze.yaml",
    )

    parser.add_argument(
        "--dataset",
        default="results/dataset_D.pkl",
    )

    parser.add_argument(
        "--prior-checkpoint",
        default="results/prior_checkpoint.pt",
    )

    parser.add_argument(
        "--pi-d-star",
        default="results/pi_d_star_empirical.pkl",
    )

    parser.add_argument(
        "--reference-config",
        default="configs/reference.yaml",
    )

    parser.add_argument(
        "--best-config",
        default="configs/ppo_fixed_d_best_config.yaml",
    )

    parser.add_argument(
        "--reuse-checkpoint",
        default=None,
        help=(
            "Path to a *.pt saved by a previous run of this script. "
            "If given, retraining is skipped and this checkpoint is used "
            "directly -- e.g. to re-run against a different --pi-d-star "
            "without retraining the same network twice."
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Only used when retraining.",
    )

    parser.add_argument(
        "--episodes-per-state",
        type=int,
        default=20,
        help=(
            "Independent rollouts per state for the discounted-return "
            "estimate. Higher = more precise but slower; see the "
            "per-state stderr in the output CSV to judge if it's enough."
        ),
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=500,
        help=(
            "Episode count for the retrain's OWN live-eval checkpointing "
            "(unrelated to --episodes-per-state, which is for the "
            "per-state rollout comparison after training)."
        ),
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=24680,
    )

    parser.add_argument(
        "--out-dir",
        default="results/analysis/policy_agreement",
    )

    parser.add_argument(
        "--z-threshold",
        type=float,
        default=2.0,
        help=(
            "Statistical significance threshold used only for the "
            "`statistically_significant` diagnostic. A state is marked "
            "significant when actions differ and "
            "value_gap > z-threshold * stderr. This threshold does NOT "
            "suppress or modify the heatmap severity."
        ),
    )

    args = parser.parse_args()

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    ref_cfg = ReferenceConfig.from_yaml(args.reference_config)

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
    )

    dataset = load_dataset(args.dataset)

    state_actions_seen: dict[int, set[int]] = {}
    for tr in dataset.trajectories:
        for s, a in zip(tr.states.tolist(), tr.actions.tolist()):
            state_actions_seen.setdefault(int(s), set()).add(int(a))

    covered_states = set(state_actions_seen.keys())

    # Pair-level (state, action) sample counts -- distinct from
    # n_actions_covered (which only counts how many of the 4 actions were
    # sampled at all). A state can look well-covered overall while the one
    # specific action a policy ends up preferring there was seen only a
    # handful of times: this is what scripts/analyze_disagreement_factors.py's
    # pair-level factors (and the new plot below) actually test, following
    # up on the "sparse-signal overfitting" hypothesis in the README.
    sa_counts = compute_sa_counts(dataset)

    print(
        f"Loaded D: {len(dataset)} transitions, "
        f"D covers {len(covered_states)}/{env.n_states} states."
    )

    prior_ckpt = torch.load(
        args.prior_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    prior_state_dict = prior_ckpt["state_dict"]

    print(
        f"Loaded prior checkpoint "
        f"(final eval: {prior_ckpt['final_eval']})"
    )

    with open(args.pi_d_star, "rb") as f:
        ref = pickle.load(f)

    print(
        f"Loaded pi_D* ({ref.kind}) from {args.pi_d_star}, "
        f"gamma={ref_cfg.gamma}"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: get the best-config policy, either by retraining (default)
    # or by reusing an already-saved checkpoint (--reuse-checkpoint).
    # ------------------------------------------------------------------
    if args.reuse_checkpoint:
        best_ckpt_path = Path(args.reuse_checkpoint)

        print(
            f"\n--reuse-checkpoint given: skipping retrain, "
            f"loading {best_ckpt_path} directly."
        )

        best_ckpt = torch.load(
            best_ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

    else:
        cfg = PPOHyperparams.from_yaml(args.best_config)

        set_global_seed(cfg.seed)

        best_ckpt_path = out_dir / "best_config_checkpoint.pt"

        ceiling_success_rate = compute_ceiling_success_rate(
            env,
            args.pi_d_star,
            args.eval_episodes,
            args.eval_seed,
        )

        print(
            f"pi_D* ceiling under this run's eval protocol: "
            f"success_rate={ceiling_success_rate:.3f}"
        )

        print(
            f"\nRetraining the best configuration ({args.best_config}) "
            f"for up to {cfg.epochs} epochs, tracking the best-observed "
            f"checkpoint (not just the final one)...\n"
        )

        retrain_summary = run_single_analysis(
            eval_env=env,
            dataset=dataset,
            prior_state_dict=prior_state_dict,
            ceiling_success_rate=ceiling_success_rate,
            cfg=cfg,
            checkpoint_every=args.checkpoint_every,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            out_dir=out_dir,
            prefix="best_config_retrain",
            title_suffix="best configuration retrain",
            verbose=True,
            save_best_checkpoint_path=best_ckpt_path,
        )

        print(
            f"\nRetrain done: best={retrain_summary['best']:.3f} "
            f"mean={retrain_summary['mean']:.3f} "
            f"(best checkpoint at epoch "
            f"{retrain_summary['best_checkpoint_epoch']})"
        )

        best_ckpt = torch.load(
            best_ckpt_path,
            map_location="cpu",
            weights_only=False,
        )

    print(
        f"Using the checkpoint from epoch {best_ckpt['epoch']} "
        f"(success_rate={best_ckpt['success_rate']:.3f}) "
        f"as 'the best configuration's policy' "
        f"for the rest of this analysis.\n"
    )

    best_net = ActorCritic(
        best_ckpt["obs_dim"],
        best_ckpt["n_actions"],
        best_ckpt["hidden_sizes"],
    )

    best_net.load_state_dict(best_ckpt["state_dict"])
    best_net.eval()

    act_fn = make_neural_act_fn(
        best_net,
        deterministic=True,
    )

    prior_net = ActorCritic(
        prior_ckpt["obs_dim"],
        prior_ckpt["n_actions"],
        prior_ckpt["hidden_sizes"],
    )

    prior_net.load_state_dict(prior_state_dict)
    prior_net.eval()

    # ------------------------------------------------------------------
    # Step 2: cheap per-state quantities (no rollout): logits, argmax,
    # rank correlation, pi_beta's own critic accuracy.
    # ------------------------------------------------------------------
    terminal_states = {
        s
        for s in range(env.n_states)
        if env.is_terminal_state(s)
    }

    non_terminal = [
        s
        for s in range(env.n_states)
        if s not in terminal_states
    ]

    obs_batch = np.stack(
        [env.state_to_obs(s) for s in non_terminal]
    ).astype(np.float32)

    obs_t = torch.as_tensor(obs_batch)

    with torch.no_grad():
        best_logits, _ = best_net.forward(obs_t)

        best_logits_np = best_logits.numpy()

        prior_logits, prior_values = prior_net.forward(obs_t)

        prior_action_probs_np = (
            torch.softmax(prior_logits, dim=-1).numpy()
        )

        prior_critic_pred_np = prior_values.numpy()

    prior_action_probs_full = np.zeros(
        (env.n_states, env.n_actions),
        dtype=np.float64,
    )

    prior_critic_pred_full = np.zeros(
        env.n_states,
        dtype=np.float64,
    )

    for i, s in enumerate(non_terminal):
        prior_action_probs_full[s] = prior_action_probs_np[i]
        prior_critic_pred_full[s] = prior_critic_pred_np[i]

    print(
        "Computing exact V^pi_beta(s) "
        "(for the later critic-error diagnostic, not used here)..."
    )

    true_value_pi_beta = compute_true_value_of_policy(
        env,
        prior_action_probs_full,
        gamma=env_cfg.gamma,
    )

    # ------------------------------------------------------------------
    # Step 3: expensive part -- discounted rollout from every state.
    # ------------------------------------------------------------------
    print(
        f"\nRolling out the best-config policy from all "
        f"{len(non_terminal)} non-terminal states, "
        f"{args.episodes_per_state} episodes each "
        f"({len(non_terminal) * args.episodes_per_state} "
        f"episodes total)...\n"
    )

    rows = []

    for k, s in enumerate(non_terminal):
        q_star = ref.Q[s]

        pi_d_star_action = int(np.argmax(q_star))
        pi_d_star_V = float(ref.V[s])

        best_config_action = int(
            np.argmax(best_logits_np[k])
        )

        argmax_disagree = int(
            pi_d_star_action != best_config_action
        )

        rank_corr = spearman_corr(
            q_star,
            best_logits_np[k],
        )

        mean_return, stderr = discounted_rollout_from_state(
            env,
            act_fn,
            s,
            args.episodes_per_state,
            gamma=ref_cfg.gamma,
            seed=args.eval_seed,
        )

        value_gap = pi_d_star_V - mean_return

        # --------------------------------------------------------------
        # New disagreement definition:
        #
        # A state is a disagreement state when:
        #   1. actions differ
        #   2. PPO is actually worse in expected discounted return
        #
        # Statistical significance is deliberately NOT required here.
        # It is stored separately below.
        # --------------------------------------------------------------
        is_disagreement = int(
            argmax_disagree and value_gap > 0.0
        )

        # Raw severity is the positive value gap ONLY when actions differ.
        # This prevents same-action states from becoming red simply
        # because the empirical rollout estimate happens to be noisy.
        severity_raw = (
            max(0.0, value_gap)
            if argmax_disagree
            else 0.0
        )

        # Statistical significance is retained as a separate diagnostic.
        statistically_significant = int(
            argmax_disagree
            and value_gap > args.z_threshold * stderr
        )

        r, c = env.layout.rc(s)

        n_pi_d_star_action = sa_counts.get((s, pi_d_star_action), 0)
        n_best_config_action = sa_counts.get((s, best_config_action), 0)
        pair_min_samples = min(n_pi_d_star_action, n_best_config_action)

        rows.append(
            {
                "state": s,
                "row": r,
                "col": c,
                "covered": s in covered_states,
                "n_actions_covered": len(state_actions_seen.get(s, ())),
                "pi_d_star_V": pi_d_star_V,
                "pi_d_star_action": pi_d_star_action,
                "best_config_action": best_config_action,
                "argmax_disagree": argmax_disagree,
                "rank_correlation": rank_corr,
                "ppo_discounted_return": mean_return,
                "ppo_discounted_return_stderr": stderr,
                "value_gap": value_gap,
                "is_disagreement": is_disagreement,
                "severity_raw": severity_raw,
                "severity": 0.0,
                "statistically_significant": statistically_significant,
                "true_value_pi_beta": float(
                    true_value_pi_beta[s]
                ),
                "critic_pred_pi_beta": float(
                    prior_critic_pred_full[s]
                ),
                "critic_abs_error": float(
                    abs(
                        prior_critic_pred_full[s]
                        - true_value_pi_beta[s]
                    )
                ),
                "n_pi_d_star_action": n_pi_d_star_action,
                "n_best_config_action": n_best_config_action,
                "pair_min_samples": pair_min_samples,
            }
        )

        if (k + 1) % 100 == 0 or (k + 1) == len(non_terminal):
            print(
                f"  {k + 1}/{len(non_terminal)} states done..."
            )

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Step 4: normalize severity for visualization.
    #
    # IMPORTANT:
    #   - Only positive severity_raw values are considered.
    #   - Negative gaps, including states where pi_D* ~= -50 and PPO
    #     performs much better, do not affect the scale.
    #   - 95th percentile is used instead of max so one extreme state
    #     cannot compress the rest of the heatmap into green.
    # ------------------------------------------------------------------
    positive_severities = df.loc[
        df["severity_raw"] > 0.0,
        "severity_raw",
    ].to_numpy(dtype=np.float64)

    if len(positive_severities) > 0:
        severity_scale = float(
            np.quantile(
                positive_severities,
                0.95,
            )
        )

        severity_scale = max(
            severity_scale,
            1e-12,
        )

        df["severity"] = np.clip(
            df["severity_raw"] / severity_scale,
            0.0,
            1.0,
        )

    else:
        severity_scale = 1.0
        df["severity"] = 0.0

    csv_path = out_dir / "policy_agreement.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    n_disagree = int(
        df["is_disagreement"].sum()
    )

    n_significant = int(
        df["statistically_significant"].sum()
    )

    print("\n=== Summary ===")

    print(
        f"States: {len(df)} total, "
        f"{df['covered'].sum()} covered by D"
    )

    print(
        f"Argmax disagreement alone: "
        f"{df['argmax_disagree'].sum()} states"
    )

    print(
        f"Actual disagreement "
        f"(argmax differs AND PPO has a positive value gap): "
        f"{n_disagree} states"
    )

    print(
        f"  of which covered by D: "
        f"{int((df['is_disagreement'] & df['covered']).sum())}"
    )

    print(
        f"Statistically significant disagreements "
        f"(z-threshold={args.z_threshold}): "
        f"{n_significant} states"
    )

    print(
        f"Severity normalization scale "
        f"(95th percentile of positive gaps): "
        f"{severity_scale:.6g}"
    )

    print(f"Saved {csv_path}")

    # ------------------------------------------------------------------
    # Plot: maze map colored by disagreement severity.
    #
    # Color:
    #   green -> no cost from disagreement
    #   yellow -> small positive cost
    #   orange -> moderate positive cost
    #   red -> large positive cost
    #   blank (white, no border) -> state NOT covered by D at all; no
    #     severity signal exists there, so it is left empty rather than
    #     colored, instead of being marked with a border as before.
    #
    # Marker size encodes n_actions_covered (0-4): how many distinct
    # actions D actually sampled at that state, not just whether the
    # state is covered at all. A floor keeps every square legible even
    # at the lowest coverage. The figure itself is sized up from the
    # original 8x8 so that this size variation stays readable across the
    # full 30x30 grid.
    #
    # All of the above (color scale via colorbar, hazard/goal markers,
    # blank meaning) is documented in the legend/colorbar rather than
    # crammed into the title, so the title itself stays short and fully
    # visible.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(11, 11)
    )

    covered_mask = df["covered"].to_numpy(
        dtype=bool
    )

    cmap = matplotlib.colormaps["RdYlGn_r"]

    norm = mcolors.Normalize(
        vmin=0.0,
        vmax=1.0,
    )

    face_colors = cmap(
        norm(df["severity"].to_numpy(dtype=float))
    )

    face_colors[~covered_mask] = mcolors.to_rgba("white")

    edgecolors = np.where(
        covered_mask,
        "0.3",
        "none",
    )

    linewidths = np.where(
        covered_mask,
        0.3,
        0.0,
    )

    MIN_MARKER_SIZE = 45
    MAX_MARKER_SIZE = 170

    n_actions_covered = df["n_actions_covered"].to_numpy(dtype=float)

    marker_sizes = MIN_MARKER_SIZE + (
        MAX_MARKER_SIZE - MIN_MARKER_SIZE
    ) * (n_actions_covered / env.n_actions)

    sc = ax.scatter(
        df["col"],
        df["row"],
        c=face_colors,
        s=marker_sizes,
        marker="s",
        edgecolors=edgecolors,
        linewidths=linewidths,
    )

    hz_rows = [
        r
        for (r, c) in env.layout.hazards
    ]

    hz_cols = [
        c
        for (r, c) in env.layout.hazards
    ]

    hazard_handle = ax.scatter(
        hz_cols,
        hz_rows,
        marker="x",
        s=120,
        color="black",
        linewidths=2,
        label="hazard",
    )

    goal_handle = ax.scatter(
        [env.layout.goal[1]],
        [env.layout.goal[0]],
        marker="*",
        s=200,
        color="gold",
        label="goal",
        edgecolors="black",
    )

    ax.invert_yaxis()

    ax.set_xlabel("col")
    ax.set_ylabel("row")

    ax.set_title(
        f"Disagreement severity by maze cell\n"
        f"(\u03c0D* {ref.kind} vs. best-config PPO)",
        fontsize=12,
    )

    # `sc` no longer carries its own colormap/norm (colors were assigned
    # manually so uncovered states could be overridden to white), so the
    # colorbar needs its own explicit ScalarMappable rather than reusing
    # `sc` directly.
    severity_mappable = matplotlib.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    severity_mappable.set_array([])

    fig.colorbar(
        severity_mappable,
        ax=ax,
        label=(
            "normalized disagreement severity\n"
            "(green=agree/no cost, red=larger PPO value loss)"
        ),
    )

    # Blank-by-not-covered is not a scatter series of its own (it's a
    # facecolor/edge override on the main `sc` scatter), so it needs a
    # manual legend proxy artist rather than relying on a `label=` kwarg.
    # A faint gray outline is used here ONLY so the blank square is
    # visible as an entry in the legend -- on the map itself, uncovered
    # states have no border at all.
    not_covered_handle = plt.Line2D(
        [0],
        [0],
        marker="s",
        markersize=11,
        markerfacecolor="white",
        markeredgecolor="0.7",
        markeredgewidth=0.8,
        linestyle="None",
        label="blank = not covered by D",
    )

    # Different sizes legend
    size_handles = [
        plt.Line2D(
            [0],
            [0],
            markersize=11 * (n / env.n_actions),
            marker="s",
            markerfacecolor="green",
            markeredgecolor="0.7",
            markeredgewidth=0.8,
            linestyle="None",
            label= str(n) + " action" + ("s" if n > 1 else "") + " covered by D",
        )
        for n in range(1, env.n_actions+1)
    ]

    ax.legend(
        handles=[hazard_handle, goal_handle, not_covered_handle, *size_handles],
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.2, 1.0),
        labelspacing=1.0,
    )

    fig.tight_layout()

    plot_path = out_dir / "policy_agreement_maze_map"

    fig.savefig(
        plot_path.with_suffix(".svg")
    )

    fig.savefig(
        plot_path.with_suffix(".png"),
        dpi=150,
    )

    plt.close(fig)

    print(
        f"Saved {plot_path}.svg/.png"
    )

    # ------------------------------------------------------------------
    # Plot 2: severity vs. pair-level coverage (n_best_config_action) --
    # the "sparse-signal overfitting" hypothesis in one picture: is
    # disagreement severity concentrated where D showed the LOSING action
    # only a handful of times, regardless of how well-covered the state
    # is overall? x-axis is the sample count for whichever action the
    # best-config policy actually ends up preferring (log1p-scaled, since
    # this is heavily right-skewed like every other coverage measure in
    # this project) -- for agreement states this is simply n(s, a*), for
    # disagreement states it's specifically the count behind PPO's wrong
    # choice. A cluster of high-severity points at the low-count end would
    # support the hypothesis; severity spread evenly across the x-axis
    # would not.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    covered_df = df[df["covered"]]
    disagree_mask_plot = covered_df["is_disagreement"].to_numpy(dtype=bool)
    log_n_best = np.log1p(covered_df["n_best_config_action"].to_numpy(dtype=float))
    ax.scatter(
        log_n_best[~disagree_mask_plot], covered_df["severity"][~disagree_mask_plot],
        s=16, alpha=0.5, color="0.6", label="agreement / no cost",
    )
    ax.scatter(
        log_n_best[disagree_mask_plot], covered_df["severity"][disagree_mask_plot],
        s=32, alpha=0.85, color="tab:red", label="disagreement state",
    )
    ax.set_xlabel("log(1 + samples in D for whichever action best-config actually chose)")
    ax.set_ylabel("normalized disagreement severity")
    ax.set_title(
        "Is severity concentrated where PPO's chosen action was rarely observed?\n"
        f"(\u03c0D* {ref.kind} vs. best-config PPO, covered states only)",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot2_path = out_dir / "policy_agreement_severity_vs_pair_coverage"
    fig.savefig(plot2_path.with_suffix(".svg"))
    fig.savefig(plot2_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot2_path}.svg/.png")


if __name__ == "__main__":
    main()