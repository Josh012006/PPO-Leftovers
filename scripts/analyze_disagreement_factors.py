"""What actually predicts disagreement severity (see scripts/
analyze_policy_agreement.py for the definition: argmax differs AND
pi_D* expects meaningfully more than the best-config policy delivers)?

Tests several candidate structural factors against `severity`, each
generalizable beyond this specific maze:
  - coverage        (log(1 + total D samples at this state)) -- generalizes
                     to "how much experience does the learner have here".
                     Reported directly (raw correlation only): it's the
                     one factor expected a priori to matter (less data,
                     less signal), not something to control for.
  - goal_distance    (BFS hops to the goal) -- generalizes to "how far
                     into the task / how long-horizon is this state".
  - hazard_distance  (BFS hops to the nearest hazard) -- generalizes to
                     "proximity to an irreversible failure mode".
  - local_connectivity (open walls at this state, 0-4) -- generalizes to
                     "how much structural redundancy/flexibility exists
                     here" (more open walls = more alternative routes).
  - action_sample_gap (log1p(n_samples(s, best_config_action)) -
                     log1p(n_samples(s, pi_d_star_action)), from D's raw
                     (state, action) counts) -- generalizes to "how much
                     more reinforcement did the optimizer's preferred
                     action get, purely from data volume, than the
                     actually-better action". Positive means D showed the
                     PPO-preferred action more often than pi_D*'s.
  - log_n_best_config_action (log1p(n_samples(s, best_config_action))) --
                     the RAW sparsity of whichever action the optimizer
                     ends up preferring, not the gap against pi_D*'s
                     action. Generalizes to "how much direct evidence
                     supports the specific choice the learner converged
                     on" -- a state can have a large action_sample_gap
                     while still having plenty of absolute samples for
                     both actions, or a small gap while both actions were
                     seen only a handful of times each. Tests the
                     "sparse-signal overfitting" hypothesis directly: does
                     PPO end up confidently wrong specifically where its
                     own preferred action had little direct reinforcement
                     to begin with, regardless of how that compares to
                     pi_D*'s action.
  - log_pair_min_samples (log1p(min(n_samples(s, pi_d_star_action),
                     n_samples(s, best_config_action)))) -- the sparser of
                     the two competing actions specifically, as opposed to
                     `coverage` (all 4 actions summed). A state can look
                     well-covered in aggregate while the two actions that
                     actually matter for this comparison were each seen
                     only a few times.
  - pi_beta_prob_gap (pi_beta(best_config_action|s) -
                     pi_beta(pi_d_star_action|s), from the behavior
                     policy's own actor, not its critic) -- generalizes to
                     "how much did the *ratio denominator* in PPO's
                     clipped objective already favor the optimizer's
                     action over the better one, independent of how many
                     samples happened to land". Distinct from
                     action_sample_gap: one is realized sample counts
                     (noisy, finite-D), the other is the underlying
                     behavior-policy probability that sets the scale of
                     PPO's importance ratio for that state-action pair.

action_sample_gap and pi_beta_prob_gap are both computed only at states
where the argmax differs (0 elsewhere, matching severity's own
definition), and both stay strictly inside what fixed-D PPO's own
optimization mechanism consumes (D's sample counts, the frozen
pi_old actor) -- neither touches pi_beta's critic.

pi_beta's critic accuracy is deliberately NOT a tested factor here. That
the critic is imperfect is already an accepted premise of this whole
project, not a hypothesis to re-confirm via a weak correlation -- the
actual question this project asks is how well fixed-D PPO can still reach
pi_D* while working with D and this potentially-biased critic *as given*,
not whether the critic itself could be made more accurate. start_distance
(BFS hops from the start) is also dropped: under this script's per-state
protocol, every state IS the rollout start (see
analyze_policy_agreement.py), so "distance from the start" has no
meaningful interpretation here.

For every factor EXCEPT coverage, both the RAW correlation and the
PARTIAL correlation (controlling for coverage) against `severity` are
reported -- coverage is the one variable everything else gets checked
"on top of", the same way goal_distance was controlled for earlier (see
README, "Policy agreement"). A factor whose partial correlation survives
controlling for coverage is doing real, independent work; one that
vanishes was likely just riding along with coverage.

Requires policy_agreement.csv from a previous run of
scripts/analyze_policy_agreement.py -- by default the STRICT columns
(`severity_strict`, `is_disagreement_strict`; see that script's own
docstring, "Why a second reference") are used as the response variable,
with automatic fallback to the single-reference `severity`/
`is_disagreement` if the strict columns aren't present (older CSVs, or a
policy_agreement.csv run with --pi-d-star-cross-check ""). Pass
--severity-column / --disagreement-column to override explicitly. Also
needs the prior checkpoint (for pi_beta_prob_gap -- a fresh, cheap forward
pass over all states, no rollout).

Outputs, under --out-dir (default results/analysis/disagreement_factors/):
  disagreement_factors.csv        -- per-state: every factor + severity
                                      + argmax_disagree, joined
  disagreement_factors_summary.csv -- one row per factor: raw and partial
                                      Pearson r against severity
  disagreement_factors_bars.svg/png -- horizontal bar chart of raw vs.
                                      partial correlation per factor

Usage:
    python scripts/analyze_disagreement_factors.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --policy-agreement-csv results/analysis/policy_agreement/policy_agreement.csv \
        --out-dir results/analysis/disagreement_factors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _analysis_lib import compute_sa_counts, goal_bfs_distance, hazard_bfs_distance, local_connectivity
from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig

FACTOR_LABELS = {
    "goal_distance": "distance to goal",
    "hazard_distance": "distance to nearest hazard",
    "local_connectivity": "local connectivity (open walls)",
    "action_sample_gap": "action sample-count gap (log, best-config \u2212 \u03c0D*)",
    "log_n_best_config_action": "raw sparsity of PPO's chosen action (log samples)",
    "log_pair_min_samples": "sparser of the two competing actions (log samples)",
    "pi_beta_prob_gap": "\u03c0\u03b2 action-prob gap (best-config \u2212 \u03c0D*)",
}


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y with the linear effect of x removed (OLS residuals)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument(
        "--policy-agreement-csv", default="results/analysis/policy_agreement/policy_agreement.csv"
    )
    parser.add_argument("--out-dir", default="results/analysis/disagreement_factors")
    parser.add_argument(
        "--severity-column",
        default="severity_strict",
        help="Response variable. Falls back to 'severity' automatically if this column isn't "
        "in --policy-agreement-csv (e.g. an older run, or one made with "
        "--pi-d-star-cross-check '').",
    )
    parser.add_argument(
        "--disagreement-column",
        default="is_disagreement_strict",
        help="Same fallback behavior as --severity-column, to 'is_disagreement'.",
    )
    parser.add_argument(
        "--covered-only",
        action="store_true",
        default=True,
        help="Restrict to states D actually covers (default: on). Disagreement/severity on "
        "uncovered states reflects pi_D*'s own uninformed tie-break there too, not a meaningful "
        "comparison.",
    )
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
    )

    pa = pd.read_csv(args.policy_agreement_csv)
    print(f"Loaded {args.policy_agreement_csv}: {len(pa)} states.")

    severity_col = args.severity_column
    if severity_col not in pa.columns:
        print(f"'{severity_col}' not found in this CSV -- falling back to 'severity'.")
        severity_col = "severity"
    disagreement_col = args.disagreement_column
    if disagreement_col not in pa.columns:
        print(f"'{disagreement_col}' not found in this CSV -- falling back to 'is_disagreement'.")
        disagreement_col = "is_disagreement"
    print(f"Using '{severity_col}' as the response variable, '{disagreement_col}' for masking.")

    dataset = load_dataset(args.dataset)
    sa_counts = compute_sa_counts(dataset)
    total_samples = {s: sum(sa_counts.get((s, a), 0) for a in range(env.n_actions)) for s in pa["state"]}

    prior_ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    print(f"Loaded prior checkpoint (final eval: {prior_ckpt['final_eval']})")

    prior_net = ActorCritic(prior_ckpt["obs_dim"], prior_ckpt["n_actions"], prior_ckpt["hidden_sizes"])
    prior_net.load_state_dict(prior_ckpt["state_dict"])
    prior_net.eval()

    non_terminal = [s for s in range(env.n_states) if not env.is_terminal_state(s)]
    obs_batch = np.stack([env.state_to_obs(s) for s in non_terminal]).astype(np.float32)
    with torch.no_grad():
        prior_logits, _ = prior_net.forward(torch.as_tensor(obs_batch))
        prior_action_probs_np = torch.softmax(prior_logits, dim=-1).numpy()
    prior_action_probs_full = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    for i, s in enumerate(non_terminal):
        prior_action_probs_full[s] = prior_action_probs_np[i]

    goal_d = goal_bfs_distance(env)
    hazard_d = hazard_bfs_distance(env)
    conn = local_connectivity(env)

    df = pa.copy()
    df["total_samples"] = df["state"].map(total_samples)
    df["log_coverage"] = np.log1p(df["total_samples"])
    df["goal_distance"] = df["state"].map(goal_d)
    df["hazard_distance"] = df["state"].map(hazard_d)
    df["local_connectivity"] = df["state"].map(conn)

    # action_sample_gap / pi_beta_prob_gap: both compare pi_D*'s preferred
    # action against best-config PPO's preferred action at each state. When
    # the two agree (no argmax_disagree), both gaps are exactly 0 -- no
    # special-casing needed, since a==a gives log1p(n)-log1p(n)=0 and
    # prob(a)-prob(a)=0 automatically.
    states_arr = df["state"].to_numpy(dtype=int)
    pi_d_star_actions = df["pi_d_star_action"].to_numpy(dtype=int)
    best_config_actions = df["best_config_action"].to_numpy(dtype=int)

    n_star = np.array([sa_counts.get((s, a), 0) for s, a in zip(states_arr, pi_d_star_actions)])
    n_best = np.array([sa_counts.get((s, a), 0) for s, a in zip(states_arr, best_config_actions)])
    df["action_sample_gap"] = np.log1p(n_best) - np.log1p(n_star)
    df["log_n_best_config_action"] = np.log1p(n_best)
    df["log_pair_min_samples"] = np.log1p(np.minimum(n_star, n_best))

    prob_star = prior_action_probs_full[states_arr, pi_d_star_actions]
    prob_best = prior_action_probs_full[states_arr, best_config_actions]
    df["pi_beta_prob_gap"] = prob_best - prob_star

    if args.covered_only:
        df = df[df["covered"]].copy()
    print(f"Analyzing {len(df)} states ({'covered only' if args.covered_only else 'all'}).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "disagreement_factors.csv", index=False)

    severity = df[severity_col].values
    coverage = df["log_coverage"].values

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    if np.std(severity) == 0:
        print(
            "\nWARNING: severity is constant (0 disagreement states found) -- every correlation "
            "below is undefined (NaN), not a bug. Re-run scripts/analyze_policy_agreement.py with "
            "more --episodes-per-state or a lower --z-threshold if this happens unexpectedly on "
            "the real dataset.\n"
        )

    summary_rows = []
    print("\n=== coverage (baseline factor, raw correlation only) ===")
    r_cov = safe_corr(coverage, severity)
    print(f"  log_coverage vs severity: r = {r_cov:+.3f}")
    summary_rows.append({"factor": "coverage", "raw_r": r_cov, "partial_r_controlling_coverage": None})

    print("\n=== other factors: raw vs. partial (controlling for coverage) ===")
    for factor in [
        "goal_distance",
        "hazard_distance",
        "local_connectivity",
        "action_sample_gap",
        "log_n_best_config_action",
        "log_pair_min_samples",
        "pi_beta_prob_gap",
    ]:
        x = df[factor].values.astype(float)
        raw_r = safe_corr(x, severity)
        if np.std(severity) == 0 or np.std(coverage) == 0:
            partial_r = float("nan")
        else:
            resid_x = residualize(x, coverage)
            resid_sev = residualize(severity, coverage)
            partial_r = safe_corr(resid_x, resid_sev)
        label = FACTOR_LABELS[factor]
        print(f"  {label:32s}  raw r = {raw_r:+.3f}   partial r (net of coverage) = {partial_r:+.3f}")
        summary_rows.append({"factor": factor, "raw_r": raw_r, "partial_r_controlling_coverage": partial_r})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "disagreement_factors_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {out_dir / 'disagreement_factors.csv'}")
    print(f"Saved {summary_path}")

    # --- Bar chart: raw vs partial correlation per factor. ---
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(summary_df) + 1.5))
    y = np.arange(len(summary_df))
    labels = [FACTOR_LABELS.get(f, f) for f in summary_df["factor"]]
    raw_vals = summary_df["raw_r"].values
    partial_vals = summary_df["partial_r_controlling_coverage"].fillna(summary_df["raw_r"]).values
    height = 0.35
    ax.barh(y + height / 2, raw_vals, height=height, color="0.6", label="raw correlation")
    ax.barh(y - height / 2, partial_vals, height=height, color="tab:blue", label="partial (net of coverage)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("correlation with disagreement severity")
    ax.set_title("What predicts disagreement severity?\n(coverage has no separate partial bar)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = out_dir / "disagreement_factors_bars"
    fig.savefig(plot_path.with_suffix(".svg"))
    fig.savefig(plot_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}.svg/.png")


if __name__ == "__main__":
    main()