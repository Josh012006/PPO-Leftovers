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
  - start_distance   (BFS hops from the start) -- generalizes to "how
                     deep into an episode this state typically sits".
  - hazard_distance  (BFS hops to the nearest hazard) -- generalizes to
                     "proximity to an irreversible failure mode".
  - local_connectivity (open walls at this state, 0-4) -- generalizes to
                     "how much structural redundancy/flexibility exists
                     here" (more open walls = more alternative routes).
  - critic_abs_error (|pi_beta's predicted V - exact true V|, from
                     scripts/analyze_critic_accuracy.py's diagnostic) --
                     generalizes to "how wrong is the learned value
                     function here".

For every factor EXCEPT coverage, both the RAW correlation and the
PARTIAL correlation (controlling for coverage) against `severity` are
reported -- coverage is the one variable everything else gets checked
"on top of", the same way goal_distance was controlled for earlier (see
README, "Policy agreement"). A factor whose partial correlation survives
controlling for coverage is doing real, independent work; one that
vanishes was likely just riding along with coverage.

Requires policy_agreement.csv from a previous run of
scripts/analyze_policy_agreement.py (any --pi-d-star variant; state-level
severity doesn't depend on which one was used for that run's own
disagreement flags, though results may differ slightly run to run).

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
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _analysis_lib import goal_bfs_distance, hazard_bfs_distance, local_connectivity, start_bfs_distance
from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.utils.config import MazeEnvConfig

FACTOR_LABELS = {
    "goal_distance": "distance to goal",
    "start_distance": "distance from start",
    "hazard_distance": "distance to nearest hazard",
    "local_connectivity": "local connectivity (open walls)",
    "critic_abs_error": "critic error under \u03c0\u03b2",
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
    parser.add_argument(
        "--policy-agreement-csv", default="results/analysis/policy_agreement/policy_agreement.csv"
    )
    parser.add_argument("--out-dir", default="results/analysis/disagreement_factors")
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

    dataset = load_dataset(args.dataset)
    sa_counts: Counter[tuple[int, int]] = Counter()
    for tr in dataset.trajectories:
        for s, a in zip(tr.states.tolist(), tr.actions.tolist()):
            sa_counts[(int(s), int(a))] += 1
    total_samples = {s: sum(sa_counts.get((s, a), 0) for a in range(env.n_actions)) for s in pa["state"]}

    goal_d = goal_bfs_distance(env)
    start_d = start_bfs_distance(env)
    hazard_d = hazard_bfs_distance(env)
    conn = local_connectivity(env)

    df = pa.copy()
    df["total_samples"] = df["state"].map(total_samples)
    df["log_coverage"] = np.log1p(df["total_samples"])
    df["goal_distance"] = df["state"].map(goal_d)
    df["start_distance"] = df["state"].map(start_d)
    df["hazard_distance"] = df["state"].map(hazard_d)
    df["local_connectivity"] = df["state"].map(conn)

    if args.covered_only:
        df = df[df["covered"]].copy()
    print(f"Analyzing {len(df)} states ({'covered only' if args.covered_only else 'all'}).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "disagreement_factors.csv", index=False)

    severity = df["severity"].values
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
    for factor in ["goal_distance", "start_distance", "hazard_distance", "local_connectivity", "critic_abs_error"]:
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
