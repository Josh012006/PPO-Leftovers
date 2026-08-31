"""Prior Asymmetry -> Learning Bias (README, "Next Steps: Testing Whether
the Imperfect Behavior Policy Limits PPO's Correction").

Hypothesis under test: because pi_beta is imperfect (~35% success), its
action distribution already encodes errors at some states. Since
pi_old = pi_beta stays frozen for the whole fixed-D PPO window, those
errors are the *reference point* the update ratio is measured against.
The prediction is that PPO systematically recovers a *smaller fraction*
of pi_beta's initial error at states where that error is *larger* --
not just that large-error states end up with more residual error in
absolute terms (which would be true almost by construction), but that
the *fraction corrected* itself shrinks.

For every state, with a* = pi_D*'s preferred action (from
--policy-agreement-csv, so this works for either pi_D* definition):

  p_beta_star  = pi_beta(a* | s)            -- behavior policy's actor,
                                                cheap forward pass, no
                                                rollout
  p_theta_star = pi_theta,final(a* | s)     -- the SAME best-config
                                                checkpoint already trained
                                                for analyze_policy_agree-
                                                ment.py, reused via
                                                --best-config-checkpoint,
                                                no retrain
  delta_p_star = p_theta_star - p_beta_star -- signed absolute movement
                                                PPO made toward a*
  prior_error  = 1 - p_beta_star            -- how far pi_beta was from
                                                *certainty* on a*. pi_D*
                                                is a deterministic
                                                argmax-of-Q* policy (see
                                                pi_d_star_action in
                                                analyze_policy_agreement.
                                                py), so it implicitly
                                                assigns probability 1 to
                                                a* -- prior_error is
                                                exactly the gap PPO would
                                                need to close to match
                                                pi_D* on this one action.
  C(s)         = delta_p_star / prior_error -- the FRACTION of that gap
                                                PPO actually closed. C=1:
                                                fully corrected. C=0: no
                                                movement at all. C<0:
                                                moved away from a*. C>1:
                                                overshot past pi_D*'s
                                                implied certainty.

C(s) is only defined where prior_error clears --min-prior-error (default
0.02): if pi_beta was already almost certain about a*, there is barely
any "error" to correct, and a small denominator would blow up an
otherwise-tiny movement into a meaningless extreme ratio. States below
the threshold are kept in the CSV (for transparency) with C left blank,
but excluded from the regression and plots.

The key prediction, exactly as stated in the README:

    larger initial pi_beta error  ==>  smaller correction fraction C

is tested with the regression

    C(s) ~ prior_error(s) + log_coverage(s)

(log_coverage controls for how much D itself supports the state, since a
poorly-covered state gives PPO less to work with independent of pi_beta's
own error). A negative, significant prior_error coefficient after
controlling for coverage is the signature this experiment is looking for.
Regression inference (std. error, 95% CI, two-sided p-value) uses a
normal approximation to the coefficient's sampling distribution rather
than an exact Student-t -- deliberately, to avoid adding scipy as a
project dependency; this project's own correlation code
(analyze_disagreement_factors.py) makes the same kind of choice (plain
numpy, no scipy). The approximation is standard practice and safe at the
sample sizes here (n in the hundreds).

The raw and coverage-partial Pearson correlation between prior_error and
C are also reported directly, for a reading consistent with the rest of
this project's "raw vs. partial, controlling for coverage" convention
(see analyze_disagreement_factors.py, and README "Policy agreement").

NOT implemented here (both explicitly flagged in the README as follow-on
work, contingent on this observational result, not part of it):
  - the per-epoch training-trajectory analysis (would need pi_theta
    checkpoints saved *during* training, which the current pipeline
    doesn't save -- only the final/best checkpoint exists today);
  - the controlled experiment that deliberately manipulates pi_beta's
    initial action preference.

Outputs, under --out-dir (default results/analysis/prior_correction/):
  prior_correction.csv            -- per-state: p_beta_star, p_theta_star,
                                      delta_p_star, prior_error, C (blank
                                      where undefined), plus context
                                      columns (severity, is_disagreement,
                                      covered, log_coverage, ...)
  prior_correction_regression.csv -- OLS term, coef, std_err, t_stat,
                                      p_value (normal approx), 95% CI,
                                      plus n and R^2 as a trailing row
  prior_correction_summary.csv    -- raw/partial Pearson r, same
                                      convention as disagreement_factors
  prior_correction_scatter.svg/png    -- prior_error vs. C, raw
  prior_correction_partial.svg/png    -- the same relationship after
                                      residualizing out log_coverage

Usage (run once per pi_D* definition):
    python scripts/analyze_prior_correction.py --env-config configs/env_maze.yaml --dataset results/dataset_D.pkl --prior-checkpoint results/prior_checkpoint.pt --best-config-checkpoint results/analysis/policy_agreement/best_config_checkpoint.pt --policy-agreement-csv results/analysis/policy_agreement/policy_agreement.csv --out-dir results/analysis/prior_correction

    python scripts/analyze_prior_correction.py --env-config configs/env_maze.yaml --dataset results/dataset_D.pkl --prior-checkpoint results/prior_checkpoint.pt --best-config-checkpoint results/analysis/policy_agreement_true_restricted/best_config_checkpoint.pt --policy-agreement-csv results/analysis/policy_agreement_true_restricted/policy_agreement.csv --out-dir results/analysis/prior_correction_true_restricted
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig

# State 320 is the counterexample the README calls out explicitly: pi_beta
# strongly favored pi_D*'s action there (pi_beta_prob_gap = -0.956) yet the
# final policy still disagrees at high severity (0.845). Annotated on the
# scatter plots, when present, rather than treated as a special case in
# any of the numbers.
HIGHLIGHT_STATE = 320


def load_action_probs(checkpoint_path: str, env: StochasticMazeEnv) -> np.ndarray:
    """Full (n_states, n_actions) softmax action-probability table for a
    saved actor-critic checkpoint. A single cheap forward pass over every
    non-terminal state -- no rollout."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden_sizes"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    non_terminal = [s for s in range(env.n_states) if not env.is_terminal_state(s)]
    obs_batch = np.stack([env.state_to_obs(s) for s in non_terminal]).astype(np.float32)
    with torch.no_grad():
        logits, _ = net.forward(torch.as_tensor(obs_batch))
        probs_np = torch.softmax(logits, dim=-1).numpy()

    probs_full = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    for i, s in enumerate(non_terminal):
        probs_full[s] = probs_np[i]

    tag = ckpt.get("final_eval", ckpt.get("success_rate"))
    print(f"Loaded {checkpoint_path} (success/eval: {tag}, epoch: {ckpt.get('epoch', 'n/a')})")
    return probs_full


def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y with the linear effect of x removed (OLS residuals). Same
    convention as analyze_disagreement_factors.py."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ coef


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in np.atleast_1d(z)])


def ols_with_inference(features: dict[str, np.ndarray], y: np.ndarray) -> tuple[pd.DataFrame, float, int]:
    """OLS of y on the named features plus an intercept. Returns a
    per-term table (coef, std_err, t_stat, two-sided p-value via a normal
    approximation, 95% CI) along with R^2 and n. See module docstring for
    why a normal approximation is used instead of an exact Student-t."""
    names = list(features.keys()) + ["intercept"]
    X = np.column_stack([features[k] for k in features] + [np.ones_like(y)])
    n, k = X.shape

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    dof = max(n - k, 1)
    sigma2 = rss / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(xtx_inv))
    t_stat = beta / se
    p_val = 2.0 * (1.0 - _norm_cdf(np.abs(t_stat)))
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se

    rows = [
        {
            "term": name,
            "coef": beta[i],
            "std_err": se[i],
            "t_stat": t_stat[i],
            "p_value_normal_approx": p_val[i],
            "ci95_low": ci_lo[i],
            "ci95_high": ci_hi[i],
        }
        for i, name in enumerate(names)
    ]
    return pd.DataFrame(rows), r2, n


def scatter_with_fit(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    disagree_mask: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    highlight_xy: tuple[float, float] | None,
) -> None:
    ax.scatter(x[~disagree_mask], y[~disagree_mask], s=18, color="0.6", alpha=0.6, label="agreement state")
    ax.scatter(x[disagree_mask], y[disagree_mask], s=28, color="tab:red", alpha=0.85, label="disagreement state")

    if np.std(x) > 0:
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(coef, xs), color="black", linewidth=1.5, label="linear fit")

    if highlight_xy is not None:
        hx, hy = highlight_xy
        ax.annotate(
            f"state {HIGHLIGHT_STATE}",
            xy=(hx, hy),
            xytext=(15, 15),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=8,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument(
        "--best-config-checkpoint", default="results/analysis/policy_agreement/best_config_checkpoint.pt"
    )
    parser.add_argument("--policy-agreement-csv", default="results/analysis/policy_agreement/policy_agreement.csv")
    parser.add_argument("--out-dir", default="results/analysis/prior_correction")
    parser.add_argument(
        "--covered-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict to states D actually covers (default: on), consistent with "
        "analyze_disagreement_factors.py.",
    )
    parser.add_argument(
        "--min-prior-error",
        type=float,
        default=0.02,
        help="C(s) requires prior_error(s) = 1 - pi_beta(a*|s) to be at least this large; "
        "below it, a tiny denominator would turn a negligible absolute movement into a "
        "meaningless extreme ratio. States below the threshold are kept in the CSV with "
        "C left blank, but excluded from the regression, correlations, and plots.",
    )
    parser.add_argument(
        "--clip-c-plot",
        type=float,
        default=4.0,
        help="Clip |C| to this bound for the SCATTER PLOTS only (the CSV and regression "
        "always use the unclipped values) -- keeps a handful of extreme-ratio states from "
        "compressing the rest of the plot into unreadability.",
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
    total_samples: dict[int, int] = {}
    for tr in dataset.trajectories:
        for s in tr.states.tolist():
            total_samples[int(s)] = total_samples.get(int(s), 0) + 1

    prior_probs = load_action_probs(args.prior_checkpoint, env)
    theta_probs = load_action_probs(args.best_config_checkpoint, env)

    df = pa.copy()
    states_arr = df["state"].to_numpy(dtype=int)
    a_star = df["pi_d_star_action"].to_numpy(dtype=int)

    df["total_samples"] = df["state"].map(lambda s: total_samples.get(int(s), 0))
    df["log_coverage"] = np.log1p(df["total_samples"])

    df["p_beta_star"] = prior_probs[states_arr, a_star]
    df["p_theta_star"] = theta_probs[states_arr, a_star]
    df["delta_p_star"] = df["p_theta_star"] - df["p_beta_star"]
    df["prior_error"] = 1.0 - df["p_beta_star"]

    valid_denom = (df["prior_error"] >= args.min_prior_error).to_numpy()
    c_vals = np.full(len(df), np.nan)
    np.divide(
        df["delta_p_star"].to_numpy(dtype=float),
        df["prior_error"].to_numpy(dtype=float),
        out=c_vals,
        where=valid_denom,
    )
    df["C"] = c_vals

    if args.covered_only:
        df = df[df["covered"]].copy()
    n_total = len(df)
    n_valid = int(df["C"].notna().sum())
    print(
        f"Analyzing {n_total} states ({'covered only' if args.covered_only else 'all'}); "
        f"{n_valid} have prior_error >= {args.min_prior_error} and a defined C(s), "
        f"{n_total - n_valid} excluded (pi_beta already near-certain on pi_D*'s action)."
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "prior_correction.csv", index=False)
    print(f"Saved {out_dir / 'prior_correction.csv'}")

    valid = df[df["C"].notna()].copy()
    prior_error = valid["prior_error"].to_numpy(dtype=float)
    coverage = valid["log_coverage"].to_numpy(dtype=float)
    C = valid["C"].to_numpy(dtype=float)
    disagree_mask = valid["is_disagreement"].to_numpy(dtype=bool)

    # --- Regression: C ~ prior_error + log_coverage ---
    reg_df, r2, n_reg = ols_with_inference({"prior_error": prior_error, "log_coverage": coverage}, C)
    reg_df.to_csv(out_dir / "prior_correction_regression.csv", index=False)
    print(f"\n=== Regression: C(s) ~ prior_error(s) + log_coverage(s)  (n={n_reg}, R^2={r2:.3f}) ===")
    print(reg_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(
        "  (p-values use a normal approximation, not an exact Student-t -- see module "
        "docstring; safe at this sample size, avoids adding scipy as a dependency.)"
    )

    # --- Raw / partial Pearson r, same convention as analyze_disagreement_factors.py ---
    raw_r = safe_corr(prior_error, C)
    resid_pe = residualize(prior_error, coverage)
    resid_C = residualize(C, coverage)
    partial_r = safe_corr(resid_pe, resid_C)
    print("\n=== prior_error vs. C: raw vs. partial (controlling for coverage) ===")
    print(f"  raw r = {raw_r:+.3f}   partial r (net of coverage) = {partial_r:+.3f}")
    pd.DataFrame(
        [{"factor": "prior_error", "raw_r": raw_r, "partial_r_controlling_coverage": partial_r}]
    ).to_csv(out_dir / "prior_correction_summary.csv", index=False)
    print(f"Saved {out_dir / 'prior_correction_summary.csv'}")

    prediction_holds = reg_df.loc[reg_df["term"] == "prior_error", "coef"].iloc[0] < 0
    print(
        f"\nKey prediction (larger prior_error -> smaller C) is "
        f"{'SUPPORTED (negative coefficient)' if prediction_holds else 'NOT supported (coefficient is non-negative)'} "
        f"by this run's sign alone -- check the p-value/CI above before drawing a conclusion."
    )

    highlight_row = valid[valid["state"] == HIGHLIGHT_STATE]
    highlight_xy = (
        (float(highlight_row["prior_error"].iloc[0]), float(np.clip(highlight_row["C"].iloc[0], -args.clip_c_plot, args.clip_c_plot)))
        if len(highlight_row) > 0
        else None
    )
    highlight_xy_partial = None
    if len(highlight_row) > 0:
        hi = valid.index.get_loc(highlight_row.index[0])
        highlight_xy_partial = (float(resid_pe[hi]), float(resid_C[hi]))

    C_plot = np.clip(C, -args.clip_c_plot, args.clip_c_plot)
    n_clipped = int(np.sum(np.abs(C) > args.clip_c_plot))

    # --- Plot 1: raw prior_error vs. C ---
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_with_fit(
        ax,
        prior_error,
        C_plot,
        disagree_mask,
        xlabel="prior_error(s) = 1 \u2212 \u03c0\u03b2(a*|s)",
        ylabel=f"C(s) = correction fraction{' (clipped to \u00b1' + str(args.clip_c_plot) + ')' if n_clipped else ''}",
        title="Does a larger initial \u03c0\u03b2 error predict a smaller correction fraction?",
        highlight_xy=highlight_xy,
    )
    ax.axhline(1.0, color="tab:green", linestyle="--", linewidth=1, alpha=0.7, label="C=1: fully corrected")
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1, alpha=0.7, label="C=0: no correction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "prior_correction_scatter.svg")
    fig.savefig(out_dir / "prior_correction_scatter.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'prior_correction_scatter'}.svg/.png")

    # --- Plot 2: same relationship with log_coverage residualized out ---
    fig, ax = plt.subplots(figsize=(7, 6))
    resid_C_plot = np.clip(resid_C, -args.clip_c_plot, args.clip_c_plot)
    scatter_with_fit(
        ax,
        resid_pe,
        resid_C_plot,
        disagree_mask,
        xlabel="prior_error(s), residualized on log_coverage",
        ylabel="C(s), residualized on log_coverage",
        title="Same relationship, controlling for coverage",
        highlight_xy=highlight_xy_partial,
    )
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1, alpha=0.7, label="no residual correction effect")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "prior_correction_partial.svg")
    fig.savefig(out_dir / "prior_correction_partial.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'prior_correction_partial'}.svg/.png")

    if n_clipped:
        print(f"\n(Note: {n_clipped} state(s) had |C| > {args.clip_c_plot} and were clipped for the plots only.)")


if __name__ == "__main__":
    main()
