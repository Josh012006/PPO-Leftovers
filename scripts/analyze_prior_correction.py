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

C(s) is only guarded by --min-prior-error (default 1e-3) against a
literal near-zero denominator; this is now purely a numerical safety
floor, not the scientific fix for the problem described next. States
below it are kept in the CSV with C left blank.

## Why a hard cutoff on prior_error is the wrong fix, with formulas

C(s) = delta_p_star(s) / prior_error(s). delta_p_star is a difference of
two probabilities, bounded in [-1, 1], and its typical size reflects
training noise (minibatch composition, epoch-to-epoch drift) that has no
structural reason to shrink as prior_error shrinks. Write that noise's
scale as a roughly constant sigma_delta. Then, by simple error
propagation:

    std(C(s) | prior_error(s) = x)  ~=  sigma_delta / x

C's variance is not roughly constant across states -- it blows up like
1/x^2 as prior_error -> 0, by construction of the ratio, regardless of
whether the underlying hypothesis is true or false. Binning the observed
C by prior_error confirms this directly: std(C) falls from ~4.7 (states
with prior_error in [0.02, 0.1]) to ~0.2 (states with prior_error in
[0.6, 1.0]) -- roughly the 1/x pattern predicted above. An unweighted OLS
fit is dominated by exactly the handful of states where this variance is
largest, which is also where the ratio is least trustworthy. That is why
the raw coefficient can come out positive (wrong sign) at a lax cutoff,
flip to negative somewhere around cutoff~0.1, and stay negative beyond
it: not because 0.1 is the "right" threshold, but because *any* cutoff
that excludes enough of the high-variance tail will show the same sign.
Picking one specific cutoff post hoc, because it gives the expected sign,
is exactly the kind of researcher-degree-of-freedom this project avoids
elsewhere (see the z-threshold correction in the Policy agreement
section). Two things address that directly, without hand-picking a
number:

1. **Weighted least squares (the main fix).** Down-weight each state by
   its own expected variance instead of hard-excluding it: weight(s) =
   prior_error(s)^2 (the inverse of sigma_delta^2/prior_error(s)^2, with
   the constant sigma_delta^2 folding into the overall residual scale
   estimate, so it doesn't need to be estimated separately to get the
   point estimate right). States near prior_error=0 are then
   automatically given almost no influence -- smoothly, not via a
   cutoff -- while still being included and visible in the output.
   Implemented as ordinary WLS: beta = (X^T W X)^-1 X^T W y, with
   W = diag(weight(s)); standard errors from
   sigma^2 (X^T W X)^-1, sigma^2 = sum(weight(s) * residual(s)^2) / dof
   (the usual WLS variance estimator under correctly-specified weights).
   This is now the PRIMARY reported regression (see wls_with_inference).
   The plain OLS is still computed and reported alongside it, labeled
   accordingly, purely so the distortion above stays visible for
   comparison -- it is not the number to draw conclusions from.

2. **Threshold-sensitivity check (the robustness diagnostic).** Sweep
   --min-prior-error over a grid (default 0.01 to 0.30) and re-fit BOTH
   OLS and WLS at each cutoff. If the WLS coefficient is doing its job,
   it should stay close to stable (in sign and magnitude, within
   overlapping confidence intervals) across the whole grid, since the
   weighting -- not the cutoff -- is what's controlling the
   high-variance tail's influence. The OLS coefficient, by contrast, is
   expected to swing with the cutoff, visibly reproducing the artifact
   above. This is a plot to check for stability, not a way to select a
   cutoff -- if WLS is *also* unstable across the grid, that is evidence
   against the hypothesis (or against the weighting model), not a reason
   to keep searching for a threshold that "works."

The key prediction, exactly as stated in the README:

    larger initial pi_beta error  ==>  smaller correction fraction C

is tested via the WLS regression

    C(s) ~ prior_error(s) + log_coverage(s)      [weight = prior_error^2]

(log_coverage controls for how much D itself supports the state, since a
poorly-covered state gives PPO less to work with independent of pi_beta's
own error). A negative, significant prior_error coefficient, stable
across the sensitivity grid, is the signature this experiment is looking
for. Regression inference (std. error, 95% CI, two-sided p-value) uses a
normal approximation to the coefficient's sampling distribution rather
than an exact Student-t -- deliberately, to avoid adding scipy as a
project dependency; this project's own correlation code
(analyze_disagreement_factors.py) makes the same kind of choice (plain
numpy, no scipy). The approximation is standard practice and safe at the
sample sizes here (n in the hundreds).

The raw and coverage-partial Pearson correlation between prior_error and
C are also reported directly (unweighted, for a reading consistent with
the rest of this project's "raw vs. partial, controlling for coverage"
convention -- see analyze_disagreement_factors.py, and README "Policy
agreement"), but the WLS regression is the number that should actually
be trusted for this specific relationship, precisely because of the
heteroscedasticity described above.

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
  prior_correction_regression.csv -- one row per (method, term): method is
                                      "ols" or "wls" -- coef, std_err,
                                      t_stat, p_value (normal approx),
                                      95% CI, plus n and R^2. Read the
                                      "wls" rows for the actual conclusion.
  prior_correction_summary.csv    -- raw/partial Pearson r (unweighted),
                                      same convention as disagreement_factors
  prior_correction_sensitivity.csv    -- one row per --min-prior-error
                                      value swept: ols/wls coef + 95% CI
                                      at that cutoff, and n retained
  prior_correction_scatter.svg/png    -- prior_error vs. C, both the OLS
                                      and WLS fit lines overlaid
  prior_correction_partial.svg/png    -- the same relationship after
                                      residualizing out log_coverage
                                      (unweighted -- illustrative only,
                                      see WLS regression for the actual
                                      statistical conclusion)
  prior_correction_sensitivity.svg/png -- OLS vs. WLS coefficient (with
                                      95% CI band) as a function of the
                                      cutoff -- the robustness check
                                      described above

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


def wls_with_inference(
    features: dict[str, np.ndarray], y: np.ndarray, weights: np.ndarray
) -> tuple[pd.DataFrame, float, int]:
    """Weighted least squares of y on the named features plus an
    intercept, weight(s) supplied by the caller (here, prior_error(s)^2 --
    see module docstring for why). Same per-term output shape as
    ols_with_inference: coef, std_err, t_stat, two-sided p-value (normal
    approximation), 95% CI, plus a weighted R^2 and n.

    beta = (X^T W X)^-1 X^T W y, W = diag(weights). Standard errors from
    sigma^2 (X^T W X)^-1 with sigma^2 = sum(weights * residual^2) / dof --
    the usual WLS variance estimator when the weights correctly capture
    the *relative* variance across observations (an overall constant
    factor in `weights` cancels out of beta and is absorbed into sigma^2,
    so it never needs to be estimated separately)."""
    names = list(features.keys()) + ["intercept"]
    X = np.column_stack([features[k] for k in features] + [np.ones_like(y)])
    n, k = X.shape

    w = np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(w)
    Xw = X * sqrt_w[:, None]
    yw = y * sqrt_w

    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    weighted_rss = float(np.sum(w * resid**2))

    y_bar_w = float(np.sum(w * y) / np.sum(w))
    weighted_tss = float(np.sum(w * (y - y_bar_w) ** 2))
    r2 = 1.0 - weighted_rss / weighted_tss if weighted_tss > 0 else float("nan")

    dof = max(n - k, 1)
    sigma2 = weighted_rss / dof
    xtwx_inv = np.linalg.pinv(Xw.T @ Xw)
    se = np.sqrt(sigma2 * np.diag(xtwx_inv))
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


def threshold_sensitivity(
    df_valid: pd.DataFrame, thresholds: np.ndarray, min_n: int = 20
) -> pd.DataFrame:
    """Re-fit both OLS and WLS (weight = prior_error^2) at each
    --min-prior-error cutoff in `thresholds`. See module docstring: this
    is a robustness check, not a way to pick a cutoff. Cutoffs that would
    leave fewer than `min_n` states are skipped rather than reported as
    an unreliable regression."""
    rows = []
    for thresh in thresholds:
        sub = df_valid[df_valid["prior_error"] >= thresh]
        if len(sub) < min_n:
            continue
        pe = sub["prior_error"].to_numpy(dtype=float)
        cov = sub["log_coverage"].to_numpy(dtype=float)
        c = sub["C"].to_numpy(dtype=float)

        ols_df, _, n = ols_with_inference({"prior_error": pe, "log_coverage": cov}, c)
        wls_df, _, _ = wls_with_inference({"prior_error": pe, "log_coverage": cov}, c, pe**2)

        ols_row = ols_df[ols_df["term"] == "prior_error"].iloc[0]
        wls_row = wls_df[wls_df["term"] == "prior_error"].iloc[0]
        rows.append(
            {
                "min_prior_error": thresh,
                "n": n,
                "ols_coef": ols_row["coef"],
                "ols_ci95_low": ols_row["ci95_low"],
                "ols_ci95_high": ols_row["ci95_high"],
                "wls_coef": wls_row["coef"],
                "wls_ci95_low": wls_row["ci95_low"],
                "wls_ci95_high": wls_row["ci95_high"],
            }
        )
    return pd.DataFrame(rows)



def scatter_with_fit(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    disagree_mask: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    highlight_xy: tuple[float, float] | None,
    ols_coef: np.ndarray | None = None,
    wls_coef: np.ndarray | None = None,
) -> None:
    ax.scatter(x[~disagree_mask], y[~disagree_mask], s=18, color="0.6", alpha=0.6, label="agreement state")
    ax.scatter(x[disagree_mask], y[disagree_mask], s=28, color="tab:red", alpha=0.85, label="disagreement state")

    xs = np.linspace(x.min(), x.max(), 100) if np.std(x) > 0 else None

    if xs is not None and ols_coef is not None:
        # ols_coef comes from ols_with_inference's [prior_error, log_coverage,
        # intercept] fit; log_coverage is held at its sample mean to draw a
        # 1D slice through the fitted plane.
        ax.plot(
            xs,
            ols_coef[0] * xs + ols_coef[2],
            color="0.3",
            linestyle="--",
            linewidth=1.5,
            label="OLS fit (biased near prior_error=0)",
        )

    if xs is not None and wls_coef is not None:
        ax.plot(
            xs,
            wls_coef[0] * xs + wls_coef[2],
            color="tab:blue",
            linewidth=2.0,
            label="WLS fit (weight = prior_error\u00b2)",
        )

    if xs is not None and ols_coef is None and wls_coef is None:
        coef = np.polyfit(x, y, deg=1)
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
        default=1e-3,
        help="Purely a numerical safety floor against a literal near-zero denominator in "
        "C(s) = delta_p_star(s) / prior_error(s) -- NOT the fix for the heteroscedasticity "
        "problem (see module docstring): that is handled by weighting the regression by "
        "prior_error(s)^2 (WLS), not by this cutoff. States below the floor are kept in "
        "the CSV with C left blank.",
    )
    parser.add_argument(
        "--sensitivity-max-threshold",
        type=float,
        default=0.30,
        help="Upper end of the --min-prior-error grid swept for the threshold-sensitivity "
        "diagnostic (checks that the WLS coefficient is stable across cutoffs, rather than "
        "picking one cutoff because it gives the expected sign -- see module docstring).",
    )
    parser.add_argument("--sensitivity-step", type=float, default=0.01)
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

    # --- Regression: C ~ prior_error + log_coverage, OLS and WLS ---
    ols_df, ols_r2, n_reg = ols_with_inference({"prior_error": prior_error, "log_coverage": coverage}, C)
    wls_df, wls_r2, _ = wls_with_inference({"prior_error": prior_error, "log_coverage": coverage}, C, prior_error**2)
    ols_df.insert(0, "method", "ols")
    wls_df.insert(0, "method", "wls")
    ols_df["r2"] = ols_r2
    wls_df["r2"] = wls_r2
    reg_df = pd.concat([ols_df, wls_df], ignore_index=True)
    reg_df.to_csv(out_dir / "prior_correction_regression.csv", index=False)

    print(f"\n=== OLS: C(s) ~ prior_error(s) + log_coverage(s)  (n={n_reg}, R^2={ols_r2:.3f}) ===")
    print(ols_df.drop(columns=["method", "r2"]).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("  (kept for comparison only -- see module docstring on why this is biased here.)")
    print(f"\n=== WLS, weight=prior_error\u00b2: C(s) ~ prior_error(s) + log_coverage(s)  (n={n_reg}, R^2={wls_r2:.3f}) ===")
    print(wls_df.drop(columns=["method", "r2"]).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(
        "  (this is the number to draw a conclusion from. p-values use a normal "
        "approximation, not an exact Student-t -- see module docstring.)"
    )

    # --- Threshold-sensitivity diagnostic: is the WLS coefficient stable across cutoffs? ---
    thresholds = np.arange(args.min_prior_error, args.sensitivity_max_threshold + 1e-9, args.sensitivity_step)
    sens_df = threshold_sensitivity(valid, thresholds)
    sens_df.to_csv(out_dir / "prior_correction_sensitivity.csv", index=False)
    print(f"Saved {out_dir / 'prior_correction_sensitivity.csv'} ({len(sens_df)} cutoffs tested)")

    # --- Raw / partial Pearson r, same convention as analyze_disagreement_factors.py ---
    raw_r = safe_corr(prior_error, C)
    resid_pe = residualize(prior_error, coverage)
    resid_C = residualize(C, coverage)
    partial_r = safe_corr(resid_pe, resid_C)
    print("\n=== prior_error vs. C: raw vs. partial (controlling for coverage), UNWEIGHTED ===")
    print(f"  raw r = {raw_r:+.3f}   partial r (net of coverage) = {partial_r:+.3f}")
    pd.DataFrame(
        [{"factor": "prior_error", "raw_r": raw_r, "partial_r_controlling_coverage": partial_r}]
    ).to_csv(out_dir / "prior_correction_summary.csv", index=False)
    print(f"Saved {out_dir / 'prior_correction_summary.csv'}")

    wls_coef_pe = wls_df.loc[wls_df["term"] == "prior_error", "coef"].iloc[0]
    wls_p_pe = wls_df.loc[wls_df["term"] == "prior_error", "p_value_normal_approx"].iloc[0]
    prediction_holds = wls_coef_pe < 0
    print(
        f"\nKey prediction (larger prior_error -> smaller C), from WLS: "
        f"{'SUPPORTED (negative, p=' + f'{wls_p_pe:.4f})' if prediction_holds else 'NOT supported (coefficient is non-negative)'} "
        f"-- check prior_correction_sensitivity.csv/.svg for stability across cutoffs "
        f"before drawing a conclusion."
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

    ols_coef_arr = ols_df["coef"].to_numpy(dtype=float)
    wls_coef_arr = wls_df["coef"].to_numpy(dtype=float)

    # --- Plot 1: raw prior_error vs. C, both fits overlaid ---
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
        ols_coef=ols_coef_arr,
        wls_coef=wls_coef_arr,
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
    # Unweighted / illustrative only -- read the WLS regression above (and the
    # sensitivity plot below) for the actual statistical conclusion.
    fig, ax = plt.subplots(figsize=(7, 6))
    resid_C_plot = np.clip(resid_C, -args.clip_c_plot, args.clip_c_plot)
    scatter_with_fit(
        ax,
        resid_pe,
        resid_C_plot,
        disagree_mask,
        xlabel="prior_error(s), residualized on log_coverage",
        ylabel="C(s), residualized on log_coverage",
        title="Same relationship, controlling for coverage (unweighted, illustrative)",
        highlight_xy=highlight_xy_partial,
    )
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1, alpha=0.7, label="no residual correction effect")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "prior_correction_partial.svg")
    fig.savefig(out_dir / "prior_correction_partial.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'prior_correction_partial'}.svg/.png")

    # --- Plot 3: threshold-sensitivity -- is the WLS coefficient actually stable? ---
    if len(sens_df) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.axhline(0.0, color="0.6", linestyle="-", linewidth=1)
        ax.plot(sens_df["min_prior_error"], sens_df["ols_coef"], color="0.3", linestyle="--", label="OLS coefficient")
        ax.fill_between(
            sens_df["min_prior_error"], sens_df["ols_ci95_low"], sens_df["ols_ci95_high"], color="0.3", alpha=0.15
        )
        ax.plot(sens_df["min_prior_error"], sens_df["wls_coef"], color="tab:blue", linewidth=2, label="WLS coefficient")
        ax.fill_between(
            sens_df["min_prior_error"],
            sens_df["wls_ci95_low"],
            sens_df["wls_ci95_high"],
            color="tab:blue",
            alpha=0.2,
        )
        ax.set_xlabel("--min-prior-error cutoff used")
        ax.set_ylabel("coefficient on prior_error (95% CI band)")
        ax.set_title("Is the prior_error coefficient stable across cutoffs, or cutoff-dependent?")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "prior_correction_sensitivity.svg")
        fig.savefig(out_dir / "prior_correction_sensitivity.png", dpi=150)
        plt.close(fig)
        print(f"Saved {out_dir / 'prior_correction_sensitivity'}.svg/.png")
    else:
        print("Skipped the sensitivity plot: no cutoff in the grid retained >= min_n states.")

    if n_clipped:
        print(f"\n(Note: {n_clipped} state(s) had |C| > {args.clip_c_plot} and were clipped for the plots only.)")


if __name__ == "__main__":
    main()