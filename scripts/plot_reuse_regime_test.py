"""Recompute the pre-registered statistics of the heavy-reuse regime test from the raw per-run CSVs
and draw the summary figure (mean curves, paired scatter, paired differences).

Usage: python scripts/plot_reuse_regime_test.py [--dir results/phase3/analysis/reuse_regime_test]

Reads seed_<k>_{none,kl_anchored}.csv only (never sweep_summary.csv), so the numbers printed here are
an independent check of the ones produced by run_reuse_regime_test.py. Same rule: late_mean = mean of the
last 25% of the evaluation points; bootstrap 95% CI of the mean paired difference; equivalence margin 0.05.
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

MARGIN, FRAC = 0.05, 0.25
COL = {"none": "#4c72b0", "kl_anchored": "#dd8452"}


def load(d):
    runs, curves = [], {"none": {}, "kl_anchored": {}}
    for arm in COL:
        for f in glob.glob(os.path.join(d, f"seed_*_{arm}.csv")):
            seed = int(os.path.basename(f).split("_")[1])
            df = pd.read_csv(f)
            n = max(1, int(round(len(df) * FRAC)))
            runs.append(dict(arm=arm, seed=seed, late_mean=df.success_rate.iloc[-n:].mean(), best=df.success_rate.max(),
                             best_reeval=df.best_reeval.iloc[0], final=df.success_rate.iloc[-1], full_mean=df.success_rate.mean()))
            curves[arm][seed] = df.success_rate.to_numpy()
            windows = df.window.to_numpy()
    return pd.DataFrame(runs), curves, windows


def boot_ci(d, rng, n=10000):
    bt = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return np.percentile(bt, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase3/analysis/reuse_regime_test")
    d = ap.parse_args().dir
    R, curves, w = load(d)
    a = R[R.arm == "none"].set_index("seed").sort_index()
    b = R[R.arm == "kl_anchored"].set_index("seed").sort_index()
    assert (a.index == b.index).all(), "arms are not paired"
    rng = np.random.default_rng(0)
    print(f"n = {len(a)} paired seeds")
    for col in ["late_mean", "best", "best_reeval", "final", "full_mean"]:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        diff = y - x
        lo, hi = boot_ci(diff, rng)
        print(f"{col:12s} none={x.mean():.4f} kl_anchored={y.mean():.4f} diff={diff.mean():+.4f} "
              f"CI=[{lo:+.4f},{hi:+.4f}] wins={int((diff > 0).sum())}/{len(diff)} Wilcoxon p={stats.wilcoxon(y, x).pvalue:.4f}")
    diff = (b.late_mean - a.late_mean).to_numpy()
    lo, hi = boot_ci(diff, np.random.default_rng(1))
    p = stats.wilcoxon(b.late_mean, a.late_mean).pvalue
    if lo > 0 and p < 0.05: v = "POSITIVE"
    elif hi < 0: v = "NEGATIVE"
    elif lo >= -MARGIN and hi <= MARGIN: v = "NULL"
    else: v = "INCONCLUSIVE"
    print(f"VERDICT on late_mean: {v}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for arm, cs in curves.items():
        c = np.array([cs[s] for s in sorted(cs)])
        m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
        ax[0].plot(w, m, color=COL[arm], label=arm)
        ax[0].fill_between(w, m - 1.96 * se, m + 1.96 * se, color=COL[arm], alpha=.2)
    n_tail = max(1, int(round(len(w) * FRAC)))
    ax[0].axvspan(w[-n_tail], w[-1], color="grey", alpha=.12, label="late window (metric)")
    ax[0].set(xlabel="window", ylabel="success rate", title=f"Mean ± 95% CI over {len(a)} seeds"); ax[0].legend()
    ax[1].scatter(a.late_mean, b.late_mean, s=14, alpha=.7); ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set(xlabel="none late_mean", ylabel="kl_anchored late_mean", title="Paired by seed", xlim=(0, .9), ylim=(0, .9))
    ax[2].hist(diff, bins=22, color="#888", alpha=.8); ax[2].axvline(0, color="k")
    ax[2].axvline(diff.mean(), color="r", label=f"mean {diff.mean():+.4f}")
    ax[2].axvspan(-MARGIN, MARGIN, color="green", alpha=.1, label=f"equivalence margin ±{MARGIN}")
    ax[2].axvspan(lo, hi, ymax=.06, color="r", label=f"95% CI [{lo:+.3f},{hi:+.3f}]")
    ax[2].set(xlabel="kl_anchored − none (late_mean)", title="Paired differences"); ax[2].legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(d, "reuse_regime_means_and_paired.png")
    plt.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
