"""Phase 2 design requirement 3 (README, "Phase 2"): overfitting to D must
be directly observable, not merely assumed absent. This is the positive
control: does the covered-vs-held-out success_rate gap infrastructure
(eval/evaluate.py's eval_start_states, wired through
scripts/05_evaluate_all.py's --start-tiers-config) actually detect
overfitting when it is deliberately induced?

This script does NOT retrain anything itself -- it reads the gap report
scripts/05_evaluate_all.py already produces (with --start-tiers-config
given) and checks/plots the specific comparison this whole requirement is
about: does a policy trained with configs/phase2/ppo_fixed_d_overfit_prone.yaml
(no entropy pressure, far more epochs than phase 1's baseline -- pushed
hard toward memorizing whatever D emphasizes) show a LARGE gap between
its covered-tier and held-out-tier success_rate, while pi_beta (trained
with a uniform start distribution, see README "Phase 2", requirement 3)
shows close to NO gap between the same two?

Full sequence to produce the report this script reads (all using the
existing, unmodified scripts 01/02/03/04/05 -- see README, "Phase 2" for
why no new training code was needed):
    python scripts/01_train_prior.py --env-config configs/phase2/env_maze.yaml \
        --prior-config configs/phase2/prior_training.yaml --out results/phase2/prior_checkpoint.pt
    python scripts/02_collect_dataset.py --env-config configs/phase2/env_maze.yaml \
        --checkpoint results/phase2/prior_checkpoint.pt \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --out results/phase2/dataset_D.pkl
    python scripts/03_compute_pi_d_star.py --env-config configs/phase2/env_maze.yaml \
        --reference-config configs/phase2/reference.yaml --dataset results/phase2/dataset_D.pkl \
        --out-empirical results/phase2/pi_d_star_empirical.pkl \
        --out-true-restricted results/phase2/pi_d_star_true_restricted.pkl
    python scripts/04_train_fixed_d_ppo.py --dataset results/phase2/dataset_D.pkl \
        --ppo-config configs/phase2/ppo_fixed_d_overfit_prone.yaml \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --out results/phase2/ppo_overfit_prone_on_D.pt
    python scripts/05_evaluate_all.py --env-config configs/phase2/env_maze.yaml \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --pi-d-star-empirical results/phase2/pi_d_star_empirical.pkl \
        --pi-d-star-true-restricted results/phase2/pi_d_star_true_restricted.pkl \
        --ppo-checkpoints overfit_prone=results/phase2/ppo_overfit_prone_on_D.pt \
        --out results/phase2/gap_report.csv

Then:
    python scripts/verify_overfitting_detectable.py \
        --gap-report results/phase2/gap_report.csv \
        --policy-name overfit_prone \
        --out-dir results/phase2/analysis/overfitting_detectable

Outputs, under --out-dir:
  overfitting_detectable.svg/png -- covered vs. held-out success_rate,
                                     the overfit-prone policy next to
                                     pi_beta for contrast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-report", default="results/phase2/gap_report.csv")
    parser.add_argument(
        "--policy-name",
        default="overfit_prone",
        help="The --ppo-checkpoints name used when running scripts/05_evaluate_all.py with the "
        "overfit-prone config (this script reads '{name}_covered' and '{name}_held_out' rows).",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.15,
        help="The covered-minus-held-out success_rate gap the overfit-prone policy must clear for "
        "this check to pass -- i.e. how large a gap counts as clearly detectable, not noise.",
    )
    parser.add_argument("--out-dir", default="results/phase2/analysis/overfitting_detectable")
    args = parser.parse_args()

    report = pd.read_csv(args.gap_report, index_col="policy")
    required = [f"{args.policy_name}_covered", f"{args.policy_name}_held_out", "prior_pi_beta_covered", "prior_pi_beta_held_out"]
    missing = [r for r in required if r not in report.index]
    if missing:
        raise SystemExit(
            f"Missing rows in {args.gap_report}: {missing}. Was scripts/05_evaluate_all.py run "
            f"with --start-tiers-config, and does --policy-name match the --ppo-checkpoints name "
            f"used there?"
        )

    overfit_covered = report.loc[f"{args.policy_name}_covered", "success_rate"]
    overfit_held_out = report.loc[f"{args.policy_name}_held_out", "success_rate"]
    beta_covered = report.loc["prior_pi_beta_covered", "success_rate"]
    beta_held_out = report.loc["prior_pi_beta_held_out", "success_rate"]

    overfit_gap = overfit_covered - overfit_held_out
    beta_gap = beta_covered - beta_held_out

    print(f"=== Overfitting detectability check ===")
    print(f"{args.policy_name}: covered={overfit_covered:.3f}, held_out={overfit_held_out:.3f}, gap={overfit_gap:+.3f}")
    print(f"prior_pi_beta:  covered={beta_covered:.3f}, held_out={beta_held_out:.3f}, gap={beta_gap:+.3f}")
    print()
    if overfit_gap >= args.min_gap and overfit_gap > beta_gap:
        print(
            f"PASS: {args.policy_name}'s gap ({overfit_gap:+.3f}) clears --min-gap ({args.min_gap}) "
            f"and exceeds pi_beta's own gap ({beta_gap:+.3f}) -- the covered/held-out split "
            f"detects deliberately-induced overfitting, and pi_beta (uniform starts during its "
            f"own training) serves as the near-zero-gap baseline it should be."
        )
    else:
        print(
            f"FAIL: either the gap is under --min-gap or pi_beta's own gap is not clearly smaller. "
            f"This means the current infrastructure/parameters do NOT reliably demonstrate "
            f"detectable overfitting -- reconsider before trusting this metric for real analysis "
            f"(e.g. a stronger overfit-prone config, more start tiers, or a larger held-out tier)."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0, 1]
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], [beta_covered, beta_held_out], width, label="\u03c0\u03b2 (prior)", color="tab:gray")
    ax.bar([xi + width / 2 for xi in x], [overfit_covered, overfit_held_out], width, label=args.policy_name, color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(["covered tier", "held-out tier"])
    ax.set_ylabel("success_rate")
    ax.set_ylim(0, 1)
    ax.set_title("Is a deliberately overfit-prone policy's covered/held-out gap detectable?")
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "overfitting_detectable"
    fig.savefig(plot_path.with_suffix(".svg"))
    fig.savefig(plot_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved {plot_path}.svg/.png")


if __name__ == "__main__":
    main()
