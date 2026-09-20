"""Re-evaluates an already-trained fixed-D PPO run's saved checkpoints
(from `--save-all-checkpoints`, e.g. scripts/analyze_policy_agreement.py
or scripts/_analysis_lib.py's run_single_analysis) under a DIFFERENT
eval_seed/eval_episodes, WITHOUT retraining.

Motivating case (see README, "A note on eval seeds"): fixed-D training
here is fully deterministic given D, pi_beta, and the training seed --
CPU only, no other source of randomness in the trainer -- so the saved
checkpoints already ARE exactly what a fresh retrain would produce.
scripts/analyze_policy_agreement.py's own retrain used eval_seed=24680
(this project's SWEEP/exploration seed, meant for checkpoint SELECTION)
for its downstream disagreement analysis too, which should have used 999
(this project's FINAL-REPORT seed, for measuring an already-fixed
artifact). Rather than repeat a real, possibly long retrain purely to
get the same weights again under a different seed, this reloads the
SAME saved checkpoints and re-evaluates them -- only the evaluation
episodes differ, not the trained weights.

Reproduces the same summary CSV / success_return / weighted plots
run_single_analysis would have produced, under the corrected seed. Does
NOT reproduce the clip_frac/entropy plot -- those are training-time
statistics, not recoverable from saved weights alone (they were never
saved per-checkpoint; only the evaluation-relevant metadata was).

Also selects whichever checkpoint has the highest weighted_success_rate
under THIS eval_seed and saves it to `--out-dir/best_checkpoint_<eval_
seed>.pt`, in the exact format scripts/analyze_policy_agreement.py's
`--reuse-checkpoint` expects (state_dict, epoch, weighted_success_rate,
obs_dim, n_actions, hidden_sizes) -- feed that path there directly to
run the disagreement analysis under the corrected seed too, with no
retraining at any step of the whole pipeline.

Outputs, under --out-dir:
  reevaluated.csv                     -- epoch, success_rate_overall/
                                          covered/held_out(+stderr),
                                          weighted_success_rate
  reevaluated_success_return.svg/png
  reevaluated_weighted.svg/png
  best_checkpoint_<eval_seed>.pt

Usage:
    python scripts/reevaluate_saved_checkpoints.py \
        --env-config configs/phase2/env_maze.yaml \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --pi-d-star-empirical results/phase2/pi_d_star_empirical.pkl \
        --checkpoints-dir results/phase2/analysis/best_config/all_checkpoints \
        --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase2/analysis/best_config_reeval_999

Then, for the disagreement analysis under the same corrected seed:
    python scripts/analyze_policy_agreement.py \
        ... (same env/dataset/pi-d-star/tiers args) \
        --reuse-checkpoint results/phase2/analysis/best_config_reeval_999/best_checkpoint_999.pt \
        --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase2/analysis/best_config_reeval_999
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch

from _analysis_lib import compute_ceiling_success_rates, plot_success_return, plot_weighted_success
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import (
    DEFAULT_EVAL_WEIGHTS,
    evaluate_policy_weighted,
    get_tier_start_lists,
    make_neural_act_fn,
)
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, StartTierConfig

CKPT_EPOCH_RE = re.compile(r"checkpoint_epoch_(\d+)\.pt$")


def epoch_of(path: Path) -> int:
    m = CKPT_EPOCH_RE.search(path.name)
    if m is None:
        raise ValueError(f"Filename doesn't match checkpoint_epoch_<N>.pt: {path.name}")
    return int(m.group(1))


def load_checkpoint_net(path: Path) -> tuple[ActorCritic, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["n_actions"], ckpt["hidden_sizes"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument("--start-tiers-config", default="configs/phase2/start_tiers.yaml")
    parser.add_argument("--pi-d-star-empirical", default="results/phase2/pi_d_star_empirical.pkl")
    parser.add_argument(
        "--checkpoints-dir",
        required=True,
        help="Directory of checkpoint_epoch_<N>.pt files, e.g. results/phase2/analysis/"
        "best_config/all_checkpoints (from a prior --save-all-checkpoints run).",
    )
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument(
        "--eval-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_EVAL_WEIGHTS),
        metavar=("OVERALL", "COVERED", "HELD_OUT"),
        help=f"Weights for the three eval modes, must sum to 1.0 (default {DEFAULT_EVAL_WEIGHTS}).",
    )
    parser.add_argument("--out-dir", required=True)
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
    print(
        f"Loaded start tiers: {len(covered_starts)} covered, {len(held_out_starts)} held-out. "
        f"Weights (overall/covered/held_out): {weights}."
    )

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_paths = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"), key=epoch_of)
    if not ckpt_paths:
        raise SystemExit(f"No checkpoint_epoch_*.pt files found in {ckpt_dir}")
    epochs_found = [epoch_of(p) for p in ckpt_paths]
    print(f"Found {len(ckpt_paths)} checkpoints in {ckpt_dir} (epoch {epochs_found[0]} to {epochs_found[-1]}).")

    ceiling_success_rates = compute_ceiling_success_rates(
        env, args.pi_d_star_empirical, args.eval_episodes, args.eval_seed, covered_starts, held_out_starts, weights=weights
    )
    print(
        f"pi_D* (empirical) ceiling under eval_seed={args.eval_seed}, n={args.eval_episodes}: "
        f"overall={ceiling_success_rates['overall']:.3f}, covered={ceiling_success_rates['covered']:.3f}, "
        f"held_out={ceiling_success_rates['held_out']:.3f}, weighted={ceiling_success_rates['weighted']:.3f}"
    )

    rows = []
    best = {"weighted_success_rate": -1.0, "epoch": None, "ckpt": None, "net_state_dict": None}
    for p, epoch_num in zip(ckpt_paths, epochs_found):
        net, raw_ckpt = load_checkpoint_net(p)
        w = evaluate_policy_weighted(
            env, make_neural_act_fn(net, deterministic=True), args.eval_episodes, args.eval_seed,
            covered_starts, held_out_starts, weights=weights,
        )
        rows.append(
            {
                "epoch": epoch_num,
                "success_rate_overall": w["overall"]["success_rate"],
                "success_rate_overall_stderr": w["overall"]["success_rate_stderr"],
                "success_rate_covered": w["covered"]["success_rate"],
                "success_rate_covered_stderr": w["covered"]["success_rate_stderr"],
                "success_rate_held_out": w["held_out"]["success_rate"],
                "success_rate_held_out_stderr": w["held_out"]["success_rate_stderr"],
                "weighted_success_rate": w["weighted_success_rate"],
            }
        )
        print(
            f"[epoch {epoch_num:4d}] weighted_sr={w['weighted_success_rate']:.3f} "
            f"(overall={w['overall']['success_rate']:.3f}, covered={w['covered']['success_rate']:.3f}, "
            f"held_out={w['held_out']['success_rate']:.3f})"
        )
        if w["weighted_success_rate"] > best["weighted_success_rate"]:
            best.update(
                weighted_success_rate=w["weighted_success_rate"], epoch=epoch_num,
                ckpt=raw_ckpt, net_state_dict=net.state_dict(),
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    df.to_csv(out_dir / "reevaluated.csv", index=False)

    row0 = df.iloc[0]
    prior_rates = {
        "overall": row0["success_rate_overall"], "covered": row0["success_rate_covered"], "held_out": row0["success_rate_held_out"],
    }
    achieved_best_overall = plot_success_return(
        df, out_dir / "reevaluated_success_return",
        f"success_rate (overall/covered/held-out) vs. epoch (re-eval, seed={args.eval_seed}, n={args.eval_episodes})",
        prior_success_rates=prior_rates, ceiling_success_rates=ceiling_success_rates,
    )
    achieved_best_weighted = plot_weighted_success(
        df, out_dir / "reevaluated_weighted",
        f"weighted_success_rate vs. epoch (re-eval, seed={args.eval_seed}, n={args.eval_episodes})",
        prior_weighted=row0["weighted_success_rate"], ceiling_weighted=ceiling_success_rates["weighted"], weights=weights,
    )

    best_ckpt_path = out_dir / f"best_checkpoint_{args.eval_seed}.pt"
    torch.save(
        {
            "state_dict": best["net_state_dict"],
            "epoch": best["epoch"],
            "weighted_success_rate": best["weighted_success_rate"],
            "obs_dim": best["ckpt"]["obs_dim"],
            "n_actions": best["ckpt"]["n_actions"],
            "hidden_sizes": best["ckpt"]["hidden_sizes"],
        },
        best_ckpt_path,
    )

    print(f"\nSaved {out_dir}/reevaluated.csv and _success_return/_weighted .svg/.png")
    print(f"Best under eval_seed={args.eval_seed}: epoch {best['epoch']}, weighted_success_rate={best['weighted_success_rate']:.4f}")
    print(f"Saved that checkpoint to {best_ckpt_path} -- pass this to scripts/analyze_policy_agreement.py's --reuse-checkpoint.")
    print(f"(achieved_best_overall this run: {achieved_best_overall:.4f})")


if __name__ == "__main__":
    main()