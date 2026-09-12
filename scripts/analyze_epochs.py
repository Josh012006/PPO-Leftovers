"""H4 ablation: how far does the epochs hyperparameter alone move things?

Trains a SINGLE long fixed-D PPO run (default 300 epochs, e.g. `configs/
ppo_fixed_d_epochs_analysis.yaml`) on the already-frozen `D` / `pi_beta`
from the baseline run, evaluating live success_rate on `--eval-episodes`
FIXED held-out episodes every `--checkpoint-every` epochs (default 5),
plus once before any training at all (epoch 0 = pi_beta itself,
unmodified). This directly probes whether an epochs-count null result at
small scale (e.g. 10 vs 30) holds all the way out to 300, or whether it
was just an early-training artifact. Re-run with different `--ppo-config`
files (varying e.g. clip_eps) to probe other hyperparameters against the
same epoch-count axis -- see README, "Results analysis" for the
clip_eps=0.1/0.2/0.3/0.4 sweep this was used for. For sweeping several
hyperparameters (including cross/grid sweeps) in one invocation, see
scripts/analyze_h7.py instead -- this script always trains exactly one
config per run.

Every checkpoint is evaluated THREE ways -- overall/covered/held-out --
and combined into weighted_success_rate (see README, "Our new starting
point"); this weighted number, not the raw overall one, is what decides
which epoch counts as "best" for --save-best-checkpoint. No hidden
overfitting risk in these curves either way: D is fixed and these eval
episodes are never part of D or of any fixed-D training batch. The eval
seed used here (`--eval-seed`, default 24680) is also deliberately
distinct from every other eval seed already in use elsewhere in this
project (tracking=12345, confirmation=54321, script 05's final
report=999) -- this is its own independent held-out set, not a reuse of
any seed a checkpoint was ever selected against. The pi_D* ceiling
reference lines plotted alongside success_rate are evaluated under this
SAME seed/episode-count, for the same reason -- comparing them against a
number computed under a different eval sample (e.g. script 05's report,
seed=999) would reintroduce exactly the kind of cross-sample noise this
project has been careful to keep separate elsewhere.

Outputs, all under `--out-dir` (default results/analysis/), each as both
.svg and .png:
  <prefix>.csv                    -- epoch, mean_return_overall(+stderr),
                                      success_rate_overall/covered/held_out
                                      (+stderr each), weighted_success_rate,
                                      clip_frac, entropy
  <prefix>_success_return.svg/png -- the three raw success_rate curves
                                      (overall/covered/held-out) vs epoch,
                                      with reference lines for the prior
                                      and the pi_D* ceiling in each's own
                                      color
  <prefix>_weighted.svg/png       -- weighted_success_rate vs epoch alone
                                      -- the actual number used to decide
                                      "best", with prior/ceiling/best
                                      reference lines
  <prefix>_clip_entropy.svg/png   -- clip_frac & entropy vs epoch

This is a long run by design (up to `epochs` epochs of training, plus
epochs/checkpoint_every + 1 live evaluations of `eval_episodes` episodes
each, now THREE evaluations per checkpoint instead of one) -- expect it
to take a while.

Usage:
    python scripts/analyze_epochs.py \
        --env-config configs/phase2/env_maze.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --pi-d-star-empirical results/phase2/pi_d_star_empirical.pkl \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --ppo-config configs/phase2/ppo_fixed_d_epochs_analysis.yaml \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 24680 \
        --out-dir results/phase2/analysis/epochs --prefix epochs_analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from _analysis_lib import compute_ceiling_success_rates, run_single_analysis
from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import DEFAULT_EVAL_WEIGHTS, get_tier_start_lists
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams, StartTierConfig
from ppo_exploitation.utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument("--dataset", default="results/phase2/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/phase2/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star-empirical", default="results/phase2/pi_d_star_empirical.pkl")
    parser.add_argument(
        "--start-tiers-config",
        required=True,
        help="Required -- every checkpoint is scored by weighted_success_rate now (see README, "
        "'Our new starting point'), which needs the covered/held-out tier split to compute.",
    )
    parser.add_argument(
        "--eval-weights",
        type=float,
        nargs=3,
        default=list(DEFAULT_EVAL_WEIGHTS),
        metavar=("OVERALL", "COVERED", "HELD_OUT"),
        help=f"Weights for the three eval modes, must sum to 1.0 (default {DEFAULT_EVAL_WEIGHTS}).",
    )
    parser.add_argument("--ppo-config", default="configs/phase2/ppo_fixed_d_epochs_analysis.yaml")
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=24680,
        help="Deliberately distinct from every other eval seed already used in this project "
        "(tracking=12345, confirmation=54321, script 05's report=999).",
    )
    parser.add_argument("--out-dir", default="results/phase2/analysis/epochs")
    parser.add_argument(
        "--prefix",
        default="epochs_analysis",
        help="Filename prefix, e.g. 'epochs_analysis_clip_0_3' when sweeping clip_eps across runs "
        "into the same --out-dir.",
    )
    args = parser.parse_args()

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    cfg = PPOHyperparams.from_yaml(args.ppo_config)
    set_global_seed(cfg.seed)
    weights = tuple(args.eval_weights)

    def make_env():
        return StochasticMazeEnv(
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

    eval_env = make_env()
    tier_cfg = StartTierConfig.from_yaml(args.start_tiers_config)
    covered_starts, held_out_starts = get_tier_start_lists(eval_env, tier_cfg)
    print(
        f"Loaded start tiers from {args.start_tiers_config}: {len(covered_starts)} covered, "
        f"{len(held_out_starts)} held-out. Weights (overall/covered/held_out): {weights}."
    )

    dataset = load_dataset(args.dataset)
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes.")

    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_state_dict = ckpt["state_dict"]
    print(f"theta and pi_old both start from the prior checkpoint (final eval: {ckpt['final_eval']})")

    ceiling_success_rates = compute_ceiling_success_rates(
        eval_env, args.pi_d_star_empirical, args.eval_episodes, args.eval_seed,
        covered_starts, held_out_starts, weights=weights,
    )
    print(
        f"pi_D* (empirical) ceiling under this run's own eval protocol "
        f"(seed={args.eval_seed}, n={args.eval_episodes}): "
        f"overall={ceiling_success_rates['overall']:.3f}, covered={ceiling_success_rates['covered']:.3f}, "
        f"held_out={ceiling_success_rates['held_out']:.3f}, weighted={ceiling_success_rates['weighted']:.3f} "
        f"(deliberately re-evaluated here rather than reused from script 05's report, which used a "
        f"different eval seed)"
    )

    print(
        f"\nTraining {cfg.epochs} epochs total, evaluating every {args.checkpoint_every} on "
        f"{args.eval_episodes} fixed held-out episodes (x3 populations)...\n"
    )
    summary = run_single_analysis(
        eval_env=eval_env,
        dataset=dataset,
        prior_state_dict=prior_state_dict,
        ceiling_success_rates=ceiling_success_rates,
        cfg=cfg,
        checkpoint_every=args.checkpoint_every,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        out_dir=Path(args.out_dir),
        prefix=args.prefix,
        covered_starts=covered_starts,
        held_out_starts=held_out_starts,
        weights=weights,
        title_suffix=f"n={args.eval_episodes}/check, clip_eps={cfg.clip_eps}",
        verbose=True,
    )

    print(f"\nSaved {summary['csv_path']}")
    print(f"Saved {args.out_dir}/{args.prefix}_success_return.svg/.png (best observed overall: {summary['best_overall']:.3f})")
    print(f"Saved {args.out_dir}/{args.prefix}_weighted.svg/.png (best observed weighted: {summary['best']:.3f})")
    print(f"Saved {args.out_dir}/{args.prefix}_clip_entropy.svg/.png")


if __name__ == "__main__":
    main()