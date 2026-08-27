"""Multi-hyperparameter (grid/cross) sweep on top of the same single-window
analysis scripts/analyze_epochs.py and scripts/_analysis_lib.py already use.

Takes ONE YAML config where every PPOHyperparams field is either a single
value (held fixed across the whole sweep) or a YAML list (swept). If more
than one field is a list, this runs the full cartesian product across all
of them -- a genuine cross/grid sweep, not just one axis at a time. This is
how scripts/analyze_epochs.py's separate clip_eps / entropy_coef / gae_lambda
/ value_coef sweeps could have been written as a single invocation each, and
is built for exactly this project's next step: a max_grad_norm x value_coef
cross sweep, to check whether a tighter max_grad_norm surfaces the
gradient-clipping interference a value_coef-only sweep couldn't trigger at
the default max_grad_norm=0.5 (see README, "Value coefficient sweep").

`hidden_sizes` is special-cased, since a single hidden_sizes value is
ALREADY a YAML list (e.g. `[64, 64]`) and would otherwise be
indistinguishable from a genuine sweep list. Convention:
  hidden_sizes: [64, 64]              -> one fixed architecture
  hidden_sizes: [[64, 64], [128, 128]] -> a 2-value sweep over architectures
(a list of lists = sweep; a flat list of numbers = one fixed value).

The setup shared by every combination (loading D, the prior checkpoint,
and computing the pi_D* ceiling under this run's own eval protocol) is
done ONCE, not once per combination -- both for speed and because these
values must be identical across every point in the grid for the comparison
to mean anything.

Resumable: if `<prefix>.csv` for a given combination already exists in
--out-dir, that combination is skipped (its existing CSV is read back in
for the summary table) unless --force is passed. Useful for a grid large
enough that you might want to stop and resume it.

Outputs, all under --out-dir (default results/analysis/h7/):
  one <base-prefix>_<swept-field>_<value>[..._<swept-field>_<value>].csv
  and matching _success_return/_clip_entropy .svg/.png PER COMBINATION
  (exactly as scripts/analyze_epochs.py produces for a single run), plus:
  h7_sweep_summary.csv -- one row per combination: swept field values,
                           best/mean/std/final success_rate, csv path

Usage:
    python scripts/analyze_h7.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --pi-d-star-empirical results/pi_d_star_empirical.pkl \
        --sweep-config configs/ppo_fixed_d_h7_sweep.yaml \
        --base-prefix h7_clip_0_3_ent_0_01_gae_0_90 \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 24680 \
        --out-dir results/analysis/h7
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
import yaml

from _analysis_lib import compute_ceiling_success_rate, run_single_analysis
from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed

SHORT_NAMES = {
    "epochs": "epochs",
    "minibatch_size": "mb",
    "clip_eps": "clip",
    "gamma": "gamma",
    "gae_lambda": "gae",
    "entropy_coef": "ent",
    "value_coef": "val",
    "max_grad_norm": "maxgrad",
    "normalize_advantages": "normadv",
    "lr": "lr",
    "hidden_sizes": "hid",
    "seed": "seed",
}


def _format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value}".replace(".", "_").replace("-", "neg")
    if isinstance(value, (list, tuple)):
        return "x".join(str(v) for v in value)
    return str(value)


def parse_sweep_yaml(raw: dict) -> tuple[dict, dict]:
    """Split a raw YAML dict into (fixed, swept). `swept` maps field name
    to the list of values to sweep; `fixed` maps field name to its single
    held value. See module docstring for the hidden_sizes special case."""
    fixed: dict = {}
    swept: dict = {}
    for key, value in raw.items():
        if key == "hidden_sizes":
            if len(value) > 0 and isinstance(value[0], (list, tuple)):
                swept[key] = [tuple(v) for v in value]
            else:
                fixed[key] = tuple(value)
        elif isinstance(value, list):
            swept[key] = value
        else:
            fixed[key] = value
    return fixed, swept


def generate_combinations(fixed: dict, swept: dict) -> list[dict]:
    if not swept:
        return [dict(fixed)]
    keys = list(swept.keys())
    value_lists = [swept[k] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        combo = dict(fixed)
        combo.update(dict(zip(keys, values)))
        combos.append(combo)
    return combos


def make_combo_prefix(base_prefix: str, swept_keys: list[str], combo: dict) -> str:
    parts = [base_prefix]
    for key in swept_keys:
        short = SHORT_NAMES.get(key, key)
        parts.append(f"{short}_{_format_value(combo[key])}")
    return "_".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star-empirical", default="results/pi_d_star_empirical.pkl")
    parser.add_argument("--sweep-config", required=True)
    parser.add_argument(
        "--base-prefix",
        required=True,
        help="Prefix stem encoding the FIXED context (e.g. 'h7_clip_0_3_ent_0_01_gae_0_90'). "
        "Each swept field's short name and value is appended automatically per combination.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=24680)
    parser.add_argument("--out-dir", default="results/analysis/h7")
    parser.add_argument("--force", action="store_true", help="Re-run combinations even if their CSV already exists.")
    args = parser.parse_args()

    with open(args.sweep_config, "r") as f:
        raw = yaml.safe_load(f)
    fixed, swept = parse_sweep_yaml(raw)
    combos = generate_combinations(fixed, swept)
    swept_keys = list(swept.keys())

    print(f"=== H7 grid sweep: {len(combos)} combination(s) ===")
    if swept_keys:
        print(f"Swept fields: {swept_keys}")
        for key in swept_keys:
            print(f"  {key}: {swept[key]}")
    else:
        print("No list-valued fields found -- running the single fixed config once.")
    print(f"Fixed fields: { {k: v for k, v in fixed.items()} }\n")

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)

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
        )

    eval_env = make_env()
    dataset = load_dataset(args.dataset)
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes.")

    ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_state_dict = ckpt["state_dict"]
    print(f"theta and pi_old both start from the prior checkpoint (final eval: {ckpt['final_eval']})")

    ceiling_success_rate = compute_ceiling_success_rate(
        eval_env, args.pi_d_star_empirical, args.eval_episodes, args.eval_seed
    )
    print(
        f"pi_D* (empirical) ceiling under this sweep's eval protocol "
        f"(seed={args.eval_seed}, n={args.eval_episodes}): success_rate={ceiling_success_rate:.3f}\n"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    combo_times = []
    t_start = time.time()

    for i, combo in enumerate(combos, start=1):
        prefix = make_combo_prefix(args.base_prefix, swept_keys, combo)
        combo_desc = ", ".join(f"{k}={combo[k]}" for k in swept_keys) if swept_keys else "(single config)"
        log_tag = f"[{i}/{len(combos)} {prefix}] "

        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            print(f"{log_tag}SKIPPING -- {csv_path} already exists (use --force to re-run)")
            df_existing = pd.read_csv(csv_path)
            results.append(
                {
                    **{k: combo[k] for k in swept_keys},
                    "prefix": prefix,
                    "best": float(df_existing["success_rate"].max()),
                    "mean": float(df_existing["success_rate"].mean()),
                    "std": float(df_existing["success_rate"].std()),
                    "final": float(df_existing.iloc[-1]["success_rate"]),
                    "csv_path": str(csv_path),
                    "status": "skipped (already existed)",
                }
            )
            continue

        cfg = PPOHyperparams(**combo)
        set_global_seed(cfg.seed)

        elapsed = time.time() - t_start
        avg = sum(combo_times) / len(combo_times) if combo_times else None
        eta_str = f", ~{avg * (len(combos) - i + 1) / 60:.1f} min remaining (est.)" if avg else ""
        print(f"\n{log_tag}Starting -- {combo_desc}  [{elapsed / 60:.1f} min elapsed so far{eta_str}]")

        t0 = time.time()
        summary = run_single_analysis(
            eval_env=eval_env,
            dataset=dataset,
            prior_state_dict=prior_state_dict,
            ceiling_success_rate=ceiling_success_rate,
            cfg=cfg,
            checkpoint_every=args.checkpoint_every,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            out_dir=out_dir,
            prefix=prefix,
            title_suffix=combo_desc,
            verbose=True,
            log_prefix=log_tag,
        )
        dt = time.time() - t0
        combo_times.append(dt)

        print(f"{log_tag}Done in {dt / 60:.1f} min -- best={summary['best']:.3f} mean={summary['mean']:.3f} std={summary['std']:.3f}")
        results.append({**{k: combo[k] for k in swept_keys}, **summary, "status": "ran"})

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "h7_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    total_min = (time.time() - t_start) / 60
    print(f"\n=== Sweep complete in {total_min:.1f} min total ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()