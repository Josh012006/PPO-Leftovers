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

Parallel execution (--cpu-count, default 1): with `--cpu-count N > 1`, up
to N combinations train concurrently in separate worker PROCESSES (this
is CPU-bound PyTorch work -- threads would not help, the GIL serializes
them anyway). The "spawn" start method is used explicitly on every OS
(`multiprocessing.get_context("spawn")`), not whatever the platform
default happens to be -- Windows and macOS already default to spawn, but
Linux defaults to "fork", which can occasionally deadlock when combined
with PyTorch's internal thread pools (OpenMP/MKL) inherited mid-state
through the fork. Forcing spawn everywhere means identical, tested
behavior regardless of OS, at the cost of spawn's own requirements (every
worker function is module-level, not a closure, and every argument
crossing the process boundary is picklable) -- both already satisfied
here. Each worker loads the environment, `D`, and the prior
checkpoint exactly ONCE when it starts (via ProcessPoolExecutor's
`initializer`, not once per combination it processes), and reuses the
SAME environment layout built once in the main process (passing the
already-built `layout` object, not regenerating it -- regenerating a
phase-2 layout runs a rejection-sampling safety check that can take real
time by itself; paying that N times over for N workers would be wasted
work for what is otherwise an identical layout, same layout_seed).
Logging stays consistent by construction, not by coincidence: each
worker's PER-EPOCH output (everything `run_single_analysis`'s
`verbose=True` prints) is redirected to its own `<out_dir>/<prefix>.log`
file, never to the shared console -- only the main process ever prints
to stdout, one line per combination as it completes, so concurrent
workers can never interleave mid-line. A failure in one combination is
caught, logged to that combination's own `.log` file, and reported in
the final summary table's `status` column as `FAILED: ...` -- it does
not stop or corrupt the other combinations in flight. Each worker's log
file is line-buffered (`open(..., buffering=1)`), so lines appear on disk
as each epoch checkpoint happens, not only once that combination finishes
-- open a `.log` file at any point during a run to see live progress
(e.g. `Get-Content -Wait <path>` in PowerShell to watch it update). The
main process also prints a heartbeat line every 30 seconds while nothing
has completed yet, so a long-running sweep never goes silent even before
the first combination finishes -- Task Manager (or `htop`/`ps` on
Unix) showing `cpu-count` busy python processes is another immediate,
no-log-file way to confirm work is happening. `--cpu-count` is
clamped to both the number of pending (non-skipped) combinations and
`os.cpu_count()`, whichever is smaller, with a note printed if it had to
be. Memory note: each worker holds its own full copy of `D` in memory --
size `--cpu-count` accordingly for a large `D` on a memory-constrained
machine.

Outputs, all under --out-dir (default results/analysis/h7/):
  one <base-prefix>_<swept-field>_<value>[..._<swept-field>_<value>].csv
  and matching _success_return/_clip_entropy .svg/.png PER COMBINATION
  (exactly as scripts/analyze_epochs.py produces for a single run), plus
  a per-combination .log file (see "Parallel execution" above -- written
  regardless of --cpu-count, so sequential and parallel runs produce the
  same artifacts), plus:
  h7_sweep_summary.csv -- one row per combination: swept field values,
                           best/mean/std/final weighted_success_rate, csv path

Every checkpoint in every combination is evaluated THREE ways --
overall/covered/held-out -- and combined into weighted_success_rate (see
README, "Our new starting point"); the summary table's best/mean/std/final
columns are now this weighted number, not the raw overall one -- so
picking "the best combination" from this sweep already accounts for
overfitting risk, not just raw performance.

Usage:
    python scripts/analyze_h7.py \
        --env-config configs/phase2/env_maze.yaml \
        --dataset results/phase2/dataset_D.pkl \
        --prior-checkpoint results/phase2/prior_checkpoint.pt \
        --pi-d-star-empirical results/phase2/pi_d_star_empirical.pkl \
        --start-tiers-config configs/phase2/start_tiers.yaml \
        --sweep-config configs/phase2/ppo_fixed_d_h7_sweep.yaml \
        --base-prefix h7_clip_0_3_ent_0_01_gae_0_90 \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 24680 \
        --out-dir results/phase2/analysis/h7 \
        --cpu-count 4
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
import yaml

from _analysis_lib import compute_ceiling_success_rates, run_single_analysis
from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import DEFAULT_EVAL_WEIGHTS, get_tier_start_lists
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams, StartTierConfig
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

# --------------------------------------------------------------------------
# Parallel worker machinery. Must be module-level (not nested inside main())
# so it can be pickled by reference -- Windows' multiprocessing uses
# "spawn", which re-imports this module in each worker process and looks
# up these functions by name, not by closure. _WORKER holds everything a
# worker loads ONCE at startup (via ProcessPoolExecutor's initializer),
# reused across every combination that worker goes on to process.
# --------------------------------------------------------------------------
_WORKER: dict = {}


def _init_worker(env_cfg_dict, layout, dataset_path, prior_state_dict, ceiling_success_rates, covered_starts, held_out_starts):
    env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    dataset = load_dataset(dataset_path)
    _WORKER.update(
        env=env, dataset=dataset, prior_state_dict=prior_state_dict,
        ceiling_success_rates=ceiling_success_rates, covered_starts=covered_starts, held_out_starts=held_out_starts,
    )


def _run_one_combo(task: dict) -> dict:
    prefix = task["prefix"]
    out_dir = Path(task["out_dir"])
    log_path = out_dir / f"{prefix}.log"
    base = {k: task["combo"][k] for k in task["swept_keys"]}
    try:
        cfg = PPOHyperparams(**task["combo"])
        set_global_seed(cfg.seed)
        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            summary = run_single_analysis(
                eval_env=_WORKER["env"],
                dataset=_WORKER["dataset"],
                prior_state_dict=_WORKER["prior_state_dict"],
                ceiling_success_rates=_WORKER["ceiling_success_rates"],
                cfg=cfg,
                checkpoint_every=task["checkpoint_every"],
                eval_episodes=task["eval_episodes"],
                eval_seed=task["eval_seed"],
                out_dir=out_dir,
                prefix=prefix,
                covered_starts=_WORKER["covered_starts"],
                held_out_starts=_WORKER["held_out_starts"],
                weights=task["weights"],
                title_suffix=task["combo_desc"],
                verbose=True,
                log_prefix=task["log_tag"],
            )
        return {**base, **summary, "status": "ran", "log_path": str(log_path)}
    except Exception as e:
        with open(log_path, "a") as logf:
            logf.write(f"\n[ERROR] {e}\n{traceback.format_exc()}\n")
        return {**base, "prefix": prefix, "status": f"FAILED: {e}", "csv_path": None, "log_path": str(log_path)}


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
    parser.add_argument("--out-dir", default="results/phase2/analysis/h7")
    parser.add_argument("--force", action="store_true", help="Re-run combinations even if their CSV already exists.")
    parser.add_argument(
        "--cpu-count",
        type=int,
        default=1,
        help="Number of combinations to train concurrently, in separate worker processes (default 1, "
        "sequential, unchanged behavior). See module docstring, 'Parallel execution', for how logging "
        "and layout reuse are handled.",
    )
    args = parser.parse_args()
    weights = tuple(args.eval_weights)

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
        f"pi_D* (empirical) ceiling under this sweep's eval protocol "
        f"(seed={args.eval_seed}, n={args.eval_episodes}): overall={ceiling_success_rates['overall']:.3f}, "
        f"covered={ceiling_success_rates['covered']:.3f}, held_out={ceiling_success_rates['held_out']:.3f}, "
        f"weighted={ceiling_success_rates['weighted']:.3f}\n"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    pending = []  # (i, combo, prefix, combo_desc) not yet run
    for i, combo in enumerate(combos, start=1):
        prefix = make_combo_prefix(args.base_prefix, swept_keys, combo)
        combo_desc = ", ".join(f"{k}={combo[k]}" for k in swept_keys) if swept_keys else "(single config)"
        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            print(f"[{i}/{len(combos)} {prefix}] SKIPPING -- {csv_path} already exists (use --force to re-run)")
            df_existing = pd.read_csv(csv_path)
            results.append(
                {
                    **{k: combo[k] for k in swept_keys},
                    "prefix": prefix,
                    "best": float(df_existing["weighted_success_rate"].max()),
                    "mean": float(df_existing["weighted_success_rate"].mean()),
                    "std": float(df_existing["weighted_success_rate"].std()),
                    "final": float(df_existing.iloc[-1]["weighted_success_rate"]),
                    "csv_path": str(csv_path),
                    "status": "skipped (already existed)",
                }
            )
        else:
            pending.append((i, combo, prefix, combo_desc))

    t_start = time.time()

    if args.cpu_count <= 1 or len(pending) <= 1:
        # --- Sequential path: unchanged from before cpu-count existed. ---
        combo_times = []
        for i, combo, prefix, combo_desc in pending:
            log_tag = f"[{i}/{len(combos)} {prefix}] "
            cfg = PPOHyperparams(**combo)
            set_global_seed(cfg.seed)

            elapsed = time.time() - t_start
            avg = sum(combo_times) / len(combo_times) if combo_times else None
            eta_str = f", ~{avg * (len(pending) - len(combo_times)) / 60:.1f} min remaining (est.)" if avg else ""
            print(f"\n{log_tag}Starting -- {combo_desc}  [{elapsed / 60:.1f} min elapsed so far{eta_str}]")

            t0 = time.time()
            summary = run_single_analysis(
                eval_env=eval_env,
                dataset=dataset,
                prior_state_dict=prior_state_dict,
                ceiling_success_rates=ceiling_success_rates,
                cfg=cfg,
                checkpoint_every=args.checkpoint_every,
                eval_episodes=args.eval_episodes,
                eval_seed=args.eval_seed,
                out_dir=out_dir,
                prefix=prefix,
                covered_starts=covered_starts,
                held_out_starts=held_out_starts,
                weights=weights,
                title_suffix=combo_desc,
                verbose=True,
                log_prefix=log_tag,
            )
            dt = time.time() - t0
            combo_times.append(dt)

            print(f"{log_tag}Done in {dt / 60:.1f} min -- best_weighted={summary['best']:.3f} mean={summary['mean']:.3f} std={summary['std']:.3f}")
            results.append({**{k: combo[k] for k in swept_keys}, **summary, "status": "ran"})

    else:
        # --- Parallel path: see module docstring, "Parallel execution". ---
        cpu_count = max(1, args.cpu_count)
        available = os.cpu_count() or cpu_count
        clamped = min(cpu_count, len(pending), available)
        if clamped != cpu_count:
            print(
                f"Note: --cpu-count {cpu_count} clamped to {clamped} "
                f"(min of requested, {len(pending)} pending combination(s), {available} CPUs available)."
            )
        cpu_count = clamped

        env_cfg_dict = dict(
            width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
            extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
            step_penalty=env_cfg.step_penalty, goal_reward=env_cfg.goal_reward, hazard_reward=env_cfg.hazard_reward,
            max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed, num_start_states=env_cfg.num_start_states,
            gamma=env_cfg.gamma,
        )
        tasks = []
        for i, combo, prefix, combo_desc in pending:
            tasks.append(
                {
                    "combo": combo, "prefix": prefix, "combo_desc": combo_desc, "swept_keys": swept_keys,
                    "checkpoint_every": args.checkpoint_every, "eval_episodes": args.eval_episodes,
                    "eval_seed": args.eval_seed, "out_dir": str(out_dir), "weights": weights,
                    "log_tag": f"[{i}/{len(combos)} {prefix}] ",
                }
            )

        print(
            f"\nRunning {len(pending)} pending combination(s) across {cpu_count} worker process(es). "
            f"Per-epoch logs go to <out_dir>/<prefix>.log, not this console -- only one status line per "
            f"completed combination is printed here.\n"
        )
        n_done = 0
        with ProcessPoolExecutor(
            max_workers=cpu_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(env_cfg_dict, eval_env.layout, args.dataset, prior_state_dict, ceiling_success_rates, covered_starts, held_out_starts),
        ) as executor:
            futures = {executor.submit(_run_one_combo, t): t for t in tasks}
            pending_futures = set(futures.keys())
            heartbeat_seconds = 30
            while pending_futures:
                done, pending_futures = wait(pending_futures, timeout=heartbeat_seconds, return_when=FIRST_COMPLETED)
                if not done:
                    elapsed = (time.time() - t_start) / 60
                    print(
                        f"... still running ({n_done}/{len(pending)} done, {elapsed:.1f} min elapsed) -- "
                        f"per-epoch progress is in each combination's own .log file under {out_dir}",
                        flush=True,
                    )
                    continue
                for fut in done:
                    t = futures[fut]
                    n_done += 1
                    elapsed = (time.time() - t_start) / 60
                    try:
                        res = fut.result()
                    except Exception as e:  # pragma: no cover -- _run_one_combo already catches its own exceptions
                        res = {"prefix": t["prefix"], "status": f"FAILED: {e}", "csv_path": None, "log_path": f"{out_dir}/{t['prefix']}.log"}
                    results.append(res)
                    if str(res.get("status", "")).startswith("FAILED"):
                        print(f"[{n_done}/{len(pending)}] {t['prefix']}: FAILED -- {res['status']} (see {res.get('log_path')})  [{elapsed:.1f} min elapsed]")
                    else:
                        print(
                            f"[{n_done}/{len(pending)}] {t['prefix']}: done -- best_weighted={res['best']:.3f} "
                            f"mean={res['mean']:.3f} std={res['std']:.3f}  [{elapsed:.1f} min elapsed]  (log: {res.get('log_path')})"
                        )

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "h7_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    total_min = (time.time() - t_start) / 60
    n_failed = int(summary_df["status"].astype(str).str.startswith("FAILED").sum()) if len(summary_df) else 0
    print(f"\n=== Sweep complete in {total_min:.1f} min total ({n_failed} failed) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved combined summary to {summary_path}")


if __name__ == "__main__":
    main()