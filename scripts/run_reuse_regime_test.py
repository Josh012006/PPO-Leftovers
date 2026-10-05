"""Pre-registered test of the weighting mechanism in a HEAVY-REUSE regime.

Question: in from-scratch, multi-window PPO training where each window's
data is reused heavily (epochs_per_window=300, the length of the frozen
window in which the mechanism was developed), does `kl_anchored` beat
`none`?

Why this test exists. The confirmatory ablation (scripts/run_confirmatory_
ablation.py, README) found no effect distinguishable from zero with 100
fresh episodes and 30 epochs per window. The mechanism was developed, and
showed its gain, in the opposite regime: a frozen dataset reused for 300
epochs while the policy drifts away from the one that collected it. This
script tests that regime, in the from-scratch loop, and nothing else.

WHAT IS FIXED IN ADVANCE (do not edit after seeing results; editing the
report below after the run defeats the point of pre-registration):

  Two arms, default PPO hyperparameters only (the tuned profile was shown
  not to transfer to from-scratch training):
    none        -- no weighting (FixedDPPOTrainer).
    kl_anchored -- LocalCountFixedDPPOTrainer, near-exact bandwidths,
                   beta=0.9995, KL-anchored with k=0.10.
  Only the amount of data reuse differs from the confirmatory ablation:
  epochs_per_window=300 (vs 30). Everything else is unchanged (lr=0.001,
  100 episodes per window, evaluation seed 424242), so the data reuse is
  the only thing that moved.

  Budget: 60 windows, evaluated every 2 windows (31 points, window 0
  included). 100 seeds per arm, paired by seed (same seed = same
  initialization and same first collection in both arms).

  Primary metric: late_mean = mean success_rate over the last 25% of the
  evaluation points (the plateau), NOT the full-trajectory mean, which
  blends in the random-initialization warm-up. Secondary (descriptive,
  no decision attached): best, the best evaluation of the run, and
  best_reeval, that same best checkpoint re-evaluated at the end of the
  run on reeval_episodes fresh episodes with a DIFFERENT seed. `best` alone
  is selected and reported on the same episodes, hence optimistic;
  best_reeval removes that selection bias.

  Single confirmatory comparison: late_mean, kl_anchored - none, paired
  by seed. Bootstrap 95% CI of the mean paired difference (10000
  resamples), Wilcoxon signed-rank test, win rate.

  Decision rule (equivalence margin 0.05 = the smallest effect judged
  worth a method):
    POSITIVE        CI lower bound > 0 AND Wilcoxon p < 0.05. The report
                    also states whether the point estimate reaches 0.05;
                    if the whole CI lies below 0.05 the effect is real
                    but smaller than the threshold of practical interest.
    NEGATIVE        CI upper bound < 0 (the method hurts).
    NULL            the whole CI lies inside [-0.05, +0.05] (and the case
                    is not POSITIVE): no effect larger than 5 points.
    INCONCLUSIVE    anything else. Report as is. Do NOT add seeds after
                    seeing the result: optional stopping inflates false
                    positives.
  Precedence: POSITIVE, then NEGATIVE, then NULL, then INCONCLUSIVE.

FEASIBILITY PILOT (blind to the method). Before the main run, run 5 seeds
of `none` ALONE (--arms none --seeds 0 1 2 3 4, same --out-dir). With one
arm the report prints only the baseline's late_mean and the feasibility
check (median late_mean >= 0.30, i.e. the baseline learns in this
budget); it never prints any contrast. If the check fails, raise
num_windows to 100 ONCE in the YAML (delete the pilot files first: the
frozen-configuration guard below refuses to mix runs made with different
settings), re-run the pilot, then freeze. Because the pilot uses the same
out-dir and the same seeds as the main run, if it passes with the
unchanged configuration its 5 runs are reused by the main run, not redone.

FROZEN-CONFIGURATION GUARD. The settings that define a run are written to
<out_dir>/frozen_config.json on the first launch. Any later launch whose
settings differ refuses to start rather than silently mixing runs made
under different settings (resume skips a run if its CSV exists, which on
its own would not notice a changed num_windows).

DETERMINISM. evaluate_policy reseeds its RNG on every call; the policy is
deterministic (argmax). The periodic evaluation seed (424242) and the
re-evaluation seed (777777) do not collide with any collection seed
(seed*100000+window, window <= 60). The data-reuse change does not alter
the per-seed initialization: seed s starts from the same theta_0 as in
the confirmatory ablation.

Per-window approx_kl and clip_frac (last epoch of the window, i.e. the
drift accumulated over the window) are saved in each run's CSV at the
evaluation points. The best-so-far checkpoint of each run is saved as
<prefix>_best.pt (a few tens of KB).

Reads the settings from a YAML config (configs/phase3/reuse_regime_test.
yaml) except --arms, --seeds, --out-dir, --cpu-count and --force, which
stay as command-line flags. Parallel execution, resumability and logging
mirror this project's other sweep scripts.

Usage (cmd.exe, one line):
  pilot:  python scripts/run_reuse_regime_test.py --config configs/phase3/reuse_regime_test.yaml --env-config configs/phase2/env_maze.yaml --arms none --seeds 0 1 2 3 4 --out-dir results/phase3/analysis/reuse_regime_test --cpu-count 5
  main:   python scripts/run_reuse_regime_test.py --config configs/phase3/reuse_regime_test.yaml --env-config configs/phase2/env_maze.yaml --out-dir results/phase3/analysis/reuse_regime_test --cpu-count 5
"""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.ppo.fixed_d_trainer_local_count import LocalCountFixedDPPOTrainer
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed

_WORKER: dict = {}
_ARMS = ("none", "kl_anchored")
_EQUIV_MARGIN = 0.05
_FEASIBILITY_MIN_MEDIAN = 0.30


def _init_worker(env_cfg_dict, layout):
    collect_env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    eval_env = StochasticMazeEnv(layout=layout, **env_cfg_dict)
    _WORKER.update(collect_env=collect_env, eval_env=eval_env)


def _late_stats(df: pd.DataFrame, frac: float) -> tuple[float, float]:
    n_tail = max(1, int(round(len(df) * frac)))
    tail = df["success_rate"].iloc[-n_tail:]
    return float(tail.mean()), float(tail.std() if len(tail) > 1 else 0.0)


def _build_cfg(arm: str, task: dict, seed: int) -> PPOHyperparams:
    hp = task["hp"]
    common = dict(
        epochs=task["epochs_per_window"], minibatch_size=task["minibatch_size"], clip_eps=hp["clip_eps"],
        gae_lambda=hp["gae_lambda"], entropy_coef=hp["entropy_coef"], value_coef=hp["value_coef"],
        max_grad_norm=hp["max_grad_norm"], hidden_sizes=tuple(task["hidden_sizes"]), lr=task["lr"], seed=seed,
    )
    if arm == "none":
        return PPOHyperparams(**common, use_effective_sample_weighting=False)
    if arm == "kl_anchored":
        return PPOHyperparams(
            **common, use_effective_sample_weighting=True, effective_sample_beta=task["beta"],
            effective_sample_kl_anchor=True, effective_sample_kl_k=task["kl_k"],
        )
    raise ValueError(f"arm must be one of {_ARMS}, got {arm!r}")


def _run_one_combo(task: dict) -> dict:
    prefix = task["prefix"]
    out_dir = Path(task["out_dir"])
    log_path = out_dir / f"{prefix}.log"
    seed, arm = task["seed"], task["arm"]
    try:
        set_global_seed(seed)
        collect_env, eval_env = _WORKER["collect_env"], _WORKER["eval_env"]
        obs_dim = collect_env.observation_space.shape[0]
        n_actions = collect_env.n_actions
        hs = tuple(task["hidden_sizes"])

        with open(log_path, "w", buffering=1) as logf, contextlib.redirect_stdout(logf):
            net = ActorCritic(obs_dim, n_actions, hidden_sizes=hs)
            state_dict = net.state_dict()
            print(f"arm={arm} seed={seed} -- theta_0 randomly initialized")

            results = []
            best = {"value": -1.0, "window": 0}
            ckpt_path = out_dir / f"{prefix}_best.pt"

            def do_eval(window, approx_kl, clip_frac):
                eval_net = ActorCritic(obs_dim, n_actions, hidden_sizes=hs)
                eval_net.load_state_dict(state_dict)
                r = evaluate_policy(
                    eval_env, lambda o, s: eval_net.act_numpy(o, deterministic=True)[0],
                    n_episodes=task["eval_episodes"], seed=task["eval_seed"],
                )
                results.append({"window": window, "approx_kl": approx_kl, "clip_frac": clip_frac, **r})
                print(f"  window {window:4d}: success_rate={r['success_rate']:.4f}  approx_kl={approx_kl:.4f}  clip_frac={clip_frac:.3f}")
                if r["success_rate"] > best["value"]:
                    best["value"], best["window"] = r["success_rate"], window
                    torch.save({k: v.clone() for k, v in state_dict.items()}, ckpt_path)

            do_eval(0, float("nan"), float("nan"))

            for window in range(1, task["num_windows"] + 1):
                collect_net = ActorCritic(obs_dim, n_actions, hidden_sizes=hs)
                collect_net.load_state_dict(state_dict)
                dataset = collect_fixed_dataset(
                    collect_env, collect_net, n_episodes=task["episodes_per_window"],
                    seed=seed * 100_000 + window, sample_actions=True,
                )
                cfg = _build_cfg(arm, task, seed)
                if arm == "none":
                    trainer = FixedDPPOTrainer(dataset, obs_dim, n_actions, cfg, prior_state_dict=state_dict)
                else:
                    trainer = LocalCountFixedDPPOTrainer(
                        dataset, obs_dim, n_actions, cfg, prior_state_dict=state_dict,
                        bandwidth_state=task["bandwidth_state"], bandwidth_action=task["bandwidth_action"],
                        kernel=task["kernel"],
                    )
                history = trainer.train(verbose=False)
                state_dict = trainer.net.state_dict()
                if window % task["eval_every_windows"] == 0:
                    do_eval(window, history[-1]["approx_kl"], history[-1]["clip_frac"])

            # Re-evaluate the best checkpoint on fresh episodes, different seed.
            re_net = ActorCritic(obs_dim, n_actions, hidden_sizes=hs)
            re_net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            re = evaluate_policy(
                eval_env, lambda o, s: re_net.act_numpy(o, deterministic=True)[0],
                n_episodes=task["reeval_episodes"], seed=task["reeval_seed"],
            )
            print(f"  best checkpoint (window {best['window']}): selected={best['value']:.4f}  re-evaluated={re['success_rate']:.4f}")

        df = pd.DataFrame(results)
        # Stored in the run's own CSV (constant column) so that a resumed launch
        # can recover it without depending on a global summary that may be missing.
        df["best_reeval"] = float(re["success_rate"])
        df["best_selected_window"] = int(best["window"])
        csv_path = out_dir / f"{prefix}.csv"
        df.to_csv(csv_path, index=False)
        late_mean, late_std = _late_stats(df, task["late_window_fraction"])
        return {
            "seed": seed, "arm": arm, "late_mean": late_mean, "late_std": late_std,
            "best": float(best["value"]), "best_window": int(best["window"]),
            "best_reeval": float(re["success_rate"]), "final": float(df["success_rate"].iloc[-1]),
            "csv_path": str(csv_path), "status": "ran", "log_path": str(log_path),
        }
    except Exception as e:
        with open(log_path, "a") as logf:
            logf.write(f"\n[ERROR] {e}\n{traceback.format_exc()}\n")
        return {"seed": seed, "arm": arm, "status": f"FAILED: {e}", "csv_path": None, "log_path": str(log_path)}


def _iqm(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    lo, hi = np.percentile(x, 25), np.percentile(x, 75)
    mid = x[(x >= lo) & (x <= hi)]
    return float(mid.mean()) if len(mid) else float(x.mean())


def _print_pilot_report(ok: pd.DataFrame) -> None:
    v = ok[ok.arm == "none"].sort_values("seed")
    print("\n--- Feasibility pilot (baseline only, no contrast is computed or shown) ---")
    for _, r in v.iterrows():
        print(f"  seed {int(r.seed):>3}: late_mean={r.late_mean:.4f}  best={r.best:.4f}")
    med = float(v.late_mean.median())
    ok_flag = med >= _FEASIBILITY_MIN_MEDIAN
    print(f"  median late_mean = {med:.4f}  (threshold {_FEASIBILITY_MIN_MEDIAN:.2f})  ->  {'PASS: freeze the configuration and launch the main run' if ok_flag else 'FAIL: the baseline does not learn in this budget'}")
    if not ok_flag:
        print("  If this is the first pilot: set num_windows to 100 in the YAML (ONCE), delete the pilot files in the out-dir, re-run the pilot, then freeze.")


def _print_main_report(ok: pd.DataFrame, n_expected_seeds: int) -> None:
    a = ok[ok.arm == "none"].set_index("seed")
    b = ok[ok.arm == "kl_anchored"].set_index("seed")
    seeds = sorted(set(a.index) & set(b.index))
    print(f"\n--- Pre-registered analysis, n = {len(seeds)} paired seeds (expected {n_expected_seeds}) ---")
    for name, df in [("none", a), ("kl_anchored", b)]:
        v = df.loc[seeds]
        print(f"  {name:>12}: late_mean mean={v.late_mean.mean():.4f}  IQM={_iqm(v.late_mean):.4f}  std={v.late_mean.std():.4f}  "
              f"seeds with late_mean<0.05: {int((v.late_mean < 0.05).sum())}")

    x, y = a.loc[seeds].late_mean.to_numpy(), b.loc[seeds].late_mean.to_numpy()
    diff = y - x
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        p = float(stats.wilcoxon(y, x).pvalue)
    except ValueError:
        p = float("nan")
    mean_d = float(diff.mean())
    print(f"\n  PRIMARY  late_mean, kl_anchored - none: mean diff={mean_d:+.4f}  95% CI=[{lo:+.4f}, {hi:+.4f}]  "
          f"wins={int((diff > 0).sum())}/{len(diff)}  Wilcoxon p={p:.4f}")

    if lo > 0 and p < 0.05:
        verdict = "POSITIVE"
        extra = (f"point estimate {'reaches' if mean_d >= _EQUIV_MARGIN else 'is below'} the {_EQUIV_MARGIN:.2f} threshold of practical interest"
                 + ("; the whole CI lies below it: a real but smaller-than-threshold effect" if hi < _EQUIV_MARGIN else ""))
    elif hi < 0:
        verdict, extra = "NEGATIVE", "the method hurts in this regime"
    elif lo >= -_EQUIV_MARGIN and hi <= _EQUIV_MARGIN:
        verdict, extra = "NULL", f"no effect larger than {_EQUIV_MARGIN:.2f}, in either direction"
    else:
        verdict, extra = "INCONCLUSIVE", "report as is; do NOT add seeds after seeing this result"
    print(f"  VERDICT (pre-registered rule): {verdict} -- {extra}")

    print("\n  SECONDARY (descriptive, no decision attached):")
    for col, label in [("best", "best (selected on the same episodes, optimistic)"), ("best_reeval", "best_reeval (fresh episodes, different seed)")]:
        xs, ys = a.loc[seeds][col].to_numpy(), b.loc[seeds][col].to_numpy()
        d = ys - xs
        bt = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(10000)])
        l, h = np.percentile(bt, [2.5, 97.5])
        print(f"    {label}: none mean={xs.mean():.4f}  kl_anchored mean={ys.mean():.4f}  diff={d.mean():+.4f}  95% CI=[{l:+.4f}, {h:+.4f}]")


def _frozen_config(y: dict) -> dict:
    keys = ["lr", "num_windows", "episodes_per_window", "epochs_per_window", "eval_every_windows", "minibatch_size",
            "hidden_sizes", "bandwidth_state", "bandwidth_action", "kernel", "beta", "kl_k", "late_window_fraction",
            "eval_episodes", "eval_seed", "reeval_episodes", "reeval_seed", "hp"]
    return {k: y.get(k) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--arms", nargs="+", choices=_ARMS, default=list(_ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100)))
    parser.add_argument("--out-dir", default="results/phase3/analysis/reuse_regime_test")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cpu-count", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        y = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_path = out_dir / "frozen_config.json"
    frozen = json.loads(json.dumps(_frozen_config(y)))
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text())
        if previous != frozen:
            diffs = {k: (previous.get(k), frozen.get(k)) for k in frozen if previous.get(k) != frozen.get(k)}
            raise SystemExit(
                f"REFUSING TO START: {frozen_path} was written with different settings ({diffs}). "
                f"Mixing runs made under different settings would silently corrupt the comparison. "
                f"Delete the out-dir's run files and frozen_config.json to start over, or restore the original YAML."
            )
    else:
        frozen_path.write_text(json.dumps(frozen, indent=2))

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    env_cfg_dict = dict(
        width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
        extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
        step_penalty=env_cfg.step_penalty, goal_reward=env_cfg.goal_reward, hazard_reward=env_cfg.hazard_reward,
        max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed, num_start_states=env_cfg.num_start_states,
        gamma=env_cfg.gamma,
    )
    ref_env = StochasticMazeEnv(**env_cfg_dict)
    print(f"HEAVY-REUSE REGIME TEST -- arms={args.arms}, {len(args.seeds)} seed(s), lr={y['lr']} (fixed).")
    print(f"Per run: {y['num_windows']} windows x {y['episodes_per_window']} episodes x {y['epochs_per_window']} epochs; "
          f"evaluation every {y['eval_every_windows']} windows on {y['eval_episodes']} episodes.")

    combos = [{"seed": s, "arm": a} for a in args.arms for s in args.seeds]
    results, pending = [], []
    for combo in combos:
        prefix = f"seed_{combo['seed']}_{combo['arm']}"
        csv_path = out_dir / f"{prefix}.csv"
        if csv_path.exists() and not args.force:
            df_e = pd.read_csv(csv_path)
            late_mean, late_std = _late_stats(df_e, y["late_window_fraction"])
            best_reeval = float(df_e["best_reeval"].iloc[0]) if "best_reeval" in df_e.columns else float("nan")
            results.append({**combo, "late_mean": late_mean, "late_std": late_std, "best": float(df_e["success_rate"].max()),
                            "best_window": int(df_e.loc[df_e["success_rate"].idxmax(), "window"]), "best_reeval": best_reeval,
                            "final": float(df_e["success_rate"].iloc[-1]), "csv_path": str(csv_path), "status": "skipped (already existed)"})
        else:
            pending.append({**combo, "prefix": prefix})
    if results:
        print(f"{len(results)} run(s) already present, skipped (use --force to redo).")

    t_start = time.time()
    common = dict(
        out_dir=str(out_dir), lr=y["lr"], num_windows=y["num_windows"], episodes_per_window=y["episodes_per_window"],
        epochs_per_window=y["epochs_per_window"], eval_every_windows=y["eval_every_windows"], minibatch_size=y["minibatch_size"],
        hidden_sizes=y["hidden_sizes"], bandwidth_state=y["bandwidth_state"], bandwidth_action=y["bandwidth_action"],
        kernel=y["kernel"], beta=y["beta"], kl_k=y["kl_k"], eval_episodes=y["eval_episodes"], eval_seed=y["eval_seed"],
        reeval_episodes=y["reeval_episodes"], reeval_seed=y["reeval_seed"], late_window_fraction=y["late_window_fraction"],
        hp=y["hp"],
    )

    def report_line(res, tag, elapsed):
        if str(res.get("status", "")).startswith("FAILED"):
            return f"{tag} FAILED -- {res['status']} (see {res.get('log_path')})  [{elapsed:.1f} min]"
        return f"{tag} done -- late_mean={res['late_mean']:.4f} best={res['best']:.4f}  [{elapsed:.1f} min]"

    if args.cpu_count <= 1 or len(pending) <= 1:
        _init_worker(env_cfg_dict, ref_env.layout)
        for i, combo in enumerate(pending, start=1):
            t0 = time.time()
            res = _run_one_combo({**combo, **common})
            print(report_line(res, f"[{i}/{len(pending)} {combo['prefix']}]", (time.time() - t0) / 60))
            results.append(res)
    else:
        cpu_count = max(1, min(args.cpu_count, len(pending), os.cpu_count() or args.cpu_count))
        print(f"Running {len(pending)} pending run(s) across {cpu_count} worker process(es). Per-window logs go to <out_dir>/<prefix>.log.\n")
        n_done = 0
        with ProcessPoolExecutor(
            max_workers=cpu_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker, initargs=(env_cfg_dict, ref_env.layout),
        ) as executor:
            futures = {executor.submit(_run_one_combo, {**c, **common}): c for c in pending}
            waiting = set(futures)
            while waiting:
                done, waiting = wait(waiting, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    print(f"... still running ({n_done}/{len(pending)} done, {(time.time() - t_start) / 60:.1f} min elapsed)", flush=True)
                    continue
                for fut in done:
                    c = futures[fut]
                    n_done += 1
                    try:
                        res = fut.result()
                    except Exception as e:  # pragma: no cover
                        res = {"seed": c["seed"], "arm": c["arm"], "status": f"FAILED: {e}", "csv_path": None, "log_path": f"{out_dir}/{c['prefix']}.log"}
                    results.append(res)
                    print(report_line(res, f"[{n_done}/{len(pending)}] {c['prefix']}:", (time.time() - t_start) / 60))

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(out_dir / "sweep_summary.csv", index=False)
    n_failed = int(summary_df["status"].astype(str).str.startswith("FAILED").sum()) if len(summary_df) else 0
    print(f"\n=== Done in {(time.time() - t_start) / 60:.1f} min ({n_failed} failed) ===")

    ok = summary_df[summary_df["status"].isin(["ran", "skipped (already existed)"])] if not summary_df.empty else summary_df
    arms_present = set(ok.arm) if not ok.empty else set()
    if arms_present == {"none"} and len(ok) == len(combos):
        _print_pilot_report(ok)
    elif arms_present == set(_ARMS) and len(ok) == 2 * len(args.seeds) and len(args.seeds) == 100:
        _print_main_report(ok, n_expected_seeds=100)
    else:
        print(f"\n{len(ok)}/{len(combos)} requested run(s) available; the pre-registered contrast is printed only when "
              f"both arms are complete for all 100 seeds. Re-run the same command (it skips what is done) until they are.")
    print(f"\nSaved combined summary to {out_dir / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
