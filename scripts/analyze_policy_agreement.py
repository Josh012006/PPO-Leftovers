"""Where, exactly, does the best fixed-D PPO configuration's policy
underperform pi_D*'s -- not just pick a different action, but actually
get a worse expected return?

A state counts as a DISAGREEMENT state if and only if BOTH hold (an AND,
not either alone):
  1. pi_D*'s greedy action at that state differs from the best-config
     policy's greedy action.
  2. The best-config policy's actual expected return from that state is
     meaningfully BELOW pi_D*'s expected value there -- not just a
     different-but-equally-good action.

Condition 1 alone (argmax disagreement) doesn't imply a real problem: two
actions can lead to statistically indistinguishable outcomes, especially
near a fork with redundant paths. Condition 2 alone doesn't either: pi_D*
could be visiting a state where its OWN value is unreliable (e.g. thin
coverage). Requiring both is a stricter, more meaningful definition of
"the two policies actually disagree in a way that costs something."

IMPORTANT correctness point: pi_D*'s V(s) is DISCOUNTED (value iteration
uses reference.yaml's gamma=0.99). Comparing it against an UNDISCOUNTED
empirical return -- which is what evaluate_policy computes everywhere
else in this project -- would silently reproduce exactly the discounted-
vs-undiscounted mismatch this project has been careful to avoid since its
first "Baseline run" (see README). So the best-config policy's expected
return here is estimated as a DISCOUNTED return-to-go (same gamma), via
direct Monte Carlo rollout starting from each state -- not the project's
usual undiscounted mean_return metric. This is a deliberate, narrow
exception for this one apples-to-apples comparison, not a change to how
success/return is reported anywhere else.

How the rollout-from-an-arbitrary-state works: `env.set_state(s)` already
exists (used by the reference solver's own machinery) -- call
`env.reset()` (clears the step counter), then `env.set_state(s)` to
override the position, then `env.state_to_obs(s)` for the correct initial
observation, then step normally. `--episodes-per-state` independent
rollouts per state (freshly reseeded each time, so slip randomness
differs) give both a mean discounted return and its standard error.

COST WARNING: this evaluates every one of ~887 non-terminal states with
`--episodes-per-state` rollouts each (887 * episodes-per-state episodes
total) -- no network training involved, but still a lot of environment
steps. Lower --episodes-per-state for a faster, noisier pass; the
per-state standard error in the output CSV tells you how much that
costs. This is why this script is meant to be run locally, not repeatedly
in a sandbox.

Also computes (cheaply, no rollout needed) pi_beta's own critic accuracy
at each state -- reusing reference.experience_optimal.compute_true_value_of_policy,
same as scripts/analyze_critic_accuracy.py -- and saves it in the output
CSV for a later diagnostic (critic error vs. disagreement severity),
deliberately NOT plotted yet: see README, "Policy agreement" -- the
mechanism connecting disagreement to any specific state property is
being established first (scripts/analyze_disagreement_factors.py), before
revisiting that specific plot.

Outputs, under --out-dir (default results/analysis/policy_agreement/):
  best_config_checkpoint.pt (only when retraining, i.e. no --reuse-checkpoint)
  best_config_retrain.csv / _success_return.svg / _clip_entropy.svg
      (only when retraining)
  policy_agreement.csv -- per state: covered, pi_d_star_V, pi_d_star_action,
      best_config_action, argmax_disagree, rank_correlation,
      ppo_discounted_return(+stderr), value_gap, is_disagreement,
      severity, true_value_pi_beta, critic_pred_pi_beta, critic_abs_error
  policy_agreement_maze_map.svg/png -- maze cell map colored by severity
      (green = agree or no meaningful gap, red = disagree AND a real,
      statistically-significant value gap)

Usage (retrain, first run):
    python scripts/analyze_policy_agreement.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --pi-d-star results/pi_d_star_empirical.pkl \
        --reference-config configs/reference.yaml \
        --best-config configs/ppo_fixed_d_best_config.yaml \
        --episodes-per-state 20 --eval-seed 24680 \
        --out-dir results/analysis/policy_agreement

Usage (reuse an already-trained checkpoint, e.g. against true-restricted pi_D*):
    python scripts/analyze_policy_agreement.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --pi-d-star results/pi_d_star_true_restricted.pkl \
        --reference-config configs/reference.yaml \
        --reuse-checkpoint results/analysis/policy_agreement/best_config_checkpoint.pt \
        --episodes-per-state 20 --eval-seed 24680 \
        --out-dir results/analysis/policy_agreement_true_restricted
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from _analysis_lib import compute_ceiling_success_rate, run_single_analysis
from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import make_neural_act_fn
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.reference.experience_optimal import compute_true_value_of_policy
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams, ReferenceConfig
from ppo_exploitation.utils.seeding import set_global_seed


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, ties handled by average rank -- plain
    numpy, no scipy dependency for this one function."""

    def rankdata_avg(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(1, len(a) + 1)
        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j < len(a) - 1 and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
            i = j + 1
        return ranks

    rx = rankdata_avg(np.asarray(x, dtype=float))
    ry = rankdata_avg(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def discounted_rollout_from_state(env, act_fn, state: int, n_episodes: int, gamma: float, seed: int) -> tuple[float, float]:
    """Mean and stderr of the DISCOUNTED return, over n_episodes
    independent rollouts starting at `state` (via env.reset() +
    env.set_state(state)), under `act_fn`. See module docstring for why
    this must be discounted, not the project's usual undiscounted
    mean_return."""
    returns = []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        env.set_state(state)
        obs = env.state_to_obs(state)
        done = False
        g = 0.0
        discount = 1.0
        while not done:
            action = act_fn(obs, state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = info["state"]
            g += discount * reward
            discount *= gamma
            done = terminated or truncated
        returns.append(g)
    arr = np.asarray(returns, dtype=np.float64)
    mean = float(arr.mean())
    stderr = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, stderr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--pi-d-star", default="results/pi_d_star_empirical.pkl")
    parser.add_argument("--reference-config", default="configs/reference.yaml")
    parser.add_argument("--best-config", default="configs/ppo_fixed_d_best_config.yaml")
    parser.add_argument(
        "--reuse-checkpoint",
        default=None,
        help="Path to a *.pt saved by a previous run of this script. If given, retraining is "
        "skipped and this checkpoint is used directly -- e.g. to re-run against a different "
        "--pi-d-star without retraining the same network twice.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Only used when retraining.")
    parser.add_argument(
        "--episodes-per-state",
        type=int,
        default=20,
        help="Independent rollouts per state for the discounted-return estimate. Higher = more "
        "precise but slower; see the per-state stderr in the output CSV to judge if it's enough.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=500,
        help="Episode count for the retrain's OWN live-eval checkpointing (unrelated to "
        "--episodes-per-state, which is for the per-state rollout comparison after training).",
    )
    parser.add_argument("--eval-seed", type=int, default=24680)
    parser.add_argument("--out-dir", default="results/analysis/policy_agreement")
    parser.add_argument(
        "--z-threshold",
        type=float,
        default=2.0,
        help="A state is only flagged as a disagreement state if, in addition to the actions "
        "differing, value_gap exceeds z-threshold * stderr of the per-state discounted-return "
        "estimate -- guards against flagging noise from a small --episodes-per-state as a real gap.",
    )
    args = parser.parse_args()

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    ref_cfg = ReferenceConfig.from_yaml(args.reference_config)
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

    dataset = load_dataset(args.dataset)
    covered_states = {int(s) for tr in dataset.trajectories for s in tr.states}
    print(f"Loaded D: {len(dataset)} transitions, D covers {len(covered_states)}/{env.n_states} states.")

    prior_ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_state_dict = prior_ckpt["state_dict"]
    print(f"Loaded prior checkpoint (final eval: {prior_ckpt['final_eval']})")

    with open(args.pi_d_star, "rb") as f:
        ref = pickle.load(f)
    print(f"Loaded pi_D* ({ref.kind}) from {args.pi_d_star}, gamma={ref_cfg.gamma}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: get the best-config policy, either by retraining (default)
    # or by reusing an already-saved checkpoint (--reuse-checkpoint). ---
    if args.reuse_checkpoint:
        best_ckpt_path = Path(args.reuse_checkpoint)
        print(f"\n--reuse-checkpoint given: skipping retrain, loading {best_ckpt_path} directly.")
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    else:
        cfg = PPOHyperparams.from_yaml(args.best_config)
        set_global_seed(cfg.seed)
        best_ckpt_path = out_dir / "best_config_checkpoint.pt"

        ceiling_success_rate = compute_ceiling_success_rate(
            env, args.pi_d_star, args.eval_episodes, args.eval_seed
        )
        print(f"pi_D* ceiling under this run's eval protocol: success_rate={ceiling_success_rate:.3f}")
        print(
            f"\nRetraining the best configuration ({args.best_config}) for up to {cfg.epochs} epochs, "
            f"tracking the best-observed checkpoint (not just the final one)...\n"
        )
        retrain_summary = run_single_analysis(
            eval_env=env,
            dataset=dataset,
            prior_state_dict=prior_state_dict,
            ceiling_success_rate=ceiling_success_rate,
            cfg=cfg,
            checkpoint_every=args.checkpoint_every,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            out_dir=out_dir,
            prefix="best_config_retrain",
            title_suffix="best configuration retrain",
            verbose=True,
            save_best_checkpoint_path=best_ckpt_path,
        )
        print(
            f"\nRetrain done: best={retrain_summary['best']:.3f} mean={retrain_summary['mean']:.3f} "
            f"(best checkpoint at epoch {retrain_summary['best_checkpoint_epoch']})"
        )
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)

    print(
        f"Using the checkpoint from epoch {best_ckpt['epoch']} "
        f"(success_rate={best_ckpt['success_rate']:.3f}) as 'the best configuration's policy' "
        f"for the rest of this analysis.\n"
    )

    best_net = ActorCritic(best_ckpt["obs_dim"], best_ckpt["n_actions"], best_ckpt["hidden_sizes"])
    best_net.load_state_dict(best_ckpt["state_dict"])
    best_net.eval()
    act_fn = make_neural_act_fn(best_net, deterministic=True)

    prior_net = ActorCritic(prior_ckpt["obs_dim"], prior_ckpt["n_actions"], prior_ckpt["hidden_sizes"])
    prior_net.load_state_dict(prior_state_dict)
    prior_net.eval()

    # --- Step 2: cheap per-state quantities (no rollout): logits, argmax,
    # rank correlation, pi_beta's own critic accuracy. ---
    terminal_states = {s for s in range(env.n_states) if env.is_terminal_state(s)}
    non_terminal = [s for s in range(env.n_states) if s not in terminal_states]
    obs_batch = np.stack([env.state_to_obs(s) for s in non_terminal]).astype(np.float32)
    obs_t = torch.as_tensor(obs_batch)

    with torch.no_grad():
        best_logits, _ = best_net.forward(obs_t)
        best_logits_np = best_logits.numpy()
        prior_logits, prior_values = prior_net.forward(obs_t)
        prior_action_probs_np = torch.softmax(prior_logits, dim=-1).numpy()
        prior_critic_pred_np = prior_values.numpy()

    prior_action_probs_full = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    prior_critic_pred_full = np.zeros(env.n_states, dtype=np.float64)
    for i, s in enumerate(non_terminal):
        prior_action_probs_full[s] = prior_action_probs_np[i]
        prior_critic_pred_full[s] = prior_critic_pred_np[i]

    print("Computing exact V^pi_beta(s) (for the later critic-error diagnostic, not used here)...")
    true_value_pi_beta = compute_true_value_of_policy(env, prior_action_probs_full, gamma=env_cfg.gamma)

    # --- Step 3: the expensive part -- discounted rollout from every state. ---
    print(
        f"\nRolling out the best-config policy from all {len(non_terminal)} non-terminal states, "
        f"{args.episodes_per_state} episodes each ({len(non_terminal) * args.episodes_per_state} "
        f"episodes total)...\n"
    )
    rows = []
    for k, s in enumerate(non_terminal):
        q_star = ref.Q[s]
        pi_d_star_action = int(np.argmax(q_star))
        pi_d_star_V = float(ref.V[s])
        best_config_action = int(np.argmax(best_logits_np[k]))
        argmax_disagree = int(pi_d_star_action != best_config_action)
        rank_corr = spearman_corr(q_star, best_logits_np[k])

        mean_return, stderr = discounted_rollout_from_state(
            env, act_fn, s, args.episodes_per_state, gamma=ref_cfg.gamma, seed=args.eval_seed
        )
        value_gap = pi_d_star_V - mean_return
        is_disagreement = bool(argmax_disagree and value_gap > args.z_threshold * stderr)
        severity = max(0.0, value_gap) if is_disagreement else 0.0

        r, c = env.layout.rc(s)
        rows.append(
            {
                "state": s,
                "row": r,
                "col": c,
                "covered": s in covered_states,
                "pi_d_star_V": pi_d_star_V,
                "pi_d_star_action": pi_d_star_action,
                "best_config_action": best_config_action,
                "argmax_disagree": argmax_disagree,
                "rank_correlation": rank_corr,
                "ppo_discounted_return": mean_return,
                "ppo_discounted_return_stderr": stderr,
                "value_gap": value_gap,
                "is_disagreement": int(is_disagreement),
                "severity": severity,
                "true_value_pi_beta": float(true_value_pi_beta[s]),
                "critic_pred_pi_beta": float(prior_critic_pred_full[s]),
                "critic_abs_error": float(abs(prior_critic_pred_full[s] - true_value_pi_beta[s])),
            }
        )
        if (k + 1) % 100 == 0 or (k + 1) == len(non_terminal):
            print(f"  {k + 1}/{len(non_terminal)} states done...")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "policy_agreement.csv"
    df.to_csv(csv_path, index=False)

    n_disagree = int(df["is_disagreement"].sum())
    print(f"\n=== Summary ===")
    print(f"States: {len(df)} total, {df['covered'].sum()} covered by D")
    print(f"Argmax disagreement alone: {df['argmax_disagree'].sum()} states")
    print(
        f"Disagreement states (argmax differs AND value_gap > "
        f"{args.z_threshold}x stderr): {n_disagree} states"
    )
    print(f"  of which covered by D: {int((df['is_disagreement'] & df['covered']).sum())}")
    print(f"Saved {csv_path}")

    # --- Plot: maze map colored by severity. ---
    fig, ax = plt.subplots(figsize=(8, 8))
    vmax = max(df["severity"].quantile(0.95), 1e-6)
    sc = ax.scatter(
        df["col"], df["row"], c=df["severity"], cmap="RdYlGn_r", vmin=0, vmax=vmax, s=90, marker="s",
        edgecolors="0.3", linewidths=0.3,
    )
    hz_rows = [r for (r, c) in env.layout.hazards]
    hz_cols = [c for (r, c) in env.layout.hazards]
    ax.scatter(hz_cols, hz_rows, marker="x", s=120, color="black", linewidths=2, label="hazard")
    ax.scatter([env.layout.start[1]], [env.layout.start[0]], marker="*", s=200, color="blue", label="start", edgecolors="black")
    ax.scatter([env.layout.goal[1]], [env.layout.goal[0]], marker="*", s=200, color="gold", label="goal", edgecolors="black")
    ax.invert_yaxis()
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    ax.set_title(
        f"Disagreement severity by maze cell (\u03c0D* {ref.kind} vs. best-config PPO)\n"
        f"green=agree/no gap, red=disagree & \u03c0D* expects much more",
        fontsize=11,
    )
    fig.colorbar(sc, ax=ax, label="severity (value_gap where flagged as a disagreement, else 0)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.2, 1.0))
    fig.tight_layout()
    plot_path = out_dir / "policy_agreement_maze_map"
    fig.savefig(plot_path.with_suffix(".svg"))
    fig.savefig(plot_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}.svg/.png")


if __name__ == "__main__":
    main()
