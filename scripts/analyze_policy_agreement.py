"""Where, exactly, does the best fixed-D PPO configuration's policy differ
from pi_D*'s? For every non-terminal state, compares pi_D*'s exact action
ranking (from its Q-values, no training noise) against the best fixed-D
PPO configuration's ranking (from its policy logits) via Spearman rank
correlation and simple argmax agreement, and cross-references both
against whether pi_beta's own policy walks into near-certain death from
that state.

By default (no --reuse-checkpoint), this RETRAINS the best configuration
found so far (clip_eps=0.3, entropy_coef=0.01, gae_lambda=0.90,
minibatch_size=64 -- configs/ppo_fixed_d_best_config.yaml), tracking the
BEST live-eval checkpoint seen during training rather than only the final
epoch (via scripts/_analysis_lib.py's save_best_checkpoint_path option --
given how non-monotonic these runs are throughout this project, epoch 300
is not reliably where the best policy actually was).

Pass --reuse-checkpoint <path to an already-saved *.pt from a previous run
of this script> to SKIP retraining entirely and re-run just the
comparison -- e.g. against pi_D*'s true-restricted definition instead of
the empirical one (via --pi-d-star), to check whether a disagreement
pattern found against the empirical definition survives once MLE sampling
noise in the reference itself is removed (same visited (s,a) support,
exact transition probabilities instead of D's estimated ones). This is
the cheap way to ask "is this about D's coverage, or about noise in
estimating probabilities over what D already covers" without retraining
the same network twice.

Spearman rank correlation is implemented directly in numpy (ties handled
by average rank, matching the standard convention) rather than adding a
scipy dependency for this one function. Ties are common and expected on
pi_D*'s side: every unvisited (s,a) pair at a given state collapses to
the same unseen_penalty Q-value.

Also recomputes the critic-accuracy diagnostic's true_value(s) under
pi_beta (reusing reference.experience_optimal.compute_true_value_of_policy
-- no need to have run scripts/analyze_critic_accuracy.py first).

Outputs, under --out-dir:
  best_config_checkpoint.pt                    -- (only when retraining)
                                                   the tracked best-observed
                                                   network
  best_config_retrain.csv / _success_return.svg / _clip_entropy.svg
                                                 -- (only when retraining)
                                                   the retrain's own
                                                   trajectory
  policy_agreement.csv                          -- per-state: covered,
                                                   true_value_pi_beta,
                                                   near_death, pi_d_star_
                                                   action, best_config_
                                                   action, argmax_agree,
                                                   rank_correlation
  policy_agreement_vs_critic_error.svg/png      -- scatter: critic abs
                                                   error (x) vs. rank
                                                   correlation (y)
  policy_agreement_maze_map.svg/png             -- spatial map of the
                                                   maze, each state colored
                                                   by rank correlation,
                                                   hazards/start/goal marked

Usage (retrain, first run):
    python scripts/analyze_policy_agreement.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --pi-d-star results/pi_d_star_empirical.pkl \
        --best-config configs/ppo_fixed_d_best_config.yaml \
        --checkpoint-every 5 --eval-episodes 500 --eval-seed 24680 \
        --out-dir results/analysis/policy_agreement

Usage (re-run against true-restricted pi_D*, reusing the checkpoint above):
    python scripts/analyze_policy_agreement.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --pi-d-star results/pi_d_star_true_restricted.pkl \
        --reuse-checkpoint results/analysis/policy_agreement/best_config_checkpoint.pt \
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
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.reference.experience_optimal import compute_true_value_of_policy
from ppo_exploitation.utils.config import MazeEnvConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, ties handled by average rank -- plain
    numpy, no scipy dependency for this one function. Returns NaN if
    either side is fully tied (e.g. a state with all 4 actions unvisited
    in D, giving pi_D* the same Q-value for every action -- undefined
    correlation, not zero)."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument(
        "--pi-d-star",
        default="results/pi_d_star_empirical.pkl",
        help="Path to either the empirical or true-restricted pi_D* pickle -- both are "
        "ReferenceSolution objects with the same interface.",
    )
    parser.add_argument("--best-config", default="configs/ppo_fixed_d_best_config.yaml")
    parser.add_argument(
        "--reuse-checkpoint",
        default=None,
        help="Path to a *.pt saved by a previous run of this script. If given, retraining is "
        "skipped entirely and this checkpoint is used directly for the comparison -- e.g. to "
        "re-run against a different --pi-d-star without retraining the same network twice.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=24680)
    parser.add_argument("--out-dir", default="results/analysis/policy_agreement")
    parser.add_argument(
        "--near-death-threshold",
        type=float,
        default=-0.9,
        help="States with true V^pi_beta below this are flagged as near-certain-death states "
        "(hazard_reward is -1.0 by default, so -0.9 catches states pi_beta walks into a hazard "
        "from almost every time).",
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

    dataset = load_dataset(args.dataset)
    covered_states = {int(s) for tr in dataset.trajectories for s in tr.states}
    print(f"Loaded D: {len(dataset)} transitions, D covers {len(covered_states)}/{env.n_states} states.")

    prior_ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_state_dict = prior_ckpt["state_dict"]
    print(f"Loaded prior checkpoint (final eval: {prior_ckpt['final_eval']})")

    with open(args.pi_d_star, "rb") as f:
        ref = pickle.load(f)
    print(f"Loaded pi_D* ({ref.kind}) from {args.pi_d_star}")

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

    prior_net = ActorCritic(prior_ckpt["obs_dim"], prior_ckpt["n_actions"], prior_ckpt["hidden_sizes"])
    prior_net.load_state_dict(prior_state_dict)
    prior_net.eval()

    # --- Step 2: per-state comparison. ---
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

    print(f"Computing exact V^pi_beta(s) via policy evaluation against true dynamics (gamma={env_cfg.gamma})...")
    true_value = compute_true_value_of_policy(env, prior_action_probs_full, gamma=env_cfg.gamma)

    rows = []
    for i, s in enumerate(non_terminal):
        r, c = env.layout.rc(s)
        q_star = ref.Q[s]
        logit_ppo = best_logits_np[i]
        rank_corr = spearman_corr(q_star, logit_ppo)
        pi_d_star_action = int(np.argmax(q_star))
        best_config_action = int(np.argmax(logit_ppo))
        rows.append(
            {
                "state": s,
                "row": r,
                "col": c,
                "covered": s in covered_states,
                "true_value_pi_beta": float(true_value[s]),
                "near_death": bool(true_value[s] < args.near_death_threshold),
                "critic_pred_pi_beta": float(prior_critic_pred_full[s]),
                "critic_abs_error": float(abs(prior_critic_pred_full[s] - true_value[s])),
                "pi_d_star_action": pi_d_star_action,
                "best_config_action": best_config_action,
                "argmax_agree": int(pi_d_star_action == best_config_action),
                "rank_correlation": rank_corr,
            }
        )
    df = pd.DataFrame(rows)

    def report(label: str, sub: pd.DataFrame):
        print(
            f"  {label:16s} n={len(sub):4d}  argmax_agree={sub['argmax_agree'].mean():.3f}  "
            f"mean_rank_corr={sub['rank_correlation'].mean():.3f}  "
            f"(nan_rank_corr={sub['rank_correlation'].isna().sum()})"
        )

    print(f"\n=== Policy agreement: pi_D* ({ref.kind}) vs. best fixed-D PPO configuration ===")
    report("all", df)
    report("covered", df[df["covered"]])
    report("near_death", df[df["near_death"]])
    report("not near_death", df[~df["near_death"]])
    report("covered & near_death", df[df["covered"] & df["near_death"]])
    report("covered & not near_death", df[df["covered"] & ~df["near_death"]])

    csv_path = out_dir / "policy_agreement.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # --- Plot 1: critic abs error vs. rank correlation ---
    fig, ax = plt.subplots(figsize=(7, 6))
    for flag, color, label in [(False, "tab:blue", "not near-death"), (True, "tab:red", "near-death")]:
        sub = df[df["near_death"] == flag]
        ax.scatter(sub["critic_abs_error"], sub["rank_correlation"], s=14, alpha=0.6, color=color, label=label)
    ax.set_xlabel("critic abs error under \u03c0\u03b2 (|predicted - exact true value|)")
    ax.set_ylabel(f"rank correlation: \u03c0D* ({ref.kind}) vs. best-config policy")
    ax.set_title("Does critic inaccuracy line up with policy disagreement?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot1 = out_dir / "policy_agreement_vs_critic_error"
    fig.savefig(plot1.with_suffix(".svg"))
    fig.savefig(plot1.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot1}.svg/.png")

    # --- Plot 2: spatial maze map, colored by rank correlation ---
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        df["col"], df["row"], c=df["rank_correlation"], cmap="RdYlGn", vmin=-1, vmax=1, s=90, marker="s",
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
    ax.set_title(f"\u03c0D* ({ref.kind}) vs. best-config policy: rank correlation by maze cell")
    fig.colorbar(sc, ax=ax, label="rank correlation (1=agree, -1=reversed)")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.15, 1.0))
    fig.tight_layout()
    plot2 = out_dir / "policy_agreement_maze_map"
    fig.savefig(plot2.with_suffix(".svg"))
    fig.savefig(plot2.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot2}.svg/.png")


if __name__ == "__main__":
    main()
