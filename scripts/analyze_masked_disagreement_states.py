"""If we remove D's (thin) information at exactly the states where PPO
disagrees with pi_D*, does pi_D* drop down toward the best fixed-D PPO
configuration's own (imprecise) success rate?

The logic: pi_D* (empirical) already proves the information at these
states is enough to get them right, WITHOUT overfitting -- it's exact
counting, not a repeatedly-updated approximator. If that thin information
is specifically what best-config PPO fails to use correctly (rather than
some other, unrelated limitation), then taking that same information away
from pi_D* entirely should reproduce roughly the SAME ceiling PPO is
already stuck at -- pi_D* forced to be just as blind there as PPO
effectively is. If instead masked pi_D* stays much closer to its own
unmasked ceiling, the thin information at these specific states isn't
what's limiting PPO -- something else is.

Masks every one of the already-known disagreement states (is_disagreement
== 1 in policy_agreement.csv) via the `mask_states` argument re-added to
reference.experience_optimal's solvers: every action at those states is
forced to the SAME unseen_penalty fallback used for genuinely-uncovered
(s,a) pairs, regardless of how much real (thin) data D actually has there.
Masking only ever affects the DP solve, never live evaluation -- same
convention as unseen_penalty itself and as the earlier (geometric,
hazard-distance-based) masking check in this project's history.

No training involved -- solving a masked/unmasked MDP is cheap DP, and
evaluating a policy (tabular or neural) is just a live rollout under the
SAME protocol used everywhere else in this project (seed=24680, n=500 by
default). Prior and best-config PPO are re-evaluated fresh here too,
under this exact protocol, rather than reusing possibly-differently-seeded
numbers from earlier scripts -- see README, "A note on eval seeds".

Outputs, under --out-dir (default results/analysis/masked_disagreement_states/):
  masked_disagreement_states.csv       -- one row per policy compared:
                                           kind, masked (bool),
                                           success_rate(+stderr), mean_return
  masked_disagreement_states_comparison.svg/png -- horizontal dot plot of
                                           every success_rate compared

Usage:
    python scripts/analyze_masked_disagreement_states.py \
        --env-config configs/env_maze.yaml \
        --dataset results/dataset_D.pkl \
        --prior-checkpoint results/prior_checkpoint.pt \
        --reference-config configs/reference.yaml \
        --best-config-checkpoint results/analysis/policy_agreement/best_config_checkpoint.pt \
        --policy-agreement-csv results/analysis/policy_agreement/policy_agreement.csv \
        --eval-episodes 500 --eval-seed 24680 \
        --out-dir results/analysis/masked_disagreement_states

Add --also-true-restricted to additionally mask/evaluate pi_D* (true-
restricted) the same way, as a robustness check against MLE noise (same
convention as every other empirical/true-restricted comparison in this
project) -- requires --pi-d-star-true-restricted (default
results/pi_d_star_true_restricted.pkl).
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
import torch

from ppo_exploitation.data.collect import load_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy, make_neural_act_fn, make_tabular_act_fn
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.reference.experience_optimal import (
    compute_pi_d_star_empirical,
    compute_pi_d_star_true_restricted,
)
from ppo_exploitation.utils.config import MazeEnvConfig, ReferenceConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--dataset", default="results/dataset_D.pkl")
    parser.add_argument("--prior-checkpoint", default="results/prior_checkpoint.pt")
    parser.add_argument("--reference-config", default="configs/reference.yaml")
    parser.add_argument(
        "--best-config-checkpoint", default="results/analysis/policy_agreement/best_config_checkpoint.pt"
    )
    parser.add_argument("--policy-agreement-csv", default="results/analysis/policy_agreement/policy_agreement.csv")
    parser.add_argument(
        "--disagreement-column",
        default="is_disagreement_strict",
        help="Which policy_agreement.csv column selects the states to mask. Falls back to "
        "'is_disagreement' automatically if the given column isn't present (e.g. an older CSV, "
        "or one made with --pi-d-star-cross-check '').",
    )
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=24680)
    parser.add_argument("--out-dir", default="results/analysis/masked_disagreement_states")
    parser.add_argument("--also-true-restricted", action="store_true")
    parser.add_argument("--pi-d-star-true-restricted", default="results/pi_d_star_true_restricted.pkl")
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
    print(f"Loaded D: {len(dataset)} transitions, {dataset.n_episodes} episodes.")

    pa = pd.read_csv(args.policy_agreement_csv)
    disagreement_col = args.disagreement_column
    if disagreement_col not in pa.columns:
        print(f"'{disagreement_col}' not found in {args.policy_agreement_csv} -- falling back to 'is_disagreement'.")
        disagreement_col = "is_disagreement"
    mask_states = set(pa.loc[pa[disagreement_col] == 1, "state"].astype(int).tolist())
    print(f"Masking {len(mask_states)} states flagged by '{disagreement_col}' in {args.policy_agreement_csv}.")
    if len(mask_states) == 0:
        print("No disagreement states found -- nothing to mask. Exiting.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def eval_ref(ref) -> dict:
        return evaluate_policy(
            env, make_tabular_act_fn(ref), args.eval_episodes, seed=args.eval_seed, covered_states=ref.covered_states
        )

    rows = []

    print("\nSolving pi_D* (empirical), normal...")
    ref_emp = compute_pi_d_star_empirical(dataset, gamma=ref_cfg.gamma, unseen_penalty=ref_cfg.unseen_penalty)
    res = eval_ref(ref_emp)
    rows.append({"policy": "pi_D* empirical", "masked": False, **res})
    print(f"  success_rate={res['success_rate']:.3f}\u00b1{res['success_rate_stderr']:.3f}")

    print(f"Solving pi_D* (empirical), {len(mask_states)} disagreement states MASKED...")
    ref_emp_masked = compute_pi_d_star_empirical(
        dataset, gamma=ref_cfg.gamma, unseen_penalty=ref_cfg.unseen_penalty, mask_states=mask_states
    )
    res = eval_ref(ref_emp_masked)
    rows.append({"policy": "pi_D* empirical (disagreement states masked)", "masked": True, **res})
    print(f"  success_rate={res['success_rate']:.3f}\u00b1{res['success_rate_stderr']:.3f}")

    if args.also_true_restricted:
        print("\nSolving pi_D* (true-restricted), normal...")
        ref_true = compute_pi_d_star_true_restricted(
            env, dataset, gamma=ref_cfg.gamma, unseen_penalty=ref_cfg.unseen_penalty
        )
        res = eval_ref(ref_true)
        rows.append({"policy": "pi_D* true-restricted", "masked": False, **res})
        print(f"  success_rate={res['success_rate']:.3f}\u00b1{res['success_rate_stderr']:.3f}")

        print(f"Solving pi_D* (true-restricted), {len(mask_states)} disagreement states MASKED...")
        ref_true_masked = compute_pi_d_star_true_restricted(
            env, dataset, gamma=ref_cfg.gamma, unseen_penalty=ref_cfg.unseen_penalty, mask_states=mask_states
        )
        res = eval_ref(ref_true_masked)
        rows.append({"policy": "pi_D* true-restricted (disagreement states masked)", "masked": True, **res})
        print(f"  success_rate={res['success_rate']:.3f}\u00b1{res['success_rate_stderr']:.3f}")

    print("\nEvaluating pi_beta (prior) for context...")
    prior_ckpt = torch.load(args.prior_checkpoint, map_location="cpu", weights_only=False)
    prior_net = ActorCritic(prior_ckpt["obs_dim"], prior_ckpt["n_actions"], prior_ckpt["hidden_sizes"])
    prior_net.load_state_dict(prior_ckpt["state_dict"])
    prior_net.eval()
    res = evaluate_policy(env, make_neural_act_fn(prior_net, deterministic=True), args.eval_episodes, seed=args.eval_seed)
    rows.append({"policy": "\u03c0\u03b2 (prior)", "masked": None, **res})
    print(f"  success_rate={res['success_rate']:.3f}\u00b1{res['success_rate_stderr']:.3f}")

    print(f"\nEvaluating best-config PPO checkpoint ({args.best_config_checkpoint}) for context...")
    best_ckpt = torch.load(args.best_config_checkpoint, map_location="cpu", weights_only=False)
    best_net = ActorCritic(best_ckpt["obs_dim"], best_ckpt["n_actions"], best_ckpt["hidden_sizes"])
    best_net.load_state_dict(best_ckpt["state_dict"])
    best_net.eval()
    res = evaluate_policy(env, make_neural_act_fn(best_net, deterministic=True), args.eval_episodes, seed=args.eval_seed)
    rows.append({"policy": f"best-config PPO (epoch {best_ckpt['epoch']})", "masked": None, **res})
    print(f"  success_rate={res['success_rate']:.3f}\u00b1{res['success_rate_stderr']:.3f}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "masked_disagreement_states.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    best_config_sr = df.loc[df["policy"].str.startswith("best-config"), "success_rate"].iloc[0]
    masked_emp_sr = df.loc[df["policy"] == "pi_D* empirical (disagreement states masked)", "success_rate"].iloc[0]
    unmasked_emp_sr = df.loc[df["policy"] == "pi_D* empirical", "success_rate"].iloc[0]
    print(
        f"\nmasked pi_D* (empirical) = {masked_emp_sr:.3f}, best-config PPO = {best_config_sr:.3f}, "
        f"unmasked pi_D* (empirical) = {unmasked_emp_sr:.3f}."
    )
    if abs(masked_emp_sr - best_config_sr) < abs(unmasked_emp_sr - best_config_sr):
        print(
            "  -> masked pi_D* landed closer to best-config PPO than the unmasked ceiling did: "
            "consistent with the thin information at these specific states being what limits PPO."
        )
    else:
        print(
            "  -> masked pi_D* did NOT land noticeably closer to best-config PPO: the thin "
            "information at these specific states may not be the limiting factor on its own."
        )

    # --- Comparison plot ---
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(df) + 1.5))
    order = df.sort_values("success_rate").reset_index(drop=True)
    colors = ["tab:red" if m is True else ("tab:green" if m is False else "tab:gray") for m in order["masked"]]
    ax.errorbar(
        order["success_rate"], range(len(order)), xerr=order["success_rate_stderr"], fmt="o", color="black",
        ecolor="0.6", capsize=3, zorder=3,
    )
    for i, c in enumerate(colors):
        ax.scatter([order["success_rate"][i]], [i], color=c, s=90, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["policy"])
    ax.set_xlabel("success_rate (500 episodes)")
    ax.set_xlim(0, 1)
    ax.set_title("Masking the known disagreement states in D: does pi_D* drop toward best-config PPO?")
    fig.tight_layout()
    plot_path = out_dir / "masked_disagreement_states_comparison"
    fig.savefig(plot_path.with_suffix(".svg"))
    fig.savefig(plot_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}.svg/.png")


if __name__ == "__main__":
    main()