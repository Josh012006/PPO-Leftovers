"""Phase 2 design requirement 1 (README, "Phase 2"): the task must stay
hard enough to require genuine training and real policy improvement --
not solvable near-trivially.

Checks this two ways:
  1. A uniformly random policy vs. the full-information exact optimal
     policy pi* (same true-dynamics, full-coverage value iteration as
     scripts/verify_redundancy_level.py) -- a large gap here means the
     task is not solvable by luck or a near-random policy.
  2. A SHORT, genuine online PPO training run (far fewer iterations than
     scripts/01_train_prior.py's real prior-training run, which continues
     until it crosses target_success_rate) -- showing success_rate
     actually climb over these iterations is direct evidence that
     training is doing real work, not that the task was already solved
     at initialization.

No dataset D needed for either check -- (1) needs only the environment,
(2) trains a small online PPO agent from scratch, live in the
environment, exactly like scripts/01_train_prior.py's own mechanism, just
for fewer iterations and without the target-crossing stop condition.

Outputs, under --out-dir (default results/phase2/analysis/task_difficulty/):
  task_difficulty_random_vs_optimal.csv  -- the two live-rollout results
  training_curve.csv                     -- success_rate at each tracked
                                             iteration of the short run
  task_difficulty.svg/png                -- both checks in one figure:
                                             left panel bar (random vs.
                                             optimal), right panel the
                                             training curve

Usage:
    python scripts/verify_task_difficulty.py \
        --env-config configs/phase2/env_maze.yaml \
        --prior-config configs/phase2/prior_training.yaml \
        --training-iterations 80 \
        --eval-episodes 500 --eval-seed 999 \
        --out-dir results/phase2/analysis/task_difficulty
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
from ppo_exploitation.reference.experience_optimal import _value_iteration
from ppo_exploitation.utils.config import MazeEnvConfig, OnlinePPOConfig
from ppo_exploitation.utils.seeding import set_global_seed


def make_random_act_fn(n_actions: int, seed: int):
    rng = np.random.default_rng(seed)

    def act_fn(obs, state):
        return int(rng.integers(0, n_actions))

    return act_fn


def make_optimal_act_fn(policy: np.ndarray):
    def act_fn(obs, state):
        return int(policy[state])

    return act_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/phase2/env_maze.yaml")
    parser.add_argument("--prior-config", default="configs/phase2/prior_training.yaml")
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=80,
        help="Deliberately short -- this is a difficulty check, not a real prior-training run "
        "(that's scripts/01_train_prior.py, which runs until target_success_rate is crossed).",
    )
    parser.add_argument("--track-every", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=999)
    parser.add_argument("--out-dir", default="results/phase2/analysis/task_difficulty")
    args = parser.parse_args()

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)

    def make_env(layout=None):
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
            layout=layout,  # same layout_seed => identical layout; skip regenerating when given
        )

    env = make_env()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Check 1: random vs. full-information optimal ---
    print("Evaluating a uniformly random policy...")
    random_res = evaluate_policy(
        env, make_random_act_fn(env.n_actions, seed=0), args.eval_episodes, seed=args.eval_seed
    )
    print(f"  random: success_rate={random_res['success_rate']:.3f}")

    print("Solving pi* exactly (full coverage, true dynamics)...")
    n_states, n_actions = env.n_states, env.n_actions
    terminal_states = {s for s in range(n_states) if env.is_terminal_state(s)}
    transition_model = {}
    for s in range(n_states):
        if s in terminal_states:
            continue
        for a in range(n_actions):
            probs = env.true_transition_probs(s, a)
            transition_model[(s, a)] = [(sp, p, env.true_reward(s, a, sp)) for sp, p in probs.items()]
    V, Q, policy = _value_iteration(
        transition_model, terminal_states, n_states, n_states, n_actions, env_cfg.gamma, 1e-8, 100_000
    )
    optimal_res = evaluate_policy(env, make_optimal_act_fn(policy), args.eval_episodes, seed=args.eval_seed)
    print(f"  optimal (pi*): success_rate={optimal_res['success_rate']:.3f}")

    gap_df = pd.DataFrame(
        [
            {"policy": "random", **random_res},
            {"policy": "optimal (pi*)", **optimal_res},
        ]
    )
    gap_df.to_csv(out_dir / "task_difficulty_random_vs_optimal.csv", index=False)
    print(f"\nGap (optimal - random) success_rate: {optimal_res['success_rate'] - random_res['success_rate']:.3f}")
    if optimal_res["success_rate"] - random_res["success_rate"] < 0.3:
        print("  -> WARNING: gap under 0.3 -- the task may be too easy to require real training.")
    else:
        print("  -> large gap: a random policy is nowhere near sufficient, consistent with requirement 1.")

    # --- Check 2: a short, genuine online-training run ---
    prior_cfg = OnlinePPOConfig.from_yaml(args.prior_config)
    set_global_seed(prior_cfg.seed)
    agent = OnlinePPOAgent(env.observation_space.shape[0], env.n_actions, prior_cfg)

    # Built ONCE, reused for the whole run -- see scripts/01_train_prior.py's
    # identical fix for why rebuilding these every iteration (as this
    # script originally did) turns a short verification run into
    # something that can take hours: phase 2's environment now runs a
    # rejection-sampling safety check at construction time that can take
    # tens of seconds by itself, and collect_rollout previously called
    # each env_fns[i]() fresh on every single call.
    rollout_envs = [make_env(layout=env.layout) for _ in range(prior_cfg.n_envs)]
    env_fns = [(lambda e=e: e) for e in rollout_envs]
    track_env = make_env(layout=env.layout)

    training_rows = []

    def track(iteration: int):
        res = evaluate_policy(track_env, lambda obs, s: agent.net.act_numpy(obs, deterministic=True)[0],
                               args.eval_episodes, seed=args.eval_seed)
        training_rows.append({"iteration": iteration, "success_rate": res["success_rate"]})
        print(f"  [iter {iteration:3d}] success_rate={res['success_rate']:.3f}")

    print(f"\nRunning {args.training_iterations} iterations of genuine online PPO training...")
    track(0)
    for it in range(1, args.training_iterations + 1):
        trajectories = agent.collect_rollout(env_fns)
        agent.update(trajectories)
        if it % args.track_every == 0 or it == args.training_iterations:
            track(it)

    train_df = pd.DataFrame(training_rows)
    train_df.to_csv(out_dir / "training_curve.csv", index=False)
    start_sr = train_df.iloc[0]["success_rate"]
    end_sr = train_df.iloc[-1]["success_rate"]
    print(f"\nsuccess_rate: iteration 0 = {start_sr:.3f}, iteration {args.training_iterations} = {end_sr:.3f}")
    if end_sr - start_sr < 0.1:
        print("  -> WARNING: less than 0.1 improvement over this short run -- check learning is happening at all.")
    else:
        print("  -> clear improvement from training alone, over far fewer iterations than a full prior-training run.")

    # --- Combined figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.bar(
        ["random", "optimal (\u03c0*)"],
        [random_res["success_rate"], optimal_res["success_rate"]],
        yerr=[random_res["success_rate_stderr"], optimal_res["success_rate_stderr"]],
        color=["tab:red", "tab:green"], capsize=5,
    )
    ax.set_ylabel("success_rate")
    ax.set_ylim(0, 1)
    ax.set_title("Random vs. full-information optimal")

    ax = axes[1]
    ax.plot(train_df["iteration"], train_df["success_rate"], marker="o", markersize=3, color="tab:blue")
    ax.axhline(random_res["success_rate"], color="tab:red", linestyle="--", linewidth=1, label="random")
    ax.axhline(optimal_res["success_rate"], color="tab:green", linestyle="--", linewidth=1, label="optimal (\u03c0*)")
    ax.set_xlabel("online PPO training iteration")
    ax.set_ylabel("success_rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"Short training run ({args.training_iterations} iterations)")
    ax.legend(fontsize=8)

    fig.suptitle("Is the task hard enough to require genuine training?")
    fig.tight_layout()
    plot_path = out_dir / "task_difficulty"
    fig.savefig(plot_path.with_suffix(".svg"))
    fig.savefig(plot_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved {plot_path}.svg/.png")


if __name__ == "__main__":
    main()