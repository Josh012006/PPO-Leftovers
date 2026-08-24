"""Train standard online PPO on the live maze environment until eval
success_rate is *maintained* at or above `target_success_rate` for
`success_streak_length` CONSECUTIVE eval checks -- not on the first
crossing. A single eval batch clearing the bar isn't evidence of a stable
policy (small maze + limited eval episodes both add variance); requiring a
streak filters that out.

The checkpoint actually SAVED is the EARLIEST one in that streak (the
network state at the first eval check that hit target), not whatever
`theta` had drifted to once the streak was confirmed -- training keeps
going past that point only to verify the streak holds, and the resulting
extra iterations should not be allowed to change which weights end up in
`pi_beta`.

This checkpoint (pi_beta) is the ONLY thing scripts/02 is allowed to use to
collect D.

Usage:
    python scripts/01_train_prior.py \
        --env-config configs/env_maze.yaml \
        --prior-config configs/prior_training.yaml \
        --out results/prior_checkpoint.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import evaluate_policy, make_neural_act_fn
from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
from ppo_exploitation.utils.config import MazeEnvConfig, OnlinePPOConfig
from ppo_exploitation.utils.seeding import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_maze.yaml")
    parser.add_argument("--prior-config", default="configs/prior_training.yaml")
    parser.add_argument("--out", default="results/prior_checkpoint.pt")
    args = parser.parse_args()

    env_cfg = MazeEnvConfig.from_yaml(args.env_config)
    prior_cfg = OnlinePPOConfig.from_yaml(args.prior_config)
    set_global_seed(prior_cfg.seed)

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

    probe_env = make_env()
    obs_dim = probe_env.observation_space.shape[0]
    n_actions = probe_env.n_actions

    agent = OnlinePPOAgent(obs_dim, n_actions, prior_cfg)
    eval_env = make_env()

    print(
        f"Stopping rule: success_rate >= {prior_cfg.target_success_rate:.2f} maintained for "
        f"{prior_cfg.success_streak_length} consecutive eval checks (every {prior_cfg.eval_every} "
        f"iterations, {prior_cfg.eval_episodes} eval episodes each). The checkpoint saved will be "
        f"the EARLIEST one in that streak, not the latest.\n"
    )

    # (iteration, state_dict, eval_res) for the current run of consecutive
    # eval checks at/above target. Reset the moment any check falls below.
    streak: list[tuple[int, dict, dict]] = []
    last_eval_res = None
    chosen_it, chosen_state_dict, chosen_eval_res = None, None, None

    t0 = time.time()
    for it in range(prior_cfg.total_iterations):
        env_fns = [make_env for _ in range(prior_cfg.n_envs)]
        trajectories = agent.collect_rollout(env_fns)
        stats = agent.update(trajectories)

        is_last_iter = it == prior_cfg.total_iterations - 1
        if it % prior_cfg.eval_every == 0 or is_last_iter:
            act_fn = make_neural_act_fn(agent.net, deterministic=True)
            eval_res = evaluate_policy(eval_env, act_fn, prior_cfg.eval_episodes, seed=12345)
            last_eval_res = eval_res
            elapsed = time.time() - t0
            print(
                f"[iter {it:4d} | {elapsed:6.1f}s] "
                f"success_rate={eval_res['success_rate']:.3f} "
                f"mean_return={eval_res['mean_return']:.3f} "
                f"policy_loss={stats['policy_loss']:.4f} entropy={stats['entropy']:.3f}"
            )

            if eval_res["success_rate"] >= prior_cfg.target_success_rate:
                streak.append((it, agent.state_dict(), eval_res))
                print(f"  >= target: streak {len(streak)}/{prior_cfg.success_streak_length}")
                if len(streak) >= prior_cfg.success_streak_length:
                    chosen_it, chosen_state_dict, chosen_eval_res = streak[0]
                    print(
                        f"\nStreak complete. Saving the EARLIEST checkpoint in it: "
                        f"iteration {chosen_it} (success_rate={chosen_eval_res['success_rate']:.3f})."
                    )
                    break
            else:
                if streak:
                    print(f"  < target: streak broken at length {len(streak)}, resetting")
                streak = []
    else:
        # Loop exhausted total_iterations without ever completing a full streak.
        if streak:
            chosen_it, chosen_state_dict, chosen_eval_res = streak[0]
            print(
                f"\nReached total_iterations ({prior_cfg.total_iterations}) without completing a full "
                f"{prior_cfg.success_streak_length}-check streak (best streak reached: {len(streak)}). "
                f"Using the earliest checkpoint from that partial streak: iteration {chosen_it}."
            )
        else:
            chosen_it, chosen_state_dict, chosen_eval_res = None, agent.state_dict(), last_eval_res
            print(
                f"\nReached total_iterations ({prior_cfg.total_iterations}) without ever reaching "
                f"target_success_rate ({prior_cfg.target_success_rate:.2f}). Saving the final network as-is "
                f"-- consider raising total_iterations."
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": chosen_state_dict,
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "hidden_sizes": prior_cfg.hidden_sizes,
            "env_config_path": args.env_config,
            "final_eval": chosen_eval_res,
            "checkpoint_iteration": chosen_it,
            "success_streak_length_required": prior_cfg.success_streak_length,
        },
        out_path,
    )
    print(f"Saved prior checkpoint to {out_path}")


if __name__ == "__main__":
    main()