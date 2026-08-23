"""Train standard online PPO on the live maze environment until it crosses
`target_success_rate` (default 60%), then freeze and save that checkpoint.
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

    t0 = time.time()
    for it in range(prior_cfg.total_iterations):
        env_fns = [make_env for _ in range(prior_cfg.n_envs)]
        trajectories = agent.collect_rollout(env_fns)
        stats = agent.update(trajectories)

        if it % prior_cfg.eval_every == 0 or it == prior_cfg.total_iterations - 1:
            act_fn = make_neural_act_fn(agent.net, deterministic=True)
            eval_res = evaluate_policy(eval_env, act_fn, prior_cfg.eval_episodes, seed=12345)
            elapsed = time.time() - t0
            print(
                f"[iter {it:4d} | {elapsed:6.1f}s] "
                f"success_rate={eval_res['success_rate']:.3f} "
                f"mean_return={eval_res['mean_return']:.3f} "
                f"policy_loss={stats['policy_loss']:.4f} entropy={stats['entropy']:.3f}"
            )
            if eval_res["success_rate"] >= prior_cfg.target_success_rate:
                print(
                    f"Target success rate {prior_cfg.target_success_rate:.2f} reached "
                    f"at iteration {it}. Stopping."
                )
                break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": agent.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "hidden_sizes": prior_cfg.hidden_sizes,
            "env_config_path": args.env_config,
            "final_eval": eval_res,
        },
        out_path,
    )
    print(f"Saved prior checkpoint to {out_path}")


if __name__ == "__main__":
    main()
