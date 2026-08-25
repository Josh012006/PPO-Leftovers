"""Train standard online PPO on the live maze environment until eval
success_rate crosses `target_success_rate` AND an immediate, independent
re-check on a genuinely different sample of episodes ALSO clears it.

The logic, per eval check:
  1. Evaluate on `tracking_eval_seed` (same seed every check -- cheap,
     comparable signal for whether a crossing has happened at all).
  2. If success_rate < target: keep training. If this value is BELOW but
     within `near_target_zone_width` of target, and we weren't already
     checking every iteration, switch to checking every single iteration
     from here on -- this exists because a real jump between two
     eval_every-spaced checks can leap straight over target_success_rate
     without ever landing near it (observed: 33.6% -> 68.4% in one 5-
     iteration gap). Once close, checking more often makes it much less
     likely training overshoots far past target before the next check.
     This does not guarantee landing exactly on target -- a big enough
     jump can still clear the zone AND the target in one gap -- it only
     reduces how far past it training is likely to land.
  3. If success_rate >= target: STOP training and immediately re-evaluate
     that exact same checkpoint on `confirmation_eval_seed` (a genuinely
     different sample of the same number of episodes).
       - If the confirmation ALSO clears target: accept this checkpoint,
         save it, done.
       - If the confirmation does NOT clear target: the first crossing was
         not robust to a different sample -- resume training and keep
         looking (per-iteration checking, if already active, stays active).

This directly tests robustness to eval-sampling noise (would a different
batch of episodes have told the same story?), which repeating the SAME
fixed seed many times over -- however many times -- structurally cannot
do, since it always asks the same questions.

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
from ppo_exploitation.ppo.networks import ActorCritic
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
        f"Stopping rule: the FIRST time success_rate >= {prior_cfg.target_success_rate:.2f} "
        f"(tracking seed={prior_cfg.tracking_eval_seed}), immediately re-check the same checkpoint "
        f"on {prior_cfg.eval_episodes} INDEPENDENT episodes (seed={prior_cfg.confirmation_eval_seed}). "
        f"If that also clears target: stop and save. If not: resume training and keep looking.\n"
        f"Checks run every {prior_cfg.eval_every} iterations until success_rate first comes within "
        f"{prior_cfg.near_target_zone_width:.2f} of target from below, then every iteration.\n"
    )

    last_eval_res = None
    chosen_it, chosen_state_dict, chosen_eval_res, confirm_eval_res = None, None, None, None
    effective_eval_every = prior_cfg.eval_every
    in_near_target_zone = False

    t0 = time.time()
    for it in range(prior_cfg.total_iterations):
        env_fns = [make_env for _ in range(prior_cfg.n_envs)]
        trajectories = agent.collect_rollout(env_fns)
        stats = agent.update(trajectories)

        is_last_iter = it == prior_cfg.total_iterations - 1
        if it % effective_eval_every == 0 or is_last_iter:
            act_fn = make_neural_act_fn(agent.net, deterministic=True)
            eval_res = evaluate_policy(eval_env, act_fn, prior_cfg.eval_episodes, seed=prior_cfg.tracking_eval_seed)
            last_eval_res = eval_res
            elapsed = time.time() - t0
            zone_tag = " [zone]" if in_near_target_zone else ""
            print(
                f"[iter {it:4d} | {elapsed:6.1f}s]{zone_tag} "
                f"success_rate={eval_res['success_rate']:.3f}"
                f"\u00b1{eval_res['success_rate_stderr']:.3f} "
                f"mean_return={eval_res['mean_return']:.3f} "
                f"policy_loss={stats['policy_loss']:.4f} entropy={stats['entropy']:.3f}"
            )

            if eval_res["success_rate"] >= prior_cfg.target_success_rate:
                print(
                    f"  >= target: confirming on {prior_cfg.eval_episodes} INDEPENDENT episodes "
                    f"(seed={prior_cfg.confirmation_eval_seed})..."
                )
                confirm_net = ActorCritic(obs_dim, n_actions, prior_cfg.hidden_sizes)
                confirm_net.load_state_dict(agent.state_dict())
                confirm_act_fn = make_neural_act_fn(confirm_net, deterministic=True)
                candidate_confirm_res = evaluate_policy(
                    eval_env, confirm_act_fn, prior_cfg.eval_episodes, seed=prior_cfg.confirmation_eval_seed
                )
                print(
                    f"    confirmation success_rate={candidate_confirm_res['success_rate']:.3f}"
                    f"\u00b1{candidate_confirm_res['success_rate_stderr']:.3f}"
                )
                if candidate_confirm_res["success_rate"] >= prior_cfg.target_success_rate:
                    chosen_it = it
                    chosen_state_dict = agent.state_dict()
                    chosen_eval_res = eval_res
                    confirm_eval_res = candidate_confirm_res
                    print(f"\nConfirmed at iteration {chosen_it}. Stopping.")
                    break
                else:
                    print(
                        f"    Confirmation did NOT clear target ({candidate_confirm_res['success_rate']:.3f} < "
                        f"{prior_cfg.target_success_rate:.2f}) -- the tracking-seed crossing wasn't robust "
                        f"to a different sample. Resuming training.\n"
                    )
            elif not in_near_target_zone:
                distance_below = prior_cfg.target_success_rate - eval_res["success_rate"]
                if 0 <= distance_below <= prior_cfg.near_target_zone_width:
                    in_near_target_zone = True
                    effective_eval_every = 1
                    print(
                        f"  Within {prior_cfg.near_target_zone_width:.2f} of target from below -- "
                        f"switching to per-iteration checks.\n"
                    )
    else:
        # Loop exhausted total_iterations without ever landing a confirmed crossing.
        chosen_it, chosen_state_dict, chosen_eval_res, confirm_eval_res = None, agent.state_dict(), last_eval_res, None
        print(
            f"\nReached total_iterations ({prior_cfg.total_iterations}) without a confirmed crossing "
            f"of target_success_rate ({prior_cfg.target_success_rate:.2f}). Saving the final network as-is "
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
            "confirmation_eval": confirm_eval_res,
            "confirmation_eval_seed": prior_cfg.confirmation_eval_seed if confirm_eval_res is not None else None,
            "checkpoint_iteration": chosen_it,
            "reached_near_target_zone": in_near_target_zone,
        },
        out_path,
    )
    print(f"Saved prior checkpoint to {out_path}")


if __name__ == "__main__":
    main()