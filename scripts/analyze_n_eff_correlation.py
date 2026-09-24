"""Does ensemble_n_eff (ppo/fixed_d_trainer_ensemble.py) actually correlate
with the TRUE, exact (state, action) sample count fixed_d_trainer.py
computes -- checked on testbeds we did NOT tune any of this project's
beta/k choices against.

Motivation (see README, "An attempt to close the gap" and "How well does
it scale"): every beta/k result in this whole project was measured on
ONE specific 30x30 maze (layout_seed=0). Testing the ensemble's n_eff
only on that same maze would risk the correlation being an artifact of
this project's own environment implementation and data-collection
process, not a general property of ensemble disagreement as a confidence
signal -- exactly the kind of self-referential validation that would be
easy to write plausible-looking documentation for without it actually
meaning anything. This script deliberately tests on THREE testbeds of
increasing independence from this project's own setup:

  - current-maze: the actual project D/prior on the 30x30 maze used
    throughout this study. Included as a reference point (does the
    correlation hold at all, here), NOT as the main evidence.
  - new-maze-layout: a FRESH, smaller stochastic maze with a DIFFERENT
    layout_seed, its own freshly-trained prior and freshly-collected D.
    Same environment CLASS as the main study, a different concrete
    instance -- tests whether the correlation is specific to this
    project's one maze geometry or holds across others.
  - taxi: gymnasium's Taxi-v3, an environment this project did not build
    and whose structure differs qualitatively from grid navigation --
    500 states combining taxi position, passenger location, and
    destination, several actions (e.g. "pick up", "drop off") that are
    only ever sensible in narrow subsets of states, forcing a very
    different (state, action) sparsity pattern than either maze. The
    strongest test against this project's own implementation quirks.

For each testbed: collect a dataset, build a plain FixedDPPOTrainer
(cfg.use_effective_sample_weighting=True) purely to get its EXACT
per-transition count (trainer._n_per_transition) and its GAE returns
(trainer._compute_advantages_and_returns()) for free, then feed those
SAME (obs, actions, returns) into fixed_d_trainer_ensemble.py's
train_ensemble_and_compute_n_eff to get ensemble_n_eff on an identical
footing. Reports Spearman rank correlation (the relevant one: a later
recalibration of beta/k only needs n_eff to preserve the true count's
RELATIVE ordering, not match its absolute scale) and, for reference,
Pearson correlation in log-log space (both quantities are heavy-tailed).

Usage:
    python scripts/analyze_n_eff_correlation.py \
        --testbeds current-maze new-maze-layout taxi \
        --ensemble-n-heads 5 --ensemble-epochs 30 \
        --out-dir results/phase2/analysis/n_eff_correlation

Any single testbed can be run alone (e.g. --testbeds taxi) for a quicker
check while iterating.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "tier0"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as scipy_stats

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.ppo.fixed_d_trainer_ensemble import train_ensemble_and_compute_n_eff
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
from ppo_exploitation.utils.config import MazeEnvConfig, OnlinePPOConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


# --------------------------------------------------------------------------
# Taxi-v3 wrapper -- deliberately minimal (only what collect_fixed_dataset /
# OnlinePPOAgent actually need: reset, step, get_state, n_states, n_actions,
# observation_space). No true_transition_probs / is_terminal_state: this
# script never computes pi_D* or needs exact dynamics, only raw counts and
# GAE returns from a frozen critic, so there's nothing to hand-derive about
# Taxi's own dynamics -- gymnasium's own step()/terminated flag is enough,
# same principle tests/tier0/frozen_lake_env.py states for why it reads
# gymnasium's own P table rather than re-deriving FrozenLake's slip rule.
# --------------------------------------------------------------------------
class TaxiWrapper:
    def __init__(self, max_steps: int = 200):
        import gymnasium as gym
        from gymnasium import spaces

        # Taxi-v4 deprecated v3 in newer gymnasium releases; try v4 first,
        # fall back to v3 for older installs, rather than hard-coding one
        # and breaking on whichever gymnasium version the user has.
        try:
            self._env = gym.make("Taxi-v4", max_episode_steps=max_steps)
        except gym.error.DeprecatedEnv:
            self._env = gym.make("Taxi-v3", max_episode_steps=max_steps)
        self.n_states = int(self._env.observation_space.n)
        self.n_actions = int(self._env.action_space.n)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_states,), dtype=np.float32)
        self._state = 0

    def _one_hot(self, state: int) -> np.ndarray:
        v = np.zeros(self.n_states, dtype=np.float32)
        v[state] = 1.0
        return v

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        self._state = int(obs)
        return self._one_hot(self._state), {"state": self._state}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._state = int(obs)
        return self._one_hot(self._state), float(reward), bool(terminated), bool(truncated), {"state": self._state}

    def get_state(self) -> int:
        return self._state


# --------------------------------------------------------------------------
# Testbed builders. Each returns (label, obs_dim, n_actions, dataset,
# prior_state_dict).
# --------------------------------------------------------------------------
def build_current_maze_testbed(args) -> tuple[str, int, int, object, dict, tuple[int, ...]]:
    """Reuses the project's OWN existing artifacts -- no fresh training.
    Included as a reference point, not the main evidence (see module
    docstring)."""
    from ppo_exploitation.data.collect import load_dataset

    dataset = load_dataset(args.current_maze_dataset)
    ckpt = torch.load(args.current_maze_prior, map_location="cpu", weights_only=False)
    hidden_sizes = tuple(ckpt.get("hidden_sizes", (64, 64)))
    return "current-maze", dataset.obs_dim, dataset.n_actions, dataset, ckpt["state_dict"], hidden_sizes


def build_new_maze_layout_testbed(args) -> tuple[str, int, int, object, dict, tuple[int, ...]]:
    """A fresh, smaller stochastic maze with a DIFFERENT layout_seed than
    the one used everywhere else in this project (default: layout_seed=0)
    -- same environment class, a different concrete instance."""
    set_global_seed(args.seed)
    env_cfg = MazeEnvConfig(
        width=args.new_maze_size, height=args.new_maze_size, slip_prob=0.1, extra_connection_prob=0.08,
        num_hazards=max(2, args.new_maze_size // 5), step_penalty=-0.03, max_steps=args.new_maze_size * 6,
        layout_seed=args.new_maze_layout_seed, num_start_states=1,
    )
    env = StochasticMazeEnv(
        width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
        extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
        step_penalty=env_cfg.step_penalty, max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed,
        num_start_states=env_cfg.num_start_states, gamma=env_cfg.gamma,
    )
    print(f"  [new-maze-layout] training a short online-PPO prior (layout_seed={args.new_maze_layout_seed})...")
    prior_cfg = OnlinePPOConfig(
        total_iterations=args.prior_iterations, rollout_steps=512, n_envs=4, epochs=4, minibatch_size=128,
        entropy_coef=0.02, hidden_sizes=(32, 32), eval_every=args.prior_iterations + 1, seed=args.seed,
    )
    agent = OnlinePPOAgent(env.observation_space.shape[0], env.n_actions, prior_cfg)

    def make_env():
        return StochasticMazeEnv(
            width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
            extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
            step_penalty=env_cfg.step_penalty, max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed,
            num_start_states=env_cfg.num_start_states, gamma=env_cfg.gamma,
        )

    for _ in range(prior_cfg.total_iterations):
        trajectories = agent.collect_rollout([make_env for _ in range(prior_cfg.n_envs)])
        agent.update(trajectories)

    dataset = collect_fixed_dataset(env, agent.net, n_episodes=args.n_episodes, seed=args.seed + 1, sample_actions=True)
    return "new-maze-layout", dataset.obs_dim, dataset.n_actions, dataset, agent.net.state_dict(), (32, 32)


def build_taxi_testbed(args) -> tuple[str, int, int, object, dict, tuple[int, ...]]:
    """gymnasium's Taxi-v3 -- an environment this project did not build,
    with a qualitatively different (state, action) sparsity structure
    than grid navigation (see module docstring)."""
    set_global_seed(args.seed)
    env = TaxiWrapper(max_steps=200)
    print("  [taxi] training a short online-PPO prior...")
    prior_cfg = OnlinePPOConfig(
        total_iterations=args.prior_iterations, rollout_steps=512, n_envs=4, epochs=4, minibatch_size=128,
        entropy_coef=0.02, hidden_sizes=(32, 32), eval_every=args.prior_iterations + 1, seed=args.seed,
    )
    agent = OnlinePPOAgent(env.observation_space.shape[0], env.n_actions, prior_cfg)
    for _ in range(prior_cfg.total_iterations):
        trajectories = agent.collect_rollout([lambda: TaxiWrapper(max_steps=200) for _ in range(prior_cfg.n_envs)])
        agent.update(trajectories)

    dataset = collect_fixed_dataset(env, agent.net, n_episodes=args.n_episodes, seed=args.seed + 1, sample_actions=True)
    return "taxi", dataset.obs_dim, dataset.n_actions, dataset, agent.net.state_dict(), (32, 32)


TESTBED_BUILDERS = {
    "current-maze": build_current_maze_testbed,
    "new-maze-layout": build_new_maze_layout_testbed,
    "taxi": build_taxi_testbed,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbeds", nargs="+", choices=list(TESTBED_BUILDERS), default=list(TESTBED_BUILDERS))
    parser.add_argument("--current-maze-dataset", default="results/phase2/dataset_D.pkl")
    parser.add_argument("--current-maze-prior", default="results/phase2/prior_checkpoint.pt")
    parser.add_argument("--new-maze-size", type=int, default=10)
    parser.add_argument("--new-maze-layout-seed", type=int, default=7, help="Anything != 0, the seed used everywhere else in this project.")
    parser.add_argument("--prior-iterations", type=int, default=60, help="Online-PPO iterations for the fresh-prior testbeds (new-maze-layout, taxi).")
    parser.add_argument("--n-episodes", type=int, default=500, help="Episodes of D to collect for the fresh testbeds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-n-heads", type=int, default=5)
    parser.add_argument("--ensemble-hidden-sizes", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--ensemble-epochs", type=int, default=30)
    parser.add_argument("--ensemble-lr", type=float, default=1e-3)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--out-dir", default="results/phase2/analysis/n_eff_correlation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    fig, axes = plt.subplots(1, len(args.testbeds), figsize=(6 * len(args.testbeds), 5.5), squeeze=False)
    axes = axes[0]

    for ax, testbed_name in zip(axes, args.testbeds):
        print(f"=== {testbed_name} ===")
        label, obs_dim, n_actions, dataset, prior_state_dict, hidden_sizes = TESTBED_BUILDERS[testbed_name](args)
        print(f"  D: {len(dataset)} transitions, {dataset.n_episodes} episodes, obs_dim={obs_dim}, n_actions={n_actions}")

        exact_cfg = PPOHyperparams(
            epochs=1, minibatch_size=args.minibatch_size, hidden_sizes=hidden_sizes,
            use_effective_sample_weighting=True, effective_sample_beta=0.9, seed=args.seed,
        )
        exact_trainer = FixedDPPOTrainer(
            dataset, obs_dim=obs_dim, n_actions=n_actions, cfg=exact_cfg, prior_state_dict=prior_state_dict
        )
        exact_n = exact_trainer._n_per_transition.copy()
        _, returns = exact_trainer._compute_advantages_and_returns()

        ensemble_cfg = PPOHyperparams(
            minibatch_size=args.minibatch_size, ensemble_n_heads=args.ensemble_n_heads,
            ensemble_hidden_sizes=tuple(args.ensemble_hidden_sizes), ensemble_epochs=args.ensemble_epochs,
            ensemble_lr=args.ensemble_lr,
        )
        print(f"  training the {args.ensemble_n_heads}-head ensemble for {args.ensemble_epochs} epochs...")
        n_eff, _ = train_ensemble_and_compute_n_eff(
            obs_arr=exact_trainer._obs, actions_arr=exact_trainer._actions, returns=returns,
            obs_dim=obs_dim, n_actions=n_actions, cfg=ensemble_cfg,
        )

        spearman_r, spearman_p = scipy_stats.spearmanr(exact_n, n_eff)
        pearson_r, pearson_p = scipy_stats.pearsonr(np.log(exact_n), np.log(n_eff))
        print(f"  Spearman rho={spearman_r:.3f} (p={spearman_p:.4g})   Pearson (log-log) r={pearson_r:.3f} (p={pearson_p:.4g})")
        results.append(
            {"testbed": label, "n_transitions": len(exact_n), "spearman_rho": spearman_r, "spearman_p": spearman_p,
             "pearson_log_log_r": pearson_r, "pearson_log_log_p": pearson_p}
        )

        ax.scatter(exact_n, n_eff, alpha=0.25, s=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("exact (state, action) count")
        ax.set_ylabel("ensemble n_eff")
        ax.set_title(f"{label}\nSpearman ρ={spearman_r:.3f}, Pearson (log-log) r={pearson_r:.3f}")

    fig.suptitle("Does ensemble n_eff track the true exact count?")
    fig.tight_layout()
    fig.savefig(out_dir / "n_eff_correlation.svg")
    fig.savefig(out_dir / "n_eff_correlation.png", dpi=150)

    import pandas as pd

    pd.DataFrame(results).to_csv(out_dir / "n_eff_correlation_summary.csv", index=False)
    print(f"\nSaved {out_dir}/n_eff_correlation_summary.csv and n_eff_correlation.svg/.png")


if __name__ == "__main__":
    main()
