"""Does ensemble_n_eff (ppo/fixed_d_trainer_ensemble.py) actually correlate
with the TRUE, exact (state, action) sample count fixed_d_trainer.py
computes -- checked on testbeds we did NOT tune any of this project's
beta/k choices against, and checked at the level that actually matters
for training: the LOSS-WEIGHT MULTIPLIER each signal would produce, not
just the raw signal itself.

Motivation for the testbeds (see README, "An attempt to close the gap"
and "How well does it scale"): every beta/k result in this whole project
was measured on ONE specific 30x30 maze (layout_seed=0). Testing the
ensemble's n_eff only on that same maze would risk the correlation being
an artifact of this project's own environment implementation, not a
general property of ensemble disagreement as a confidence signal. This
script tests on SIX testbeds of increasing independence from this
project's own setup:

  - current-maze: the actual project D/prior on the 30x30 maze used
    throughout this study. A reference point, not the main evidence.
  - new-maze-layout: a FRESH, smaller stochastic maze, DIFFERENT
    layout_seed, its own freshly-trained prior and freshly-collected D.
  - frozen-lake: gymnasium's FrozenLake-v1 (tests/tier0's own wrapper,
    reused not duplicated). Plain, well-behaved: normal episode
    lengths, a simple non-spiky reward.
  - river-swim: a hand-implemented RiverSwim (Strehl & Littman, 2008;
    Osband et al., 2013) -- a 1D chain, not a 2D grid, and thematically
    fitting (the same exploration/uncertainty literature Bootstrapped
    DQN belongs to). NOTE: with a short prior-training budget, the
    agent may barely explore past the first few states of a long chain
    (this IS RiverSwim's whole point), leaving very few (state, action)
    pairs actually visited -- inspect n_pairs in the output before
    trusting any correlation number from this testbed alone.
  - taxi: gymnasium's Taxi-v3/v4. Structurally unrelated to grid
    navigation, but its reward is highly heterogeneous per (state,
    action) pair (-1 ordinary step, -10 illegal pickup/dropoff, +20
    successful dropoff), so realized return VARIANCE differs a lot
    between pairs independent of how often they're seen -- the ensemble
    tracks that variance too (a real, informative signal, but one that
    weakens the simple exact-count correlation this script measures).
    --taxi-encoding {onehot, decomposed} exists from an earlier,
    rejected hypothesis (a structured decomposition made the
    correlation WORSE, 0.068 vs. 0.333 -- kept as a documented negative
    result).
  - blackjack: gymnasium's Blackjack-v1, a card game. Episodes are only
    1-3 steps, so it needs a much larger --n-episodes (thousands) than
    the other testbeds. Reward is small and bounded (+1/-1/0), given
    only at episode end -- the complementary extreme to taxi.

MEMORY: each testbed is run in its OWN subprocess (spawn context) that
exits when done, so its dataset/ensemble/torch tensors are released back
to the OS -- not just garbage-collected within a long-lived Python
process, which PyTorch's allocator does not reliably do. Only small
summary numbers cross back to the main process; each testbed's own plot
is written to disk by its own subprocess before exiting. Testbeds run
ONE AT A TIME (not in parallel) specifically to keep peak memory low --
this is a fix for exactly the failure mode of taxi's 500-dim one-hot
encoding times a large --n-episodes exhausting an 8GB machine.

WEIGHT-LEVEL COMPARISON (not just n_eff vs. exact count): the number that
actually matters for training is the loss-weight multiplier
effective_sample_weight(n, beta)/n, not n itself. This script computes,
per transition: weight_exact (from the true count), weight_no_threshold
(from n_eff = 1/sqrt(variance), fixed_d_trainer_ensemble.py's un-gated
signal), and weight_thresholded (from n_eff after compute_thresholded_
n_eff's F-test gate, which pins the multiplier to EXACTLY 1 for pairs
that aren't significantly above the estimated noise floor). All three
use the SAME --weight-beta (default 0.9995, this project's own validated
choice) so the comparison reflects what a real training run would
actually apply. Reports Spearman AND Pearson correlation between each
ensemble-derived weight array and weight_exact, letting the threshold
mechanism's real, isolated contribution be read off directly (compare
the thresholded row against the un-gated row) rather than assumed.

Usage:
    python scripts/analyze_n_eff_correlation.py \
        --testbeds current-maze new-maze-layout frozen-lake river-swim taxi blackjack \
        --n-episodes 8000 --ensemble-n-heads 5 --ensemble-epochs 30 --weight-beta 0.9995 \
        --out-dir results/phase2/analysis/n_eff_correlation

Any single testbed can be run alone (e.g. --testbeds blackjack) for a
quicker check while iterating -- also the natural way to work around a
memory-constrained machine: run taxi (or any single heavy testbed) with
a smaller --n-episodes on its own, separately from the others.
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "tier0"))

import numpy as np
import pandas as pd

from ppo_exploitation.utils.config import PPOHyperparams


def build_current_maze_testbed(args):
    from ppo_exploitation.data.collect import load_dataset
    import torch

    dataset = load_dataset(args.current_maze_dataset)
    ckpt = torch.load(args.current_maze_prior, map_location="cpu", weights_only=False)
    hidden_sizes = tuple(ckpt.get("hidden_sizes", (64, 64)))
    return "current-maze", dataset.obs_dim, dataset.n_actions, dataset, ckpt["state_dict"], hidden_sizes


def build_new_maze_layout_testbed(args):
    from ppo_exploitation.data.collect import collect_fixed_dataset
    from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
    from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
    from ppo_exploitation.utils.config import MazeEnvConfig, OnlinePPOConfig
    from ppo_exploitation.utils.seeding import set_global_seed

    set_global_seed(args.seed)
    env_cfg = MazeEnvConfig(
        width=args.new_maze_size, height=args.new_maze_size, slip_prob=0.1, extra_connection_prob=0.08,
        num_hazards=max(2, args.new_maze_size // 5), step_penalty=-0.03, max_steps=args.new_maze_size * 6,
        layout_seed=args.new_maze_layout_seed, num_start_states=1,
    )

    def make_env():
        return StochasticMazeEnv(
            width=env_cfg.width, height=env_cfg.height, slip_prob=env_cfg.slip_prob,
            extra_connection_prob=env_cfg.extra_connection_prob, num_hazards=env_cfg.num_hazards,
            step_penalty=env_cfg.step_penalty, max_steps=env_cfg.max_steps, layout_seed=env_cfg.layout_seed,
            num_start_states=env_cfg.num_start_states, gamma=env_cfg.gamma,
        )

    env = make_env()
    print(f"  [new-maze-layout] training a short online-PPO prior (layout_seed={args.new_maze_layout_seed})...")
    prior_cfg = OnlinePPOConfig(
        total_iterations=args.prior_iterations, rollout_steps=512, n_envs=4, epochs=4, minibatch_size=128,
        entropy_coef=0.02, hidden_sizes=(32, 32), eval_every=args.prior_iterations + 1, seed=args.seed,
    )
    agent = OnlinePPOAgent(env.observation_space.shape[0], env.n_actions, prior_cfg)
    for _ in range(prior_cfg.total_iterations):
        trajectories = agent.collect_rollout([make_env for _ in range(prior_cfg.n_envs)])
        agent.update(trajectories)

    dataset = collect_fixed_dataset(env, agent.net, n_episodes=args.n_episodes, seed=args.seed + 1, sample_actions=True)
    return "new-maze-layout", dataset.obs_dim, dataset.n_actions, dataset, agent.net.state_dict(), (32, 32)


def build_frozen_lake_testbed(args):
    from frozen_lake_env import FrozenLakeWrapper
    from ppo_exploitation.data.collect import collect_fixed_dataset
    from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
    from ppo_exploitation.utils.config import OnlinePPOConfig
    from ppo_exploitation.utils.seeding import set_global_seed

    set_global_seed(args.seed)
    env = FrozenLakeWrapper(map_name="4x4", is_slippery=True, max_steps=100)
    print("  [frozen-lake] training a short online-PPO prior...")
    prior_cfg = OnlinePPOConfig(
        total_iterations=args.prior_iterations, rollout_steps=512, n_envs=4, epochs=4, minibatch_size=128,
        entropy_coef=0.02, hidden_sizes=(32, 32), eval_every=args.prior_iterations + 1, seed=args.seed,
    )
    agent = OnlinePPOAgent(env.observation_space.shape[0], env.n_actions, prior_cfg)
    for _ in range(prior_cfg.total_iterations):
        trajectories = agent.collect_rollout(
            [lambda: FrozenLakeWrapper(map_name="4x4", is_slippery=True, max_steps=100) for _ in range(prior_cfg.n_envs)]
        )
        agent.update(trajectories)

    dataset = collect_fixed_dataset(env, agent.net, n_episodes=args.n_episodes, seed=args.seed + 1, sample_actions=True)
    return "frozen-lake", dataset.obs_dim, dataset.n_actions, dataset, agent.net.state_dict(), (32, 32)


class RiverSwimWrapper:
    """RiverSwim (Strehl & Littman, 2008; standard parameterization
    following Osband et al., 2013). See module docstring for why this
    testbed is included and its known limitation with a short
    prior-training budget."""

    def __init__(self, n_states: int = 6, max_steps: int = 20):
        from gymnasium import spaces

        self.n_states = n_states
        self.n_actions = 2
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n_states,), dtype=np.float32)
        self._state = 0
        self._t = 0

    def _one_hot(self, state: int) -> np.ndarray:
        v = np.zeros(self.n_states, dtype=np.float32)
        v[state] = 1.0
        return v

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._state = 0
        self._t = 0
        return self._one_hot(self._state), {"state": self._state}

    def step(self, action: int):
        s = self._state
        if action == 0:
            next_state = max(0, s - 1)
            reward = 0.005 if s == 0 else 0.0
        else:
            if s == 0:
                probs = [0.4, 0.6, 0.0]
            elif s == self.n_states - 1:
                probs = [0.65, 0.0, 0.35]
            else:
                probs = [0.60, 0.35, 0.05]
            outcome = np.random.choice(["stay", "right", "left"], p=probs)
            next_state = {"stay": s, "right": s + 1, "left": s - 1}[outcome]
            reward = 1.0 if (s == self.n_states - 1) else 0.0
        self._state = next_state
        self._t += 1
        truncated = self._t >= self.max_steps
        return self._one_hot(self._state), float(reward), False, truncated, {"state": self._state}

    def get_state(self) -> int:
        return self._state

    def close(self):
        pass  # no real gym env to close


def build_river_swim_testbed(args):
    from ppo_exploitation.data.collect import collect_fixed_dataset
    from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
    from ppo_exploitation.utils.config import OnlinePPOConfig
    from ppo_exploitation.utils.seeding import set_global_seed

    set_global_seed(args.seed)
    env = RiverSwimWrapper(n_states=args.river_swim_n_states, max_steps=args.river_swim_max_steps)
    print(f"  [river-swim] training a short online-PPO prior (n_states={args.river_swim_n_states})...")
    prior_cfg = OnlinePPOConfig(
        total_iterations=args.prior_iterations, rollout_steps=512, n_envs=4, epochs=4, minibatch_size=128,
        entropy_coef=0.02, hidden_sizes=(32, 32), eval_every=args.prior_iterations + 1, seed=args.seed,
    )
    agent = OnlinePPOAgent(env.observation_space.shape[0], env.n_actions, prior_cfg)
    for _ in range(prior_cfg.total_iterations):
        trajectories = agent.collect_rollout(
            [lambda: RiverSwimWrapper(n_states=args.river_swim_n_states, max_steps=args.river_swim_max_steps) for _ in range(prior_cfg.n_envs)]
        )
        agent.update(trajectories)

    dataset = collect_fixed_dataset(env, agent.net, n_episodes=args.n_episodes, seed=args.seed + 1, sample_actions=True)
    return "river-swim", dataset.obs_dim, dataset.n_actions, dataset, agent.net.state_dict(), (32, 32)


class TaxiWrapper:
    """gymnasium's Taxi-v3/v4. See module docstring for the return-
    variance finding and the (rejected) decomposed-encoding hypothesis."""

    def __init__(self, max_steps: int = 200, encoding: str = "onehot"):
        import gymnasium as gym
        from gymnasium import spaces

        if encoding not in ("onehot", "decomposed"):
            raise ValueError(f"encoding must be 'onehot' or 'decomposed', got {encoding!r}")
        self.encoding = encoding
        try:
            self._env = gym.make("Taxi-v4", max_episode_steps=max_steps)
        except gym.error.DeprecatedEnv:
            self._env = gym.make("Taxi-v3", max_episode_steps=max_steps)
        self.n_states = int(self._env.observation_space.n)
        self.n_actions = int(self._env.action_space.n)
        self.action_space = spaces.Discrete(self.n_actions)
        obs_dim = self.n_states if encoding == "onehot" else (5 + 5 + 5 + 4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self._state = 0

    def _encode(self, state: int) -> np.ndarray:
        if self.encoding == "onehot":
            v = np.zeros(self.n_states, dtype=np.float32)
            v[state] = 1.0
            return v
        taxi_row, taxi_col, passenger_loc, destination = self._env.unwrapped.decode(state)
        v = np.zeros(19, dtype=np.float32)
        v[taxi_row] = 1.0
        v[5 + taxi_col] = 1.0
        v[10 + passenger_loc] = 1.0
        v[15 + destination] = 1.0
        return v

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        self._state = int(obs)
        return self._encode(self._state), {"state": self._state}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._state = int(obs)
        return self._encode(self._state), float(reward), bool(terminated), bool(truncated), {"state": self._state}

    def get_state(self) -> int:
        return self._state

    def close(self):
        self._env.close()


def build_taxi_testbed(args):
    from ppo_exploitation.data.collect import collect_fixed_dataset
    from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
    from ppo_exploitation.utils.config import OnlinePPOConfig
    from ppo_exploitation.utils.seeding import set_global_seed

    set_global_seed(args.seed)
    env = TaxiWrapper(max_steps=200, encoding=args.taxi_encoding)
    print(f"  [taxi] training a short online-PPO prior (encoding={args.taxi_encoding})...")
    prior_cfg = OnlinePPOConfig(
        total_iterations=args.prior_iterations, rollout_steps=512, n_envs=4, epochs=4, minibatch_size=128,
        entropy_coef=0.02, hidden_sizes=(32, 32), eval_every=args.prior_iterations + 1, seed=args.seed,
    )
    agent = OnlinePPOAgent(env.observation_space.shape[0], env.n_actions, prior_cfg)
    for _ in range(prior_cfg.total_iterations):
        trajectories = agent.collect_rollout(
            [lambda: TaxiWrapper(max_steps=200, encoding=args.taxi_encoding) for _ in range(prior_cfg.n_envs)]
        )
        agent.update(trajectories)

    dataset = collect_fixed_dataset(env, agent.net, n_episodes=args.taxi_n_episodes, seed=args.seed + 1, sample_actions=True)
    return f"taxi ({args.taxi_encoding})", dataset.obs_dim, dataset.n_actions, dataset, agent.net.state_dict(), (32, 32)


class BlackjackWrapper:
    """gymnasium's Blackjack-v1. See module docstring for why episodes
    being 1-3 steps means --n-episodes needs to be much larger here."""

    def __init__(self, max_steps: int = 20):
        import gymnasium as gym
        from gymnasium import spaces

        self._env = gym.make("Blackjack-v1", natural=False, sab=False, max_episode_steps=max_steps)
        self._player_sum_n, self._dealer_n, self._ace_n = 32, 11, 2
        self.n_states = self._player_sum_n * self._dealer_n * self._ace_n
        self.n_actions = int(self._env.action_space.n)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_states,), dtype=np.float32)
        self._state = 0

    def _flatten(self, obs) -> int:
        player_sum, dealer_showing, usable_ace = int(obs[0]), int(obs[1]), int(obs[2])
        return (player_sum * self._dealer_n + dealer_showing) * self._ace_n + usable_ace

    def _one_hot(self, state: int) -> np.ndarray:
        v = np.zeros(self.n_states, dtype=np.float32)
        v[state] = 1.0
        return v

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        self._state = self._flatten(obs)
        return self._one_hot(self._state), {"state": self._state}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._state = self._flatten(obs)
        return self._one_hot(self._state), float(reward), bool(terminated), bool(truncated), {"state": self._state}

    def get_state(self) -> int:
        return self._state

    def close(self):
        self._env.close()


def build_blackjack_testbed(args):
    from ppo_exploitation.data.collect import collect_fixed_dataset
    from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
    from ppo_exploitation.utils.config import OnlinePPOConfig
    from ppo_exploitation.utils.seeding import set_global_seed

    set_global_seed(args.seed)
    env = BlackjackWrapper()
    print("  [blackjack] training a short online-PPO prior...")
    prior_cfg = OnlinePPOConfig(
        total_iterations=args.prior_iterations, rollout_steps=512, n_envs=4, epochs=4, minibatch_size=128,
        entropy_coef=0.02, hidden_sizes=(32, 32), eval_every=args.prior_iterations + 1, seed=args.seed,
    )
    agent = OnlinePPOAgent(env.observation_space.shape[0], env.n_actions, prior_cfg)
    for _ in range(prior_cfg.total_iterations):
        trajectories = agent.collect_rollout([lambda: BlackjackWrapper() for _ in range(prior_cfg.n_envs)])
        agent.update(trajectories)

    dataset = collect_fixed_dataset(env, agent.net, n_episodes=args.n_episodes, seed=args.seed + 1, sample_actions=True)
    return "blackjack", dataset.obs_dim, dataset.n_actions, dataset, agent.net.state_dict(), (32, 32)


TESTBED_BUILDERS = {
    "current-maze": build_current_maze_testbed,
    "new-maze-layout": build_new_maze_layout_testbed,
    "frozen-lake": build_frozen_lake_testbed,
    "river-swim": build_river_swim_testbed,
    "taxi": build_taxi_testbed,
    "blackjack": build_blackjack_testbed,
}
DEFAULT_TESTBEDS = ["current-maze", "new-maze-layout", "frozen-lake", "river-swim", "taxi", "blackjack"]


def _run_one_testbed_worker(testbed_name, args, result_queue):
    try:
        result = _run_one_testbed(testbed_name, args)
        result_queue.put(("ok", result))
    except Exception as e:
        import traceback

        result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))


def _run_one_testbed(testbed_name, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer, effective_sample_weight
    from ppo_exploitation.ppo.fixed_d_trainer_ensemble import compute_thresholded_n_eff, train_ensemble_and_compute_n_eff

    def weight_multiplier(n, beta):
        return effective_sample_weight(n, beta) / n

    label, obs_dim, n_actions, dataset, prior_state_dict, hidden_sizes = TESTBED_BUILDERS[testbed_name](args)
    print(f"  D: {len(dataset)} transitions, {dataset.n_episodes} episodes, obs_dim={obs_dim}, n_actions={n_actions}")

    exact_cfg = PPOHyperparams(
        epochs=1, minibatch_size=args.minibatch_size, hidden_sizes=hidden_sizes,
        use_effective_sample_weighting=True, effective_sample_beta=0.9, seed=args.seed,
    )
    exact_trainer = FixedDPPOTrainer(dataset, obs_dim=obs_dim, n_actions=n_actions, cfg=exact_cfg, prior_state_dict=prior_state_dict)
    exact_n = exact_trainer._n_per_transition.copy()
    _, returns = exact_trainer._compute_advantages_and_returns()

    ensemble_cfg = PPOHyperparams(
        minibatch_size=args.minibatch_size, ensemble_n_heads=args.ensemble_n_heads,
        ensemble_hidden_sizes=tuple(args.ensemble_hidden_sizes), ensemble_epochs=args.ensemble_epochs,
        ensemble_lr=args.ensemble_lr, ensemble_threshold_quantile=args.ensemble_threshold_quantile,
        ensemble_threshold_alpha=args.ensemble_threshold_alpha,
        ensemble_threshold_min_reference_pairs=args.ensemble_threshold_min_reference_pairs,
    )
    print(f"  training the {args.ensemble_n_heads}-head ensemble for {args.ensemble_epochs} epochs...")
    n_eff_no_threshold, variance, _ = train_ensemble_and_compute_n_eff(
        obs_arr=exact_trainer._obs, actions_arr=exact_trainer._actions, returns=returns,
        obs_dim=obs_dim, n_actions=n_actions, cfg=ensemble_cfg,
    )
    n_eff_thresholded, diag = compute_thresholded_n_eff(
        obs_arr=exact_trainer._obs, actions_arr=exact_trainer._actions, variance=variance, cfg=ensemble_cfg
    )
    n_significant = int(diag["pair_significant"].sum()) if diag.get("pair_significant") is not None else None
    skipped_reason = diag.get("skipped_reason")

    spearman_raw, spearman_raw_p = scipy_stats.spearmanr(exact_n, n_eff_no_threshold)
    pearson_raw, pearson_raw_p = scipy_stats.pearsonr(np.log(exact_n), np.log(n_eff_no_threshold))

    weight_exact = weight_multiplier(exact_n, args.weight_beta)
    weight_no_threshold = weight_multiplier(n_eff_no_threshold, args.weight_beta)
    weight_thresholded = weight_multiplier(n_eff_thresholded, args.weight_beta)

    spearman_w_no_thresh, p_w_no_thresh = scipy_stats.spearmanr(weight_exact, weight_no_threshold)
    pearson_w_no_thresh, pp_w_no_thresh = scipy_stats.pearsonr(weight_exact, weight_no_threshold)
    spearman_w_thresh, p_w_thresh = scipy_stats.spearmanr(weight_exact, weight_thresholded)
    pearson_w_thresh, pp_w_thresh = scipy_stats.pearsonr(weight_exact, weight_thresholded)

    print(
        f"  Spearman rho (n_eff vs. count) = {spearman_raw:.3f}   "
        f"weight corr, no threshold: Spearman={spearman_w_no_thresh:.3f} Pearson={pearson_w_no_thresh:.3f}   "
        f"WITH threshold: Spearman={spearman_w_thresh:.3f} Pearson={pearson_w_thresh:.3f}"
    )
    if skipped_reason:
        print(f"  (threshold skipped: {skipped_reason})")
    elif n_significant is not None:
        print(f"  {n_significant}/{diag['n_pairs']} pairs flagged significant")

    # Min-max normalize each weight array to [0, 1] SEPARATELY, for the
    # plot only -- the correlation numbers above are computed on the raw
    # weight multipliers and are unaffected either way (both Spearman and
    # Pearson are invariant to a monotonic/affine rescaling like this).
    # Without it, weight_exact clusters tightly near 1 for most testbeds
    # (most (state, action) pairs' exact counts sit well below the
    # saturation scale 1/(1-beta), so their weight is close to the
    # unsaturated ceiling) while only a few heavily-saturated pairs pull
    # it down -- visually this crams almost every point into a thin strip
    # at the top of the y-axis, hiding the actual shape of the
    # relationship. Rescaling each array to use its own full range makes
    # the same underlying correlation visible instead of buried in a
    # sliver.
    def normalize_for_plot(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    weight_exact_norm = normalize_for_plot(weight_exact)
    weight_no_threshold_norm = normalize_for_plot(weight_no_threshold)
    weight_thresholded_norm = normalize_for_plot(weight_thresholded)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].scatter(exact_n, n_eff_no_threshold, alpha=0.25, s=10)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("exact (state, action) count")
    axes[0].set_ylabel("ensemble n_eff (no threshold)")
    axes[0].set_title(f"Raw signal\nSpearman rho={spearman_raw:.3f}, Pearson (log-log)={pearson_raw:.3f}")

    axes[1].scatter(weight_exact_norm, weight_no_threshold_norm, alpha=0.25, s=10, color="tab:orange")
    axes[1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("weight multiplier from exact count (normalized to [0,1])")
    axes[1].set_ylabel("weight multiplier from n_eff, no threshold (normalized)")
    axes[1].set_title(f"Loss weight, un-gated\nSpearman rho={spearman_w_no_thresh:.3f}, Pearson={pearson_w_no_thresh:.3f}")

    axes[2].scatter(weight_exact_norm, weight_thresholded_norm, alpha=0.25, s=10, color="tab:green")
    axes[2].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("weight multiplier from exact count (normalized to [0,1])")
    axes[2].set_ylabel("weight multiplier from n_eff, thresholded (normalized)")
    axes[2].set_title(f"Loss weight, WITH threshold\nSpearman rho={spearman_w_thresh:.3f}, Pearson={pearson_w_thresh:.3f}")

    fig.suptitle(f"{label}: does ensemble n_eff track the true exact count and its loss-weight behavior?")
    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"{safe_label}_n_eff_correlation.png", dpi=150)
    plt.close(fig)

    return {
        "testbed": label, "n_transitions": len(exact_n),
        "n_pairs": diag.get("n_pairs"), "n_significant_pairs": n_significant, "threshold_skipped_reason": skipped_reason,
        "spearman_n_eff_vs_count": spearman_raw, "spearman_n_eff_vs_count_p": spearman_raw_p,
        "pearson_log_log_n_eff_vs_count": pearson_raw, "pearson_log_log_n_eff_vs_count_p": pearson_raw_p,
        "spearman_weight_no_threshold": spearman_w_no_thresh, "spearman_weight_no_threshold_p": p_w_no_thresh,
        "pearson_weight_no_threshold": pearson_w_no_thresh, "pearson_weight_no_threshold_p": pp_w_no_thresh,
        "spearman_weight_thresholded": spearman_w_thresh, "spearman_weight_thresholded_p": p_w_thresh,
        "pearson_weight_thresholded": pearson_w_thresh, "pearson_weight_thresholded_p": pp_w_thresh,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbeds", nargs="+", choices=list(TESTBED_BUILDERS), default=DEFAULT_TESTBEDS)
    parser.add_argument("--current-maze-dataset", default="results/phase2/dataset_D.pkl")
    parser.add_argument("--current-maze-prior", default="results/phase2/prior_checkpoint.pt")
    parser.add_argument("--new-maze-size", type=int, default=10)
    parser.add_argument("--new-maze-layout-seed", type=int, default=7, help="Anything != 0, the seed used everywhere else in this project.")
    parser.add_argument("--taxi-encoding", choices=["onehot", "decomposed"], default="onehot", help="See TaxiWrapper.")
    parser.add_argument("--river-swim-max-steps", type=int, default=20)
    parser.add_argument(
        "--river-swim-n-states", type=int, default=25,
        help="The original benchmark uses 6, but with only 12 total (state, action) pairs there's little "
        "sparsity gradient left once D is reasonably sized -- 25 is used here instead.",
    )
    parser.add_argument("--prior-iterations", type=int, default=60)
    parser.add_argument(
        "--n-episodes", type=int, default=500,
        help="Episodes of D to collect for the fresh testbeds. Blackjack needs several thousand (short "
        "episodes). Taxi uses its own --taxi-n-episodes instead (see below), not this flag.",
    )
    parser.add_argument(
        "--taxi-n-episodes", type=int, default=500,
        help="Episodes of D to collect for taxi specifically -- kept separate from --n-episodes because "
        "taxi's 500-dim one-hot encoding makes its memory footprint per episode much larger than the other "
        "testbeds (500 at max_steps=200 was the largest that stayed comfortably under 2GB peak in testing; "
        "raise cautiously).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ensemble-n-heads", type=int, default=5)
    parser.add_argument("--ensemble-hidden-sizes", type=int, nargs="+", default=[32, 32])
    parser.add_argument("--ensemble-epochs", type=int, default=30)
    parser.add_argument("--ensemble-lr", type=float, default=1e-3)
    parser.add_argument("--ensemble-threshold-quantile", type=float, default=0.2)
    parser.add_argument("--ensemble-threshold-alpha", type=float, default=0.05)
    parser.add_argument("--ensemble-threshold-min-reference-pairs", type=int, default=20)
    parser.add_argument(
        "--weight-beta", type=float, default=0.9995,
        help="beta used to convert n (exact count or n_eff) into an actual loss-weight multiplier for "
        "the weight-level correlation -- this project's own validated choice by default.",
    )
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--out-dir", default="results/phase2/analysis/n_eff_correlation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    results = []
    for testbed_name in args.testbeds:
        print(f"=== {testbed_name} (in its own subprocess) ===")
        result_queue = ctx.Queue()
        p = ctx.Process(target=_run_one_testbed_worker, args=(testbed_name, args, result_queue))
        p.start()
        status, payload = result_queue.get()
        p.join()
        if status == "error":
            print(f"  FAILED: {payload}")
            results.append({"testbed": testbed_name, "error": payload})
        else:
            results.append(payload)

    summary_df = pd.DataFrame(results)
    summary_path = out_dir / "n_eff_correlation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {summary_path} and one <testbed>_n_eff_correlation.png per testbed in {out_dir}")


if __name__ == "__main__":
    main()
