"""Tier-0 validation environment: gymnasium's FrozenLake-v1, adapted to the
minimal interface the fixed-D pipeline expects (get_state(), n_states,
n_actions, observation_space, is_terminal_state, true_transition_probs,
true_reward).

This file is deliberately kept OUT of src/ppo_exploitation. FrozenLake
exists here for exactly one purpose: answering "does the D -> pi_D* ->
fixed-D-PPO -> evaluate pipeline actually work on an environment we did NOT
build ourselves, with dynamics we did NOT hand-design?" It is not part of
the research library and is not meant to be imported by anything under
src/. If validating this ever required special-casing pipeline code for
FrozenLake, that would itself be a sign the pipeline had become overfit to
the custom maze -- so far it hasn't needed any.

FrozenLake's own observation IS already the ground-truth integer state
(Discrete(n_states)), so unlike the custom maze there is no real
generalization gap between "ground truth" and "what PPO sees" here -- we
one-hot encode it purely so the existing ActorCritic network (which expects
a continuous vector) can consume it unmodified. That's fine: Tier 0 is
about mechanical correctness, not about being a hard learning problem.

The exact transition/reward model is read directly from gymnasium's own
`env.unwrapped.P` (a dict `P[state][action] -> [(prob, next_state, reward,
terminated), ...]`), which FrozenLakeEnv exposes natively -- this avoids
hand-deriving FrozenLake's slip convention (which direction is "left/right"
of the intended action, etc.) and any risk of getting it subtly wrong.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FrozenLakeWrapper:
    def __init__(self, map_name: str = "4x4", is_slippery: bool = True, max_steps: int = 100):
        self._env = gym.make(
            "FrozenLake-v1", map_name=map_name, is_slippery=is_slippery, max_episode_steps=max_steps
        )
        desc = self._env.unwrapped.desc  # grid of bytes: b'S' start, b'F' frozen, b'H' hole, b'G' goal
        self.n_states = int(self._env.observation_space.n)
        self.n_actions = int(self._env.action_space.n)
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_states,), dtype=np.float32)

        flat = [c.decode("utf-8") if isinstance(c, bytes) else c for row in desc for c in row]
        self._terminal_states = {i for i, c in enumerate(flat) if c in ("H", "G")}

        self._state = 0

    def _one_hot(self, state: int) -> np.ndarray:
        v = np.zeros(self.n_states, dtype=np.float32)
        v[state] = 1.0
        return v

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._state = int(obs)
        return self._one_hot(self._state), {"state": self._state}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(int(action))
        self._state = int(obs)
        return self._one_hot(self._state), float(reward), bool(terminated), bool(truncated), {"state": self._state}

    def get_state(self) -> int:
        return self._state

    def is_terminal_state(self, state: int) -> bool:
        return state in self._terminal_states

    # ------------------------------------------------------------------
    # Exact dynamics, read directly from gymnasium -- used for the
    # true-dynamics, support-restricted pi_D* definition (see
    # reference/experience_optimal.py). No hand-derived slip model here.
    # ------------------------------------------------------------------
    def true_transition_probs(self, state: int, action: int) -> dict[int, float]:
        if self.is_terminal_state(state):
            return {state: 1.0}
        outcomes = self._env.unwrapped.P[state][action]
        probs: dict[int, float] = {}
        for prob, next_state, _reward, _terminated in outcomes:
            probs[next_state] = probs.get(next_state, 0.0) + prob
        return probs

    def true_reward(self, state: int, action: int, next_state: int) -> float:
        if self.is_terminal_state(state):
            return 0.0
        for prob, ns, reward, _terminated in self._env.unwrapped.P[state][action]:
            if ns == next_state:
                return float(reward)
        # Only reachable if called with a (s, a, s') triple gymnasium's own
        # model says is impossible -- surface that loudly rather than
        # silently returning 0, since it would indicate a real mismatch.
        raise ValueError(f"(state={state}, action={action}, next_state={next_state}) is not a reachable transition")