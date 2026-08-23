import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv, ACTIONS


def _bfs_reachable(env: StochasticMazeEnv) -> set[int]:
    start = env.layout.state_id(*env.layout.start)
    q = deque([start])
    seen = {start}
    while q:
        s = q.popleft()
        r, c = env.layout.rc(s)
        for a in ACTIONS:
            if not env.layout.open_walls[r, c, a]:
                continue
            from ppo_exploitation.envs.stochastic_maze import _ACTION_DELTA

            dr, dc = _ACTION_DELTA[a]
            nr, nc = r + dr, c + dc
            ns = env.layout.state_id(nr, nc)
            if ns not in seen:
                seen.add(ns)
                q.append(ns)
    return seen


@pytest.mark.parametrize("layout_seed", [0, 1, 2, 7, 42])
def test_maze_start_can_reach_goal(layout_seed):
    env = StochasticMazeEnv(width=12, height=12, num_hazards=6, layout_seed=layout_seed)
    reachable = _bfs_reachable(env)
    goal_state = env.layout.state_id(*env.layout.goal)
    assert goal_state in reachable, f"goal unreachable from start for layout_seed={layout_seed}"


@pytest.mark.parametrize("layout_seed", [0, 3, 9])
def test_maze_fully_connected(layout_seed):
    """The DFS spanning tree alone guarantees full connectivity; extra
    connections and hazards must not break it."""
    env = StochasticMazeEnv(width=10, height=10, num_hazards=5, layout_seed=layout_seed)
    reachable = _bfs_reachable(env)
    assert len(reachable) == env.n_states


def test_true_transition_probs_sum_to_one():
    env = StochasticMazeEnv(width=8, height=8, slip_prob=0.15, num_hazards=4, layout_seed=1)
    for s in range(env.n_states):
        if env.is_terminal_state(s):
            continue
        for a in ACTIONS:
            probs = env.true_transition_probs(s, a)
            total = sum(probs.values())
            assert abs(total - 1.0) < 1e-9, f"state={s} action={a} probs sum to {total}"


def test_true_transition_probs_deterministic_when_no_slip():
    env = StochasticMazeEnv(width=8, height=8, slip_prob=0.0, num_hazards=4, layout_seed=2)
    for s in range(env.n_states):
        if env.is_terminal_state(s):
            continue
        for a in ACTIONS:
            probs = env.true_transition_probs(s, a)
            assert len(probs) == 1, f"expected deterministic transition, got {probs}"
            assert abs(next(iter(probs.values())) - 1.0) < 1e-9


def test_step_reward_matches_true_reward():
    """Sample a bunch of live steps and check the reward returned by
    env.step matches env.true_reward(s, a, s') exactly -- these two paths
    must never disagree, since true_reward is what the reference solver
    relies on."""
    env = StochasticMazeEnv(width=10, height=10, slip_prob=0.2, num_hazards=6, layout_seed=3, max_steps=500)
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(500):
        s = env.get_state()
        a = int(rng.integers(0, 4))
        obs, reward, terminated, truncated, info = env.step(a)
        sp = info["state"]
        assert abs(reward - env.true_reward(s, a, sp)) < 1e-9
        if terminated or truncated:
            obs, info = env.reset(seed=int(rng.integers(0, 10_000)))


def test_observation_bounds_and_shape():
    env = StochasticMazeEnv(width=6, height=6, num_hazards=2, layout_seed=4)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)
    for a in ACTIONS:
        obs2, *_ = env.step(a)
        assert env.observation_space.contains(obs2)


def test_wall_blocks_movement():
    """Taking an action into a wall must leave the agent's state unchanged."""
    env = StochasticMazeEnv(width=6, height=6, slip_prob=0.0, num_hazards=0, layout_seed=5)
    env.reset(seed=0)
    r, c = env.layout.start
    for a in ACTIONS:
        if not env.layout.open_walls[r, c, a]:
            env.set_state(env.layout.state_id(r, c))
            before = env.get_state()
            env.step(a)
            after = env.get_state()
            assert before == after
            return
    pytest.skip("start cell had no walls to test against (unlikely but not impossible)")
