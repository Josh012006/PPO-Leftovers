import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ppo_exploitation.ppo.buffer import FixedDataset, Trajectory
from ppo_exploitation.reference.experience_optimal import compute_pi_d_star_empirical


def _traj(states, actions, rewards, next_states, terminated_final, obs_dim=1):
    n = len(states)
    return Trajectory(
        states=np.array(states, dtype=np.int64),
        obs=np.zeros((n, obs_dim), dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        log_probs=np.zeros(n, dtype=np.float32),
        next_states=np.array(next_states, dtype=np.int64),
        next_obs=np.zeros((n, obs_dim), dtype=np.float32),
        terminated_final=terminated_final,
    )


def test_deterministic_two_state_mdp_matches_hand_solution():
    """state 0 = start, state 1 = terminal goal. action 1 deterministically
    advances 0->1 with reward 1.0 and is visited; action 0 is never visited
    at all. Hand solution: V(0) = 1.0, policy(0) = 1."""
    tr = _traj(states=[0], actions=[1], rewards=[1.0], next_states=[1], terminated_final=True)
    dataset = FixedDataset(trajectories=[tr], n_states=2, n_actions=2, obs_dim=1)

    ref = compute_pi_d_star_empirical(dataset, gamma=0.9, unseen_penalty=-50.0)
    assert ref.policy[0] == 1
    assert abs(ref.V[0] - 1.0) < 1e-6
    assert 1 in ref.terminal_states  # state 1 is the terminal one
    assert 0 in ref.covered_states
    assert 1 not in ref.covered_states  # never a starting state in D


def test_stochastic_self_loop_matches_closed_form_fixed_point():
    """state 0, action 1 leads 50% of the time to terminal state 1 with
    reward +1.0, and 50% of the time back to state 0 (non-terminal) with
    reward -0.1. action 0 is unvisited.

    Closed form: Q(0,1) = 0.5*(1.0 + gamma*0) + 0.5*(-0.1 + gamma*V(0))
    With V(0) = Q(0,1) (since action 0 is penalized into irrelevance):
        V(0) = 0.5 - 0.05 + 0.45*V(0)  =>  V(0) = 0.45 / 0.55
    """
    gamma = 0.9
    tr_a = _traj(states=[0], actions=[1], rewards=[1.0], next_states=[1], terminated_final=True)
    tr_b = _traj(states=[0], actions=[1], rewards=[-0.1], next_states=[0], terminated_final=False)
    dataset = FixedDataset(trajectories=[tr_a, tr_b], n_states=2, n_actions=2, obs_dim=1)

    ref = compute_pi_d_star_empirical(dataset, gamma=gamma, unseen_penalty=-50.0, theta=1e-10, max_iter=200_000)
    expected_v0 = 0.45 / 0.55
    assert abs(ref.V[0] - expected_v0) < 1e-4
    assert ref.policy[0] == 1


def test_unseen_action_is_dominated_when_alternative_exists():
    """If action 0 leads to a mediocre-but-visited outcome and action 1 is
    completely unvisited, the greedy policy must pick action 0 (the visited
    one), never the unseen one, provided unseen_penalty is very negative."""
    tr = _traj(states=[0], actions=[0], rewards=[-0.05], next_states=[1], terminated_final=True)
    dataset = FixedDataset(trajectories=[tr], n_states=2, n_actions=2, obs_dim=1)
    ref = compute_pi_d_star_empirical(dataset, gamma=0.9, unseen_penalty=-50.0)
    assert ref.policy[0] == 0
    assert abs(ref.V[0] - (-0.05)) < 1e-6


def test_act_returns_none_for_fully_uncovered_state():
    tr = _traj(states=[0], actions=[1], rewards=[1.0], next_states=[1], terminated_final=True)
    dataset = FixedDataset(trajectories=[tr], n_states=3, n_actions=2, obs_dim=1)  # state 2 never appears at all
    ref = compute_pi_d_star_empirical(dataset, gamma=0.9, unseen_penalty=-50.0)
    assert ref.act(2) is None
    assert ref.act(0) == 1
