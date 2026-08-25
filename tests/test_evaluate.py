"""Tests for eval/evaluate.py, in particular the uncovered-state behavior:
a tabular policy must never be forced to terminate or penalized a second
time for landing on a state D didn't cover -- it always takes a real
action and the real environment continues, with the frequency of that
happening surfaced as a diagnostic instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.eval.evaluate import build_gap_report, evaluate_policy, make_tabular_act_fn
from ppo_exploitation.ppo.buffer import FixedDataset, Trajectory
from ppo_exploitation.reference.experience_optimal import compute_pi_d_star_empirical
import numpy as np


def test_evaluate_policy_never_forces_termination_on_uncovered_state():
    """Build a deliberately tiny/sparse D on the real maze (a handful of
    steps from the start only), so most states are guaranteed uncovered,
    then confirm a live rollout runs full episodes (reaching max_steps or a
    true terminal) rather than being cut short the moment it wanders off
    D's covered states."""
    env = StochasticMazeEnv(width=6, height=6, slip_prob=0.0, num_hazards=0, max_steps=40, layout_seed=0)
    obs, info = env.reset(seed=0)
    states, obss, actions, rewards, next_states, next_obss = [], [], [], [], [], []
    # Two forced steps only -- almost the entire maze stays uncovered.
    for a in [3, 1]:  # RIGHT, DOWN (whichever these resolve to on this layout)
        s = env.get_state()
        next_obs, r, terminated, truncated, nfo = env.step(a)
        states.append(s)
        obss.append(obs)
        actions.append(a)
        rewards.append(r)
        next_states.append(nfo["state"])
        next_obss.append(next_obs)
        obs = next_obs
        if terminated or truncated:
            break

    tr = Trajectory(
        states=np.array(states, dtype=np.int64),
        obs=np.array(obss, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        log_probs=np.zeros(len(states), dtype=np.float32),
        next_states=np.array(next_states, dtype=np.int64),
        next_obs=np.array(next_obss, dtype=np.float32),
        terminated_final=False,
    )
    dataset = FixedDataset(trajectories=[tr], n_states=env.n_states, n_actions=env.n_actions, obs_dim=8)
    ref = compute_pi_d_star_empirical(dataset, gamma=0.99, unseen_penalty=-50.0)

    result = evaluate_policy(
        env, make_tabular_act_fn(ref), n_episodes=10, seed=1, covered_states=ref.covered_states
    )
    # With such sparse coverage and max_steps=40, episodes should run to
    # the time limit rather than being cut short at the first uncovered
    # state -- mean_length should be well above the 2-step covered window.
    assert result["mean_length"] > 5
    # And the diagnostic should confirm most of the rollout was, in fact,
    # spent on uncovered states -- this IS the honest signal now, not a
    # forced penalty.
    assert result["uncovered_state_step_rate"] > 0.35


def test_evaluate_policy_reports_no_coverage_diagnostic_without_covered_states():
    env = StochasticMazeEnv(width=5, height=5, num_hazards=1, max_steps=30, layout_seed=0)

    def always_action_0(obs, state):
        return 0

    result = evaluate_policy(env, always_action_0, n_episodes=5, seed=0)
    assert "uncovered_state_step_rate" not in result
    assert "uncovered_state_episode_rate" not in result


def test_success_rate_stderr_matches_bernoulli_formula():
    """SE(p_hat) = sqrt(p_hat * (1 - p_hat) / n) for a Bernoulli proportion
    -- documented here as an explicit, checkable claim rather than left
    implicit in eval/evaluate.py's implementation."""
    env = StochasticMazeEnv(width=6, height=6, slip_prob=0.0, num_hazards=1, max_steps=30, layout_seed=0)

    def always_action_0(obs, state):
        return 0

    result = evaluate_policy(env, always_action_0, n_episodes=200, seed=0)
    p = result["success_rate"]
    n = result["n_episodes"]
    expected_stderr = (p * (1 - p) / n) ** 0.5
    assert np.isclose(result["success_rate_stderr"], expected_stderr, atol=1e-12)


def test_build_gap_report_includes_coverage_columns_only_when_present():
    results_with = {
        "pi_D*_empirical": {
            "mean_return": 1.0, "stderr_return": 0.1, "success_rate": 1.0, "success_rate_stderr": 0.0,
            "mean_length": 10.0, "uncovered_state_step_rate": 0.2, "uncovered_state_episode_rate": 0.3,
        },
        "ppo": {
            "mean_return": 0.8, "stderr_return": 0.1, "success_rate": 0.8, "success_rate_stderr": 0.02,
            "mean_length": 12.0,
        },
    }
    report = build_gap_report(results_with, reference_key="pi_D*_empirical")
    assert "uncovered_state_step_rate" in report.columns
    assert "success_rate_stderr" in report.columns
    assert report.loc["pi_D*_empirical", "uncovered_state_step_rate"] == 0.2
    assert report.loc["ppo", "uncovered_state_step_rate"] is None or np.isnan(report.loc["ppo", "uncovered_state_step_rate"])

    results_without = {
        "pi_D*_empirical": {
            "mean_return": 1.0, "stderr_return": 0.1, "success_rate": 1.0, "success_rate_stderr": 0.0,
            "mean_length": 10.0,
        },
        "ppo": {
            "mean_return": 0.8, "stderr_return": 0.1, "success_rate": 0.8, "success_rate_stderr": 0.02,
            "mean_length": 12.0,
        },
    }
    report2 = build_gap_report(results_without, reference_key="pi_D*_empirical")
    assert "uncovered_state_step_rate" not in report2.columns
    assert "success_rate_stderr" in report2.columns