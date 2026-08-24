"""Tier-0 pipeline validation.

Distinct in purpose from tests/test_*.py (the Tier-1 unit tests, which ask
"is this specific piece of code correct" -- e.g. does value iteration match
a hand-derived closed form). This suite asks a narrower, more mechanical
question: does the FULL pipeline (collect D -> compute pi_D* -> train
fixed-D PPO -> evaluate everything under one protocol) run start to finish,
stay internally consistent, and produce well-formed output on an
environment we did not build ourselves? Every function called below is
imported unmodified from src/ppo_exploitation -- this file adds no new
pipeline logic, only a FrozenLake adapter (frozen_lake_env.py) and
assertions.

Kept small and fast (4x4 map, short training budgets) because this runs on
every push/PR via .github/workflows/tests.yml. It is not trying to produce
a scientifically meaningful exploitation-gap number -- for that, see the
real Tier-1 study (scripts/run_pipeline.sh on the 30x30 maze, configs/*).
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))                        # for frozen_lake_env
sys.path.insert(0, str(_THIS_DIR.parents[1] / "src"))      # for ppo_exploitation

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from frozen_lake_env import FrozenLakeWrapper

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.eval.evaluate import build_gap_report, evaluate_policy, make_neural_act_fn, make_tabular_act_fn
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.ppo.online_agent import OnlinePPOAgent
from ppo_exploitation.reference.experience_optimal import (
    compute_pi_d_star_empirical,
    compute_pi_d_star_true_restricted,
)
from ppo_exploitation.utils.config import OnlinePPOConfig, PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed

GAMMA = 0.99
UNSEEN_PENALTY = -50.0


def _make_env():
    return FrozenLakeWrapper(map_name="4x4", is_slippery=True, max_steps=100)


@pytest.fixture(scope="module")
def prior_agent():
    """A short online PPO run -- enough for D to contain a handful of
    successful (goal-reaching) episodes. Not tuned for strong performance;
    scripts/01_train_prior.py's target-success-rate loop is what the real
    study uses for that. This just needs to not be degenerate."""
    set_global_seed(0)
    probe = _make_env()
    cfg = OnlinePPOConfig(
        total_iterations=60,
        rollout_steps=512,
        n_envs=4,
        epochs=4,
        minibatch_size=128,
        entropy_coef=0.02,
        hidden_sizes=(32, 32),
        eval_episodes=50,
        eval_every=60,  # only evaluate at the end -- keep this fixture fast
        seed=0,
    )
    agent = OnlinePPOAgent(probe.observation_space.shape[0], probe.n_actions, cfg)
    for _ in range(cfg.total_iterations):
        trajectories = agent.collect_rollout([_make_env for _ in range(cfg.n_envs)])
        agent.update(trajectories)
    return agent


@pytest.fixture(scope="module")
def dataset(prior_agent):
    env = _make_env()
    return collect_fixed_dataset(env, prior_agent.net, n_episodes=400, seed=1, sample_actions=True)


def test_prior_collects_some_successful_episodes(dataset):
    """Precondition for everything below: if D never reaches the goal,
    pi_D* is degenerate (V(s0) == unseen_penalty everywhere reachable) and
    the rest of this suite wouldn't be testing anything meaningful."""
    n_goal_episodes = sum(1 for tr in dataset.trajectories if tr.terminated_final and tr.rewards[-1] > 0)
    assert n_goal_episodes >= 1, (
        f"prior policy never reached the goal in {dataset.n_episodes} episodes of D -- "
        f"increase the prior training budget in the `prior_agent` fixture"
    )


def test_pi_d_star_empirical_solves_without_error(dataset):
    ref = compute_pi_d_star_empirical(dataset, gamma=GAMMA, unseen_penalty=UNSEEN_PENALTY)
    assert ref.V.shape == (dataset.n_states,)
    assert ref.policy.shape == (dataset.n_states,)
    assert np.all(np.isfinite(ref.V))
    assert len(ref.covered_states) > 0


def test_pi_d_star_true_restricted_solves_without_error(dataset):
    """Exercises the OTHER reference-solver code path -- the one that reads
    the environment's exact dynamics (FrozenLakeWrapper.true_transition_probs
    / true_reward, sourced from gymnasium's own `env.unwrapped.P`) rather
    than estimating them from D. Not run on the real Tier-1 study's
    FrozenLake stand-in in scripts/03 (that script is maze-specific), but
    the underlying library function is identical."""
    env = _make_env()
    ref = compute_pi_d_star_true_restricted(env, dataset, gamma=GAMMA, unseen_penalty=UNSEEN_PENALTY)
    assert ref.V.shape == (dataset.n_states,)
    assert np.all(np.isfinite(ref.V))


def test_empirical_and_true_restricted_agree_reasonably(dataset):
    """Both pi_D* definitions are solving 'the same' underlying problem
    restricted to D's support; with a non-trivial number of episodes they
    should not be wildly different. This is a soft sanity check, not a
    tight bound -- see README on why these two numbers can legitimately
    diverge when D is sparse."""
    env = _make_env()
    ref_empirical = compute_pi_d_star_empirical(dataset, gamma=GAMMA, unseen_penalty=UNSEEN_PENALTY)
    ref_true = compute_pi_d_star_true_restricted(env, dataset, gamma=GAMMA, unseen_penalty=UNSEEN_PENALTY)
    start_state = 0  # FrozenLake's 'S' tile is always state 0
    assert abs(ref_empirical.V[start_state] - ref_true.V[start_state]) < 0.5


def test_fixed_d_ppo_trains_without_error(dataset, prior_agent):
    cfg = PPOHyperparams(epochs=5, minibatch_size=128, hidden_sizes=(32, 32), seed=0)
    trainer = FixedDPPOTrainer(
        dataset,
        obs_dim=dataset.obs_dim,
        n_actions=dataset.n_actions,
        cfg=cfg,
        prior_state_dict=prior_agent.net.state_dict(),
    )
    history = trainer.train(verbose=False)
    assert len(history) == cfg.epochs
    assert all(np.isfinite(row["policy_loss"]) for row in history)
    assert all(np.isfinite(row["value_loss"]) for row in history)


def test_full_pipeline_produces_well_formed_gap_report(prior_agent, dataset):
    """The end-to-end check: prior, pi_D*, and a fixed-D PPO run, evaluated
    through the SAME evaluate_policy/build_gap_report code the real Tier-1
    study uses, must produce a well-formed table -- finite returns, success
    rates in [0, 1], and (for the tabular policy) a well-formed coverage
    diagnostic rather than a forced-termination penalty."""
    env = _make_env()
    ref = compute_pi_d_star_empirical(dataset, gamma=GAMMA, unseen_penalty=UNSEEN_PENALTY)

    cfg = PPOHyperparams(epochs=5, minibatch_size=128, hidden_sizes=(32, 32), seed=0)
    trainer = FixedDPPOTrainer(
        dataset,
        obs_dim=dataset.obs_dim,
        n_actions=dataset.n_actions,
        cfg=cfg,
        prior_state_dict=prior_agent.net.state_dict(),
    )
    trainer.train(verbose=False)

    results = {
        "prior": evaluate_policy(env, make_neural_act_fn(prior_agent.net, deterministic=True), 100, seed=42),
        "pi_D*_empirical": evaluate_policy(
            env, make_tabular_act_fn(ref), 100, seed=42, covered_states=ref.covered_states
        ),
        "fixed_d_ppo": evaluate_policy(env, make_neural_act_fn(trainer.net, deterministic=True), 100, seed=42),
    }
    for name, res in results.items():
        assert np.isfinite(res["mean_return"]), f"{name} produced a non-finite mean_return"
        assert 0.0 <= res["success_rate"] <= 1.0, f"{name} produced an out-of-range success_rate"
    assert 0.0 <= results["pi_D*_empirical"]["uncovered_state_step_rate"] <= 1.0
    assert 0.0 <= results["pi_D*_empirical"]["uncovered_state_episode_rate"] <= 1.0
    assert "uncovered_state_step_rate" not in results["prior"]  # neural policies don't get this diagnostic

    report = build_gap_report(results, reference_key="pi_D*_empirical")
    assert "gap_vs_reference_return" in report.columns
    assert np.isfinite(report["gap_vs_reference_return"]).all()
    assert np.isfinite(report["gap_vs_reference_success_rate"]).all()