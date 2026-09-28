"""Tier-1 tests for the kernel-smoothed LOCAL count
(ppo/fixed_d_trainer_local_count.py). The properties that matter, and that
the whole approach rests on: (1) it is a strict GENERALIZATION of the exact
count -- bandwidth -> 0 reproduces it exactly; (2) it equals an independent
brute-force kernel sum; (3) it never falls below the exact count and only
grows with bandwidth; (4) actions are matched exactly, only states are
smoothed. Mirrors tests/test_ensemble_uncertainty.py's style and scope.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.fixed_d_trainer_local_count import (
    LocalCountFixedDPPOTrainer,
    _standardize_by_distinct_states,
    compute_local_counts,
    nearest_neighbor_scale,
)
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


def _brute_force(obs, actions, h, kernel):
    z = _standardize_by_distinct_states(obs)
    out = np.zeros(len(z))
    for i in range(len(z)):
        d = np.linalg.norm(z - z[i], axis=1)
        w = np.exp(-0.5 * (d / h) ** 2) if kernel == "gaussian" else (d <= h).astype(float)
        out[i] = w[actions == actions[i]].sum()
    return out


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(50, 3))
    obs = np.concatenate([base, base[:20], base[:5]])  # exact duplicates -> multiplicity > 1
    actions = rng.integers(0, 3, size=len(obs))
    return obs, actions


# --------------------------------------------------------------------------
# the core computation
# --------------------------------------------------------------------------
def test_tiny_bandwidth_reproduces_the_exact_count():
    obs, actions = _synthetic()
    exact_counts = Counter(zip(map(bytes, np.ascontiguousarray(obs).view(np.uint8).reshape(len(obs), -1)), actions.tolist()))
    exact = np.array([exact_counts[(bytes(np.ascontiguousarray(o).view(np.uint8)), int(a))] for o, a in zip(obs, actions)], dtype=float)
    local = compute_local_counts(obs, actions, bandwidth=1e-9)
    assert np.array_equal(local, exact), "h -> 0 must give EXACTLY the (state, action) count"


@pytest.mark.parametrize("kernel", ["gaussian", "ball"])
@pytest.mark.parametrize("h", [0.3, 1.0])
def test_matches_independent_brute_force_kernel_sum(kernel, h):
    obs, actions = _synthetic()
    got = compute_local_counts(obs, actions, h, kernel=kernel, cutoff_sigmas=60.0)  # cutoff so large nothing is truncated
    ref = _brute_force(obs, actions, h, kernel)
    assert np.allclose(got, ref, atol=1e-9), f"max abs diff {np.abs(got - ref).max():.2e}"


def test_never_below_exact_and_monotone_in_bandwidth():
    obs, actions = _synthetic()
    exact = compute_local_counts(obs, actions, 1e-9)
    prev = exact
    for h in (0.1, 0.3, 0.6, 1.2):
        cur = compute_local_counts(obs, actions, h)
        assert np.all(cur >= exact - 1e-9), "local count fell below the exact count"
        assert np.all(cur >= prev - 1e-9), f"local count decreased when bandwidth grew to {h}"
        prev = cur
    assert np.all(exact >= 1)  # every transition at least counts itself


def test_actions_are_matched_exactly_not_smoothed():
    obs = np.array([[0.0, 0.0], [0.0, 0.0], [0.05, 0.0], [1.0, 1.0]])
    actions = np.array([0, 1, 1, 0])
    # rows 0 and 1 share a state but differ in action; with a wide kernel they
    # must still NOT pool with each other (different action), only rows 1 and 2 pool.
    local = compute_local_counts(obs, actions, bandwidth=10.0, kernel="ball")
    assert local[0] == 2.0  # itself + row 3 (same action 0); NOT row 1
    assert local[1] == 2.0  # itself + row 2 (same action 1); NOT row 0
    assert local[2] == 2.0


def test_nearest_neighbor_scale_two_points():
    # two distinct points -> standardized to +-1 -> distance 2
    obs = np.array([[0.0], [1.0], [1.0], [0.0]])
    assert nearest_neighbor_scale(obs) == pytest.approx(2.0)


def test_rejects_bad_arguments():
    obs, actions = _synthetic()
    with pytest.raises(ValueError):
        compute_local_counts(obs, actions, bandwidth=0.0)
    with pytest.raises(ValueError):
        compute_local_counts(obs, actions, bandwidth=-1.0)
    with pytest.raises(ValueError):
        compute_local_counts(obs, actions, bandwidth=0.5, kernel="triangle")
    with pytest.raises(ValueError):
        compute_local_counts(obs, actions[:-1], bandwidth=0.5)


# --------------------------------------------------------------------------
# the trainer, on the project's own (small) maze
# --------------------------------------------------------------------------
def _make_dataset_and_prior(n_episodes=60, seed=0):
    env = StochasticMazeEnv(width=5, height=5, slip_prob=0.1, num_hazards=2, max_steps=80, layout_seed=0)
    prior_net = ActorCritic(env.observation_space.shape[0], env.n_actions, hidden_sizes=(16, 16))
    dataset = collect_fixed_dataset(env, prior_net, n_episodes=n_episodes, seed=seed, sample_actions=True)
    return env, prior_net, dataset


def _cfg(**over):
    kw = dict(
        epochs=3, minibatch_size=64, hidden_sizes=(16, 16), use_effective_sample_weighting=True,
        effective_sample_beta=0.9, effective_sample_kl_anchor=True, effective_sample_kl_k=0.1,
    )
    kw.update(over)
    return PPOHyperparams(**kw)


def test_trainer_requires_effective_sample_weighting_flag():
    _, prior_net, dataset = _make_dataset_and_prior(n_episodes=5)
    with pytest.raises(ValueError):
        LocalCountFixedDPPOTrainer(
            dataset, dataset.obs_dim, dataset.n_actions, _cfg(use_effective_sample_weighting=False),
            prior_state_dict=prior_net.state_dict(), bandwidth=0.2,
        )


def test_trainer_rejects_bad_bandwidth_and_kernel():
    _, prior_net, dataset = _make_dataset_and_prior(n_episodes=5)
    for kwargs in ({"bandwidth": 0.0}, {"bandwidth": -1.0}, {"bandwidth": 0.2, "kernel": "triangle"}):
        with pytest.raises(ValueError):
            LocalCountFixedDPPOTrainer(
                dataset, dataset.obs_dim, dataset.n_actions, _cfg(), prior_state_dict=prior_net.state_dict(), **kwargs
            )


def test_trainer_tiny_bandwidth_is_exactly_the_exact_count_trainer():
    set_global_seed(0)
    _, prior_net, dataset = _make_dataset_and_prior()
    trainer = LocalCountFixedDPPOTrainer(
        dataset, dataset.obs_dim, dataset.n_actions, _cfg(), prior_state_dict=prior_net.state_dict(), bandwidth=1e-9
    )
    assert np.array_equal(trainer._n_per_transition, trainer.exact_n_per_transition.astype(np.float64)), (
        "at h -> 0 the trainer must consume EXACTLY the exact count -- otherwise this class is not a "
        "generalization of the already-validated mechanism"
    )


def test_trainer_larger_bandwidth_pools_and_training_stays_finite():
    set_global_seed(1)
    _, prior_net, dataset = _make_dataset_and_prior(n_episodes=80, seed=2)
    trainer = LocalCountFixedDPPOTrainer(
        dataset, dataset.obs_dim, dataset.n_actions, _cfg(), prior_state_dict=prior_net.state_dict(), bandwidth=0.8
    )
    exact = trainer.exact_n_per_transition.astype(np.float64)
    local = trainer._n_per_transition
    assert local.shape == exact.shape and np.all(np.isfinite(local))
    assert np.all(local >= exact - 1e-9)
    assert (local > exact + 1e-9).any(), "a wide kernel should pool at least some neighboring states"

    for k, v in prior_net.state_dict().items():  # theta still starts at pi_beta (inherited design)
        assert torch.equal(v, trainer.net.state_dict()[k])

    history = trainer.train(verbose=False)
    assert len(history) == 3
    for row in history:
        for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
            assert np.isfinite(row[key]), f"non-finite {key} in {row}"
