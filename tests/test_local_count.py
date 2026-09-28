"""Tests for the product-kernel LOCAL count
(ppo/fixed_d_trainer_local_count.py). Properties that matter: (1) both
bandwidths -> 0 reproduces the exact count exactly; (2) matches an
independent brute-force product-kernel sum, for both kernels, for several
(bandwidth_state, bandwidth_action) pairs, including MULTI-DIMENSIONAL
continuous actions; (3) never falls below the exact count and only grows
with either bandwidth; (4) either bandwidth alone can be driven to ~0 while
the other stays positive (exact state match with smoothed actions, or
vice-versa). Mirrors tests/test_ensemble_uncertainty.py's style and scope.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.fixed_d_trainer_local_count import (
    LocalCountFixedDPPOTrainer,
    _standardize_by_distinct_rows,
    compute_local_counts,
    nearest_neighbor_scale,
)
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


def _brute_force_product_kernel(obs, actions, hs, ha, kernel):
    zs = _standardize_by_distinct_rows(obs)
    za = _standardize_by_distinct_rows(actions)
    n = len(zs)
    out = np.zeros(n)
    for i in range(n):
        ds = np.linalg.norm(zs - zs[i], axis=1)
        da = np.linalg.norm(za - za[i], axis=1)
        if kernel == "gaussian":
            w = np.exp(-0.5 * (ds / hs) ** 2) * np.exp(-0.5 * (da / ha) ** 2)
        else:
            w = (ds <= hs).astype(float) * (da <= ha).astype(float)
        out[i] = w.sum()
    return out


def _exact_count(obs, actions_2d):
    key = np.concatenate([obs, actions_2d], axis=1)
    n = len(key)
    return np.array([np.all(key == key[i], axis=1).sum() for i in range(n)], dtype=float)


def _synthetic_discrete_action(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(50, 3))
    obs = np.concatenate([base, base[:20], base[:5]])  # exact duplicates -> multiplicity > 1
    actions = rng.integers(0, 3, size=len(obs))
    return obs, actions


def _synthetic_continuous_action(seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(60, 4))
    obs = np.concatenate([base, base[:15]])
    actions = rng.normal(size=(len(obs), 2))  # 2-D continuous action
    return obs, actions


# --------------------------------------------------------------------------
# the core computation
# --------------------------------------------------------------------------
def test_both_bandwidths_tiny_reproduces_the_exact_count_discrete_action():
    obs, actions = _synthetic_discrete_action()
    exact = _exact_count(obs, actions.reshape(-1, 1))
    local = compute_local_counts(obs, actions, bandwidth_state=1e-9, bandwidth_action=1e-9)
    assert np.array_equal(local, exact)


def test_both_bandwidths_tiny_reproduces_the_exact_count_continuous_action():
    obs, actions = _synthetic_continuous_action()
    exact = _exact_count(obs, actions)
    local = compute_local_counts(obs, actions, bandwidth_state=1e-9, bandwidth_action=1e-9)
    assert np.array_equal(local, exact)


@pytest.mark.parametrize("kernel", ["gaussian", "ball"])
@pytest.mark.parametrize("hs,ha", [(0.3, 0.5), (0.6, 0.2), (0.2, 0.2)])
def test_matches_independent_brute_force_product_kernel(kernel, hs, ha):
    obs, actions = _synthetic_continuous_action()
    got = compute_local_counts(obs, actions, hs, ha, kernel=kernel, cutoff_sigmas=60.0)
    ref = _brute_force_product_kernel(obs, actions, hs, ha, kernel)
    assert np.allclose(got, ref, atol=1e-9), f"max abs diff {np.abs(got - ref).max():.2e}"


def test_exact_action_match_with_smoothed_state():
    # bandwidth_action -> 0 should recover EXACT action matching regardless
    # of how large bandwidth_state is: each row's count must equal a sum
    # restricted to OTHER rows sharing the same action, weighted by the
    # (unrestricted) state kernel.
    obs, actions = _synthetic_discrete_action()
    local = compute_local_counts(obs, actions, bandwidth_state=5.0, bandwidth_action=1e-9)
    zs = _standardize_by_distinct_rows(obs)
    for a in np.unique(actions):
        idx = np.where(actions == a)[0]
        ds = np.linalg.norm(zs[idx][:, None, :] - zs[idx][None, :, :], axis=-1)
        ref = np.exp(-0.5 * (ds / 5.0) ** 2).sum(axis=1)
        assert np.allclose(local[idx], ref, atol=1e-6), "smoothing must never pool across different actions when bandwidth_action -> 0"


def test_exact_state_match_with_smoothed_action():
    # bandwidth_state -> 0 should recover EXACT state matching regardless of
    # how large bandwidth_action is: each row's count must equal a sum
    # restricted to OTHER rows sharing the exact same state, weighted by the
    # (unrestricted) action kernel.
    obs, actions = _synthetic_continuous_action()
    local = compute_local_counts(obs, actions, bandwidth_state=1e-9, bandwidth_action=5.0)
    za = _standardize_by_distinct_rows(actions)
    _, groups = np.unique(obs, axis=0, return_inverse=True)
    groups = np.asarray(groups).reshape(-1)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        da = np.linalg.norm(za[idx][:, None, :] - za[idx][None, :, :], axis=-1)
        ref = np.exp(-0.5 * (da / 5.0) ** 2).sum(axis=1)
        assert np.allclose(local[idx], ref, atol=1e-6)


def test_never_below_exact_and_monotone_in_each_bandwidth():
    obs, actions = _synthetic_continuous_action()
    exact = compute_local_counts(obs, actions, 1e-9, 1e-9)
    assert np.all(exact >= 1)
    prev = exact
    for hs, ha in [(0.1, 0.1), (0.3, 0.2), (0.6, 0.5), (1.2, 1.0)]:
        cur = compute_local_counts(obs, actions, hs, ha)
        assert np.all(cur >= exact - 1e-9)
        assert np.all(cur >= prev - 1e-9), f"local count decreased at (hs={hs}, ha={ha})"
        prev = cur


def test_nearest_neighbor_scale_two_points():
    obs = np.array([[0.0], [1.0], [1.0], [0.0]])
    assert nearest_neighbor_scale(obs) == pytest.approx(2.0)


def test_rejects_bad_arguments():
    obs, actions = _synthetic_discrete_action()
    with pytest.raises(ValueError):
        compute_local_counts(obs, actions, bandwidth_state=0.0, bandwidth_action=0.2)
    with pytest.raises(ValueError):
        compute_local_counts(obs, actions, bandwidth_state=0.2, bandwidth_action=-1.0)
    with pytest.raises(ValueError):
        compute_local_counts(obs, actions, bandwidth_state=0.5, bandwidth_action=0.5, kernel="triangle")
    with pytest.raises(ValueError):
        compute_local_counts(obs, actions[:-1], bandwidth_state=0.5, bandwidth_action=0.5)


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
            prior_state_dict=prior_net.state_dict(), bandwidth_state=0.2, bandwidth_action=0.2,
        )


def test_trainer_rejects_bad_bandwidths_and_kernel():
    _, prior_net, dataset = _make_dataset_and_prior(n_episodes=5)
    bad = (
        {"bandwidth_state": 0.0, "bandwidth_action": 0.2},
        {"bandwidth_state": 0.2, "bandwidth_action": -1.0},
        {"bandwidth_state": 0.2, "bandwidth_action": 0.2, "kernel": "triangle"},
    )
    for kwargs in bad:
        with pytest.raises(ValueError):
            LocalCountFixedDPPOTrainer(
                dataset, dataset.obs_dim, dataset.n_actions, _cfg(), prior_state_dict=prior_net.state_dict(), **kwargs
            )


def test_trainer_tiny_bandwidths_is_exactly_the_exact_count_trainer():
    set_global_seed(0)
    _, prior_net, dataset = _make_dataset_and_prior()
    trainer = LocalCountFixedDPPOTrainer(
        dataset, dataset.obs_dim, dataset.n_actions, _cfg(), prior_state_dict=prior_net.state_dict(),
        bandwidth_state=1e-9, bandwidth_action=1e-9,
    )
    assert np.array_equal(trainer._n_per_transition, trainer.exact_n_per_transition.astype(np.float64)), (
        "at bandwidth_state, bandwidth_action -> 0 the trainer must consume EXACTLY the exact count"
    )


def test_trainer_larger_bandwidths_pool_and_training_stays_finite():
    set_global_seed(1)
    _, prior_net, dataset = _make_dataset_and_prior(n_episodes=80, seed=2)
    trainer = LocalCountFixedDPPOTrainer(
        dataset, dataset.obs_dim, dataset.n_actions, _cfg(), prior_state_dict=prior_net.state_dict(),
        bandwidth_state=0.8, bandwidth_action=1e-9,  # smooth states, keep actions exact (discrete maze actions)
    )
    exact = trainer.exact_n_per_transition.astype(np.float64)
    local = trainer._n_per_transition
    assert local.shape == exact.shape and np.all(np.isfinite(local))
    assert np.all(local >= exact - 1e-9)
    assert (local > exact + 1e-9).any(), "a wide state kernel should pool at least some neighboring states"

    for k, v in prior_net.state_dict().items():  # theta still starts at pi_beta (inherited design)
        assert torch.equal(v, trainer.net.state_dict()[k])

    history = trainer.train(verbose=False)
    assert len(history) == 3
    for row in history:
        for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
            assert np.isfinite(row[key]), f"non-finite {key} in {row}"
