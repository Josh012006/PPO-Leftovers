"""Tier-1 library smoke tests for the ensemble-based confidence signal
(ppo/networks.py:CriticEnsemble and ppo/fixed_d_trainer_ensemble.py:
EnsembleFixedDPPOTrainer) -- see that module's docstring for the full
rationale. Mirrors tests/test_fixed_d_trainer.py's style and scope: checks
the mechanics run correctly and produce numerically sane output, not
whether the resulting n_eff is a GOOD confidence signal in any absolute
sense (that's the calibration study this class's docstring explicitly
defers to later, comparing n_eff against fixed_d_trainer.py's exact
counts on this project's own maze).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.fixed_d_trainer_ensemble import (
    EnsembleFixedDPPOTrainer,
    train_ensemble_and_compute_n_eff,
)
from ppo_exploitation.ppo.networks import ActorCritic, CriticEnsemble
from ppo_exploitation.utils.config import PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


# --------------------------------------------------------------------------
# CriticEnsemble (the network itself)
# --------------------------------------------------------------------------
def test_critic_ensemble_forward_shape():
    set_global_seed(0)
    ensemble = CriticEnsemble(obs_dim=6, n_actions=4, hidden_sizes=(16, 16), n_heads=5)
    obs = torch.randn(10, 6)
    out = ensemble(obs)
    assert out.shape == (5, 10, 4), f"expected (n_heads=5, batch=10, n_actions=4), got {tuple(out.shape)}"


def test_critic_ensemble_heads_are_independent_at_init():
    """Independent initialization is the whole source of diversity here
    (see module docstring: no data sub-sampling) -- so two heads should
    not, in general, produce identical output right after construction."""
    set_global_seed(0)
    ensemble = CriticEnsemble(obs_dim=6, n_actions=4, hidden_sizes=(16, 16), n_heads=5)
    obs = torch.randn(3, 6)
    with torch.no_grad():
        out = ensemble(obs)
    head0, head1 = out[0], out[1]
    assert not torch.allclose(head0, head1), "two independently-initialized heads produced identical output"


def test_critic_ensemble_rejects_fewer_than_two_heads():
    with pytest.raises(ValueError):
        CriticEnsemble(obs_dim=4, n_actions=2, hidden_sizes=(8, 8), n_heads=1)


# --------------------------------------------------------------------------
# train_ensemble_and_compute_n_eff (the training + n_eff extraction logic,
# tested on small synthetic data so the ground truth is known exactly --
# no maze environment needed for this part)
# --------------------------------------------------------------------------
def test_ensemble_training_reduces_loss_on_easy_synthetic_data():
    """A simple, learnable regression target (return depends linearly on
    a single obs feature, same target regardless of action) -- just
    checks the training loop actually fits something, not a claim about
    real D."""
    set_global_seed(0)
    n = 500
    obs = np.random.randn(n, 3).astype(np.float32)
    actions = np.random.randint(0, 2, size=n)
    returns = (2.0 * obs[:, 0] + 0.5).astype(np.float32)  # easy, low-noise target

    cfg = PPOHyperparams(ensemble_n_heads=3, ensemble_hidden_sizes=(16, 16), ensemble_epochs=1, ensemble_lr=1e-2, minibatch_size=64)
    _, _, ensemble = train_ensemble_and_compute_n_eff(obs, actions, returns, obs_dim=3, n_actions=2, cfg=cfg)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    with torch.no_grad():
        preds = ensemble(obs_t)
        gather_idx = actions_t.view(1, -1, 1).expand(preds.shape[0], -1, 1)
        preds_taken = preds.gather(2, gather_idx).squeeze(-1)
        loss_after_1_epoch = torch.nn.functional.mse_loss(preds_taken, returns_t.unsqueeze(0).expand_as(preds_taken)).item()

    cfg_more = PPOHyperparams(ensemble_n_heads=3, ensemble_hidden_sizes=(16, 16), ensemble_epochs=60, ensemble_lr=1e-2, minibatch_size=64)
    set_global_seed(0)
    _, _, ensemble_more = train_ensemble_and_compute_n_eff(obs, actions, returns, obs_dim=3, n_actions=2, cfg=cfg_more)
    with torch.no_grad():
        preds2 = ensemble_more(obs_t)
        preds_taken2 = preds2.gather(2, gather_idx).squeeze(-1)
        loss_after_60_epochs = torch.nn.functional.mse_loss(preds_taken2, returns_t.unsqueeze(0).expand_as(preds_taken2)).item()

    assert np.isfinite(loss_after_1_epoch) and np.isfinite(loss_after_60_epochs)
    assert loss_after_60_epochs < loss_after_1_epoch, (
        f"loss did not improve with more training: 1 epoch={loss_after_1_epoch:.4f}, "
        f"60 epochs={loss_after_60_epochs:.4f}"
    )


def test_n_eff_ranks_well_supported_pair_above_noisy_sparse_pair():
    """The core sanity check for the whole mechanism: a (state, action)
    pair seen often with a CONSISTENT target should end up with higher
    n_eff (lower ensemble variance) than a pair seen rarely with
    CONFLICTING targets across its few occurrences -- exactly the
    state-717-style situation this project's README documents (a sparse,
    noisy signal should look less trustworthy than a dense, consistent
    one). Uses two distinct, fixed observation vectors so "which pair" is
    unambiguous."""
    set_global_seed(0)
    obs_dim, n_actions = 4, 3
    well_supported_obs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    noisy_sparse_obs = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    action_of_interest = 0

    rows_obs, rows_actions, rows_returns = [], [], []
    # 200 occurrences of (well_supported_obs, action 0), consistent target
    for _ in range(200):
        rows_obs.append(well_supported_obs)
        rows_actions.append(action_of_interest)
        rows_returns.append(1.0 + np.random.normal(scale=0.01))
    # 3 occurrences of (noisy_sparse_obs, action 0), wildly conflicting targets
    for target in [5.0, -5.0, 0.2]:
        rows_obs.append(noisy_sparse_obs)
        rows_actions.append(action_of_interest)
        rows_returns.append(target)
    # a little filler data on OTHER actions so the heads see a non-degenerate problem
    for _ in range(50):
        rows_obs.append(well_supported_obs)
        rows_actions.append(1)
        rows_returns.append(-1.0 + np.random.normal(scale=0.01))

    obs = np.stack(rows_obs).astype(np.float32)
    actions = np.array(rows_actions, dtype=np.int64)
    returns = np.array(rows_returns, dtype=np.float32)

    cfg = PPOHyperparams(ensemble_n_heads=6, ensemble_hidden_sizes=(32, 32), ensemble_epochs=150, ensemble_lr=5e-3, minibatch_size=64)
    n_eff, _, _ = train_ensemble_and_compute_n_eff(obs, actions, returns, obs_dim=obs_dim, n_actions=n_actions, cfg=cfg)

    well_supported_mask = (actions == action_of_interest) & np.all(obs == well_supported_obs, axis=1)
    noisy_sparse_mask = (actions == action_of_interest) & np.all(obs == noisy_sparse_obs, axis=1)
    assert well_supported_mask.sum() == 200
    assert noisy_sparse_mask.sum() == 3

    mean_n_eff_well_supported = n_eff[well_supported_mask].mean()
    mean_n_eff_noisy_sparse = n_eff[noisy_sparse_mask].mean()
    assert np.isfinite(mean_n_eff_well_supported) and np.isfinite(mean_n_eff_noisy_sparse)
    assert mean_n_eff_well_supported > mean_n_eff_noisy_sparse, (
        f"expected the dense, consistent pair to show higher n_eff than the sparse, conflicting "
        f"one, got well_supported={mean_n_eff_well_supported:.2f} vs noisy_sparse={mean_n_eff_noisy_sparse:.2f}"
    )


def test_train_ensemble_rejects_mismatched_row_counts():
    cfg = PPOHyperparams(ensemble_n_heads=2, ensemble_epochs=1)
    obs = np.zeros((10, 3), dtype=np.float32)
    actions = np.zeros(10, dtype=np.int64)
    returns = np.zeros(5, dtype=np.float32)  # deliberately mismatched
    with pytest.raises(ValueError):
        train_ensemble_and_compute_n_eff(obs, actions, returns, obs_dim=3, n_actions=2, cfg=cfg)


# --------------------------------------------------------------------------
# EnsembleFixedDPPOTrainer (the full integration, on the project's own maze)
# --------------------------------------------------------------------------
def _make_dataset_and_prior(n_episodes=60, seed=0):
    env = StochasticMazeEnv(width=5, height=5, slip_prob=0.1, num_hazards=2, max_steps=80, layout_seed=0)
    prior_net = ActorCritic(env.observation_space.shape[0], env.n_actions, hidden_sizes=(16, 16))
    dataset = collect_fixed_dataset(env, prior_net, n_episodes=n_episodes, seed=seed, sample_actions=True)
    return env, prior_net, dataset


def test_ensemble_fixed_d_trainer_runs_and_stays_finite():
    set_global_seed(0)
    env, prior_net, dataset = _make_dataset_and_prior()

    cfg = PPOHyperparams(
        epochs=4,
        minibatch_size=64,
        hidden_sizes=(16, 16),
        use_effective_sample_weighting=True,
        effective_sample_beta=0.9,
        use_ensemble_uncertainty=True,
        ensemble_n_heads=3,
        ensemble_hidden_sizes=(16, 16),
        ensemble_epochs=5,
    )
    trainer = EnsembleFixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=prior_net.state_dict()
    )
    initial_state_dict = {k: v.clone() for k, v in trainer.net.state_dict().items()}

    history = trainer.train(verbose=False)

    assert len(history) == cfg.epochs
    for row in history:
        for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
            assert np.isfinite(row[key]), f"non-finite {key}={row[key]} in history row {row}"

    trained_state_dict = trainer.state_dict()
    changed = any(not torch.equal(v, trained_state_dict[k]) for k, v in initial_state_dict.items())
    assert changed, "network weights are identical to initialization after training -- update likely a no-op"


def test_ensemble_n_eff_exposed_and_row_aligned_with_exact_count():
    """The whole point of exposing BOTH ensemble_n_eff and
    exact_n_per_transition (rather than silently discarding the exact
    count) is to let a later calibration study compare them row-by-row
    -- check that alignment actually holds (same length, same dtype
    family, both finite and positive)."""
    set_global_seed(1)
    env, prior_net, dataset = _make_dataset_and_prior(n_episodes=40, seed=2)

    cfg = PPOHyperparams(
        epochs=2,
        minibatch_size=64,
        hidden_sizes=(16, 16),
        use_effective_sample_weighting=True,
        effective_sample_beta=0.9,
        use_ensemble_uncertainty=True,
        ensemble_n_heads=4,
        ensemble_hidden_sizes=(16, 16),
        ensemble_epochs=5,
    )
    trainer = EnsembleFixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=prior_net.state_dict()
    )

    n = trainer._obs.shape[0]
    assert trainer.ensemble_n_eff.shape == (n,)
    assert trainer.exact_n_per_transition.shape == (n,)
    assert np.all(np.isfinite(trainer.ensemble_n_eff))
    assert np.all(trainer.ensemble_n_eff > 0)
    assert np.all(trainer.exact_n_per_transition >= 1)  # every stored transition was observed at least once by definition
    # the inherited training loop must actually be using n_eff, not the exact count
    assert np.array_equal(trainer._n_per_transition, trainer.ensemble_n_eff)


def test_ensemble_fixed_d_trainer_requires_effective_sample_weighting_flag():
    _, prior_net, dataset = _make_dataset_and_prior(n_episodes=5)
    cfg = PPOHyperparams(hidden_sizes=(16, 16), use_effective_sample_weighting=False, use_ensemble_uncertainty=True)
    with pytest.raises(ValueError):
        EnsembleFixedDPPOTrainer(
            dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=prior_net.state_dict()
        )


def test_ensemble_fixed_d_trainer_requires_ensemble_uncertainty_flag():
    _, prior_net, dataset = _make_dataset_and_prior(n_episodes=5)
    cfg = PPOHyperparams(hidden_sizes=(16, 16), use_effective_sample_weighting=True, use_ensemble_uncertainty=False)
    with pytest.raises(ValueError):
        EnsembleFixedDPPOTrainer(
            dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=prior_net.state_dict()
        )


def test_ensemble_fixed_d_trainer_theta_still_starts_at_pi_beta():
    """Inherited from FixedDPPOTrainer -- re-checked here since this is
    the property the whole single-window design depends on, and this
    class overrides __init__."""
    _, prior_net, dataset = _make_dataset_and_prior(n_episodes=10)
    cfg = PPOHyperparams(
        hidden_sizes=(16, 16),
        use_effective_sample_weighting=True,
        use_ensemble_uncertainty=True,
        ensemble_n_heads=3,
        ensemble_hidden_sizes=(16, 16),
        ensemble_epochs=3,
    )
    trainer = EnsembleFixedDPPOTrainer(
        dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=prior_net.state_dict()
    )
    for k, v in prior_net.state_dict().items():
        assert torch.equal(v, trainer.net.state_dict()[k])
        assert torch.equal(v, trainer.pi_beta.state_dict()[k])
