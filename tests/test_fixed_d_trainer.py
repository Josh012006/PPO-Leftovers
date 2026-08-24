"""Tier-1 library smoke test: does FixedDPPOTrainer run without error, stay
numerically finite, and actually move its own weights, using the project's
own custom maze environment. This checks the training LOOP's mechanics in
isolation -- it is not a cross-environment pipeline validation. For that
(the full D -> pi_D* -> fixed-D-PPO -> evaluate pipeline exercised against
an environment we did not build, gymnasium's FrozenLake), see
tests/tier0/test_tier0_pipeline.py instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppo_exploitation.data.collect import collect_fixed_dataset
from ppo_exploitation.envs.stochastic_maze import StochasticMazeEnv
from ppo_exploitation.ppo.fixed_d_trainer import FixedDPPOTrainer
from ppo_exploitation.ppo.networks import ActorCritic
from ppo_exploitation.utils.config import PPOHyperparams
from ppo_exploitation.utils.seeding import set_global_seed


def test_fixed_d_trainer_runs_and_stays_finite():
    set_global_seed(0)
    env = StochasticMazeEnv(width=5, height=5, slip_prob=0.1, num_hazards=2, max_steps=80, layout_seed=0)
    prior_net = ActorCritic(env.observation_space.shape[0], env.n_actions, hidden_sizes=(16, 16))

    dataset = collect_fixed_dataset(env, prior_net, n_episodes=60, seed=0, sample_actions=True)
    assert len(dataset) > 0
    assert dataset.n_episodes == 60

    cfg = PPOHyperparams(epochs=4, minibatch_size=64, hidden_sizes=(16, 16))
    trainer = FixedDPPOTrainer(
        dataset,
        obs_dim=dataset.obs_dim,
        n_actions=dataset.n_actions,
        cfg=cfg,
        prior_state_dict=prior_net.state_dict(),
    )
    initial_state_dict = {k: v.clone() for k, v in trainer.net.state_dict().items()}

    history = trainer.train(verbose=False)

    assert len(history) == cfg.epochs
    for row in history:
        for key in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
            assert np.isfinite(row[key]), f"non-finite {key}={row[key]} in history row {row}"

    # weights must have actually changed from initialization
    trained_state_dict = trainer.state_dict()
    changed = any(not torch.equal(v, trained_state_dict[k]) for k, v in initial_state_dict.items())
    assert changed, "network weights are identical to initialization after training -- update likely a no-op"


def test_fixed_d_trainer_pi_old_equals_pi_beta_throughout():
    """The whole point of the rigorous single-window design: the ratio's
    denominator (old_logprobs) must be pi_beta's REAL log-probabilities,
    taken directly from D, never recomputed from a separately-drifting
    snapshot. Check that the trainer's stored old_logprobs exactly match
    what pi_beta itself reports for the same (obs, action) pairs."""
    set_global_seed(0)
    env = StochasticMazeEnv(width=5, height=5, slip_prob=0.1, num_hazards=2, max_steps=80, layout_seed=0)
    prior_net = ActorCritic(env.observation_space.shape[0], env.n_actions, hidden_sizes=(16, 16))
    dataset = collect_fixed_dataset(env, prior_net, n_episodes=30, seed=1, sample_actions=True)

    cfg = PPOHyperparams(epochs=2, minibatch_size=64, hidden_sizes=(16, 16))
    trainer = FixedDPPOTrainer(
        dataset,
        obs_dim=dataset.obs_dim,
        n_actions=dataset.n_actions,
        cfg=cfg,
        prior_state_dict=prior_net.state_dict(),
    )

    obs_t = torch.as_tensor(trainer._obs, dtype=torch.float32)
    actions_t = torch.as_tensor(trainer._actions, dtype=torch.int64)
    with torch.no_grad():
        recomputed_log_prob, _, _ = trainer.pi_beta.evaluate_actions(obs_t, actions_t)
    stored = torch.as_tensor(trainer._old_logprobs, dtype=torch.float32)
    assert torch.allclose(recomputed_log_prob, stored, atol=1e-5), (
        "pi_beta's own log-probabilities disagree with D's stored log_probs -- "
        "pi_old is supposed to be exactly pi_beta, not an approximation of it"
    )


def test_fixed_d_trainer_theta_starts_at_pi_beta():
    """theta must start at pi_beta's exact weights -- otherwise 'pi_old =
    pi_beta' and 'theta continues from pi_beta' would be two different,
    inconsistent claims about the same run."""
    env = StochasticMazeEnv(width=5, height=5, num_hazards=1, max_steps=50, layout_seed=0)
    prior_net = ActorCritic(env.observation_space.shape[0], env.n_actions, hidden_sizes=(8, 8))
    dataset = collect_fixed_dataset(env, prior_net, n_episodes=5, seed=0)
    cfg = PPOHyperparams(hidden_sizes=(8, 8))
    trainer = FixedDPPOTrainer(
        dataset,
        obs_dim=dataset.obs_dim,
        n_actions=dataset.n_actions,
        cfg=cfg,
        prior_state_dict=prior_net.state_dict(),
    )
    for k, v in prior_net.state_dict().items():
        assert torch.equal(v, trainer.net.state_dict()[k])
        assert torch.equal(v, trainer.pi_beta.state_dict()[k])


def test_fixed_d_trainer_requires_prior_state_dict():
    env = StochasticMazeEnv(width=5, height=5, num_hazards=1, max_steps=50, layout_seed=0)
    net = ActorCritic(env.observation_space.shape[0], env.n_actions, hidden_sizes=(8, 8))
    dataset = collect_fixed_dataset(env, net, n_episodes=5, seed=0)
    cfg = PPOHyperparams()
    with pytest.raises(ValueError):
        FixedDPPOTrainer(
            dataset, obs_dim=dataset.obs_dim, n_actions=dataset.n_actions, cfg=cfg, prior_state_dict=None
        )