"""Unit test 7 (no leakage) and smoke test 2 (environment-free improvement)."""

import pytest
import torch

from diffusion_policy.beta import improvement_weights
from diffusion_policy.config import Config
from diffusion_policy.denoiser import Denoiser, ResBlock
from diffusion_policy.diffusion import Diffusion
from diffusion_policy.train import Trainer


def tiny_config(**overrides):
    cfg = Config(
        hidden_sizes=(4,),
        K=2,
        segment_multiple=1,
        max_steps=1000,
        num_envs=3,
        eps_kl=0.5,
        diffusion_steps=10,
        sampling_steps=5,
        embed_dim=16,
        encoder_layers=1,
        encoder_heads=2,
        unet_channels=8,
        minibatch=4,
        grad_steps=2,
        bootstrap_size=64,
        bootstrap_steps=2,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return cfg


def test_no_leakage():
    """The batch never holds an H_i containing a transition produced by theta_i."""
    trainer = Trainer(tiny_config())
    trainer.collect()  # fills the history with the first policy's transitions

    before, _ = trainer.history.snapshot()
    batch, _ = trainer.collect()
    after, _ = trainer.history.snapshot()  # with L >= K: only theta_i's transitions

    torch.testing.assert_close(batch["H"], before)
    for env in range(3):
        for own in after[env]:
            assert not any(torch.allclose(own, old) for old in batch["H"][env])


def test_segment_stops_at_episode_end_and_history_resets():
    trainer = Trainer(tiny_config(K=2, segment_multiple=2, max_steps=6))  # L = 4
    trainer.collect()
    assert trainer.env.t == 4
    _, stats = trainer.collect()  # only 2 steps left in the episode
    assert "episode_return" in stats
    assert trainer.env.t == 0
    assert not trainer.history.mask.any()


def test_bootstrap_leaves_h_weights_at_zero():
    """The bootstrap only sees the zero null embedding, so FiLM's h-columns get no gradient."""
    trainer = Trainer(tiny_config(bootstrap_steps=20))
    trainer.bootstrap()
    time_dim = trainer.model.denoiser.time_dim
    blocks = [m for m in trainer.model.denoiser.modules() if isinstance(m, ResBlock)]
    assert blocks
    for block in blocks:
        assert (block.film.weight[:, time_dim:] == 0).all()
        assert (block.film.weight[:, :time_dim] != 0).any()  # the time columns did train


def test_stored_z_is_the_sampled_z():
    trainer = Trainer(tiny_config())
    batch, _ = trainer.collect()
    theta = trainer.model.normalizer.unnormalize(batch["z"])
    torch.testing.assert_close(trainer.model.normalizer.normalize(theta), batch["z"])


def test_one_training_iteration_runs():
    trainer = Trainer(tiny_config())
    trainer.bootstrap()
    batch, _ = trainer.collect()
    stats = trainer.refit(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


@pytest.mark.smoke
def test_environment_free_improvement():
    """R(theta) = -||theta - theta*||^2 with H held constant.

    The sample mean should move toward theta* while the spread stays above zero.
    """
    torch.manual_seed(0)
    target = torch.tensor([1.5, -1.0, 0.5, 1.0])
    diffusion = Diffusion(T=100)
    denoiser = Denoiser(data_dim=4, cond_dim=8, base_channels=32)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    h = torch.zeros(1, 8)

    def fit(data, weights, steps):
        for _ in range(steps):
            idx = torch.multinomial(weights, 64, replacement=True)
            loss = diffusion.per_sample_loss(denoiser, data[idx], h.expand(64, -1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def sample(n):
        return diffusion.sample(denoiser, h.expand(n, -1), 4, num_steps=20)

    start = torch.randn(4096, 4)
    fit(start, torch.ones(len(start)), steps=1000)
    start_distance = (sample(1000).mean(dim=0) - target).norm()

    for _ in range(6):
        theta = sample(128)
        returns = -((theta - target) ** 2).sum(dim=-1)
        weights, _ = improvement_weights(returns, eps_kl=1.0)
        fit(theta, weights, steps=200)

    final = sample(1000)
    assert (final.mean(dim=0) - target).norm() < 0.5 * start_distance
    assert final.std(dim=0).mean() > 0.05
