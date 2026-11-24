"""Unit test 5 (loss reduces to standard DDPM) and smoke test 1 (diffusion sanity)."""

import pytest
import torch

from diffusion_policy.denoiser import Denoiser
from diffusion_policy.diffusion import Diffusion


def test_loss_reduces_to_standard_ddpm():
    torch.manual_seed(0)
    diffusion = Diffusion(T=100)
    denoiser = Denoiser(data_dim=5, cond_dim=4, hidden=32, depth=2)
    z0 = torch.randn(10, 5)
    h = torch.randn(10, 4)
    t = torch.randint(1, 101, (10,))
    eps = torch.randn(10, 5)

    per_sample = diffusion.per_sample_loss(denoiser, z0, h, t, eps)

    # Plain DDPM loss, written out by hand.
    ab = diffusion.alpha_bar[t].unsqueeze(-1)
    z_t = ab.sqrt() * z0 + (1 - ab).sqrt() * eps
    ddpm = ((eps - denoiser(z_t, t.float() / 100, h)) ** 2).mean(dim=-1)
    torch.testing.assert_close(per_sample, ddpm)

    uniform = torch.full((10,), 0.1)
    torch.testing.assert_close((uniform * per_sample).sum(), ddpm.mean())


def test_alpha_bar_schedule():
    ab = Diffusion(T=100).alpha_bar
    assert ab[0] == 1.0
    assert (ab[1:] < ab[:-1]).all()
    assert 0 < ab[-1] < 1e-3  # ends close to pure noise
    # Flat end: the first DDIM step (t = 100 -> 95) rescales the sample by a small factor.
    assert (ab[95] / ab[100]).sqrt() < 3


@pytest.mark.smoke
def test_diffusion_reproduces_gaussian_mixture_modes():
    torch.manual_seed(0)
    modes = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    diffusion = Diffusion(T=100)
    denoiser = Denoiser(data_dim=2, cond_dim=4, hidden=128, depth=3)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
    # Decay the learning rate to zero: at a constant rate the final weights jitter,
    # and DDIM turns that jitter into a shift of probability mass between modes.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)
    h = torch.zeros(256, 4)

    for _ in range(3000):
        data = modes[torch.randint(4, (256,))] + 0.1 * torch.randn(256, 2)
        loss = diffusion.per_sample_loss(denoiser, data, h).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    samples = diffusion.sample(denoiser, torch.zeros(2000, 4), 2, num_steps=50)
    distance, nearest = torch.cdist(samples, modes).min(dim=-1)
    assert (distance < 0.4).float().mean() > 0.9
    shares = torch.bincount(nearest, minlength=4).float() / len(samples)
    assert (shares > 0.1).all()  # no mode dropped (a fair share is 0.25)
