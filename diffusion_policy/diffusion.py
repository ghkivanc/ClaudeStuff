"""Gaussian diffusion: eps-prediction loss and DDIM sampling."""

import torch


def linear_alpha_bar(T):
    """alpha_bar[t] for t = 0..T, with alpha_bar[0] = 1.

    The standard DDPM linear schedule (1e-4 to 0.02 over 1000 steps), with betas
    rescaled by 1000 / T so any T covers the same noise range. Its last steps
    are flat, unlike the cosine schedule's, so the first DDIM step from pure
    noise does not amplify the denoiser's error.
    """
    scale = 1000 / T
    betas = torch.linspace(scale * 1e-4, scale * 0.02, T, dtype=torch.float64).clamp(max=0.999)
    return torch.cat([torch.ones(1, dtype=torch.float64), torch.cumprod(1 - betas, dim=0)]).float()


class Diffusion:
    def __init__(self, T, device="cpu"):
        self.T = T
        self.alpha_bar = linear_alpha_bar(T).to(device)

    def add_noise(self, z0, t, eps):
        ab = self.alpha_bar[t].unsqueeze(-1)
        return ab.sqrt() * z0 + (1 - ab).sqrt() * eps

    def per_sample_loss(self, denoiser, z0, h, t=None, eps=None):
        """|| eps - eps_phi(z_t, t, h) ||^2 averaged over coordinates, one value per sample.

        t ~ Uniform{1..T} and eps ~ N(0, I) unless given.
        """
        if t is None:
            t = torch.randint(1, self.T + 1, (z0.shape[0],), device=z0.device)
        if eps is None:
            eps = torch.randn_like(z0)
        predicted = denoiser(self.add_noise(z0, t, eps), t.float() / self.T, h)
        return ((eps - predicted) ** 2).mean(dim=-1)

    @torch.no_grad()
    def sample(self, denoiser, h, data_dim, num_steps, clip_x0=None):
        """Deterministic DDIM from pure noise to a clean sample, one per row of h."""
        n = h.shape[0]
        times = torch.linspace(self.T, 0, num_steps + 1).round().long().tolist()
        z = torch.randn(n, data_dim, device=h.device)
        for t, t_prev in zip(times[:-1], times[1:]):
            ab, ab_prev = self.alpha_bar[t], self.alpha_bar[t_prev]
            eps = denoiser(z, torch.full((n,), t / self.T, device=h.device), h)
            x0 = (z - (1 - ab).sqrt() * eps) / ab.sqrt()
            if clip_x0 is not None:
                x0 = x0.clamp(-clip_x0, clip_x0)
                eps = (z - ab.sqrt() * x0) / (1 - ab).sqrt()
            z = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return z
