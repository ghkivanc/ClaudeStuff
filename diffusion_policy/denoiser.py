import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t, dim):
    """Sinusoidal embedding of diffusion time t in [0, 1]: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    angles = 1000.0 * t.unsqueeze(-1) * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class FiLMBlock(nn.Module):
    """Residual MLP block whose normalised activations are scaled and shifted by (t, h).

    The FiLM layer starts at zero. During the bootstrap h is the null embedding,
    which is also zero, so the weights that read h get no gradient and stay
    zero. A real h therefore starts with no effect, and the sampler only learns
    to use it during the improvement iterations.
    """

    def __init__(self, hidden, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.film = nn.Linear(cond_dim, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        y = self.norm(x) * (1 + scale) + shift
        return x + self.linear(F.silu(y))


class Denoiser(nn.Module):
    """eps_phi(z_t, t, h): an MLP over the flat weight vector, conditioned with FiLM."""

    def __init__(self, data_dim, cond_dim, hidden=512, depth=3, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.input = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList([FiLMBlock(hidden, hidden + cond_dim) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, data_dim))

    def forward(self, z_t, t, h):
        """z_t (B, D), t (B,) in [0, 1], h (B, cond_dim) -> predicted noise (B, D)."""
        t_emb = self.time_mlp(timestep_embedding(t, self.time_dim))
        x = self.input(z_t) + t_emb
        cond = torch.cat([t_emb, h], dim=-1)
        for block in self.blocks:
            x = block(x, cond)
        return self.output(x)
