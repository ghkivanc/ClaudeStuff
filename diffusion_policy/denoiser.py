"""eps_phi(z_t, t, h): a small U-Net over the policy weights laid out as an image.

The flat weight vector (every layer's W and b, vectorised and concatenated) is
zero-padded and reshaped row by row into a square image. The layout is
arbitrary: weights that end up next to each other need not be related. The
output is cropped back to a flat vector, so callers never see the image.
(t, h) enter every residual block through FiLM.
"""

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


def image_side(data_dim):
    """Smallest multiple of 4 (the U-Net halves the image twice) whose square holds data_dim values."""
    side = math.ceil(math.sqrt(data_dim))
    return 4 * math.ceil(side / 4)


class ResBlock(nn.Module):
    """Two norm-SiLU-conv steps with a skip connection; FiLM from (t, h) on the second norm.

    The FiLM layer starts at zero. During the bootstrap h is the null embedding,
    which is also zero, so the weights that read h get no gradient and stay
    zero. A real h therefore starts with no effect, and the sampler only learns
    to use it during the improvement iterations.
    """

    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels, affine=False)
        self.film = nn.Linear(cond_dim, 2 * out_channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, cond):
        y = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(cond)[:, :, None, None].chunk(2, dim=1)
        y = self.norm2(y) * (1 + scale) + shift
        y = self.conv2(F.silu(y))
        return self.skip(x) + y


class Denoiser(nn.Module):
    """U-Net with two resolution halvings. base_channels must be a multiple of 8 (GroupNorm)."""

    def __init__(self, data_dim, cond_dim, base_channels=32, time_dim=128):
        super().__init__()
        c1, c2, c3 = base_channels, 2 * base_channels, 4 * base_channels
        cond = time_dim + cond_dim
        self.data_dim = data_dim
        self.side = image_side(data_dim)
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )

        self.input = nn.Conv2d(1, c1, 3, padding=1)
        self.down1 = ResBlock(c1, c1, cond)  # full size
        self.halve1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.down2 = ResBlock(c1, c2, cond)  # 1/2 size
        self.halve2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)
        self.middle = ResBlock(c2, c3, cond)  # 1/4 size
        self.double2 = nn.Conv2d(c3, c3, 3, padding=1)  # applied after nearest upsampling
        self.up2 = ResBlock(c3 + c2, c2, cond)  # 1/2 size
        self.double1 = nn.Conv2d(c2, c2, 3, padding=1)
        self.up1 = ResBlock(c2 + c1, c1, cond)  # full size
        self.output = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, z_t, t, h):
        """z_t (B, D), t (B,) in [0, 1], h (B, cond_dim) -> predicted noise (B, D)."""
        n = z_t.shape[0]
        cond = torch.cat([self.time_mlp(timestep_embedding(t, self.time_dim)), h], dim=-1)
        x = F.pad(z_t, (0, self.side**2 - self.data_dim)).view(n, 1, self.side, self.side)

        x = self.input(x)
        skip1 = self.down1(x, cond)
        skip2 = self.down2(self.halve1(skip1), cond)
        x = self.middle(self.halve2(skip2), cond)
        x = self.double2(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.up2(torch.cat([x, skip2], dim=1), cond)
        x = self.double1(F.interpolate(x, scale_factor=2, mode="nearest"))
        x = self.up1(torch.cat([x, skip1], dim=1), cond)
        return self.output(x).flatten(1)[:, : self.data_dim]
