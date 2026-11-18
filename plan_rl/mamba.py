"""Minimal Mamba (selective state space model) in pure PyTorch.

The selective scan is a plain Python loop over the sequence. That is slow
for long sequences but fine for the short windows (K ~ 8) used here, and it
keeps the recurrence easy to read:

    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
    y_t = C_t . h_t + D * x_t

where dt_t, B_t, C_t are computed from the input (the "selective" part).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        # Depthwise causal convolution over time.
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))          # A = -exp(A_log) < 0, stable
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):
        """x: (batch, length, d_model) -> (batch, length, d_model)"""
        length = x.shape[1]
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2)  # keep causal part
        u = F.silu(u)
        y = self.selective_scan(u)
        y = y * F.silu(gate)
        return self.out_proj(y)

    def selective_scan(self, u):
        """u: (batch, length, d_inner)"""
        batch, length, d_inner = u.shape
        A = -torch.exp(self.A_log)                                       # (d_inner, n)
        dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                # (b, l, d_inner)

        h = torch.zeros(batch, d_inner, self.d_state, device=u.device)
        ys = []
        for t in range(length):
            dA = torch.exp(dt[:, t, :, None] * A)                        # (b, d_inner, n)
            dBu = dt[:, t, :, None] * B[:, t, None, :] * u[:, t, :, None]
            h = dA * h + dBu
            ys.append((h * C[:, t, None, :]).sum(-1))                    # (b, d_inner)
        y = torch.stack(ys, dim=1)
        return y + u * self.D


class ResidualMamba(nn.Module):
    """Pre-norm residual wrapper: x + Mamba(LayerNorm(x))."""

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, **kwargs)

    def forward(self, x):
        return x + self.mamba(self.norm(x))
