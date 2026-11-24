"""The policy pi_theta: a small ReLU MLP whose weights are one flat vector theta.

The network maps an observation to an action mean and log-std. Every function
here takes a batch of weight vectors, so N different policies run at once.
Flat layout, layer by layer: W (out x in, row-major), then b (out).
"""

import math

import torch

from .pendulum import MAX_TORQUE


class PolicySpec:
    def __init__(self, obs_dim, hidden_sizes, act_dim):
        self.sizes = [obs_dim, *hidden_sizes, 2 * act_dim]  # output: mean and log-std
        self.layer_shapes = [(out, inp) for inp, out in zip(self.sizes[:-1], self.sizes[1:])]
        self.num_params = sum(out * inp + out for out, inp in self.layer_shapes)

    def unflatten(self, theta):
        """theta (..., P) -> list of (W (..., out, in), b (..., out))."""
        layers = []
        i = 0
        for out, inp in self.layer_shapes:
            W = theta[..., i : i + out * inp].reshape(*theta.shape[:-1], out, inp)
            i += out * inp
            b = theta[..., i : i + out]
            i += out
            layers.append((W, b))
        return layers

    def flatten(self, layers):
        return torch.cat([torch.cat([W.flatten(-2), b], dim=-1) for W, b in layers], dim=-1)


def random_init(spec, n, device="cpu"):
    """n weight vectors with PyTorch's default Linear initialisation."""
    parts = []
    for out, inp in spec.layer_shapes:
        bound = 1.0 / math.sqrt(inp)
        parts.append(bound * (2 * torch.rand(n, out * inp, device=device) - 1))
        parts.append(bound * (2 * torch.rand(n, out, device=device) - 1))
    return torch.cat(parts, dim=-1)


def policy_forward(spec, theta, obs):
    """theta (N, P), obs (N, obs_dim) -> mean (N, act_dim), log_std (N, act_dim)."""
    layers = spec.unflatten(theta)
    x = obs
    for i, (W, b) in enumerate(layers):
        x = torch.einsum("noi,ni->no", W, x) + b
        if i < len(layers) - 1:
            x = torch.relu(x)
    mean, log_std = x.chunk(2, dim=-1)
    return mean, log_std.clamp(-5.0, 2.0)


def sample_action(spec, theta, obs):
    mean, log_std = policy_forward(spec, theta, obs)
    action = mean + log_std.exp() * torch.randn_like(mean)
    return action.clamp(-MAX_TORQUE, MAX_TORQUE)
