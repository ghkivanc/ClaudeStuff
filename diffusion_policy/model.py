import torch
from torch import nn

from .denoiser import Denoiser
from .encoder import HistoryEncoder


class WeightNormalizer(nn.Module):
    """Per-coordinate standardisation of canonical weights, fitted on the bootstrap set."""

    def __init__(self, num_params):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_params))
        self.register_buffer("std", torch.ones(num_params))

    def fit(self, theta):
        self.mean.copy_(theta.mean(dim=0))
        self.std.copy_(theta.std(dim=0).clamp_min(1e-6))

    def normalize(self, theta):
        return (theta - self.mean) / self.std

    def unnormalize(self, z):
        return z * self.std + self.mean


class SamplerModel(nn.Module):
    """Everything with state: normaliser, history encoder, denoiser and value head V(h).

    The value loss trains the encoder too, not only the value head.
    """

    def __init__(self, num_params, tuple_dim, cfg):
        super().__init__()
        self.normalizer = WeightNormalizer(num_params)
        self.encoder = HistoryEncoder(
            tuple_dim, cfg.K, cfg.embed_dim, cfg.encoder_layers, cfg.encoder_heads
        )
        self.denoiser = Denoiser(num_params, cfg.embed_dim, cfg.denoiser_hidden, cfg.denoiser_depth)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, 128), nn.SiLU(), nn.Linear(128, 1)
        )

    def value(self, h):
        return self.value_head(h).squeeze(-1)
