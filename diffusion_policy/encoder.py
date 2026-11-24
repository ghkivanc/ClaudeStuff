import torch
from torch import nn


class HistoryEncoder(nn.Module):
    """Encodes the last K (s, a, r, s') tuples into one vector h.

    Per-tuple MLP embedding, plus learned positions, then a small transformer
    and a mean over the real (unmasked) tuples. A history with no tuples maps
    to the learned null embedding.
    """

    def __init__(self, tuple_dim, K, d_model=64, num_layers=2, num_heads=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(tuple_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.positions = nn.Parameter(0.02 * torch.randn(K, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=4 * d_model, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        # Starts at zero so that bootstrap training, which only ever sees the null
        # embedding, leaves the denoiser's h-weights at zero (see FiLMBlock).
        self.null = nn.Parameter(torch.zeros(d_model))

    def forward(self, tuples, mask):
        """tuples (B, K, tuple_dim), mask (B, K) bool, True = real tuple -> h (B, d_model)."""
        empty = ~mask.any(dim=-1)
        # Attention over a row that is all padding divides by zero. Let such rows
        # attend to their padding; their output is replaced by the null embedding.
        padding = ~mask & ~empty.unsqueeze(-1)

        x = self.embed(tuples) + self.positions
        x = self.transformer(x, src_key_padding_mask=padding)

        weights = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.where(empty.unsqueeze(-1), self.null.expand_as(pooled), pooled)
