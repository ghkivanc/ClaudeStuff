"""Unit test 6: encoder masking."""

import torch

from diffusion_policy.encoder import HistoryEncoder


def make_encoder():
    torch.manual_seed(0)
    encoder = HistoryEncoder(tuple_dim=8, K=6, d_model=32, num_layers=2, num_heads=4)
    with torch.no_grad():
        encoder.null.normal_()  # nonzero, so the empty-history check means something
    return encoder


def test_padded_content_does_not_change_h():
    encoder = make_encoder()
    tuples = torch.randn(4, 6, 8)
    mask = torch.tensor([[False, False, True, True, True, True]] * 4)
    changed = tuples.clone()
    changed[:, :2] = 100 * torch.randn(4, 2, 8)
    torch.testing.assert_close(encoder(changed, mask), encoder(tuples, mask))


def test_real_content_does_change_h():
    encoder = make_encoder()
    tuples = torch.randn(4, 6, 8)
    mask = torch.ones(4, 6, dtype=torch.bool)
    changed = tuples.clone()
    changed[:, -1] += 1.0
    assert not torch.allclose(encoder(changed, mask), encoder(tuples, mask))


def test_empty_history_gives_null_embedding():
    encoder = make_encoder()
    mask = torch.tensor([[False] * 6, [True] * 6])
    h = encoder(torch.randn(2, 6, 8), mask)
    assert torch.isfinite(h).all()
    torch.testing.assert_close(h[0], encoder.null)
    assert not torch.allclose(h[1], encoder.null)
