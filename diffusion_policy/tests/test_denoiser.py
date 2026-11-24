import torch
from torch import nn

from diffusion_policy.denoiser import Denoiser, SelfAttention, image_side


def test_image_side():
    assert image_side(1250) == 36  # the default policy: 35.4^2 = 1250
    assert image_side(2) == 4
    assert image_side(16) == 4
    assert image_side(17) == 8


def test_output_has_input_shape():
    torch.manual_seed(0)
    for dim in (2, 5, 1250):
        denoiser = Denoiser(dim, cond_dim=4, base_channels=8)
        out = denoiser(torch.randn(3, dim), torch.rand(3), torch.randn(3, 4))
        assert out.shape == (3, dim)
        assert torch.isfinite(out).all()


def test_h_has_no_effect_at_initialisation():
    torch.manual_seed(0)
    denoiser = Denoiser(50, cond_dim=4, base_channels=8)
    z, t = torch.randn(3, 50), torch.rand(3)
    torch.testing.assert_close(denoiser(z, t, torch.randn(3, 4)), denoiser(z, t, torch.randn(3, 4)))


def test_attention_connects_opposite_corners():
    torch.manual_seed(0)
    attention = SelfAttention(32)
    attention.norm = nn.Identity()  # GroupNorm also couples all cells; test attention alone
    x = torch.randn(1, 32, 9, 9)
    changed = x.clone()
    changed[0, :, 0, 0] += 1.0
    assert not torch.allclose(attention(changed)[0, :, 8, 8], attention(x)[0, :, 8, 8])
