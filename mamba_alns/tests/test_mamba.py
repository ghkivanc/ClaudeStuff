import torch

from mamba_alns.mamba import MambaBlock, ResidualMamba


def test_output_shape():
    block = ResidualMamba(32)
    x = torch.randn(4, 8, 32)
    assert block(x).shape == (4, 8, 32)


def test_causal():
    """Changing future tokens must not change earlier outputs."""
    torch.manual_seed(0)
    block = MambaBlock(32)
    x = torch.randn(2, 10, 32)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 4, 32)
    assert torch.allclose(block(x)[:, :6], block(x2)[:, :6], atol=1e-6)
    assert not torch.allclose(block(x)[:, 6:], block(x2)[:, 6:])


def test_gradients_flow():
    block = MambaBlock(16)
    x = torch.randn(2, 5, 16, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None
    assert block.A_log.grad is not None
