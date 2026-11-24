"""Unit tests 1-3: the canonicaliser preserves function, is idempotent, removes symmetry."""

import torch

from diffusion_policy.canonicalize import canonicalize
from diffusion_policy.policy import PolicySpec, policy_forward

SPEC = PolicySpec(obs_dim=3, hidden_sizes=(16, 16, 8), act_dim=1)


def random_theta(n=32):
    torch.manual_seed(0)
    return torch.randn(n, SPEC.num_params, dtype=torch.float64)


def apply_symmetry(spec, theta):
    """Random hidden-unit permutation P and positive per-unit scaling S in every hidden layer."""
    layers = [[W.clone(), b.clone()] for W, b in spec.unflatten(theta)]
    for l in range(len(layers) - 1):
        n = layers[l][1].shape[-1]
        perm = torch.randperm(n)
        scale = 0.2 + 3 * torch.rand(n, dtype=theta.dtype)
        layers[l][0] = (layers[l][0] * scale.unsqueeze(-1))[..., perm, :]
        layers[l][1] = (layers[l][1] * scale)[..., perm]
        layers[l + 1][0] = (layers[l + 1][0] / scale)[..., :, perm]
    return spec.flatten(layers)


def test_canonicalizer_preserves_function():
    theta = random_theta()
    canonical = canonicalize(SPEC, theta)
    for _ in range(5):
        obs = torch.randn(len(theta), 3, dtype=torch.float64)
        mean, log_std = policy_forward(SPEC, theta, obs)
        mean_c, log_std_c = policy_forward(SPEC, canonical, obs)
        torch.testing.assert_close(mean_c, mean)
        torch.testing.assert_close(log_std_c, log_std)


def test_symmetry_helper_preserves_function():
    """Checks the test helper itself, so test 3 compares equivalent networks."""
    theta = random_theta()
    transformed = apply_symmetry(SPEC, theta)
    obs = torch.randn(len(theta), 3, dtype=torch.float64)
    torch.testing.assert_close(policy_forward(SPEC, transformed, obs), policy_forward(SPEC, theta, obs))


def test_canonicalizer_is_idempotent():
    canonical = canonicalize(SPEC, random_theta())
    torch.testing.assert_close(canonicalize(SPEC, canonical), canonical)


def test_canonicalizer_removes_symmetry():
    theta = random_theta()
    torch.testing.assert_close(canonicalize(SPEC, apply_symmetry(SPEC, theta)), canonicalize(SPEC, theta))


def test_dead_unit_does_not_break_canonicalizer():
    theta = random_theta(n=1)
    # First hidden layer: W is 16 x 3 (entries 0..47), then b (entries 48..63).
    theta[0, 0:3] = 0.0  # unit 0's incoming row
    theta[0, 48] = 0.0  # unit 0's bias
    canonical = canonicalize(SPEC, theta)
    assert torch.isfinite(canonical).all()
    obs = torch.randn(1, 3, dtype=torch.float64)
    torch.testing.assert_close(policy_forward(SPEC, canonical, obs), policy_forward(SPEC, theta, obs))
