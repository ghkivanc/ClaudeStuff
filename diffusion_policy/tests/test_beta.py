"""Unit test 4: the beta tuner hits its KL target."""

import math

import pytest
import torch

from diffusion_policy.beta import improvement_weights, kl_to_uniform, tune_beta


def test_tuned_beta_hits_kl_target():
    torch.manual_seed(0)
    scores = torch.randn(64) * 30 - 100
    for eps_kl in (0.1, 1.0, 3.0):
        weights, _ = improvement_weights(scores, eps_kl)
        assert kl_to_uniform(weights) == pytest.approx(eps_kl, abs=1e-4)


def test_kl_is_monotone_in_beta():
    torch.manual_seed(0)
    scores = torch.randn(64, dtype=torch.float64)
    betas = torch.logspace(-4, 3, 50, dtype=torch.float64)
    kls = [kl_to_uniform(torch.softmax(scores / beta, dim=0)) for beta in betas]
    assert all(a >= b - 1e-12 for a, b in zip(kls, kls[1:]))
    assert kls[0] > 0.9 * math.log(64)  # nearly all weight on the best sample
    assert kls[-1] == pytest.approx(0.0, abs=1e-4)


def test_equal_scores_give_uniform_weights():
    weights, beta = improvement_weights(torch.full((16,), 3.0), 1.0)
    assert beta == math.inf
    torch.testing.assert_close(weights, torch.full((16,), 1 / 16))


def test_target_outside_range_is_rejected():
    with pytest.raises(ValueError):
        tune_beta(torch.randn(8), math.log(8))
