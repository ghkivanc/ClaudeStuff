"""Choose beta each iteration so the batch weights sit exactly eps_KL from uniform.

w_i(beta) = softmax(score_i / beta), and KL(w || uniform) = log N - H(w) falls
monotonically from log N (beta -> 0) to 0 (beta -> inf). Bisection on log beta.
"""

import math

import torch


def kl_to_uniform(weights):
    weights = weights.double()
    entropy = -(weights * weights.clamp_min(1e-300).log()).sum()  # 0 log 0 = 0
    return math.log(weights.numel()) - entropy.item()


def tune_beta(scores, eps_kl, steps=60):
    """Returns beta with KL(softmax(scores / beta) || uniform) = eps_kl.

    Returns inf when all scores are equal: the weights are then uniform for any beta.
    """
    n = scores.numel()
    if not 0 < eps_kl < math.log(n):
        raise ValueError(f"eps_kl must lie in (0, log N) = (0, {math.log(n):.3f}), got {eps_kl}")

    scores = scores.double()
    spread = scores.std().item()
    if spread < 1e-12:
        return math.inf

    def kl(log_beta):
        return kl_to_uniform(torch.softmax(scores / math.exp(log_beta), dim=0))

    low, high = math.log(spread * 1e-4), math.log(spread * 1e4)
    for _ in range(steps):
        mid = (low + high) / 2
        if kl(mid) > eps_kl:
            low = mid  # too concentrated: raise beta
        else:
            high = mid
    return math.exp((low + high) / 2)


def improvement_weights(scores, eps_kl):
    """Self-normalised weights w_i proportional to exp(score_i / beta), and beta."""
    beta = tune_beta(scores, eps_kl)
    return torch.softmax(scores / beta, dim=0), beta
