"""Belief over the opponent's private card: a vector of 3 probabilities (J, Q, K)."""
import torch

from .leduc import N_RANKS, remaining_counts

LIKELIHOOD_FLOOR = 1e-6   # keeps a belief from collapsing to all zeros


def normalize(belief):
    return belief / belief.sum()


def initial_belief(my_card):
    """Card prior from the deck with my card removed."""
    counts = torch.tensor(remaining_counts(my_card), dtype=torch.float32)
    return counts / counts.sum()


def update_on_public(belief, my_card, public):
    """Exact Bayes for the public card: P(public | my card, opponent card)."""
    likelihood = torch.tensor([remaining_counts(my_card, c)[public] / 4 for c in range(N_RANKS)])
    return normalize(belief * likelihood)


def update_on_action(belief, action_probs, action):
    """Bayes for an opponent action. action_probs: (3, 3), the probability of
    each action if the opponent holds each card."""
    return normalize(belief * (action_probs[:, action] + LIKELIHOOD_FLOOR))
