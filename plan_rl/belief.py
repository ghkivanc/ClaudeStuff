"""Belief over hypotheses h = (opponent's card, opponent's type).

Hypotheses are numbered card-major: h = card * n_types + type, so a belief is
a vector of length N_RANKS * n_types. Types are slots of the snapshot
population; slots that are not filled yet get probability 0.
"""
import torch

from .leduc import N_RANKS, remaining_counts

LIKELIHOOD_FLOOR = 1e-6   # keeps a belief from collapsing to all zeros


def hypothesis_cards(n_types):
    return torch.arange(N_RANKS).repeat_interleave(n_types)


def hypothesis_types(n_types):
    return torch.arange(n_types).repeat(N_RANKS)


def normalize(belief):
    return belief / belief.sum()


def initial_belief(my_card, n_types, n_known_types):
    """Card prior from the deck (my card removed) times uniform over known types."""
    counts = torch.tensor(remaining_counts(my_card), dtype=torch.float32)
    card_prior = counts / counts.sum()
    type_prior = torch.zeros(n_types)
    type_prior[:n_known_types] = 1 / n_known_types
    return (card_prior[:, None] * type_prior[None, :]).reshape(-1)


def update_on_public(belief, my_card, public, n_types):
    """Exact Bayes for the public card: P(public | my card, opponent card)."""
    likelihood = torch.tensor([remaining_counts(my_card, c)[public] / 4 for c in range(N_RANKS)])
    return normalize(belief * likelihood[hypothesis_cards(n_types)])


def update_on_action(belief, action_probs, action):
    """Bayes for an opponent action. action_probs: (H, 3), the probability of
    each action under each hypothesis."""
    return normalize(belief * (action_probs[:, action] + LIKELIHOOD_FLOOR))
