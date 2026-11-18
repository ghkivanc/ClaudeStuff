"""Counterfactual regret minimisation (vanilla CFR) for Leduc, used as a
near-equilibrium baseline, plus exact expected value of two fixed policies.

A policy here is a function info_key -> probabilities over the 3 actions,
where info_key = (card, public, rounds) is what the acting player sees.
"""
from collections import defaultdict

import numpy as np

from .leduc import (N_ACTIONS, N_RANKS, State, apply_action, current_player, deal_public,
                    is_chance, is_terminal, legal_actions, payoff, remaining_counts)


def private_deals():
    """All (card0, card1) rank pairs with their probability."""
    deals = []
    for c0 in range(N_RANKS):
        for c1 in range(N_RANKS):
            p = 2 / 6 * remaining_counts(c0)[c1] / 5
            if p > 0:
                deals.append(((c0, c1), p))
    return deals


def public_deals(s):
    """(rank, probability) of the public card given both private cards."""
    counts = remaining_counts(*s.cards)
    return [(r, counts[r] / 4) for r in range(N_RANKS) if counts[r] > 0]


def info_key(s):
    return (s.cards[current_player(s)], s.public, s.rounds)


def _regret_matching(regret, legal):
    positive = np.zeros(N_ACTIONS)
    positive[legal] = np.maximum(regret[legal], 0)
    if positive.sum() > 0:
        return positive / positive.sum()
    uniform = np.zeros(N_ACTIONS)
    uniform[legal] = 1 / len(legal)
    return uniform


class CFR:
    def __init__(self):
        self.regret = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.strategy_sum = defaultdict(lambda: np.zeros(N_ACTIONS))

    def run(self, iterations):
        for _ in range(iterations):
            for cards, p in private_deals():
                self._walk(State(cards), np.array([1.0, 1.0]), p)

    def _walk(self, s, reach, chance_reach):
        """Returns player 0's expected value below s and updates regrets."""
        if is_terminal(s):
            return payoff(s)
        if is_chance(s):
            return sum(p * self._walk(deal_public(s, r), reach, chance_reach * p)
                       for r, p in public_deals(s))

        player = current_player(s)
        key, legal = info_key(s), legal_actions(s)
        strategy = _regret_matching(self.regret[key], legal)

        values = np.zeros(N_ACTIONS)
        for a in legal:
            child_reach = reach.copy()
            child_reach[player] *= strategy[a]
            values[a] = self._walk(apply_action(s, a), child_reach, chance_reach)
        node_value = strategy @ values

        sign = 1 if player == 0 else -1          # regrets from the acting player's view
        others_reach = reach[1 - player] * chance_reach
        self.regret[key][legal] += others_reach * sign * (values[legal] - node_value)
        self.strategy_sum[key] += reach[player] * strategy
        return node_value

    def average_policy(self):
        table = {k: v / v.sum() for k, v in self.strategy_sum.items() if v.sum() > 0}

        def policy(key, legal):
            if key in table:
                return table[key]
            uniform = np.zeros(N_ACTIONS)
            uniform[legal] = 1 / len(legal)
            return uniform
        return policy


def uniform_policy(key, legal):
    probs = np.zeros(N_ACTIONS)
    probs[legal] = 1 / len(legal)
    return probs


def expected_value(policy0, policy1):
    """Exact expected chips for player 0 when seat 0 plays policy0 and seat 1 plays policy1.
    Each policy is called as policy(info_key, legal)."""
    policies = (policy0, policy1)

    def walk(s):
        if is_terminal(s):
            return payoff(s)
        if is_chance(s):
            return sum(p * walk(deal_public(s, r)) for r, p in public_deals(s))
        legal = legal_actions(s)
        probs = policies[current_player(s)](info_key(s), legal)
        return sum(probs[a] * walk(apply_action(s, a)) for a in legal)

    return sum(p * walk(State(cards)) for cards, p in private_deals())
