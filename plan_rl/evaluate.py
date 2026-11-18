"""Evaluate a trained world model's planning agent against fixed opponents.

  random : uniform over legal actions
  cfr    : CFR average strategy (near-equilibrium). An unexploitable strategy,
           so about 0 chips/hand is the best one can expect against it on
           average, and a clearly negative result means the agent is exploitable.

Seats alternate every hand. Results are chips per hand for the agent, with the
standard error of the mean.

Run (after training saved world_model.pt):  python -m plan_rl.evaluate
"""
import random

import numpy as np
import torch

from .agent import PlanningAgent
from .cfr import CFR, expected_value, uniform_policy
from .leduc import payoff, play_hand
from .model import WorldModel
from .train import Config


class PolicyPlayer:
    """Plays a fixed policy(info_key, legal) -> probabilities."""

    def __init__(self, policy, rng):
        self.policy, self.rng = policy, rng

    def new_hand(self, seat, card):
        pass

    def act(self, info):
        probs = self.policy((info.card, info.public, info.rounds), info.legal)
        return int(self.rng.choice(3, p=probs / probs.sum()))


def match(agent, opponent, n_hands, py_rng):
    """Mean chips per hand for `agent` and its standard error."""
    results = []
    for i in range(n_hands):
        seat = i % 2
        players = (agent, opponent) if seat == 0 else (opponent, agent)
        final = play_hand(*players, py_rng)
        results.append(payoff(final) if seat == 0 else -payoff(final))
    results = np.array(results, dtype=float)
    return results.mean(), results.std(ddof=1) / np.sqrt(n_hands)


def evaluate_all(model, cfg):
    py_rng = random.Random(cfg.seed + 1)
    rng = np.random.default_rng(cfg.seed + 1)

    print(f"running CFR for {cfg.cfr_iterations} iterations ...")
    cfr = CFR()
    cfr.run(cfg.cfr_iterations)
    cfr_policy = cfr.average_policy()
    # Reference: CFR against random, exact, averaged over both seats.
    reference = (expected_value(cfr_policy, uniform_policy)
                 - expected_value(uniform_policy, cfr_policy)) / 2
    print(f"reference  CFR vs random (exact): {reference:+.3f} chips/hand")

    agent = PlanningAgent(model, cfg, rng)
    for name, policy in [("random", uniform_policy), ("cfr", cfr_policy)]:
        mean, stderr = match(agent, PolicyPlayer(policy, rng), cfg.eval_hands, py_rng)
        print(f"agent vs {name:6s}: {mean:+.3f} ± {stderr:.3f} chips/hand  ({cfg.eval_hands} hands)")


if __name__ == "__main__":
    cfg = Config()
    model = WorldModel(cfg.d_model, cfg.mamba_layers)
    model.load_state_dict(torch.load("world_model.pt"))
    evaluate_all(model, cfg)
