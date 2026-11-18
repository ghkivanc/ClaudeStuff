"""Planning agent: own world model, own belief, commit until surprised.

Each player is a separate instance with its own tree and belief, as in a real game.

  - Real belief over the opponent's card, updated after every event:
    public card -> exact Bayes from the deck;
    opponent action -> Bayes with the model's opponent head as the likelihood
    (approximate: the opponent plans too, and the head stands in for their planner).
  - Plan: build a tree, then at each of my turns sample my action from the
    plan's policy there, softmax(plan values / temperature).
  - Surprise: the opponent's action or the public card is not in the tree, or had
    probability below `surprise_threshold` there -> drop the plan, replan at my turn.
"""
import torch

from .belief import initial_belief, update_on_action, update_on_public
from .model import MY, OPP, history_events, to_event
from .plan_tree import build_tree


class PlanningAgent:
    def __init__(self, model, cfg, rng):
        self.model = model
        self.cfg = cfg
        self.rng = rng

    def new_hand(self, seat, card):
        self.seat, self.card = seat, card
        self.events = []          # event ids seen so far, from my point of view
        self.belief = initial_belief(card)
        self.node = None          # where I am in the committed plan (None = no plan)
        self.targets = {}         # event index of each of my decisions -> policy target
        self.n_replans = 0

    def act(self, info):
        history = history_events(info.seat, info.public, info.rounds)
        for kind, x in history[len(self.events):]:      # everything since my last turn
            self._observe(kind, x)

        probs = self._plan_policy(info.legal)
        if probs is None:
            self._replan(info.legal)
            probs = self._plan_policy(info.legal)
        self.targets[len(self.events)] = probs
        p = probs.double().numpy()
        action = int(self.rng.choice(3, p=p / p.sum()))   # renormalise in float64 for numpy

        self.events.append(to_event(MY, action))
        self.node = self.node.children[action][0]
        return action

    def _observe(self, kind, x):
        """An opponent action or the public card."""
        if kind == OPP:
            with torch.no_grad():
                latents = self.model.represent_hypotheses(self.seat, self.card, self.events)
                probs = self.model.predict(latents).opp_policy.softmax(-1)
            self.belief = update_on_action(self.belief, probs, x)
        else:
            self.belief = update_on_public(self.belief, self.card, x)

        surprised = (self.node is None or self.node.kind != kind or x not in self.node.children
                     or self.node.outcome_probs()[x] < self.cfg.surprise_threshold)
        self.node = None if surprised else self.node.children[x][0]
        self.events.append(to_event(kind, x))

    def _plan_policy(self, legal):
        """The committed plan's action probabilities here, restricted to the real
        legal actions. None if there is no usable plan."""
        if self.node is None or self.node.kind != MY or self.node.policy is None:
            return None
        probs = torch.zeros(3)
        probs[legal] = self.node.policy[legal]
        if probs.sum() <= 0:
            return None
        return probs / probs.sum()

    def _replan(self, legal):
        with torch.no_grad():
            latents = self.model.represent_hypotheses(self.seat, self.card, self.events)
        self.node, _ = build_tree(self.model, latents, self.belief, legal, self.cfg.n_plans,
                                  self.cfg.max_depth, self.cfg.policy_temperature, self.rng)
        self.n_replans += 1
