"""Planning agent: own world model, own belief, commit until surprised.

Each agent (learner or opponent) is a separate instance with its own tree and
belief, as in a real game.

  - Real belief over (opponent card, opponent type), updated after every event:
    public card -> exact Bayes from the deck;
    opponent action -> Bayes with the model's opponent head as the likelihood
    (approximate: the opponent plans too, and the head stands in for their planner).
  - Plan: build a tree, then follow its chosen action at each of my turns.
  - Surprise: the opponent's action or the public card is not in the tree, or had
    probability below `surprise_threshold` there -> drop the plan, replan at my turn.
"""
import torch

from .belief import initial_belief, update_on_action, update_on_public
from .model import MY, OPP, history_events, to_event
from .plan_tree import build_tree, policy_target


class PlanningAgent:
    def __init__(self, model, n_known_types, cfg, rng):
        self.model = model
        self.n_known_types = n_known_types
        self.cfg = cfg
        self.rng = rng

    def new_hand(self, seat, card):
        self.seat, self.card = seat, card
        self.events = []          # event ids seen so far, from my point of view
        self.belief = initial_belief(card, self.model.n_types, self.n_known_types)
        self.node = None          # where I am in the committed plan (None = no plan)
        self.targets = {}         # event index of each of my decisions -> policy target
        self.n_replans = 0

    def act(self, info):
        history = history_events(info.seat, info.public, info.rounds)
        for kind, x in history[len(self.events):]:      # everything since my last turn
            self._observe(kind, x)

        if not self._plan_usable(info.legal):
            self._replan(info.legal)
        action = self.node.choice
        self.targets[len(self.events)] = policy_target(self.node, self.cfg.policy_temperature)

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
            self.belief = update_on_public(self.belief, self.card, x, self.model.n_types)

        surprised = (self.node is None or self.node.kind != kind or x not in self.node.children
                     or self.node.outcome_probs()[x] < self.cfg.surprise_threshold)
        self.node = None if surprised else self.node.children[x][0]
        self.events.append(to_event(kind, x))

    def _plan_usable(self, legal):
        return (self.node is not None and self.node.kind == MY
                and self.node.choice is not None and self.node.choice in legal)

    def _replan(self, legal):
        with torch.no_grad():
            latents = self.model.represent_hypotheses(self.seat, self.card, self.events)
        self.node, _ = build_tree(self.model, latents, self.belief, legal,
                                  self.cfg.n_plans, self.cfg.max_depth, self.rng)
        self.n_replans += 1
