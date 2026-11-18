"""Learned world model (MuZero-style), always from one player's ("my") view.

  represent : (deal, opponent card, event history) -> latent  [Mamba]
  dynamics  : (latent, event) -> next latent, reward for me    [MLP]
  predict   : latent -> node kind, my policy, opponent policy,
              public-card probabilities, legal actions, value  [MLP heads]

The opponent's card is unknown, so planning keeps one latent per possible
card (a hypothesis). The rules are only used to make training targets;
planning runs entirely inside this model.

Events, numbered from my point of view:
  0..2  my action (fold, call, raise)
  3..5  opponent action
  6..8  public card dealt (J, Q, K)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .leduc import N_ACTIONS, N_RANKS
from .mamba import ResidualMamba

MY, OPP, CHANCE, TERMINAL = 0, 1, 2, 3     # node kinds
N_KINDS = 4
N_EVENTS = 9
PAD = N_EVENTS                              # padding token for batched histories


def to_event(kind, x):
    """Event id of outcome x (an action or a card rank) at a node of the given kind."""
    return {MY: 0, OPP: 3, CHANCE: 6}[kind] + x


def history_events(seat, public, rounds):
    """The public history as (node kind, outcome) pairs from `seat`'s point of view."""
    out = []
    for r, actions in enumerate(rounds):
        if r == 1:
            out.append((CHANCE, public))
        for i, a in enumerate(actions):
            out.append((MY if i % 2 == seat else OPP, a))
    return out


@dataclass
class Prediction:
    kind: torch.Tensor        # (B, 4) logits
    my_policy: torch.Tensor   # (B, 3) logits
    opp_policy: torch.Tensor  # (B, 3) logits
    chance: torch.Tensor      # (B, 3) logits over the public card
    legal: torch.Tensor       # (B, 3) logits, one per action
    value: torch.Tensor       # (B,) expected chips for me from here


def _mlp(d_in, hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


class WorldModel(nn.Module):
    def __init__(self, d_model=64, mamba_layers=2, hidden=128):
        super().__init__()
        # representation
        self.deal_proj = nn.Linear(2 * N_RANKS + 2, d_model)
        self.event_emb = nn.Embedding(N_EVENTS + 1, d_model)       # + PAD
        self.blocks = nn.ModuleList([ResidualMamba(d_model) for _ in range(mamba_layers)])
        self.norm = nn.LayerNorm(d_model)
        # dynamics
        self.dyn_event_emb = nn.Embedding(N_EVENTS, d_model)
        self.dyn = _mlp(2 * d_model, hidden, d_model)
        self.dyn_norm = nn.LayerNorm(d_model)
        self.reward = _mlp(2 * d_model, hidden, 1)
        # prediction
        self.trunk = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU())
        self.kind_head = nn.Linear(hidden, N_KINDS)
        self.my_policy_head = nn.Linear(hidden, N_ACTIONS)
        self.opp_policy_head = nn.Linear(hidden, N_ACTIONS)
        self.chance_head = nn.Linear(hidden, N_RANKS)
        self.legal_head = nn.Linear(hidden, N_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)

    def represent(self, seat, my_card, opp_card, events, lengths):
        """seat, my_card, opp_card, lengths: (B,) long.
        events: (B, L) long, right-padded with PAD.
        Returns (B, d_model): the Mamba output at each sequence's last real token."""
        deal = torch.cat([F.one_hot(my_card, N_RANKS), F.one_hot(opp_card, N_RANKS),
                          F.one_hot(seat, 2)], dim=-1).float()
        first = self.deal_proj(deal)
        tokens = torch.cat([first.unsqueeze(1), self.event_emb(events)], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # Token 0 is the deal, so the last real token is at index `lengths`.
        # Mamba is causal, so the padding after it does not change it.
        return tokens[torch.arange(tokens.shape[0]), lengths]

    def represent_hypotheses(self, seat, my_card, events):
        """One latent per possible opponent card (3) for the same real history."""
        H = N_RANKS
        return self.represent(
            torch.full((H,), seat), torch.full((H,), my_card), torch.arange(H),
            torch.tensor([list(events)] * H, dtype=torch.long).reshape(H, len(events)),
            torch.full((H,), len(events)))

    def dynamics(self, latent, event):
        """latent: (B, d), event: (B,) long -> (next latent (B, d), reward (B,))"""
        x = torch.cat([latent, self.dyn_event_emb(event)], dim=-1)
        return self.dyn_norm(latent + self.dyn(x)), self.reward(x).squeeze(-1)

    def predict(self, latent):
        h = self.trunk(latent)
        return Prediction(kind=self.kind_head(h), my_policy=self.my_policy_head(h),
                          opp_policy=self.opp_policy_head(h), chance=self.chance_head(h),
                          legal=self.legal_head(h), value=self.value_head(h).squeeze(-1))
