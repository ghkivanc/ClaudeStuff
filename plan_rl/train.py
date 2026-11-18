"""Self-play training of the planning agent on Leduc.

Each hand: the learner plays a snapshot sampled from the population (the
snapshot's slot is the opponent's "type"). Both sides plan. After the hand,
both cards are known, so the learner's view of the hand becomes a Record with
targets for every head of the world model:

  node kind, legal actions  : from the rules
  my policy                 : softmax of my tree's action values (policy_target)
  opponent policy           : the action the opponent actually took
  public card               : the card actually dealt
  reward / value            : chips I won (all reward arrives at the end)

The model is trained MuZero-style: represent a random prefix of the hand with
the TRUE hypothesis, then unroll the dynamics over the rest of the hand.

Run:  python -m plan_rl.train
"""
import copy
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .agent import PlanningAgent
from .leduc import current_player, legal_actions, payoff, play_hand, replay
from .model import CHANCE, MY, OPP, PAD, TERMINAL, WorldModel, to_event


@dataclass
class Config:
    # planning
    n_plans: int = 32                 # plans sampled per decision
    max_depth: int = 10               # events per plan (a Leduc hand has at most 9)
    surprise_threshold: float = 0.05
    policy_temperature: float = 1.0   # softmax temperature of the policy target (chips)
    # model
    n_types: int = 8                  # population size = number of opponent types
    d_model: int = 64
    mamba_layers: int = 2
    # self-play and learning
    hands: int = 5000
    train_every: int = 20             # hands between training phases
    updates_per_phase: int = 10
    batch_size: int = 64
    buffer_hands: int = 5000
    snapshot_every: int = 250         # hands between new snapshots
    lr: float = 3e-4
    log_every: int = 200
    # evaluation
    eval_hands: int = 1000
    cfr_iterations: int = 1000
    seed: int = 0


def known_types(cfg):
    """Population slots filled by the end of training."""
    return min(cfg.n_types, 1 + cfg.hands // cfg.snapshot_every)


# ---------------------------------------------------------------- records

@dataclass
class Record:
    seat: int
    my_card: int
    opp_card: int
    opp_type: int
    events: list    # event ids from my point of view
    kinds: list     # node kind before each event, plus TERMINAL at the end
    legal: list     # legal-action mask before each event ([False]*3 at the chance node)
    targets: dict   # event index of my decisions -> policy target (3,)
    payoff: float   # chips I won


def make_record(final, seat, opp_type, targets):
    events, kinds, legal = [], [], []
    for s, (what, x) in replay(final):
        if what == "chance":
            kind, mask = CHANCE, [False] * 3
        else:
            kind = MY if current_player(s) == seat else OPP
            mask = [a in legal_actions(s) for a in range(3)]
        events.append(to_event(kind, x))
        kinds.append(kind)
        legal.append(mask)
    kinds.append(TERMINAL)
    my_payoff = payoff(final) if seat == 0 else -payoff(final)
    return Record(seat, final.cards[seat], final.cards[1 - seat], opp_type,
                  events, kinds, legal, targets, float(my_payoff))


# ---------------------------------------------------------------- loss

def _masked_mean(per_item, mask):
    mask = mask.float()
    return (per_item * mask).sum() / mask.sum().clamp(min=1)


def _node_loss(pred, records, nodes):
    """Loss of the prediction heads at node index nodes[i] of records[i] (None = past the end)."""
    valid = torch.tensor([n is not None for n in nodes])
    kinds = torch.tensor([r.kinds[n] if n is not None else 0 for r, n in zip(records, nodes)])
    loss = _masked_mean(F.cross_entropy(pred.kind, kinds, reduction="none"), valid)

    value_target = torch.tensor([r.payoff if n is not None and r.kinds[n] != TERMINAL else 0.0
                                 for r, n in zip(records, nodes)])
    loss = loss + _masked_mean((pred.value - value_target) ** 2, valid)

    decision = valid & ((kinds == MY) | (kinds == OPP))
    legal = torch.tensor([r.legal[n] if d else [False] * 3
                          for r, n, d in zip(records, nodes, decision.tolist())]).float()
    legal_loss = F.binary_cross_entropy_with_logits(pred.legal, legal, reduction="none").sum(-1)
    loss = loss + _masked_mean(legal_loss, decision)

    has_target = torch.tensor([d and k == MY and n in r.targets for r, n, k, d
                               in zip(records, nodes, kinds.tolist(), decision.tolist())])
    my_target = torch.stack([r.targets[n] if h else torch.zeros(3)
                             for r, n, h in zip(records, nodes, has_target.tolist())])
    my_loss = -(my_target * F.log_softmax(pred.my_policy, -1)).sum(-1)
    loss = loss + _masked_mean(my_loss, has_target)

    # Opponent actions are events 3..5 and public cards 6..8, so the class is event % 3.
    outcome = torch.tensor([r.events[n] % 3 if n is not None and n < len(r.events) else 0
                            for r, n in zip(records, nodes)])
    opp_mask = valid & (kinds == OPP)
    loss = loss + _masked_mean(F.cross_entropy(pred.opp_policy, outcome, reduction="none"), opp_mask)
    chance_mask = valid & (kinds == CHANCE)
    loss = loss + _masked_mean(F.cross_entropy(pred.chance, outcome, reduction="none"), chance_mask)
    return loss


def model_loss(model, records, rng):
    """Represent a random prefix of each hand (true hypothesis), unroll over the rest."""
    starts = [int(rng.integers(len(r.events) + 1)) for r in records]
    prefix = torch.full((len(records), max(starts)), PAD, dtype=torch.long)
    for i, (r, k) in enumerate(zip(records, starts)):
        prefix[i, :k] = torch.tensor(r.events[:k], dtype=torch.long)
    latent = model.represent(torch.tensor([r.seat for r in records]),
                             torch.tensor([r.my_card for r in records]),
                             torch.tensor([r.opp_card for r in records]),
                             torch.tensor([r.opp_type for r in records]),
                             prefix, torch.tensor(starts))

    unroll = max(len(r.events) - k for r, k in zip(records, starts))
    loss = 0.0
    for j in range(unroll + 1):
        nodes = [k + j if k + j <= len(r.events) else None for r, k in zip(records, starts)]
        loss = loss + _node_loss(model.predict(latent), records, nodes)
        if j == unroll:
            break
        has_next = torch.tensor([n is not None and n < len(r.events) for r, n in zip(records, nodes)])
        event = torch.tensor([r.events[n] if h else 0
                              for r, n, h in zip(records, nodes, has_next.tolist())])
        reward_target = torch.tensor([r.payoff if h and r.kinds[n + 1] == TERMINAL else 0.0
                                      for r, n, h in zip(records, nodes, has_next.tolist())])
        latent, reward = model.dynamics(latent, event)
        loss = loss + _masked_mean((reward - reward_target) ** 2, has_next)
    return loss


# ---------------------------------------------------------------- population

class Population:
    """Frozen snapshots of the learner. The slot index is the opponent's type."""

    def __init__(self, n_types):
        self.n_types = n_types
        self.slots = []        # (frozen model, number of types it knew about)
        self.next_slot = 0     # slot replaced next once full (oldest first)

    def add(self, model):
        """Returns (slot, replaced)."""
        frozen = copy.deepcopy(model).eval().requires_grad_(False)
        snapshot = (frozen, max(1, len(self.slots)))
        if len(self.slots) < self.n_types:
            self.slots.append(snapshot)
            return len(self.slots) - 1, False
        slot = self.next_slot
        self.slots[slot] = snapshot
        self.next_slot = (slot + 1) % self.n_types
        return slot, True


# ---------------------------------------------------------------- training loop

def train(cfg):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    py_rng = random.Random(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = WorldModel(cfg.n_types, cfg.d_model, cfg.mamba_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    population = Population(cfg.n_types)
    population.add(model)
    buffer = deque(maxlen=cfg.buffer_hands)
    payoffs, replans, losses = [], [], []

    for hand in range(cfg.hands):
        slot = int(rng.integers(len(population.slots)))
        opp_model, opp_known = population.slots[slot]
        learner = PlanningAgent(model, len(population.slots), cfg, rng)
        opponent = PlanningAgent(opp_model, opp_known, cfg, rng)
        seat = hand % 2
        players = (learner, opponent) if seat == 0 else (opponent, learner)
        final = play_hand(*players, py_rng)

        record = make_record(final, seat, slot, learner.targets)
        buffer.append(record)
        payoffs.append(record.payoff)
        replans.append(learner.n_replans)

        if (hand + 1) % cfg.train_every == 0:
            for _ in range(cfg.updates_per_phase):
                batch = py_rng.sample(list(buffer), min(cfg.batch_size, len(buffer)))
                loss = model_loss(model, batch, rng)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(loss.item())

        if (hand + 1) % cfg.snapshot_every == 0:
            slot, replaced = population.add(model)
            if replaced:   # this type is now a different opponent: forget the old one
                buffer = deque((r for r in buffer if r.opp_type != slot), maxlen=cfg.buffer_hands)
                with torch.no_grad():
                    model.type_emb.weight[slot].normal_()

        if (hand + 1) % cfg.log_every == 0:
            print(f"hand {hand + 1:6d}  payoff/hand {np.mean(payoffs):+.3f}  "
                  f"replans/hand {np.mean(replans):.2f}  "
                  f"loss {np.mean(losses) if losses else float('nan'):.3f}  "
                  f"population {len(population.slots)}")
            payoffs, replans, losses = [], [], []
    return model


if __name__ == "__main__":
    from .evaluate import evaluate_all

    cfg = Config()
    model = train(cfg)
    torch.save(model.state_dict(), "world_model.pt")
    evaluate_all(model, cfg)
