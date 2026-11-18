import math

import numpy as np
import torch

from plan_rl.leduc import CALL, RAISE, State, apply_action, deal_public
from plan_rl.model import CHANCE, MY, OPP, TERMINAL, WorldModel
from plan_rl.train import Config, make_record, model_loss, train


def finished_hand():
    """Player 0 holds J, player 1 holds K, public Q. raise/call, check/check."""
    s = State((0, 2))
    for a in (RAISE, CALL):
        s = apply_action(s, a)
    s = deal_public(s, 1)
    for a in (CALL, CALL):
        s = apply_action(s, a)
    return s


def test_make_record_from_seat_1():
    rec = make_record(finished_hand(), seat=1, targets={1: torch.tensor([0., 1., 0.])})
    assert rec.events == [5, 1, 7, 4, 1]                     # opp raise, my call, Q, opp check, my check
    assert rec.kinds == [OPP, MY, CHANCE, OPP, MY, TERMINAL]
    assert rec.legal[0] == [False, True, True]
    assert rec.legal[1] == [True, True, True]                # facing a raise
    assert rec.legal[2] == [False, False, False]             # chance node
    assert rec.payoff == 3.0                                 # K beats J, pot 3 each
    assert (rec.my_card, rec.opp_card) == (2, 0)


def small_model():
    torch.manual_seed(0)
    return WorldModel(d_model=16, mamba_layers=1, hidden=32)


def test_model_loss_decreases_on_fixed_batch():
    model = small_model()
    records = [make_record(finished_hand(), seat, {}) for seat in (0, 1)] * 4
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = model_loss(model, records, np.random.default_rng(0)).item()
    for _ in range(40):
        loss = model_loss(model, records, np.random.default_rng(0))
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = model_loss(model, records, np.random.default_rng(0)).item()
    assert math.isfinite(first) and last < first


def test_train_smoke():
    cfg = Config(d_model=16, mamba_layers=1, n_plans=4, hands=6, train_every=3,
                 updates_per_phase=1, batch_size=4, log_every=3)
    model = train(cfg)
    assert isinstance(model, WorldModel)
