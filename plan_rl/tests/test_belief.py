import pytest
import torch

from plan_rl.belief import (hypothesis_cards, hypothesis_types, initial_belief, update_on_action,
                            update_on_public)


def test_hypothesis_numbering():
    # h = card * n_types + type
    assert hypothesis_cards(2).tolist() == [0, 0, 1, 1, 2, 2]
    assert hypothesis_types(2).tolist() == [0, 1, 0, 1, 0, 1]


def test_initial_belief():
    b = initial_belief(my_card=0, n_types=2, n_known_types=1).reshape(3, 2)
    assert b[:, 0].tolist() == pytest.approx([0.2, 0.4, 0.4])   # one J left, two Q, two K
    assert b[:, 1].tolist() == [0, 0, 0]                          # unfilled slot


def test_public_card_update():
    b = initial_belief(my_card=0, n_types=1, n_known_types=1)
    b = update_on_public(b, my_card=0, public=0, n_types=1)
    assert b.tolist() == pytest.approx([0.0, 0.5, 0.5])            # both J are out


def test_action_update():
    b = torch.full((3,), 1 / 3)
    probs = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.5, 0.5, 0.0]])
    b = update_on_action(b, probs, action=0)
    assert b.sum() == pytest.approx(1.0)
    assert b[0] > b[2] > b[1]
