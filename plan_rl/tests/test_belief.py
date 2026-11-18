import pytest
import torch

from plan_rl.belief import initial_belief, update_on_action, update_on_public


def test_initial_belief():
    assert initial_belief(my_card=0).tolist() == pytest.approx([0.2, 0.4, 0.4])  # one J, two Q, two K


def test_public_card_update():
    b = update_on_public(initial_belief(my_card=0), my_card=0, public=0)
    assert b.tolist() == pytest.approx([0.0, 0.5, 0.5])            # both J are out


def test_action_update():
    b = torch.full((3,), 1 / 3)
    probs = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.5, 0.5, 0.0]])
    b = update_on_action(b, probs, action=0)
    assert b.sum() == pytest.approx(1.0)
    assert b[0] > b[2] > b[1]
