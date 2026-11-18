import numpy as np
import pytest

from plan_rl.cfr import CFR, expected_value, private_deals, uniform_policy
from plan_rl.leduc import CALL, FOLD, N_ACTIONS


def always_call(key, legal):
    probs = np.zeros(N_ACTIONS)
    probs[CALL] = 1.0
    return probs


def test_private_deals_sum_to_one():
    assert sum(p for _, p in private_deals()) == pytest.approx(1.0)


def test_expected_value_symmetric_game():
    # Both always call: pure showdown, so by symmetry the expected value is 0.
    assert expected_value(always_call, always_call) == pytest.approx(0.0, abs=1e-12)


def test_cfr_policy_is_valid():
    cfr = CFR()
    cfr.run(10)
    policy = cfr.average_policy()
    for key in cfr.strategy_sum:
        probs = policy(key, [CALL])
        assert probs.sum() == pytest.approx(1.0)
        assert (probs >= 0).all()
    first_move = policy((0, None, ((),)), [CALL])
    assert first_move[FOLD] == 0                             # fold is illegal at the start


def test_cfr_beats_random():
    cfr = CFR()
    cfr.run(50)
    policy = cfr.average_policy()
    both_seats = (expected_value(policy, uniform_policy) - expected_value(uniform_policy, policy)) / 2
    assert both_seats > 0
