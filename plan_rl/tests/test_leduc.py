import random

from plan_rl.leduc import (CALL, FOLD, RAISE, State, apply_action, contributions, deal_public,
                           is_chance, is_terminal, legal_actions, payoff, play_hand, replay)


def play(cards, public, *rounds):
    """Build a state from per-round action lists, checking legality on the way."""
    s = State(cards)
    for r, actions in enumerate(rounds):
        if r == 1:
            assert is_chance(s)
            s = deal_public(s, public)
        for a in actions:
            assert a in legal_actions(s)
            s = apply_action(s, a)
    return s


def test_legal_actions():
    s = State((0, 1))
    assert legal_actions(s) == [CALL, RAISE]                 # nothing to fold against
    s = apply_action(s, RAISE)
    assert legal_actions(s) == [FOLD, CALL, RAISE]
    s = apply_action(s, RAISE)
    assert legal_actions(s) == [FOLD, CALL]                  # max 2 raises per round


def test_round_end_goes_to_chance():
    assert is_chance(play((0, 1), None, [CALL, CALL]))
    assert is_chance(play((0, 1), None, [RAISE, CALL]))
    assert not is_chance(play((0, 1), None, [CALL, RAISE]))


def test_fold_payoffs():
    s = play((0, 1), None, [RAISE, FOLD])
    assert is_terminal(s) and payoff(s) == 1                 # player 1 loses the ante
    s = play((0, 1), None, [CALL, RAISE, FOLD])
    assert is_terminal(s) and payoff(s) == -1                # player 0 loses the ante


def test_showdowns():
    checks = ([CALL, CALL], [CALL, CALL])
    assert payoff(play((2, 0), 1, *checks)) == 1             # higher card
    assert payoff(play((0, 2), 0, *checks)) == 1             # pair beats higher card
    assert payoff(play((1, 1), 0, *checks)) == 0             # split


def test_bet_sizes():
    s = play((0, 2), 1, [RAISE, CALL], [RAISE, CALL])
    assert contributions(s) == [7, 7]                        # 1 + 2 + 4
    assert payoff(s) == -7
    assert contributions(play((0, 2), None, [RAISE, RAISE, CALL])) == [5, 5]


def test_replay_order():
    final = play((0, 1), 2, [CALL, RAISE, CALL], [CALL, CALL])
    assert [e for _, e in replay(final)] == [
        ("action", CALL), ("action", RAISE), ("action", CALL),
        ("chance", 2), ("action", CALL), ("action", CALL)]


class AlwaysCall:
    def new_hand(self, seat, card):
        pass

    def act(self, info):
        return CALL


def test_play_hand():
    final = play_hand(AlwaysCall(), AlwaysCall(), random.Random(0))
    assert is_terminal(final)
    assert final.rounds == ((CALL, CALL), (CALL, CALL))
