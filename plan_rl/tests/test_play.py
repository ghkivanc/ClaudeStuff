import random

import pytest

from plan_rl.leduc import CALL, FOLD, RAISE, InfoSet, State, apply_action, play_hand
from plan_rl.play import (HumanPlayer, Quit, action_word, chips_in, describe_betting,
                          show_result)


def scripted_input(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))


def test_action_words():
    assert action_word(CALL, ()) == "check"
    assert action_word(RAISE, ()) == "bet"
    assert action_word(CALL, (RAISE,)) == "call"
    assert action_word(RAISE, (RAISE,)) == "raise"
    assert action_word(FOLD, (RAISE,)) == "fold"


def test_describe_betting():
    text = describe_betting(((CALL, RAISE, CALL), ()), my_seat=0)
    assert "round 1: you check, bot bet, you call" in text
    assert "round 2: -" in text


def test_chips_in():
    assert chips_in(None, ((RAISE, CALL),)) == [3, 3]


def test_human_retries_until_valid(monkeypatch):
    scripted_input(monkeypatch, ["x", "f", "r"])            # fold is not legal at the start
    human = HumanPlayer()
    human.new_hand(0, 1)
    assert human.act(InfoSet(0, 1, None, ((),), [CALL, RAISE])) == RAISE


def test_human_can_quit(monkeypatch):
    scripted_input(monkeypatch, ["q"])
    human = HumanPlayer()
    human.new_hand(0, 1)
    with pytest.raises(Quit):
        human.act(InfoSet(0, 1, None, ((),), [CALL, RAISE]))


class AlwaysCall:
    def new_hand(self, seat, card):
        pass

    def act(self, info):
        return CALL


def test_full_hand_against_scripted_bot(monkeypatch):
    scripted_input(monkeypatch, ["c", "c"])                 # check in both rounds
    final = play_hand(HumanPlayer(), AlwaysCall(), random.Random(0))
    assert final.rounds == ((CALL, CALL), (CALL, CALL))


def test_show_result_after_fold(capsys):
    final = apply_action(apply_action(State((0, 2)), RAISE), FOLD)
    show_result(final, human_seat=0, result=1)
    out = capsys.readouterr().out
    assert "Bot folded" in out and "You win 1 chips" in out
