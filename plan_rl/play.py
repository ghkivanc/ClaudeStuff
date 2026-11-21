"""Play Leduc hold'em against the trained bot in the terminal.

Run from the repository root:
    python -m plan_rl.play                      # against the planning bot (world_model.pt)
    python -m plan_rl.play --model other.pt     # another saved model
    python -m plan_rl.play --cfr                # against CFR instead

You and the bot swap seats every hand. Type f / c / r to fold, check-or-call,
bet-or-raise, and q to stop.
"""
import argparse
import random

import numpy as np
import torch

from .agent import PlanningAgent
from .cfr import CFR
from .evaluate import PolicyPlayer
from .leduc import CALL, FOLD, RAISE, RANK_NAMES, State, contributions, payoff, play_hand
from .model import WorldModel
from .train import Config

RULES = """Leduc hold'em
  Deck: J J Q Q K K. You both put in 1 chip and get one hidden card.
  Round 1: bets/raises are 2 chips. Then a public card is turned up.
  Round 2: bets/raises are 4 chips. At most 2 bets/raises per round.
  Showdown: pairing the public card wins, otherwise the higher card wins.
"""


class Quit(Exception):
    pass


def action_word(action, earlier_in_round):
    """Poker word for an action, given the actions before it in the same round."""
    facing_raise = bool(earlier_in_round) and earlier_in_round[-1] == RAISE
    if action == FOLD:
        return "fold"
    if action == CALL:
        return "call" if facing_raise else "check"
    return "raise" if RAISE in earlier_in_round else "bet"


def describe_betting(rounds, my_seat):
    lines = []
    for r, actions in enumerate(rounds):
        words = [("you" if i % 2 == my_seat else "bot") + " " + action_word(a, actions[:i])
                 for i, a in enumerate(actions)]
        lines.append(f"  round {r + 1}: " + (", ".join(words) if words else "-"))
    return "\n".join(lines)


def chips_in(public, rounds):
    """Chips each seat has put in (only the betting matters, not the cards)."""
    return contributions(State(cards=None, public=public, rounds=rounds))


class HumanPlayer:
    KEYS = {"f": FOLD, "c": CALL, "r": RAISE}

    def new_hand(self, seat, card):
        self.seat = seat
        print("\n" + "=" * 40)
        print(f"New hand. You act {'first' if seat == 0 else 'second'}. Your card: {RANK_NAMES[card]}")

    def act(self, info):
        chips = chips_in(info.public, info.rounds)
        me, bot = chips[info.seat], chips[1 - info.seat]
        public = RANK_NAMES[info.public] if info.public is not None else "not dealt yet"
        print(f"\nYour card: {RANK_NAMES[info.card]}   Public card: {public}")
        print(f"Chips in: you {me}, bot {bot}")
        print(describe_betting(info.rounds, info.seat))

        current = info.rounds[-1]
        options = {k: a for k, a in self.KEYS.items() if a in info.legal}
        menu = "  ".join(f"[{k}] {action_word(a, current)}" for k, a in options.items())
        while True:
            answer = input(f"{menu}  [q] quit > ").strip().lower()
            if answer == "q":
                raise Quit
            if answer in options:
                return options[answer]
            print("Please type one of:", ", ".join(options), "or q")


def hand_name(card, public):
    return f"pair of {RANK_NAMES[card]}s" if card == public else f"{RANK_NAMES[card]} high"


def show_result(final, human_seat, result):
    bot_seat = 1 - human_seat
    print("\n" + describe_betting(final.rounds, human_seat))
    last = final.rounds[-1]
    if last[-1] == FOLD:
        folder = "You" if (len(last) - 1) % 2 == human_seat else "Bot"
        print(f"{folder} folded. (Bot had {RANK_NAMES[final.cards[bot_seat]]}.)")
    else:
        print(f"Showdown, public card {RANK_NAMES[final.public]}: "
              f"you {hand_name(final.cards[human_seat], final.public)}, "
              f"bot {hand_name(final.cards[bot_seat], final.public)}.")
    verdict = "win" if result > 0 else "lose" if result < 0 else "split, net"
    print(f"You {verdict} {abs(result)} chips.")


def load_bot(args, cfg):
    rng = np.random.default_rng()
    if args.cfr:
        print(f"Computing the CFR strategy ({cfg.cfr_iterations} iterations), takes a few minutes ...")
        cfr = CFR()
        cfr.run(cfg.cfr_iterations)
        return PolicyPlayer(cfr.average_policy(), rng)
    model = WorldModel(cfg.d_model, cfg.mamba_layers)
    model.load_state_dict(torch.load(args.model))
    return PlanningAgent(model, cfg, rng)


def main():
    parser = argparse.ArgumentParser(description="Play Leduc hold'em against the bot.")
    parser.add_argument("--model", default="world_model.pt")
    parser.add_argument("--cfr", action="store_true", help="play against CFR instead")
    args = parser.parse_args()

    cfg = Config()
    bot, human = load_bot(args, cfg), HumanPlayer()
    print(RULES)
    total, hands, deck_rng = 0, 0, random.Random()
    try:
        while True:
            human_seat = hands % 2
            players = (human, bot) if human_seat == 0 else (bot, human)
            final = play_hand(*players, deck_rng)
            result = payoff(final) if human_seat == 0 else -payoff(final)
            show_result(final, human_seat, result)
            total, hands = total + result, hands + 1
            print(f"Score: {total:+d} chips after {hands} hands ({total / hands:+.2f} per hand)")
    except (Quit, KeyboardInterrupt, EOFError):
        pass
    if hands:
        print(f"\nFinal score: {total:+d} chips over {hands} hands ({total / hands:+.2f} per hand).")


if __name__ == "__main__":
    main()
