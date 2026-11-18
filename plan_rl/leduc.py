"""Leduc hold'em rules.

Deck: two copies each of J, Q, K (ranks 0, 1, 2). Each player antes 1 chip and
gets one private card. Two betting rounds; player 0 acts first in each.
Round 1 raises are 2 chips, round 2 raises are 4, at most 2 raises per round.
The public card is dealt between the rounds. At showdown a pair with the
public card wins, otherwise the higher card wins; equal ranks split the pot.

Actions: FOLD, CALL (also "check" when nothing is owed), RAISE.
"""
from dataclasses import dataclass

FOLD, CALL, RAISE = 0, 1, 2
N_ACTIONS = 3
N_RANKS = 3
RANK_NAMES = "JQK"
DECK = (0, 0, 1, 1, 2, 2)
RAISE_SIZE = (2, 4)
MAX_RAISES = 2


@dataclass(frozen=True)
class State:
    cards: tuple          # (player 0's card, player 1's card)
    public: object = None  # public card rank, None until dealt
    rounds: tuple = ((),)  # actions of each betting round so far


def _round_over(actions):
    return len(actions) >= 2 and actions[-1] == CALL


def is_terminal(s):
    last = s.rounds[-1]
    if last and last[-1] == FOLD:
        return True
    return len(s.rounds) == 2 and _round_over(last)


def is_chance(s):
    """True when round 1 is over and the public card is still to be dealt."""
    return len(s.rounds) == 1 and _round_over(s.rounds[0])


def current_player(s):
    return len(s.rounds[-1]) % 2


def legal_actions(s):
    actions = s.rounds[-1]
    facing_raise = bool(actions) and actions[-1] == RAISE
    legal = [FOLD] if facing_raise else []
    legal.append(CALL)
    if actions.count(RAISE) < MAX_RAISES:
        legal.append(RAISE)
    return legal


def apply_action(s, a):
    return State(s.cards, s.public, s.rounds[:-1] + (s.rounds[-1] + (a,),))


def deal_public(s, rank):
    return State(s.cards, rank, s.rounds + ((),))


def remaining_counts(*seen_ranks):
    """How many copies of each rank are left in the deck after removing seen_ranks."""
    counts = [2, 2, 2]
    for r in seen_ranks:
        counts[r] -= 1
    return counts


def contributions(s):
    """Chips each player has put in the pot."""
    contrib = [1, 1]
    for r, actions in enumerate(s.rounds):
        for i, a in enumerate(actions):
            p = i % 2
            if a == CALL:
                contrib[p] = max(contrib)
            elif a == RAISE:
                contrib[p] = max(contrib) + RAISE_SIZE[r]
    return contrib


def _strength(card, public):
    return 10 + card if card == public else card


def payoff(s):
    """Chips won by player 0 (player 1 wins the negative) at a terminal state."""
    contrib = contributions(s)
    last = s.rounds[-1]
    if last[-1] == FOLD:
        folder = (len(last) - 1) % 2
        return -contrib[0] if folder == 0 else contrib[1]
    s0, s1 = _strength(s.cards[0], s.public), _strength(s.cards[1], s.public)
    if s0 == s1:
        return 0
    return contrib[1] if s0 > s1 else -contrib[0]


def replay(final):
    """Walk a finished hand from the start.

    Returns a list of (state_before, event) with event = ("action", a) or
    ("chance", rank), in the order they happened.
    """
    steps = []
    s = State(final.cards)
    for r, actions in enumerate(final.rounds):
        if r == 1:
            steps.append((s, ("chance", final.public)))
            s = deal_public(s, final.public)
        for a in actions:
            steps.append((s, ("action", a)))
            s = apply_action(s, a)
    return steps


@dataclass(frozen=True)
class InfoSet:
    """What one player sees when it is their turn."""
    seat: int
    card: int
    public: object
    rounds: tuple
    legal: list


def infoset(s, seat):
    return InfoSet(seat, s.cards[seat], s.public, s.rounds, legal_actions(s))


def play_hand(player0, player1, rng):
    """Deal and play one hand. Players need new_hand(seat, card) and act(infoset).
    Returns the final State; payoff(final) is player 0's result."""
    c0, c1, public = rng.sample(DECK, 3)
    players = (player0, player1)
    player0.new_hand(0, c0)
    player1.new_hand(1, c1)
    s = State((c0, c1))
    while not is_terminal(s):
        if is_chance(s):
            s = deal_public(s, public)
            continue
        p = current_player(s)
        a = players[p].act(infoset(s, p))
        if a not in legal_actions(s):
            raise ValueError(f"player {p} played illegal action {a}")
        s = apply_action(s, a)
    return s
