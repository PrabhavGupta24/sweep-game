"""Sweep game engine: pure rules logic, no I/O.

Implements RULES.md exactly. Drive a game with:

    game = Game(seed=42)
    while not game.game_over:
        if game.awaiting == "declare":
            game.declare(choice_from(game.declare_options()))
        else:
            game.step(choice_from(game.legal_actions()))

Players are indices 0 and 1. All cards are ints (see cards.py).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from .cards import FULL_DECK, card_points, card_str, card_value
from .combos import groups_summing_to, maximal_capture_unions

PILE_MIN = 9
PILE_MAX = 13
SWEEP_POINTS = 50
MAJORITY_BONUS = 4
TIE_BONUS = 2
DEFAULT_WIN_LEAD = 200


class ActionKind(Enum):
    PICKUP = 0
    BUILD = 1  # create a pile, or add to an existing pile at its value
    RAISE = 2  # raise an opponent's pile to a higher value
    THROW = 3


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    card: int  # the card played from hand
    value: int  # value captured at / built to / of the thrown card
    loose: tuple = ()  # loose table cards involved, sorted
    takes_pile: bool = False  # an existing pile of `value` is captured/absorbed
    raised_from: int = 0  # RAISE only: value of the pile being raised
    sweeps: bool = False  # PICKUP only: clears the table and scores a sweep

    def __str__(self):
        played = card_str(self.card)
        loose = " ".join(card_str(c) for c in self.loose)
        if self.kind is ActionKind.THROW:
            return f"Throw {played}"
        if self.kind is ActionKind.PICKUP:
            parts = [p for p in (loose, f"the pile of {self.value}" if self.takes_pile else "") if p]
            return f"Pick up {' + '.join(parts)} with {played}" + (" (SWEEP)" if self.sweeps else "")
        if self.kind is ActionKind.BUILD:
            parts = [played] + [p for p in (loose, f"the pile of {self.value}" if self.takes_pile else "") if p]
            return f"Build {self.value} from " + " + ".join(parts)
        absorbed = f", absorbing {loose}" if loose else ""
        merged = f" and the pile of {self.value}" if self.takes_pile else ""
        return f"Raise the pile of {self.raised_from} to {self.value} with {played}{absorbed}{merged}"


@dataclass(frozen=True)
class Pile:
    value: int
    cards: tuple
    creators: frozenset  # player indices that built at the current value

    def __post_init__(self):
        total = sum(card_value(c) for c in self.cards)
        if total == 0 or total % self.value != 0:
            raise ValueError(f"pile cards sum to {total}, not a multiple of {self.value}")

    @property
    def doubled(self) -> bool:
        return sum(card_value(c) for c in self.cards) >= 2 * self.value

    def __str__(self):
        tag = " (doubled)" if self.doubled else ""
        return f"Pile of {self.value}{tag}: " + " ".join(card_str(c) for c in self.cards)


class Game:
    """A full multi-round game of Sweep between players 0 and 1."""

    def __init__(self, seed=None, win_lead=DEFAULT_WIN_LEAD):
        self.rng = random.Random(seed)
        self.win_lead = win_lead
        self.round_num = 0
        self.differential = 0  # cumulative: player 0 score minus player 1 score
        self.round_history = []  # (p0 round score, p1 round score) per finished round
        self.game_over = False
        self.winner = None
        self.first_player = self.rng.randrange(2)  # round 1: random
        self._start_round()

    # ------------------------------------------------------------------ setup

    def _start_round(self):
        self.round_num += 1
        # The losing player deals, so the leader plays first; on an exact tie
        # the same player keeps dealing (first_player unchanged).
        if self.differential > 0:
            self.first_player = 0
        elif self.differential < 0:
            self.first_player = 1

        deck = list(FULL_DECK)
        while True:
            self.rng.shuffle(deck)
            if any(card_value(c) >= PILE_MIN for c in deck[4:8]):
                break  # first player's 4 cards must contain a 9 or higher

        fp = self.first_player
        self._facedown = deck[0:4]  # table cards, hidden until the declaration
        self.hands = [[], []]
        self.hands[fp] = sorted(deck[4:8])
        self.hands[1 - fp] = sorted(deck[8:12])
        self.deck = deck[12:]
        self.table = []  # loose cards only; piles live in self.piles
        self.piles = {}  # value -> Pile (at most one per value)
        self.captured = [[], []]
        self.points = [0, 0]
        self.sweeps = [0, 0]
        self.last_capturer = None
        self.declared = None
        self.opening = False  # next play must honor the declared value
        self.turn = fp
        self.awaiting = "declare"
        self.unseen = [set(FULL_DECK) - set(self.hands[0]), set(FULL_DECK) - set(self.hands[1])]

    # ------------------------------------------------------------ declaration

    def declare_options(self):
        if self.awaiting != "declare":
            raise RuntimeError("not awaiting a declaration")
        hand = self.hands[self.first_player]
        return sorted({card_value(c) for c in hand if card_value(c) >= PILE_MIN})

    def declare(self, value):
        if value not in self.declare_options():
            raise ValueError(f"cannot declare {value}")
        self.declared = value
        self.table = sorted(self._facedown)  # table is revealed to both players
        self._facedown = None
        for unseen in self.unseen:
            unseen.difference_update(self.table)
        self.awaiting = "play"
        self.opening = True

    # ---------------------------------------------------------- legal actions

    def legal_actions(self):
        if self.awaiting != "play" or self.game_over:
            raise RuntimeError("not awaiting a play")
        actions = self._all_actions(self.turn)
        if self.opening:
            # The opening move must be made at the declared value; throwing is
            # only allowed when no capture/build at that value exists.
            actions = [a for a in actions if a.value == self.declared]
            if any(a.kind is not ActionKind.THROW for a in actions):
                actions = [a for a in actions if a.kind is not ActionKind.THROW]
        return actions

    def _all_actions(self, p):
        hand = self.hands[p]
        out = []
        for card in hand:
            v = card_value(card)
            pickups = self._pickups(p, card)
            out += pickups
            other_values = {card_value(c) for c in hand if c != card}
            pile_v = self.piles.get(v)
            if pile_v and p in pile_v.creators and v not in other_values:
                # Reserved: your last card matching your own pile may only
                # capture (its pickups necessarily include the pile).
                continue
            pile_acts, feeds_own = self._pile_ons(p, card, other_values)
            out += pile_acts
            if not pickups and not feeds_own:
                out.append(Action(ActionKind.THROW, card, v))
        return out

    def _pickups(self, p, card):
        v = card_value(card)
        has_pile = v in self.piles
        equals = tuple(c for c in self.table if card_value(c) == v)
        lows = [c for c in self.table if card_value(c) < v]
        acts = []
        for union in maximal_capture_unions(groups_summing_to(lows, v)):
            loose = tuple(sorted(equals + tuple(union)))
            if not loose and not has_pile:
                continue
            clears = len(loose) == len(self.table) and len(self.piles) == (1 if has_pile else 0)
            sweep = clears and not self._is_final_play(p)
            acts.append(Action(ActionKind.PICKUP, card, v, loose, has_pile, 0, sweep))
        return acts

    def _pile_ons(self, p, card, other_values):
        acts = {}
        feeds_own = False
        cv = card_value(card)

        def absorb_options(build_value, pool):
            """Maximal absorptions at build_value: equal loose cards + unions."""
            equals = frozenset(c for c in self.table if card_value(c) == build_value)
            return [
                union | equals
                for union in maximal_capture_unions(groups_summing_to(pool, build_value))
            ]

        # Build at value B: create a pile, or add to the existing pile of B.
        for build_value in range(PILE_MIN, PILE_MAX + 1):
            if build_value not in other_values:
                continue  # must hold another card of the pile's value
            pile_b = self.piles.get(build_value)
            lows = [c for c in self.table if card_value(c) < build_value]
            # The played card's own group must sum to the pile value.
            if cv == build_value:
                card_groups = [()]
            elif cv < build_value:
                card_groups = groups_summing_to(lows, build_value - cv)
            else:
                continue
            for group in card_groups:
                rest = [c for c in lows if c not in group]
                for absorbed in absorb_options(build_value, rest):
                    loose = frozenset(group) | absorbed
                    if not loose and pile_b is None:
                        continue  # a lone hand card is a throw, not a pile
                    key = (ActionKind.BUILD, build_value, loose)
                    if key not in acts:
                        acts[key] = Action(
                            ActionKind.BUILD, card, build_value,
                            tuple(sorted(loose)), pile_b is not None,
                        )
                    if pile_b is not None and p in pile_b.creators:
                        feeds_own = True

        # Raise an opponent's pile from W to W + cv. The increment must come
        # entirely from the hand card; loose cards never group with the pile.
        for raised_from, pile_w in self.piles.items():
            if pile_w.doubled or p in pile_w.creators:
                continue
            build_value = raised_from + cv
            if build_value > PILE_MAX or build_value not in other_values:
                continue
            pile_b = self.piles.get(build_value)  # merged in if it exists
            lows = [c for c in self.table if card_value(c) < build_value]
            for absorbed in absorb_options(build_value, lows):
                key = (ActionKind.RAISE, raised_from, build_value, absorbed)
                if key not in acts:
                    acts[key] = Action(
                        ActionKind.RAISE, card, build_value,
                        tuple(sorted(absorbed)), pile_b is not None, raised_from,
                    )
                if pile_b is not None and p in pile_b.creators:
                    feeds_own = True

        return list(acts.values()), feeds_own

    def _is_final_play(self, p):
        """True when p's next play is the very last play of the round."""
        return not self.deck and len(self.hands[p]) == 1 and not self.hands[1 - p]

    # ----------------------------------------------------------------- moves

    def step(self, action: Action):
        if self.awaiting != "play" or self.game_over:
            raise RuntimeError("not awaiting a play")
        p = self.turn
        self.hands[p].remove(action.card)  # ValueError if the card isn't held
        self.unseen[1 - p].discard(action.card)

        if action.kind is ActionKind.THROW:
            self.table.append(action.card)
        elif action.kind is ActionKind.PICKUP:
            taken = [action.card]
            for c in action.loose:
                self.table.remove(c)
                taken.append(c)
            if action.takes_pile:
                taken += self.piles.pop(action.value).cards
            self.points[p] += sum(card_points(c) for c in taken)
            self.captured[p] += taken
            self.last_capturer = p
            round_continues = self.deck or self.hands[0] or self.hands[1]
            if not self.table and not self.piles and round_continues:
                self.sweeps[p] += 1
        else:  # BUILD or RAISE
            cards = [action.card]
            for c in action.loose:
                self.table.remove(c)
                cards.append(c)
            creators = {p}
            if action.kind is ActionKind.RAISE:
                cards += self.piles.pop(action.raised_from).cards
                # the raised pile's creators built at the old value, not this one
            if action.takes_pile:
                absorbed = self.piles.pop(action.value)
                cards += absorbed.cards
                creators |= absorbed.creators
            self.piles[action.value] = Pile(action.value, tuple(cards), frozenset(creators))

        self._advance(p)

    def _advance(self, p):
        if self.opening:
            self.opening = False
            for _ in range(2):  # deal out the rest of the first half
                self._deal(self.first_player, 4)
                self._deal(1 - self.first_player, 4)
        self.turn = 1 - p
        if not self.hands[0] and not self.hands[1]:
            if self.deck:
                for _ in range(3):  # second half: 12 cards each, no table cards
                    self._deal(self.turn, 4)
                    self._deal(1 - self.turn, 4)
            else:
                self._end_round()

    def _deal(self, p, n):
        dealt = self.deck[:n]
        del self.deck[:n]
        self.hands[p] += dealt
        self.hands[p].sort()
        self.unseen[p].difference_update(dealt)

    def _end_round(self):
        if self.last_capturer is not None and (self.table or self.piles):
            leftovers = list(self.table)
            for pile in self.piles.values():
                leftovers += pile.cards
            self.points[self.last_capturer] += sum(card_points(c) for c in leftovers)
            self.captured[self.last_capturer] += leftovers
        self.table = []
        self.piles = {}

        c0, c1 = len(self.captured[0]), len(self.captured[1])
        if c0 > c1:
            self.points[0] += MAJORITY_BONUS
        elif c1 > c0:
            self.points[1] += MAJORITY_BONUS
        else:
            self.points[0] += TIE_BONUS
            self.points[1] += TIE_BONUS

        scores = (
            self.points[0] + SWEEP_POINTS * self.sweeps[0],
            self.points[1] + SWEEP_POINTS * self.sweeps[1],
        )
        self.round_history.append(scores)
        self.differential += scores[0] - scores[1]
        if abs(self.differential) >= self.win_lead:
            self.game_over = True
            self.winner = 0 if self.differential > 0 else 1
            self.awaiting = "over"
        else:
            self._start_round()

    # ------------------------------------------------------------ observation

    def view(self, p):
        """Everything player p may legitimately know, as a plain dict."""
        return {
            "hand": tuple(self.hands[p]),
            "table": tuple(self.table),
            "piles": {
                v: {
                    "cards": pile.cards,
                    "mine": p in pile.creators,
                    "opponents": (1 - p) in pile.creators,
                    "doubled": pile.doubled,
                }
                for v, pile in self.piles.items()
            },
            "unseen": frozenset(self.unseen[p]),
            "points": (self.points[p], self.points[1 - p]),
            "sweeps": (self.sweeps[p], self.sweeps[1 - p]),
            "captured_counts": (len(self.captured[p]), len(self.captured[1 - p])),
            "opp_hand_count": len(self.hands[1 - p]),
            "deck_count": len(self.deck),
            "declared": self.declared,
            "is_first_player": self.first_player == p,
            "last_capturer_is_me": None if self.last_capturer is None else self.last_capturer == p,
            "second_half": not self.deck,
            "differential": self.differential if p == 0 else -self.differential,
        }
