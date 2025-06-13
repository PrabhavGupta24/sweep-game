from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sweep_game import SweepGame


# Constants for card values
CARD_VALUES = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "J": 11, "Q": 12, "K": 13
}

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = list(CARD_VALUES.keys())

class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        self.value = CARD_VALUES[rank]
        self.points = 0

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return NotImplemented

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __lt__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value < other.value
        return NotImplemented


class Player:
    def __init__(self, name: str, is_ai: bool = False):
        self.name = name
        self.hand: List[Card] = []
        self.captured: List[Card] = []
        self.sweeps: int = 0
        self.is_ai = is_ai
        self.points: int = 0

    def __str__(self):
        return self.name


class Pile:

    def __init__(self, creators: Set[Player], cards: List[Card], value: int):
        self.creators: Set[Player] = creators
        self.value = value
        self.cards = cards

        sum_of_cards = sum(card.value for card in cards)
        self.doubled = False

        if sum_of_cards % value == 0:
            if sum_of_cards // value >= 2:
                self.doubled = True
        else:
            raise ValueError("Bad Sum of Cards in Pile")

    def __repr__(self):
        return f"Pile of {self.value}: {self.cards}"

    def __lt__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value < other.value


class ActionType(Enum):
    PICK_UP = 1
    PILE_ON = 2
    THROW = 3


class Action(ABC):

    def __init__(
        self,
        action_type: ActionType,
        played_card: Card,
        value: int,
        other_cards: List[Card] = [],
    ):
        self.action_type = action_type
        self.played_card = played_card
        self.value = value
        self.other_cards = other_cards

    @abstractmethod
    def execute(self, game: SweepGame):
        pass

    def __str__(self):
        base = f"[{self.action_type.name} (Value: {self.value})] {self.played_card}"
        if self.other_cards:
            return f"{base} with {self.other_cards}"
        return base
