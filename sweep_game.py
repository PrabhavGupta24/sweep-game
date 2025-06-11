from typing import List, Union, Dict, Set
import random
from abc import ABC, abstractmethod
from enum import Enum
from itertools import combinations
import time

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

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def __eq__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value > other.value
        elif isinstance(other, int):
            return self.value > other
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        return NotImplemented

    def __hash__(self):
        # Required if you want to use Card in sets or as dict keys, bc implemented __eq__
        return hash((self.rank, self.suit))

    def __add__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value + other.value
        elif isinstance(other, int):
            return self.value + other
        return NotImplemented

    def __radd__(self, other):
        # This lets sum() work when starting from 0 (an int)
        if isinstance(other, int):
            return other + self.value
        return NotImplemented


class Player:
    def __init__(self, name: str):
        self.name = name
        self.hand: List[Card] = []
        self.captured: List[Card] = []
        self.sweeps: int = 0

class Pile:
    def __init__(self, creator: Player, cards: List[Card], value: int):
        self.creators: Set[Player] = {creator}
        self.value = value
        self.cards = cards
        self.doubled = False

    def __repr__(self):
        return f"Pile of {self.value}: {self.cards}"

    def add_to_pile(self, creator: Player, cards: List[Card]):
        cards_val = sum(cards)
        if cards_val != self.value:
            return False
        self.cards = self.cards + cards
        self.doubled = True
        self.creators.add(creator)
        return True

    def __eq__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value > other.value
        elif isinstance(other, int):
            return self.value > other
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value < other.value
        elif isinstance(other, int):
            return self.value < other
        return NotImplemented

    def __hash__(self):
        # Required if you want to use Card in sets or as dict keys, bc implemented __eq__
        return hash((self.value, self.doubled))

    def __add__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value + other.value
        elif isinstance(other, int):
            return self.value + other
        return NotImplemented

    def __radd__(self, other):
        # This lets sum() work when starting from 0 (an int)
        if isinstance(other, int):
            return other + self.value
        return NotImplemented


class ActionType(Enum):
    PICK_UP = 1
    PILE_ON = 2
    THROW = 3

class Action:
    def __init__(self, action_type: ActionType, played_card: Card, value: int, other_cards: List[Card] = []):
        self.action_type = action_type
        self.played_card = played_card
        self.value = value
        self.other_cards = other_cards

    def __str__(self):
        if self.other_cards:
            return f'Action: {self.action_type}, Card: {self.played_card}, Value: {self.value}, Other Cards: {self.other_cards}'

        return f'Action: {self.action_type}, Card: {self.played_card}, Value: {self.value}'

# Game class
class SweepGame:
    def __init__(self):
        self.deck: List[Card] = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.table: List[Union[Card, Pile]] = []
        self.players = [Player("Player 1"), Player("Player 2")]
        self.round = 0 # Needed??
        self.piles: Dict[int, Pile] = {}
        self.last_to_pick_up = None
        self.point_differential = 0
        self.turn = 1 if self.point_differential < 0 else 0  # Index of current player

        num = 2 if self.round == 0 else 3
        for i in range(num):
            self.players[self.turn].hand.append(self.deck[0:4])
            self.players[1 - self.turn].hand.append(self.deck[4:8])
            self.deck = self.deck[8:]

    def setup_new_round(self):
        self.deck = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.table = []
        self.round += 1
        self.piles = {}
        self.last_to_pick_up = None
        self.turn = 1 if self.point_differential < 0 else 0  # Index of current player

        while True:
            random.shuffle(self.deck)
            self.table = self.deck[:4]
            self.players[self.turn].hand = self.deck[4:8]
            self.players[1 - self.turn].hand = self.deck[8:12]
            self.deck = self.deck[12:]

            if any(card.value >= 9 for card in self.players[self.turn].hand):
                break

    def first_move_finish_setup(self):
        declared = max(self.players[self.turn].hand)
        # Play the actual move

        self.turn = 1 - self.turn

        # Deal remaining cards for round one, starting with player who just played
        for i in range(2):
            self.players[1 - self.turn].hand += self.deck[:4]
            self.players[self.turn].hand += self.deck[4:8]
            self.deck = self.deck[8:]

        for player in self.players:
            player.hand.sort()

    def play_round(self):
        # self.table.append(
        #     Pile(self.players[1 - self.turn], [Card('7', SUITS[2]), Card('3', SUITS[0])], 10)
        # )
        # self.table.append(
        #     Pile(
        #         self.players[1 - self.turn],
        #         [
        #             Card("8", SUITS[2]),
        #             Card('3', SUITS[0]),
        #             Card('4', SUITS[2]),
        #             Card('7', SUITS[0]),
        #         ], 11
        #     )
        # )
        # self.table.append(
        #     Pile(self.players[self.turn], [Card('10', SUITS[2]), Card('3', SUITS[0])], 12)
        # )

        self.players[self.turn].hand = [
            # Card("2", SUITS[0]),
            # Card("7", SUITS[0]),
            # Card("8", SUITS[0]),
            Card("9", SUITS[0]),
        ]

        self.table = [
            Card("2", SUITS[1]),
            Card("7", SUITS[0]),
            Card("9", SUITS[1]),
            Card("9", SUITS[2]),
        ]

        self.get_valid_actions()
        pass

    def get_valid_actions(self):
        cards = self.players[self.turn].hand
        card_set = set(cards)
        print(cards)
        print(self.table)
        actions: List[Action] = []
        for card in cards:
            value = card.value
            can_throw = True

            # Pick Up Check
            combos = self.number_combinations(value)
            for c in combos:
                actions.append(Action(ActionType.PICK_UP, card, value, c))
                can_throw = False

            # Pile On Check
            # other_cards = list(card_set - {card})
            # for v in range(9, 14):
            #     if v in other_cards:
            #         combos = self.number_combinations(v, card)
            #         for c in combos:
            #             actions.append(Action(ActionType.PILE_ON, card, v, c))
            #             if value in self.piles and self.players[self.turn] in self.piles[value].creators:
            #                 can_throw = False

            # Throwing Actions
            # Check all other actions for:
            # - Pick Up with that value
            # - or Pile On an existing pile that you created
            if can_throw:
                actions.append(Action(ActionType.THROW, card, value))

        actions.sort(key=lambda action: action.action_type.value)
        for act in actions:
            print(act)

    def number_combinations(self, value, addition: Card = None):

        # equalities = [x for x in self.table if x == value]
        # table = [
        #     x
        #     for x in self.table
        #     if x < value
        #     and not (
        #         isinstance(x, Pile)
        #         and (x.doubled or self.players[self.turn] in x.creators)
        #     )
        # ]

        # addition_is_equal = False

        # if addition:
        #     if addition == value:
        #         equalities.append(addition)
        #         addition_is_equal = True
        #     elif addition < value:
        #         table.append(addition)
        #     else:
        #         return []

        # table.sort()

        target_value_combos = [
            list(combo)
            for r in range(1, len(self.table) + 1)
            for combo in combinations(self.table, r)
            if sum(combo) == value
        ]
        print(value, target_value_combos)

        all_combos_of_combos = [
            list(combo)
            for r in range(1, len(target_value_combos) + 1)
            for combo in combinations(target_value_combos, r)
        ]

        print(all_combos_of_combos)

        unique_maximal_combos = []
        for combo in all_combos_of_combos:
            used_cards = set()

            invalid_combo = False
            for c in combo:
                for card in c:
                    if card in used_cards:
                        invalid_combo = True
                        break
                    used_cards.add(card)

            if invalid_combo:
                continue

            # if not (addition_is_equal) and addition not in used_cards:
            #     continue

            unused_combos = [c for c in target_value_combos if c not in combo]
            print("UNUSED", unused_combos)
            for uc in unused_combos:
                repeated_card = False
                for card in uc:
                    if card in used_cards:
                        repeated_card = True
                        break
                if not repeated_card:
                    invalid_combo = True

            if not invalid_combo:
                unique_maximal_combos += combo

        # unique_maximal_combos = [combo + equalities for combo in unique_maximal_combos]
        # if not (unique_maximal_combos) and equalities:
        #     unique_maximal_combos.append(equalities)

        return unique_maximal_combos

    def run_game(self):
        self.setup_new_round()
        self.first_move_finish_setup()
        self.play_round()


if __name__ == "__main__":
    game = SweepGame()
    game.run_game()
