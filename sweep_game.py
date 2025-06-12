from typing import List, Union, Dict, Set
import random
from itertools import combinations
from actions import Action, ActionType, create_action

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

    def __lt__(self, other):
        if isinstance(other, Card) or isinstance(other, Pile):
            return self.value < other.value


class Player:
    def __init__(self, name: str):
        self.name = name
        self.hand: List[Card] = []
        self.captured: List[Card] = []
        self.sweeps: int = 0

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
        declared = max(card.value for card in self.players[self.turn].hand)
        # Play the actual move
        action_options = [action for action in self.get_valid_actions() if action.value == declared]

        self.turn = 1 - self.turn

        # Deal remaining cards for round one, starting with player who just played
        for i in range(2):
            self.players[1 - self.turn].hand += self.deck[:4]
            self.players[self.turn].hand += self.deck[4:8]
            self.deck = self.deck[8:]

        for player in self.players:
            player.hand.sort()

    def play_round(self):

        self.players[self.turn].hand = [
            Card("5", SUITS[0]),
            Card("9", SUITS[0]),
            Card("10", SUITS[3]),
            Card("10", SUITS[0]),
        ]

        self.table = [
            Card("A", SUITS[0]),
            Card("3", SUITS[1]),
            Card("7", SUITS[0]),
            Card("7", SUITS[1]),
            Card("10", SUITS[1]),
            Card("10", SUITS[2]),
        ]

        # self.table.append(
        #     Pile(self.players[1 - self.turn], [Card('7', SUITS[2]), Card('3', SUITS[0])], 10)
        # )
        self.table.append(
            Pile(
                self.players[1 - self.turn],
                [
                    Card("2", SUITS[3]),
                    Card('7', SUITS[3]),
                    Card('4', SUITS[3]),
                    Card('5', SUITS[3]),
                ], 9, True
            )
        )
        # self.table.append(
        #     Pile(self.players[self.turn], [Card('10', SUITS[2]), Card('3', SUITS[0])], 12)
        # )

        while len(self.players[self.turn].hand) > 0:
            action_options = self.get_valid_actions()
            # Get player to select an action, default first option for now
            action_to_play = action_options[0]

            # Perform action
            action_to_play.execute()

            self.turn = 1 - self.turn
            break

        pass

    def get_valid_actions(self):
        cards = self.players[self.turn].hand
        valid_actions: List[Action] = []
        for card in cards:
            value = card.value
            can_throw = True

            # Pick Up Check
            combos = self.number_combinations(value)
            for c in combos:
                valid_actions.append(create_action(ActionType.PICK_UP, card, value, c))
                can_throw = False

            # Pile On Check
            other_cards_values = [c.value for c in cards if c != card]
            for v in range(9, 14):
                if v in other_cards_values:
                    combos = self.number_combinations(v, card)
                    if (
                        combos and
                        v in self.piles
                        and self.players[self.turn] in self.piles[v].creators
                    ):
                        can_throw = False
                    for c in combos:
                        c.remove(card)
                        valid_actions.append(create_action(ActionType.PILE_ON, card, v, c))

            # Throwing Actions
            # Check all other valid_actions for:
            # - Pick Up with that value
            # - or Pile On an existing pile that you created
            if can_throw:
                valid_actions.append(create_action(ActionType.THROW, card, value))

        valid_actions.sort(key=lambda action: action.action_type.value)
        # for act in valid_actions:
        #     print(act)
        return valid_actions

    def number_combinations(self, value, addition: Card = None):
        equalities = [x for x in self.table if x.value == value]
        table = [
            x
            for x in self.table
            if x.value < value
            and not (
                isinstance(x, Pile)
                and (x.doubled or self.players[self.turn] in x.creators)
            )
        ]

        addition_is_equal = False

        if addition:
            if addition.value == value:
                addition_is_equal = True
                equalities.append(addition)
            elif addition.value < value:
                table.append(addition)
            else:
                return []

        table.sort()

        target_value_combos = [
            list(combo)
            for r in range(1, len(table) + 1)
            for combo in combinations(table, r)
            if sum(card.value for card in combo) == value
        ]

        unique_maximal_combos = []

        for r in range(1, len(target_value_combos) + 1):
            for combo in combinations(target_value_combos, r):
                used_cards = set()

                invalid_combo = False
                for summation in combo:
                    for card in summation:
                        if card in used_cards:
                            invalid_combo = True
                            break
                        used_cards.add(card)

                if invalid_combo:
                    continue

                if addition and (not addition_is_equal) and addition not in used_cards:
                    continue

                unused_combos = [c for c in target_value_combos if c not in combo]
                for uc in unused_combos:
                    if not any(card in used_cards for card in uc):
                        invalid_combo = True
                        break

                if not invalid_combo:
                    unique_maximal_combos.append([card for summation in combo for card in summation] + equalities)
        if not addition or addition_is_equal:
            if not (unique_maximal_combos) and equalities:
                unique_maximal_combos.append(equalities)

        return unique_maximal_combos

    def run_game(self):
        self.setup_new_round()
        self.first_move_finish_setup()
        self.play_round()


if __name__ == "__main__":
    game = SweepGame()
    game.run_game()
