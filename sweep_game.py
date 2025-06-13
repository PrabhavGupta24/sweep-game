from typing import List, Union, Dict
import random
from itertools import combinations
from actions import create_action
from models import Card, Pile, Action, ActionType, SUITS, RANKS, Player
import time
import os


# Game class
class SweepGame:
    def __init__(self):
        self.deck: List[Card] = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.table: List[Union[Card, Pile]] = []
        self.players = [
            Player("Player 1", is_ai=False),
            Player("Player 2 (AI)", is_ai=True),
        ]
        self.round = 0 # Needed??
        self.piles: Dict[int, Pile] = {}
        self.last_to_pick_up: Player = None
        self.point_differential = 0
        self.turn = random.randint(0, 1)  # Index of current player

        num = 2 if self.round == 0 else 3
        for i in range(num):
            self.players[self.turn].hand.append(self.deck[0:4])
            self.players[1 - self.turn].hand.append(self.deck[4:8])
            self.deck = self.deck[8:]

    def initialize_round(self):
        # set initial instance variables
        self.deck = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.table = []
        self.round += 1
        self.piles = {}
        self.last_to_pick_up = None

        # Else Case: self.turn stays as is
        if self.point_differential < 0:
            self.turn = 1
        elif self.point_differential > 0:
            self.turn = 0

        for player in self.players:
            player.hand = []
            player.captured = []
            player.sweeps = 0

        # Assign card points
        for card in self.deck:
            # SUITS[0] = spades, SUITS[2] = diamonds
            if card.suit == SUITS[0] or card.value == 1:
                card.points = card.value
            elif card.value == 10 and card.suit == SUITS[2]:
                card.points = 2

    def play_turn(self, declared_value = None):
        action_options = self.get_valid_actions()
        if declared_value:
            action_options = [action for action in action_options if action.value == declared_value]

        # Get player to select an action, default random option for now
        action_to_play = random.choice(action_options)

        print(f'{self.players[self.turn]} -- {action_to_play}')

        action_to_play.execute(self)

        self.turn = 1 - self.turn

    def first_move(self):
        print(f"\nStarting round {self.round}! {self.players[self.turn]} will play first.")

        while True:
            random.shuffle(self.deck)
            if any(card.value >= 9 for card in self.deck[4:8]):
                self.players[self.turn].hand = self.deck[4:8]
                del self.deck[4:8]
                break

        # time.sleep(2)

        # Get actual declared value
        if self.players[self.turn].is_ai:
            declared = max(card.value for card in self.players[self.turn].hand)
        else:
            declared = max(card.value for card in self.players[self.turn].hand)

        self.table = self.deck[:4]
        self.players[1 - self.turn].hand = self.deck[4:8]
        del self.deck[:8]

        self.play_turn(declared_value=declared)

        # Deal remaining cards for first half of round one, starting with player who just played
        for i in range(2):
            self.players[1 - self.turn].hand += self.deck[:4]
            self.players[self.turn].hand += self.deck[4:8]
            del self.deck[:8]

        for player in self.players:
            player.hand.sort()

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

            # Check if card is responsible for a Pile
            other_card_values = [c.value for c in cards if c != card]
            if (
                card.value in self.piles and 
                self.players[self.turn] in self.piles[card.value].creators 
                and card.value not in other_card_values
            ):
                continue

            # Pile On Check
            for v in range(9, 14):
                if v in other_card_values:
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
        if len(valid_actions) == 0:
            print("NO ACTIONS!!!")
            print(self.players[self.turn].hand)
            print(self.players[1 - self.turn].hand)
            print(self.table)
        return valid_actions

    def number_combinations(self, value, addition: Card = None):
        equalities = [x for x in self.table if x.value == value]
        table = [
            x
            for x in self.table
            if x.value < value
            and not (
                isinstance(x, Pile)
                and (not addition or x.doubled or self.players[self.turn] in x.creators)
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

        if not addition or (addition_is_equal and len(equalities) > 1):
            if not (unique_maximal_combos) and equalities:
                unique_maximal_combos.append(equalities)

        return unique_maximal_combos

    def deal_second_half(self):
        for i in range(3):
            self.players[self.turn].hand += self.deck[:4]
            self.players[1 - self.turn].hand += self.deck[4:8]
            del self.deck[:8]

        for player in self.players:
            player.hand.sort()

    def end_round(self):
        if not self.table:
            self.players[1 - self.turn].sweeps -= 1
        self.last_to_pick_up.points += sum(card.points for card in self.table)
        self.last_to_pick_up.captured += self.table

        if len(self.players[0].captured) > 26:
            self.players[0].points += 4
        elif len(self.players[1].captured) > 26:
            self.players[1].points += 4
        else:
            self.players[0].points += 2
            self.players[1].points += 2

        self.players[0].points += 2
        self.players[1].points += 2

        points_per_player = [
            player.points + (50 * player.sweeps)
            for player in self.players
        ]

        print(points_per_player)
        if points_per_player[0] == points_per_player[1]:
            print(
                f"Round Tied! ({points_per_player[0]} - {points_per_player[1]})"
            )
        else:
            max_ind = max(0, 1, key=lambda i: points_per_player[i])
            print(
                f"{self.players[max_ind]} won by {points_per_player[max_ind] - points_per_player[1 - max_ind]}! ({points_per_player[max_ind]} - {points_per_player[1 - max_ind]})"
            )

        self.point_differential += (points_per_player[0] - points_per_player[1])

    def run_game(self):
        # self.players[self.turn].hand = [
        #     Card("5", SUITS[0]),
        #     Card("9", SUITS[0]),
        #     Card("10", SUITS[3]),
        #     # Card("9", SUITS[2]),
        # ]

        # self.table = [
        #     Card("A", SUITS[0]),
        #     Card("3", SUITS[1]),
        #     Card("7", SUITS[0]),
        #     Card("7", SUITS[1]),
        #     Card("10", SUITS[1]),
        #     Card("10", SUITS[2]),
        # ]

        # self.table.append(
        #     Pile(self.players[1 - self.turn], [Card('7', SUITS[2]), Card('3', SUITS[0])], 10)
        # )
        # new_pile = Pile(
        #         {self.players[1 - self.turn]},
        #         [
        #             Card("2", SUITS[3]),
        #             Card('7', SUITS[3]),
        #             Card('4', SUITS[3]),
        #             Card('5', SUITS[3]),
        #         ], 9
        #     )
        # self.table.append(
        #     new_pile
        # )
        # self.piles[9] = new_pile
        # self.table.append(
        #     Pile(self.players[self.turn], [Card('10', SUITS[2]), Card('3', SUITS[0])], 12)
        # )

        # while abs(self.point_differential) < 200:
        # while True:
        self.initialize_round()

        self.first_move()
        print(self.players[0].hand)
        print(self.players[1].hand)
        while len(self.players[self.turn].hand) > 0:
            self.play_turn()

        self.deal_second_half()
        print(self.players[0].hand)
        print(self.players[1].hand)
        while len(self.players[self.turn].hand) > 0:
            self.play_turn()

        self.end_round()
        # print("---------------------------------------------")
        # self.end_game()

    def _clear_screen(self):
        """Clears the console screen."""
        os.system("cls" if os.name == "nt" else "clear")

    # def _display_header(self):
    #     """Displays the game title and scores."""
    #     print("=" * 40)
    #     print("          SWEEP")
    #     print("=" * 40)
    #     print(
    #         f"Scores: Team A (You & P3): {self.scores['A']} | Team B (P2 & P4): {self.scores['B']}"
    #     )
    #     print(
    #         f"Tricks this round: Team A: {self.tricks_won['A']} | Team B: {self.tricks_won['B']}"
    #     )
    #     if self.trump_suit:
    #         print(f"Trump Suit: {SUIT_NAMES[self.trump_suit]} {SUITS[self.trump_suit]}")
    #     print("-" * 40)


if __name__ == "__main__":
    game = SweepGame()
    game.run_game()
