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
            player.points = 0

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

        self._clear_screen()
        self._display_header()
        self._display_table()

        if self.players[self.turn].is_ai:
            action_to_play = random.choice(action_options)
            print(f"{self.players[self.turn]} chose action: {action_to_play}")
        else:
            self._display_player_hand()
            self._display_action_options(action_options)
            error_msg = "Invalid action choice. Please choose from the options. \n"
            while True:
                try:
                    selected_action_ind = int(
                        input(
                            f"{self.players[self.turn]}, choose an action in range 1 - {len(action_options)}: "
                        )
                    )
                    if selected_action_ind in range(1, len(action_options) + 1):
                        action_to_play = action_options[selected_action_ind - 1]
                        break
                    else:
                        print(error_msg)
                except (ValueError, IndexError):
                    print(error_msg)
                time.sleep(1)
            print()
            print(f"{self.players[self.turn]} played: {action_to_play}")

        print()

        action_to_play.execute(self)

        self._display_table(is_new=True)
        input("Press ENTER to continue.")

        self.turn = 1 - self.turn

    def first_move(self):
        self._clear_screen()
        self._display_header()

        print(f"Starting round {self.round}! {self.players[self.turn]} will play first.")

        while True:
            random.shuffle(self.deck)
            if any(card.value >= 9 for card in self.deck[4:8]):
                self.players[self.turn].hand = self.deck[4:8]
                del self.deck[4:8]
                break

        self.players[self.turn].hand.sort()

        # Get actual declared value
        if self.players[self.turn].is_ai:
            time.sleep(1)
            declared = max(card.value for card in self.players[self.turn].hand)
            print(f'{self.players[self.turn]} declares {declared}.\n')
            input("Press ENTER to continue.")
        else:
            print()
            self._display_player_hand()
            declare_choices = set(card.value for card in self.players[self.turn].hand if card.value >= 9)
            declare_choices = sorted(list(declare_choices))
            error_msg = "Invalid declaration. Please choose from the options. \n"
            while True:
                try:
                    declared = int(
                        input(
                            f"{self.players[self.turn]}, declare a value from {declare_choices}: "
                        )
                    )
                    if declared in declare_choices:
                        break
                    else:
                        print(error_msg)
                except (ValueError, IndexError):
                    print(error_msg)

                time.sleep(1)

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

        # valid_actions.sort(key=lambda action: action.played_card)
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
        print()
        if self.table:
            leftover_points = sum(card.points for card in self.table)
            self.last_to_pick_up.points += leftover_points
            self.last_to_pick_up.captured += self.table

            print(
                f"Remaining {len(self.table)} cards and {leftover_points} points go to {self.last_to_pick_up}."
            )

        if len(self.players[0].captured) > 26:
            self.players[0].points += 4
            print(f'{self.players[0]} has more cards -> +4 points')
        elif len(self.players[1].captured) > 26:
            self.players[1].points += 4
            print(f"{self.players[1]} has more cards -> +4 points")
        else:
            self.players[0].points += 2
            self.players[1].points += 2
            print(f"Both players have the same number of cards -> +2 points each")

        points_per_player = [
            player.points + (50 * player.sweeps)
            for player in self.players
        ]

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
        # print(self.players[0].hand)
        # print(self.players[1].hand)
        while len(self.players[self.turn].hand) > 0:
            self.play_turn()

        self.deal_second_half()
        # print(self.players[0].hand)
        # print(self.players[1].hand)
        while len(self.players[self.turn].hand) > 0:
            self.play_turn()

        self.end_round()
        # print("---------------------------------------------")
        # self.end_game()

    def _clear_screen(self):
        """Clears the console screen."""
        os.system("cls" if os.name == "nt" else "clear")

    def _display_header(self):
        """Displays the game title and scores."""
        print("=" * 40)
        print("                SWEEP")
        print("=" * 40)
        print("               Round", self.round)
        print("-" * 40)

        print(
            f"Cards:  {self.players[0]}: {len(self.players[0].captured)} | {self.players[1]}: {len(self.players[1].captured)}"
        )
        print(
            f"Points: {self.players[0]}: {self.players[0].points} | {self.players[1]}: {self.players[1].points}"
        )
        print(
            f"Sweeps: {self.players[0]}: {self.players[0].sweeps} | {self.players[1]}: {self.players[1].sweeps}"
        )

        print("-" * 40)
        print()

    def _display_table(self, is_new=False):
        """Displays the game table."""
        title_str = "--- New Table ---" if is_new else "--- Table ---"
        print(title_str)
        if not self.table:
            print("[Empty]")
        else:
            for item in self.table:
                print(item)
        print("-" * len(title_str), "\n")

    def _display_player_hand(self):
        player = self.players[self.turn]
        title_str = f"--- {player.name}'s Hand ---"
        print(title_str)
        print(player.hand)
        print("-" * len(title_str), '\n')

    def _display_action_options(self, action_options):
        player = self.players[self.turn]
        title_str = f"--- Action Options ---"
        print(title_str)
        for i, action in enumerate(action_options):
            print(f'({i + 1}) {action}')
        print("-" * len(title_str), "\n")


if __name__ == "__main__":
    game = SweepGame()
    game.run_game()
