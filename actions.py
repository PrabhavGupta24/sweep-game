from typing import List
from sweep_game import SweepGame, Card, Pile
from abc import ABC, abstractmethod
from enum import Enum

# These Action-related classes should technically be inside the SweepGame class
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


class PickUpAction(Action):

    def __init__(self, played_card, value, other_cards=[]):
        super().__init__(
            ActionType.PICK_UP, played_card, played_card.value, other_cards
        )

    def execute(self, game: SweepGame):
        picked_up_cards = []
        game.players[game.turn].hand.remove(self.played_card)
        picked_up_cards.append(self.played_card)

        for item in self.other_cards:
            if isinstance(item, Pile):
                del game.piles[item.value]
                picked_up_cards += item.cards
            else:
                picked_up_cards.append(item)

            game.table.remove(item)

        game.players[game.turn].hand += picked_up_cards

        if len(game.table) == 0:
            game.players[game.turn].sweeps += 1


class PileOnAction(Action):

    def __init__(self, played_card, value, other_cards=[]):
        super().__init__(ActionType.PILE_ON, played_card, value, other_cards)

    def execute(self, game: SweepGame):
        pile_cards = []
        game.players[game.turn].hand.remove(self.played_card)
        pile_cards.append(self.played_card)
        creators = set()

        for item in self.other_cards:
            if isinstance(item, Pile):
                if item.value == self.value:
                    creators = item.creators
                del game.piles[item.value]
                pile_cards += item.cards
            else:
                pile_cards.append(item)

            game.table.remove(item)

        new_pile = Pile(creators, pile_cards, self.value)
        game.table.append(new_pile)
        game.piles[self.value] = new_pile


class ThrowAction(Action):
    def __init__(self, played_card, value, other_cards=[]):
        super().__init__(ActionType.THROW, played_card, played_card.value, [])

    def execute(self, game: SweepGame):
        game.players[game.turn].hand.remove(self.played_card)
        game.table.append(self.played_card)


def create_action(
    action_type: ActionType, played_card: Card, value: int, other_cards: List[Card] = []
) -> Action:
    cls_map = {
        ActionType.PICK_UP: PickUpAction,
        ActionType.PILE_ON: PileOnAction,
        ActionType.THROW: ThrowAction,
    }
    return cls_map[action_type](played_card, value, other_cards)
