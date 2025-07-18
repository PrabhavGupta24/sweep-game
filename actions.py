from __future__ import annotations

from typing import List
from models import Card, Pile, Action, ActionType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sweep_game import SweepGame


class PickUpAction(Action):
    def __init__(self, played_card, value, other_cards=[]):
        super().__init__(
            ActionType.PICK_UP, played_card, played_card.value, other_cards
        )
        if not other_cards:
            raise ValueError("Must have other cards for PICK_UP")

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

        gained_points = sum(card.points for card in picked_up_cards)
        game.players[game.turn].points += gained_points
        game.players[game.turn].captured += picked_up_cards

        print(f"{game.players[game.turn]} picked up {len(picked_up_cards)} cards and gained {gained_points} points.")

        if len(game.table) == 0 and (len(game.deck) > 0 or any(len(player.hand) > 0 for player in game.players)):
            game.players[game.turn].sweeps += 1
            print(f"{game.players[game.turn]} sweeps!")
        
        print()

        game.last_to_pick_up = game.players[game.turn]


class PileOnAction(Action):
    def __init__(self, played_card, value, other_cards=[]):
        super().__init__(ActionType.PILE_ON, played_card, value, other_cards)
        if not other_cards:
            raise ValueError("Must have other cards for PILE_ON")

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

        creators.add(game.players[game.turn])
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
