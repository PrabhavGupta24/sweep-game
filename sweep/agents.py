"""Agent protocol and the random baseline.

An agent receives the full Game object for convenience; a *fair* agent must
only consult game.view(game.turn), game.declare_options(), and
game.legal_actions() — never the opponent's hand or the deck order.
"""

import random

from .engine import Game


class Agent:
    name = "agent"

    def declare(self, game) -> int:
        """Pick a value from game.declare_options() at the start of a round."""
        raise NotImplementedError

    def act(self, game):
        """Pick an Action from game.legal_actions()."""
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def declare(self, game):
        return self.rng.choice(game.declare_options())

    def act(self, game):
        return self.rng.choice(game.legal_actions())


def play_game(agents, seed=None, win_lead=200):
    """Play one full game between agents[0] (player 0) and agents[1]."""
    game = Game(seed=seed, win_lead=win_lead)
    while not game.game_over:
        if game.awaiting == "declare":
            game.declare(agents[game.first_player].declare(game))
        else:
            game.step(agents[game.turn].act(game))
    return game
