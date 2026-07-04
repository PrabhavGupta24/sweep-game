"""Decision-stream RL environment for Sweep: one episode == one round.

SweepEnv turns the engine's declare/play loop into a single stream of
*decisions*. Each decision is a plain dict:

    player              acting player index (0 or 1)
    obs                 float32 vector of length OBS_SIZE, from that player's
                        view (see encoders.encode_observation)
    candidates          the legal moves: engine Actions, or (DECLARE, value)
                        tuples for the round-opening declaration
    encoded_candidates  float32 array [len(candidates), ACTION_SIZE]

The env holds a Game internally, but everything it returns is built from
``game.view(player)`` / ``game.legal_actions()`` / ``game.declare_options()``
only, so agents trained on the stream can never see hidden information.

An episode ends when the round does (the engine auto-starts the next round;
the env stops there). Rewards come from the engine's authoritative
RoundResult: reward[p] = (scores[p] - scores[1-p]) / 100. Evaluation code can
opt into the same game's next round with continue_same_game().
"""

from __future__ import annotations

import random

import numpy as np

from ..engine import DEFAULT_WIN_LEAD, Game
from .encoders import ACTION_SIZE, DECLARE, encode_action, encode_observation


class SweepEnv:
    """One-round episodes over freshly seeded Games.

    Deterministic: the same env seed yields the same sequence of games, so
    identical action indices reproduce an identical decision stream.
    """

    def __init__(self, seed=None, win_lead=DEFAULT_WIN_LEAD):
        self.rng = random.Random(seed)
        self.win_lead = win_lead
        self.game = None
        self._candidates = None  # candidates of the pending decision

    def reset(self):
        """Start a fresh Game (seed drawn from the env rng); first decision."""
        self.game = Game(seed=self.rng.getrandbits(64), win_lead=self.win_lead)
        return self._decision()

    def step(self, index):
        """Apply candidates[index] from the pending decision.

        Returns (next_decision, None, False) mid-round, or
        (None, rewards, True) when the play ends the round, with rewards[p] =
        (scores[p] - scores[1-p]) / 100 from the round's RoundResult.
        """
        if self._candidates is None:
            raise RuntimeError("no pending decision — call reset() first")
        game = self.game
        finished = len(game.round_results)
        item = self._candidates[index]
        if isinstance(item, tuple):
            game.declare(item[1])
        else:
            game.step(item)
        if len(game.round_results) > finished:
            self._candidates = None
            scores = game.round_results[-1].scores
            diff = (scores[0] - scores[1]) / 100.0
            return None, (diff, -diff), True
        return self._decision(), None, False

    def continue_same_game(self):
        """Opt-in continuation: the first decision of this game's next round.

        The engine has already dealt it; episode/reward semantics are
        unchanged (the next round is scored on its own RoundResult).
        """
        if self.game is None:
            raise RuntimeError("no game — call reset() first")
        if self.game.game_over:
            raise RuntimeError("game is over — call reset() for a new one")
        return self._decision()

    # ------------------------------------------------------------- internals

    def _decision(self):
        game = self.game
        if game.awaiting == DECLARE:
            player = game.first_player
            candidates = [(DECLARE, v) for v in game.declare_options()]
        else:
            player = game.turn
            candidates = game.legal_actions()
        view = game.view(player)
        encoded = np.zeros((len(candidates), ACTION_SIZE), dtype=np.float32)
        for i, item in enumerate(candidates):
            encode_action(item, view, np_out=encoded[i])
        self._candidates = candidates
        return {
            "player": player,
            "obs": encode_observation(view, game.awaiting),
            "candidates": candidates,
            "encoded_candidates": encoded,
        }
