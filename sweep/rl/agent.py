"""NeuralAgent: a trained PolicyValueNet as a playable Agent.

A thin evaluation-side wrapper: each decision is encoded with the pure
encoders and scored by the net, exactly as during training. FAIRNESS: the
agent reads only game.view(game.turn), game.declare_options(), and
game.legal_actions(); the encoders take the plain view dict rather than the
Game, so hidden information is structurally out of reach.

This module imports torch, so (like sweep.rl.model) it is loaded lazily by
the package: ``import sweep.rl`` stays torch-free and ``sweep.rl.NeuralAgent``
triggers the import on first access.
"""

from __future__ import annotations

import numpy as np
import torch

from ..agents import Agent
from .encoders import ACTION_SIZE, DECLARE, OBS_SIZE, encode_action, encode_observation
from .model import PolicyValueNet
from .ppo import load_checkpoint


class NeuralAgent(Agent):
    """Plays with a PolicyValueNet; the value head is ignored.

    The net comes from, in order of precedence: ``net`` (an instance, used
    as-is), ``ckpt_path`` (restored via load_checkpoint), or — with neither —
    a fresh randomly-initialized PolicyValueNet, which plays arbitrarily and
    exists for testing only.

    ``temperature`` is passed to act_single: 0.0 (the default) is
    deterministic argmax; > 0 samples from softmax(logits / temperature)
    using a torch.Generator seeded from ``seed`` (``seed`` is irrelevant at
    temperature 0).
    """

    name = "neural"

    def __init__(self, ckpt_path=None, net=None, seed=None, temperature=0.0):
        if net is None:
            net = PolicyValueNet()
            if ckpt_path is not None:
                load_checkpoint(ckpt_path, net)
        self.net = net
        self.temperature = temperature
        self.generator = torch.Generator()
        if seed is None:
            self.generator.seed()  # nondeterministic, like random.Random()
        else:
            self.generator.manual_seed(seed)
        # Reused encoding buffers; the candidate buffer grows on demand.
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._cands = np.zeros((16, ACTION_SIZE), dtype=np.float32)

    def declare(self, game):
        candidates = [(DECLARE, v) for v in game.declare_options()]
        return self._choose(game, candidates)[1]

    def act(self, game):
        return self._choose(game, game.legal_actions())

    def _choose(self, game, candidates):
        view = game.view(game.turn)  # turn == first_player while declaring
        obs = encode_observation(view, game.awaiting, np_out=self._obs)
        n = len(candidates)
        if n > len(self._cands):
            self._cands = np.zeros((n, ACTION_SIZE), dtype=np.float32)
        encoded = self._cands[:n]
        for i, item in enumerate(candidates):
            encode_action(item, view, np_out=encoded[i])
        index, _, _ = self.net.act_single(
            obs, encoded, self.generator, self.temperature)
        return candidates[index]
