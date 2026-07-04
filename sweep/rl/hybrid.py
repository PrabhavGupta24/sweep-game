"""HybridAgent: ISMCTS with the trained value head replacing greedy rollouts.

Stage 5 productionizes a proven prototype: the plain ISMCTSAgent scores a
freshly-expanded leaf by playing the round out with a greedy rollout, then
reading the exact swing. HybridAgent instead evaluates a still-live leaf with
the PolicyValueNet's value head — a learned estimate of the round's full final
swing conditioned on the leaf state (the critic's training target; see
_leaf_reward) — and only falls back to the exact swing when the leaf is already
the round's end. Everything else (per-iteration determinize, availability-UCT,
negamax backprop, visit-count move choice) is inherited unchanged from
ISMCTSAgent. PUCT / policy priors are explicitly out of scope.

FAIRNESS: the value head reads only encode_observation(det.view(det.turn),
"play"), a pure function of det's public view, so hidden information stays out
of reach exactly as in NeuralAgent. The determinization itself is built by the
inherited search from the searching player's own ``unseen``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..ismcts import ISMCTSAgent, _exact_reward
from .agent import declare_through_net, resolve_net
from .encoders import OBS_SIZE, encode_observation


class HybridAgent(ISMCTSAgent):
    """ISMCTS whose leaf evaluation is the trained value head.

    The net comes from, in order of precedence: ``net``, ``ckpt_path``, or a
    fresh random-init PolicyValueNet (see resolve_net) — the last plays
    arbitrarily and exists for testing only. ``seed``, ``n_sims`` and ``c`` are
    the ISMCTSAgent knobs; the same ``c=0.35`` default holds — see the
    exploration-constant note below, which confirms it transfers to the value
    head.
    """

    # Exploration-constant check (Stage 5): the value head's leaf rewards have
    # different variance than the rollout swings c was tuned on, so c=0.35 was
    # re-measured against it. On PAIRED deals (base_seed 5000, 10 games each,
    # 200 sims vs HeuristicAgent) c=0.35 went 10/10 at +244.0/game while c=0.7
    # went 5/10 at -21.4/game — a decisive gap in the same direction Stage 3
    # found for plain rollouts. The Stage 3 tuning transfers; c=0.35 stays the
    # default. (Same-scale ~[-1, 1] rewards, so a smaller c exploits more.)

    name = "hybrid"

    def __init__(self, ckpt_path=None, net=None, seed=None, n_sims=200, c=0.35):
        super().__init__(seed=seed, n_sims=n_sims, c=c)
        self.net = resolve_net(net, ckpt_path)
        # Declarations go through the net at temperature 0 (argmax), so this
        # generator is never actually sampled from; it exists for act_single's
        # signature. Seeding it keeps declares reproducible if temperature > 0
        # is ever wired in.
        self.generator = torch.Generator()
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)
        # Reused encoding buffer for the leaf value head (mirrors
        # NeuralAgent._obs): encode_observation zeroes and refills it each
        # call, so results are identical to a fresh allocation while avoiding a
        # 253-float alloc per expanded leaf on the search hot path.
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)

    def _leaf_reward(self, det, root, hist, rng):
        """Value-head estimate of a live leaf, from ``root``'s perspective.

        The value head predicts the *mover's own* expected round swing (score
        diff / 100) — its training target — so it is already from det.turn's
        perspective and must be NEGATED when det.turn != root to express it
        from root's. When the leaf is already the round's end there is nothing
        to estimate: return the exact swing (also from root's perspective).
        ``rng`` is accepted for the hook signature but unused: no rollout runs.
        """
        if len(det.round_history) == hist and not det.game_over:
            obs = encode_observation(det.view(det.turn), "play", np_out=self._obs)
            value = self.net.value_only(obs)
            return value if det.turn == root else -value
        return _exact_reward(det, root, hist)

    def declare(self, game):
        return declare_through_net(self.net, game, self.generator)
