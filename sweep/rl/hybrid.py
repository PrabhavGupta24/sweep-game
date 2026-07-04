"""HybridAgent: ISMCTS with the trained value head replacing greedy rollouts.

Stage 5 productionizes a proven prototype: the plain ISMCTSAgent scores a
freshly-expanded leaf by playing the round out with a greedy rollout, then
reading the exact swing. HybridAgent instead evaluates a still-live leaf with
the PolicyValueNet's value head — a learned estimate of the round's full final
swing conditioned on the leaf state (the critic's training target; see
_leaf_reward) — and only falls back to the exact swing when the leaf is already
the round's end. With ``priors=False`` (the default, the shipped behavior)
everything else (per-iteration determinize, availability-UCT, negamax backprop,
visit-count move choice) is inherited unchanged from ISMCTSAgent.

PUCT (``priors=True``): the SAME net's policy head guides selection AlphaZero-
style, adapted to determinized ISMCTS. Selection maximizes, from the mover's
perspective, ``Q(a) + c_puct * P(a) * sqrt(A(a)) / (1 + N(a))`` over the
det-legal actions, where ``P`` is the (cached, renormalized) policy prior,
``A`` the availability count and ``N`` the visit count; there is no separate
expand-uniformly step — selecting an unvisited action IS the expansion. See
_puct_act. Note the availability numerator is per-child, not the parent's total
visit count: an action still without a child has A pinned to 1 (avail is bumped
only for actions already in ``children``; see _puct_act), so an unvisited action
scores a constant ``c_puct * P`` — a deliberate departure from AlphaZero's
``sqrt(sum_b N(b))`` numerator that trusts the policy prior more (a confidently
low-prior move can stay unexpanded within the budget). The ``priors=False`` code path is untouched: it delegates verbatim
to ISMCTSAgent.act, preserving the shipped rng stream and decisions
seed-for-seed.

FAIRNESS: the value head reads only encode_observation(det.view(det.turn),
"play"), a pure function of det's public view, so hidden information stays out
of reach exactly as in NeuralAgent. Priors are likewise evaluated on the
mover's own determinized view; only the root uses the real game view (the root
position IS the searching player's true information set, so it is not a leak).
The determinization itself is built by the inherited search from the searching
player's own ``unseen``.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from ..ismcts import ISMCTSAgent, _Child, _Node, _exact_reward, determinize
from .agent import declare_through_net, resolve_net
from .encoders import ACTION_SIZE, OBS_SIZE, encode_action, encode_observation


class _PriorNode(_Node):
    """A tree node that also caches the policy prior P(a) per action.

    Inherits ``children`` (Action -> _PriorChild) from _Node and adds
    ``priors`` (Action -> raw softmax probability), filled lazily the first
    time the node is reached in some determinization and topped up when a
    later determinization presents an action the cache has not seen yet
    (existing entries stay stable). Selection renormalizes over the det-legal
    subset. The base ``_Child`` visit/reward/availability accounting is reused
    unchanged (see _PriorChild).
    """

    __slots__ = ("priors",)

    def __init__(self):
        super().__init__()
        self.priors = {}  # Action -> raw prior probability (pre-renormalize)


class _PriorChild(_Child):
    """A _Child whose subtree node is a _PriorNode (so it can cache priors).

    Identical bookkeeping to _Child (n / w / avail, negamax-ready); it only
    overrides the child's ``node`` to a _PriorNode. The PUCT walk never touches
    _Child's rollout-era semantics, so subclassing keeps that accounting in one
    place instead of re-deriving it.
    """

    __slots__ = ()

    def __init__(self):
        super().__init__()
        self.node = _PriorNode()


class HybridAgent(ISMCTSAgent):
    """ISMCTS whose leaf evaluation is the trained value head.

    The net comes from, in order of precedence: ``net``, ``ckpt_path``, or a
    fresh random-init PolicyValueNet (see resolve_net) — the last plays
    arbitrarily and exists for testing only. ``seed``, ``n_sims`` and ``c`` are
    the ISMCTSAgent knobs; the same ``c=0.35`` default holds — see the
    exploration-constant note below, which confirms it transfers to the value
    head.

    ``priors=False`` (default) is the shipped value-head ISMCTS, byte-for-byte:
    act() delegates to ISMCTSAgent.act. ``priors=True`` switches selection to
    PUCT with the net's policy head as priors, tuned by ``c_puct`` (see
    _puct_act); the value-head leaf, terminal handling, negamax backprop and
    determinize are shared with the base path.
    """

    # Exploration-constant check (Stage 5): the value head's leaf rewards have
    # different variance than the rollout swings c was tuned on, so c=0.35 was
    # re-measured against it. On PAIRED deals (base_seed 5000, 10 games each,
    # 200 sims vs HeuristicAgent) c=0.35 went 10/10 at +244.0/game while c=0.7
    # went 5/10 at -21.4/game — a decisive gap in the same direction Stage 3
    # found for plain rollouts. The Stage 3 tuning transfers; c=0.35 stays the
    # default. (Same-scale ~[-1, 1] rewards, so a smaller c exploits more.)

    name = "hybrid"

    def __init__(self, ckpt_path=None, net=None, seed=None, n_sims=200, c=0.35,
                 priors=False, c_puct=1.0):
        super().__init__(seed=seed, n_sims=n_sims, c=c)
        self.priors = priors
        self.c_puct = c_puct
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

    def act(self, game):
        # priors=False is EXACTLY the shipped behavior: hand the whole decision
        # to the base UCT loop (same rng stream, same choices, seed-for-seed).
        # Only priors=True takes the PUCT walk below.
        if not self.priors:
            return super().act(game)
        return self._puct_act(game)

    def _priors_for(self, view, actions):
        """Masked softmax of the policy head over ``actions`` on ``view``.

        Evaluates the policy head ONCE — encode_observation(view, "play") plus
        encode_action for every action — and returns {action: probability},
        a valid distribution over exactly ``actions`` (the net's own softmax
        over this candidate set). ``view`` is the mover's determinized view at
        an interior node, or the real game view at the root (not a leak: the
        root is the searching player's true information set).
        """
        obs = encode_observation(view, "play")
        cands = np.zeros((len(actions), ACTION_SIZE), dtype=np.float32)
        for i, a in enumerate(actions):
            encode_action(a, view, np_out=cands[i])
        obs_t = torch.as_tensor(obs).unsqueeze(0)
        cands_t = torch.as_tensor(cands).unsqueeze(0)
        mask = torch.ones(cands_t.shape[:2], dtype=torch.bool)
        with torch.no_grad():
            logits, _ = self.net(obs_t, cands_t, mask)
            probs = F.softmax(logits[0], dim=-1)
        return {a: float(probs[i]) for i, a in enumerate(actions)}

    def _ensure_priors(self, node, view, legal):
        """Fill/top up ``node.priors`` so every ``legal`` action has an entry.

        First visit: evaluate the policy on all legal actions and cache them.
        A later determinization may present actions the cache is missing (a
        different opponent hand yields different legal moves); recompute the
        policy on THIS det and merge ONLY the new actions' priors, leaving
        existing entries stable. No-op once every legal action is cached.
        """
        missing = [a for a in legal if a not in node.priors]
        if not missing:
            return
        fresh = self._priors_for(view, legal)
        for a in missing:
            node.priors[a] = fresh[a]

    def _puct_select(self, node, legal):
        """Pick a det-legal action by the PUCT rule, from the mover's view.

        Priors are renormalized over the det-legal subset (the cache may hold
        actions illegal under this determinization). Q(a) = W/N is already from
        the mover's perspective — the child stores W from the parent-mover's
        view (negamax backprop) — with Q = 0 for an unvisited action (neutral
        first-play urgency). Selecting an unvisited action IS the expansion, so
        scores must stay finite for N = 0. An unvisited action has no child yet,
        so its avail is taken as 1 (avail is only bumped for actions already in
        ``children``): its exploration term is a constant ``c_puct * P``, not a
        parent-visit-growing bonus — see the class docstring's note on this.
        """
        children = node.children
        total = sum(node.priors[a] for a in legal)
        # Uniform fallback if the net gave the whole legal set ~0 mass (masked
        # softmax underflow); keeps priors a valid distribution over `legal`.
        norm = total if total > 0 else 0.0

        def score(a):
            child = children.get(a)
            p = (node.priors[a] / norm) if norm > 0 else 1.0 / len(legal)
            if child is None or child.n == 0:
                q, n, avail = 0.0, 0, (child.avail if child is not None else 1)
            else:
                q, n, avail = child.w / child.n, child.n, child.avail
            return q + self.c_puct * p * math.sqrt(avail) / (1 + n)

        return max(legal, key=score)

    def _puct_act(self, game):
        """PUCT search: policy priors guide selection, value head scores leaves.

        Mirrors ISMCTSAgent.act's per-iteration determinize / negamax backprop
        / visit-count move choice, but selection is PUCT (see _puct_select) and
        expansion is fused into selection: reaching an unvisited action steps
        the det, scores the value-head leaf and backs up — there is no separate
        expand-one-untried step. Root priors come from the real game view.
        """
        actions = game.legal_actions()
        if len(actions) == 1:
            return actions[0]
        root = game.turn
        hist = len(game.round_history)
        tree = _PriorNode()
        # The root is the searching player's true info set, so its priors come
        # from the real view (not a leak); interior nodes use det views.
        # Load-bearing invariant: every det-legal root action is in `actions`,
        # so _puct_select's node.priors[a] lookup never KeyErrors at the root
        # even though _ensure_priors is skipped for the root below. It holds
        # because determinize resamples only the opponent hand + deck (same
        # sizes) and legal_actions depends only on the mover's own hand, table,
        # piles, opening/declared and _is_final_play (deck emptiness + hand
        # sizes) — all identical at the root across determinizations. A future
        # determinize that alters any of these must top up root priors too.
        self._ensure_priors(tree, game.view(root), actions)
        for _ in range(self.n_sims):
            it_rng = random.Random(self.rng.getrandbits(64))
            det = determinize(game, root, it_rng)
            path = []
            node = tree
            reward = None
            while len(det.round_history) == hist and not det.game_over:
                legal = det.legal_actions()
                children = node.children
                # Availability: every det-legal action that already has a child
                # gets its avail bumped, exactly as the base class counts it.
                for a in legal:
                    if a in children:
                        children[a].avail += 1
                # Priors: root already cached; interior nodes lazily on the
                # mover's own det view (top up if this det has new actions).
                mover = det.turn
                if node is not tree:
                    self._ensure_priors(node, det.view(mover), legal)
                action = self._puct_select(node, legal)
                child = children.get(action)
                if child is None:
                    # Unvisited action: selection IS expansion. Step, score the
                    # value-head leaf, stop descending.
                    child = children[action] = _PriorChild()
                    det.step(action)
                    path.append((child, mover))
                    reward = self._leaf_reward(det, root, hist, it_rng)
                    break
                det.step(action)
                path.append((child, mover))
                node = child.node
            if reward is None:
                # Selected into a finished round without a fresh leaf: exact.
                reward = _exact_reward(det, root, hist)
            for child, mover in path:
                child.n += 1
                child.w += reward if mover == root else -reward
        return max(actions, key=lambda a: tree.children[a].n if a in tree.children else -1)

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
