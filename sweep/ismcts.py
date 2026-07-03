"""Single-observer ISMCTS agent.

Each decision builds a fresh UCT tree over the acting player's information
set: every iteration determinizes the hidden cards (opponent hand + deck are
resampled from view-legitimate ``unseen``), walks the shared tree stepping
the determinization along, and backs up the finished round's score swing.
The search never crosses a round boundary — the next declaration would leak
a fresh deal — so rewards are per-round point differentials.

FAIRNESS: only determinize() touches hidden state, and it reconstructs it
from game.unseen[player] plus public counts; the real opponent hand and deck
order are never read.
"""

import math
import random
from collections import Counter

from .agents import Agent
from .cards import card_points, card_value
from .engine import ActionKind, _hand_order


def determinize(game, player, rng):
    """Clone `game` with all information hidden from `player` resampled.

    The opponent's hand and the deck are rebuilt from game.unseen[player].
    Rule invariant honored (see HeuristicAgent.OPP_OWN_PILE_P): a live pile's
    creator always holds a capture card for it, so one unseen card of each
    opponent-created pile's value is forced into their hand (distinct cards
    for distinct values) before the remaining slots are filled uniformly.
    """
    if game.awaiting != "play":
        raise ValueError("determinize requires a position awaiting a play")
    det = game.clone(rng=rng)
    opp = 1 - player
    pool = sorted(game.unseen[player])  # sorted first: reproducible from rng
    rng.shuffle(pool)
    hand_size = len(game.hands[opp])
    hand = []
    for value in sorted(v for v, pile in game.piles.items() if opp in pile.creators):
        if len(hand) >= hand_size:
            break
        for i, c in enumerate(pool):  # pool is shuffled: first hit is uniform
            if card_value(c) == value:
                hand.append(pool.pop(i))
                break
    need = hand_size - len(hand)
    # Re-shuffle: popping the *first* hit of each forced value conditions the
    # leftover order (cards before the hit stay in front), which would bias
    # the fill against remaining same-value copies and skew the deck order.
    rng.shuffle(pool)
    hand += pool[:need]
    det.hands[opp] = sorted(hand, key=_hand_order)
    det.deck = pool[need:]  # leftover unseen cards, already in random order
    det.unseen[opp] = set(det.deck) | set(det.hands[player])
    return det


def _rollout_action(game, rng):
    """Greedy playout policy; reads `game` directly (a determinization)."""
    actions = game.legal_actions()
    pickups = [a for a in actions if a.kind is ActionKind.PICKUP]
    if pickups:
        def gain(a):
            cards = (a.card,) + a.loose
            if a.takes_pile:
                cards += game.piles[a.value].cards
            return (a.sweeps, sum(card_points(c) for c in cards), len(cards))

        return max(pickups, key=gain)
    throws = [a for a in actions if a.kind is ActionKind.THROW]
    if throws:
        return min(throws, key=lambda a: (card_points(a.card), card_value(a.card)))
    return rng.choice(actions)  # forced to build/raise


class _Node:
    __slots__ = ("children",)

    def __init__(self):
        self.children = {}  # Action -> _Child


class _Child:
    __slots__ = ("n", "w", "avail", "node")

    def __init__(self):
        self.n = 0  # visits
        self.w = 0.0  # total reward, from the mover-at-parent's perspective
        self.avail = 1  # iterations in which this action was legal
        self.node = _Node()


class ISMCTSAgent(Agent):
    """Single-observer Information Set MCTS (UCT with availability counts)."""

    name = "ismcts"

    def __init__(self, seed=None, n_sims=200, c=0.7):
        self.rng = random.Random(seed)
        self.n_sims = n_sims
        self.c = c

    def declare(self, game):
        # Same rule as HeuristicAgent: prefer a declared value held in
        # duplicate (the opening can build a guaranteed-capturable pile).
        view = game.view(game.turn)
        counts = Counter(card_value(c) for c in view["hand"])
        return max(game.declare_options(), key=lambda v: (counts[v], v))

    def act(self, game):
        actions = game.legal_actions()
        if len(actions) == 1:
            return actions[0]
        root = game.turn
        hist = len(game.round_history)  # terminal = this round finished
        tree = _Node()
        for _ in range(self.n_sims):
            it_rng = random.Random(self.rng.getrandbits(64))
            det = determinize(game, root, it_rng)
            path = []
            node = tree
            while len(det.round_history) == hist and not det.game_over:
                legal = det.legal_actions()
                children = node.children
                existing = [a for a in legal if a in children]
                for a in existing:
                    children[a].avail += 1
                untried = [a for a in legal if a not in children]
                mover = det.turn
                if untried:
                    action = it_rng.choice(untried)
                    child = children[action] = _Child()
                    det.step(action)
                    path.append((child, mover))
                    while len(det.round_history) == hist and not det.game_over:
                        det.step(_rollout_action(det, it_rng))
                    break
                action = max(existing, key=lambda a: self._uct(children[a]))
                child = children[action]
                det.step(action)
                path.append((child, mover))
                node = child.node
            scores = det.round_history[hist]
            reward = (scores[root] - scores[1 - root]) / 100.0
            for child, mover in path:
                child.n += 1
                child.w += reward if mover == root else -reward
        # Pick by visits. Root children are always available (the root state
        # is never resampled), but with n_sims < len(actions) some may still
        # be unexpanded.
        return max(actions, key=lambda a: tree.children[a].n if a in tree.children else -1)

    def _uct(self, child):
        # Selected children always have n >= 1: expansion rolls out and
        # backpropagates before the action can ever be selected again.
        return child.w / child.n + self.c * math.sqrt(math.log(child.avail) / child.n)
