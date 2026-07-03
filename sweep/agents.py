"""Agent protocol and baseline agents.

An agent receives the full Game object for convenience; a *fair* agent must
only consult game.view(game.turn), game.declare_options(), and
game.legal_actions() — never the opponent's hand or the deck order.
"""

import random
from collections import Counter

from .cards import card_points, card_value
from .combos import groups_summing_to, maximal_capture_unions
from .engine import ActionKind, Game


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


def _captured_cards(action, pile_cards):
    """All cards a PICKUP banks: the played card, loose cards, and any pile."""
    cards = (action.card,) + action.loose
    if action.takes_pile:
        cards += tuple(pile_cards[action.value])
    return cards


class GreedyAgent(Agent):
    """One-ply point maximizer.

    Captures: prefer sweeps, then most immediate card-points, then most cards.
    Otherwise: throw the lowest-point, lowest-value card. Never builds unless
    forced (a card that feeds its own pile may have no throw available).
    """

    name = "greedy"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def declare(self, game):
        return max(game.declare_options())

    def act(self, game):
        view = game.view(game.turn)
        actions = game.legal_actions()
        pile_cards = {v: info["cards"] for v, info in view["piles"].items()}

        pickups = [a for a in actions if a.kind is ActionKind.PICKUP]
        if pickups:
            def key(a):
                cards = _captured_cards(a, pile_cards)
                pts = sum(card_points(c) for c in cards)
                return (a.sweeps, pts, len(cards), self.rng.random())

            return max(pickups, key=key)

        throws = [a for a in actions if a.kind is ActionKind.THROW]
        if throws:
            def key(a):
                return (card_points(a.card), card_value(a.card), self.rng.random())

            return min(throws, key=key)

        # Forced to build/raise (e.g. the only playable card feeds our own pile).
        return self.rng.choice(actions)


class HeuristicAgent(Agent):
    """Hand-crafted player: greedy's capture instincts plus lookahead about
    what the table offers the opponent after each candidate move.

    Every legal action is scored as (immediate gain) + (equity of any pile we
    end up owning) - (expected points the opponent's next card can take from
    the resulting table), with the opponent's holdings estimated from
    view()['unseen']. Individual heuristics are commented inline.
    """

    name = "heuristic"

    # Weights, tuned by arena self-play against GreedyAgent.
    CARD_BONUS = 0.2          # each captured card is progress toward the +4 majority
    CAPTURE_TEMPO = 1.0       # capturing makes us the last capturer (claims leftovers)
    ENDGAME_TEMPO = 1.5       # ...worth more in the second half when leftovers loom
    THROW_POINT_COST = 0.8    # throwing a point card hands its points to the table
    PILE_RISK = 1.4           # swing factor if the opponent snatches our pile
    DENIAL = 0.5              # raising away an opponent pile cancels their reserve plan
    LOCK_BONUS = 1.0          # doubling our own pile makes it raise-proof
    THREAT_REST = 0.25        # secondary opponent threats matter less than the best one
    OPP_OWN_PILE_P = 1.0      # rule invariant, not an estimate: builds/raises require a second
                              # copy of the value, and the reserve rule then pins a creator's
                              # last copy to capturing the pile — so a live pile's creator
                              # always holds a capture card

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def declare(self, game):
        # Declare a value we hold in duplicate when possible: the opening can
        # then build a pile we are guaranteed to capture. Break ties high.
        view = game.view(game.turn)
        counts = Counter(card_value(c) for c in view["hand"])
        return max(game.declare_options(), key=lambda v: (counts[v], v))

    def act(self, game):
        view = game.view(game.turn)
        actions = game.legal_actions()
        p_opp = self._opp_value_probs(view)

        def key(a):
            return (self._score(a, view, p_opp), self.rng.random())

        return max(actions, key=key)

    # ------------------------------------------------------------- scoring

    def _score(self, action, view, p_opp):
        pile_cards = {v: info["cards"] for v, info in view["piles"].items()}
        score = 0.0

        if action.kind is ActionKind.PICKUP:
            cards = _captured_cards(action, pile_cards)
            # Immediate points are king; a sweep is worth a flat 50.
            score += sum(card_points(c) for c in cards) + (50.0 if action.sweeps else 0.0)
            # Card count feeds the majority bonus.
            score += self.CARD_BONUS * len(cards)
            # Becoming the last capturer claims end-of-round leftovers.
            score += self.CAPTURE_TEMPO
            if view["second_half"] and view["last_capturer_is_me"] is not True:
                score += self.ENDGAME_TEMPO

        elif action.kind is ActionKind.THROW:
            # A thrown point card is up for grabs; keep spades/aces/10♦ in hand
            # where they can still capture their own points later.
            score -= self.THROW_POINT_COST * card_points(action.card)

        else:  # BUILD or RAISE: we end up owning a pile at action.value
            new_pile = [action.card] + list(action.loose)
            if action.kind is ActionKind.RAISE:
                old = pile_cards[action.raised_from]
                new_pile += list(old)
                if view["piles"][action.raised_from]["opponents"]:
                    # Denial: the opponent reserved a capture card for this
                    # pile; raising it strands that plan.
                    score += self.DENIAL * (
                        sum(card_points(c) for c in old) + self.CARD_BONUS * len(old)
                    )
            if action.takes_pile:
                merged = view["piles"][action.value]
                new_pile += list(merged["cards"])
                if merged["mine"]:
                    # Doubling our own pile locks it: it can no longer be raised.
                    score += self.LOCK_BONUS
            pts = sum(card_points(c) for c in new_pile) + self.CARD_BONUS * len(new_pile)
            # We hold the capture card by rule, but the opponent may too (or
            # draw one): if they take the pile the points swing against us.
            p_steal = p_opp.get(action.value, 0.0)
            score += pts * (1.0 - self.PILE_RISK * p_steal)

        # Subtract what the table we leave behind offers the opponent.
        table, piles = self._after(action, view)
        score -= self._threat(table, piles, p_opp)
        return score

    # --------------------------------------------------------- table model

    @staticmethod
    def _after(action, view):
        """Simulate the loose table and piles after `action`, from view data.

        Piles map value -> (cards, mine); `mine` means we hold its reserve
        capture card, so its fate is priced in _score, not in _threat.
        """
        table = [c for c in view["table"] if c not in action.loose]
        piles = {v: (list(info["cards"]), info["mine"]) for v, info in view["piles"].items()}
        if action.kind is ActionKind.THROW:
            table.append(action.card)
        elif action.kind is ActionKind.PICKUP:
            if action.takes_pile:
                del piles[action.value]
        else:  # BUILD / RAISE
            cards = [action.card] + list(action.loose)
            if action.kind is ActionKind.RAISE:
                cards += piles.pop(action.raised_from)[0]
            if action.takes_pile:
                cards += piles.pop(action.value)[0]
            piles[action.value] = (cards, True)
        return table, piles

    def _opp_value_probs(self, view):
        """P(opponent currently holds at least one card of each value),
        hypergeometric over the unseen cards (their hand + the deck)."""
        unseen = view["unseen"]
        u, h = len(unseen), view["opp_hand_count"]
        probs = {}
        for v, k in Counter(card_value(c) for c in unseen).items():
            p_none = 1.0
            for i in range(k):
                p_none *= max(0, u - h - i) / (u - i) if u - i > 0 else 0.0
            probs[v] = 1.0 - p_none
        return probs

    def _threat(self, table, piles, p_opp):
        """Expected points the opponent's single next card takes off this table.

        For each value v they might hold, compute the best capture at v
        (equal loose cards + best maximal sum-union + any pile of v, plus 50
        if it sweeps), weight by the probability they hold a v. The best
        threat counts in full, the rest at THREAT_REST (they get one move).
        """
        gains = []
        for v in range(1, 14):
            p = p_opp.get(v, 0.0)
            pile = piles.get(v)
            if pile is not None:
                if pile[1]:
                    pile = None  # our pile's fate is priced in _score
                else:
                    p = max(p, self.OPP_OWN_PILE_P)
            if p <= 0.0:
                continue
            gain = self._best_capture(table, piles, v, pile)
            if gain > 0.0:
                gains.append(p * gain)
        if not gains:
            return 0.0
        gains.sort(reverse=True)
        return gains[0] + self.THREAT_REST * sum(gains[1:])

    def _best_capture(self, table, piles, v, pile):
        """Best card-point haul a single card of value v gets from the table."""
        equals = [c for c in table if card_value(c) == v]
        lows = [c for c in table if card_value(c) < v]
        base_pts = sum(card_points(c) for c in equals)
        base_n = len(equals)
        if pile is not None:
            base_pts += sum(card_points(c) for c in pile[0])
            base_n += len(pile[0])
        elif not equals and not lows:
            return 0.0
        best = -1.0
        for union in maximal_capture_unions(groups_summing_to(lows, v)):
            n = base_n + len(union)
            if n == 0:
                continue
            pts = base_pts + sum(card_points(c) for c in union)
            pts += self.CARD_BONUS * n
            # Clearing everything (the only pile, if any, is at v) is a sweep.
            if len(equals) + len(union) == len(table) and set(piles) <= {v}:
                pts += 50.0
            best = max(best, pts)
        return max(best, 0.0)


def play_game(agents, seed=None, win_lead=200):
    """Play one full game between agents[0] (player 0) and agents[1]."""
    game = Game(seed=seed, win_lead=win_lead)
    while not game.game_over:
        if game.awaiting == "declare":
            game.declare(agents[game.first_player].declare(game))
        else:
            game.step(agents[game.turn].act(game))
    return game
