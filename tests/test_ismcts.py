"""ISMCTS tests: cloning, determinization, legality, determinism, search."""

import random
from collections import Counter

from sweep.agents import RandomAgent
from sweep.cards import card_from, card_value
from sweep.engine import ActionKind, Game
from sweep.ismcts import ISMCTSAgent, determinize
from test_rules import C, make_game


def first_half_position(seed=0, plays=3):
    """A real early-round position: declared, a few plays in, deck alive."""
    g = Game(seed=seed)
    g.declare(g.declare_options()[0])
    rng = random.Random(seed + 1)
    for _ in range(plays):
        g.step(rng.choice(g.legal_actions()))
    assert g.awaiting == "play" and g.deck
    return g


def second_half_position(seed=0):
    """A real position with the deck exhausted (second half)."""
    g = Game(seed=seed)
    rng = random.Random(seed + 99)
    while True:
        if g.awaiting == "declare":
            g.declare(rng.choice(g.declare_options()))
        elif g.deck:
            g.step(rng.choice(g.legal_actions()))
        else:
            assert g.awaiting == "play"
            return g


def fingerprint(g):
    """Every public game field, as comparable immutable values."""
    return (
        g.win_lead, g.round_num, g.differential, tuple(g.round_history),
        g.game_over, g.winner, g.first_player,
        None if g._facedown is None else tuple(g._facedown),
        tuple(g.hands[0]), tuple(g.hands[1]), tuple(g.deck), tuple(g.table),
        tuple(sorted(g.piles.items())), tuple(g.captured[0]), tuple(g.captured[1]),
        tuple(g.points), tuple(g.sweeps), g.last_capturer, g.declared,
        g.opening, g.turn, g.awaiting, frozenset(g.unseen[0]), frozenset(g.unseen[1]),
    )


# ------------------------------------------------------------------- clone


def test_clone_copies_all_public_fields():
    for seed in range(3):
        g = first_half_position(seed)
        assert fingerprint(g.clone()) == fingerprint(g)


def test_stepping_a_clone_never_mutates_the_original():
    g = first_half_position(seed=3)
    before = fingerprint(g)
    c = g.clone(rng=random.Random(0))
    rng = random.Random(1)
    for _ in range(6):
        if c.game_over:
            break
        if c.awaiting == "declare":
            c.declare(rng.choice(c.declare_options()))
        else:
            c.step(rng.choice(c.legal_actions()))
    assert fingerprint(g) == before
    assert fingerprint(c) != before


def test_clone_seeded_rng_is_deterministic():
    # Play each clone across the round boundary: the next round's shuffle
    # comes from the clone's own rng, so equal seeds must yield equal deals.
    g = first_half_position(seed=4)
    results = []
    for _ in range(2):
        c = g.clone(rng=random.Random(9))
        rng = random.Random(5)
        while not c.round_history:
            c.step(rng.choice(c.legal_actions()))
        results.append(fingerprint(c))
    assert results[0] == results[1]


# ------------------------------------------------------------- determinize


def test_determinize_partitions_unseen():
    for seed in range(3):
        g = first_half_position(seed)
        p = g.turn
        opp = 1 - p
        view = g.view(p)
        det = determinize(g, p, random.Random(seed))
        hand, deck = set(det.hands[opp]), set(det.deck)
        assert len(hand) == len(det.hands[opp]) == view["opp_hand_count"]
        assert len(deck) == len(det.deck) == view["deck_count"]
        assert hand.isdisjoint(deck)
        assert hand | deck == set(view["unseen"])
        # Engine invariants: my state untouched, opponent's unseen recomputed.
        assert det.hands[p] == g.hands[p]
        assert det.table == g.table
        # Derived from the view, not restated from the implementation:
        # unseen-to-opponent = the sampled deck plus my (real) hand.
        assert det.unseen[opp] == (set(view["unseen"]) - hand) | set(g.hands[p])


def test_determinize_does_not_leak_the_true_hand():
    g = first_half_position(seed=1, plays=0)
    p, opp = g.turn, 1 - g.turn
    true_hand = set(g.hands[opp])
    expected = g.view(p)["opp_hand_count"] / len(g.unseen[p])
    exact = 0
    counts = Counter()
    for i in range(200):
        det = determinize(g, p, random.Random(i))
        sampled = set(det.hands[opp])
        exact += sampled == true_hand
        counts.update(sampled & true_hand)
    assert exact < 200  # the true hand must not be reproduced every time
    for c in true_hand:  # each true card sampled near its base rate
        assert expected - 0.10 <= counts[c] / 200 <= expected + 0.10


def test_determinize_honors_pile_reserve_invariant():
    # Player 1 created the 9-pile, so every determinization from player 0's
    # perspective must deal them at least one of the unseen 9s.
    g = make_game(
        hand0=["2H", "3S"],
        hand1=["9H", "2C"],
        piles=[(9, ["4C", "5C"], [1])],
        deck=["9D", "3C"],
    )
    g.unseen[0] = set(C("9H", "2C", "9D", "3C"))
    for i in range(100):
        det = determinize(g, 0, random.Random(i))
        assert any(card_value(c) == 9 for c in det.hands[1])


def test_determinize_fill_is_uniform_after_forced_reserve():
    # Two unseen 9s, opp-created 9-pile: after one 9 is forced into the hand,
    # the other must land in the remaining slots at the plain fill rate,
    # (hand_size - 1) / (pool - 1) = 3/9. Popping the first shuffled hit
    # without re-shuffling under-sampled this to ~0.13.
    g = make_game(
        hand0=["2H", "3S", "4S", "5S"],
        hand1=["9H", "2C", "3C", "4C"],
        piles=[(9, ["4D", "5D"], [1])],
        deck=["9D", "6C", "7C", "8C", "6D", "7D"],
    )
    g.unseen[0] = set(g.hands[1]) | set(g.deck)
    n = 2000
    both = sum(
        sum(card_value(c) == 9 for c in determinize(g, 0, random.Random(i)).hands[1]) == 2
        for i in range(n)
    )
    assert abs(both / n - 3 / 9) < 0.05  # ~4.7 sigma at n=2000


def test_determinize_second_half_is_exact():
    # Deck empty: the unseen set *is* the opponent's hand.
    g = second_half_position(seed=2)
    p, opp = g.turn, 1 - g.turn
    det = determinize(g, p, random.Random(0))
    assert det.hands[opp] == g.hands[opp]
    assert det.deck == []


# ------------------------------------------------------------------ agent


def test_ismcts_plays_full_legal_game():
    game = Game(seed=11, win_lead=100)
    agents = [ISMCTSAgent(seed=1, n_sims=16), RandomAgent(seed=2)]
    plays = 0
    while not game.game_over:
        if game.awaiting == "declare":
            v = agents[game.first_player].declare(game)
            assert v in game.declare_options()
            game.declare(v)
        else:
            a = agents[game.turn].act(game)
            assert a in game.legal_actions()
            game.step(a)
            plays += 1
    assert game.winner in (0, 1)
    assert plays > 0


def _trajectory_prefix(n_plays):
    game = Game(seed=6, win_lead=100)
    agents = [ISMCTSAgent(seed=3, n_sims=16), RandomAgent(seed=4)]
    trace = []
    while not game.game_over and len(trace) < n_plays:
        if game.awaiting == "declare":
            v = agents[game.first_player].declare(game)
            trace.append(("declare", v))
            game.declare(v)
        else:
            a = agents[game.turn].act(game)
            trace.append(a)
            game.step(a)
    return trace


def test_ismcts_deterministic_given_seeds():
    assert _trajectory_prefix(24) == _trajectory_prefix(24)


def test_ismcts_takes_available_sweep():
    g = make_game(hand0=["9S", "KH"], hand1=["3C", "4C"], table=["4H", "5H"], deck=())
    g.unseen = [set(g.hands[1]), set(g.hands[0])]
    action = ISMCTSAgent(seed=0, n_sims=200).act(g)
    assert action.kind is ActionKind.PICKUP
    assert action.sweeps


def test_ismcts_avoids_gifting_throw_in_second_half():
    # Deck empty, opponent hand known: throwing K♠ hands their K♥ a 13-point
    # sweep; throwing 2♣ is safe. The search must avoid the gift.
    g = make_game(hand0=["2C", "KS"], hand1=["5C", "KH"], table=(), deck=())
    g.unseen = [set(g.hands[1]), set(g.hands[0])]
    action = ISMCTSAgent(seed=1, n_sims=200).act(g)
    assert action.kind is ActionKind.THROW
    assert action.card == card_from("2C")
