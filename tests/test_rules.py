"""Rules tests: each test maps to a clause of RULES.md."""

import random

import pytest

from sweep.cards import card_from, card_value
from sweep.engine import Action, ActionKind, Game, Pile


def C(*specs):
    return [card_from(s) for s in specs]


def make_game(
    hand0=(),
    hand1=(),
    table=(),
    piles=(),
    deck=("2H", "3H"),
    turn=0,
    points=(0, 0),
    captured=(0, 0),
    last_capturer=None,
    first_player=0,
):
    """Build a Game in a handcrafted mid-round position.

    piles: iterable of (value, card_specs, creator_indices).
    captured: dummy capture-stack sizes (only their lengths matter; points are
    tracked incrementally by the engine).
    """
    g = Game(seed=0)
    g.hands = [sorted(C(*hand0)), sorted(C(*hand1))]
    g.table = sorted(C(*table))
    g.piles = {v: Pile(v, tuple(C(*cards)), frozenset(creators)) for v, cards, creators in piles}
    g.deck = C(*deck)
    g._facedown = None
    g.turn = turn
    g.points = list(points)
    g.sweeps = [0, 0]
    g.captured = [list(range(captured[0])), list(range(captured[1]))]
    g.last_capturer = last_capturer
    g.first_player = first_player
    g.awaiting = "play"
    g.opening = False
    g.declared = None
    g.unseen = [set(), set()]
    g.round_num = 1
    g.differential = 0
    g.round_history = []
    return g


def acts(g, card=None, kind=None, value=None):
    out = g.legal_actions()
    if card is not None:
        out = [a for a in out if a.card == card_from(card)]
    if kind is not None:
        out = [a for a in out if a.kind is kind]
    if value is not None:
        out = [a for a in out if a.value == value]
    return out


# --------------------------------------------------------------- capturing


def test_maximal_capture_is_mandatory():
    g = make_game(hand0=["9S", "2H"], table=["4H", "5H", "9D", "2C"])
    pickups = acts(g, card="9S", kind=ActionKind.PICKUP)
    assert len(pickups) == 1
    assert pickups[0].loose == tuple(sorted(C("4H", "5H", "9D")))


def test_overlapping_combos_yield_multiple_maximal_pickups():
    g = make_game(hand0=["9S", "2H"], table=["4H", "5H", "5D"])
    pickups = acts(g, card="9S", kind=ActionKind.PICKUP)
    captured_sets = {frozenset(a.loose) for a in pickups}
    assert captured_sets == {frozenset(C("4H", "5H")), frozenset(C("4H", "5D"))}


def test_pile_capturable_only_at_exact_value():
    # K (13) cannot absorb a 9-pile plus a 4 even though 9 + 4 = 13.
    g = make_game(
        hand0=["KS", "2H"],
        table=["4D"],
        piles=[(9, ["4C", "5C"], [1])],
    )
    assert acts(g, card="KS", kind=ActionKind.PICKUP) == []


def test_pickup_takes_pile_equals_and_groups_together():
    g = make_game(
        hand0=["9S", "2H"],
        table=["9H", "6D", "3D"],
        piles=[(9, ["4C", "5C"], [1])],
    )
    pickups = acts(g, card="9S", kind=ActionKind.PICKUP)
    assert len(pickups) == 1
    assert pickups[0].takes_pile
    assert pickups[0].loose == tuple(sorted(C("9H", "6D", "3D")))


def test_pickup_points_and_capture_stack():
    g = make_game(hand0=["9S", "2H"], table=["9H", "AD"])
    # 9♠ captures 9♥; the ace (value 1) is not part of any 9.
    [pickup] = acts(g, card="9S", kind=ActionKind.PICKUP)
    g.step(pickup)
    assert g.points[0] == 9  # 9♠ itself; 9♥ is worth nothing
    assert sorted(g.captured[0]) == sorted(C("9S", "9H"))
    assert g.last_capturer == 0


# ----------------------------------------------------------------- building


def test_build_requires_holding_another_card_of_that_value():
    g = make_game(hand0=["4S", "7D"], table=["5H"])
    assert acts(g, kind=ActionKind.BUILD) == []
    g2 = make_game(hand0=["4S", "9C"], table=["5H"])
    builds = acts(g2, card="4S", kind=ActionKind.BUILD, value=9)
    assert len(builds) == 1
    assert builds[0].loose == tuple(C("5H"))


def test_build_absorbs_all_other_combos_maximally():
    g = make_game(hand0=["4S", "9C"], table=["5H", "6D", "3D"])
    builds = acts(g, card="4S", kind=ActionKind.BUILD, value=9)
    assert len(builds) == 1
    assert builds[0].loose == tuple(sorted(C("5H", "6D", "3D")))


def test_lone_hand_card_is_not_a_pile():
    g = make_game(hand0=["9S", "9C"], table=["7D"])
    assert acts(g, kind=ActionKind.BUILD) == []


def test_build_on_equal_loose_card_creates_doubled_pile():
    g = make_game(hand0=["9S", "9C"], table=["9D"])
    builds = acts(g, card="9S", kind=ActionKind.BUILD, value=9)
    assert len(builds) == 1
    g.step(builds[0])
    assert g.piles[9].doubled
    assert g.piles[9].creators == frozenset({0})


def test_doubling_opponents_pile_adds_creator():
    g = make_game(hand0=["9S", "9H"], piles=[(9, ["4C", "5C"], [1])])
    builds = acts(g, card="9S", kind=ActionKind.BUILD, value=9)
    assert len(builds) == 1 and builds[0].takes_pile
    g.step(builds[0])
    assert g.piles[9].doubled
    assert g.piles[9].creators == frozenset({0, 1})


def test_piles_only_nine_through_thirteen():
    g = make_game(hand0=["3S", "8C", "8H"], table=["5H"])
    # 3+5=8 would be a pile of 8: never legal.
    assert acts(g, kind=ActionKind.BUILD) == []


# ------------------------------------------------------------------ raising


def test_raise_increment_must_come_from_hand():
    # Pile of 10; hand ace cannot team up with a table ace to reach 12.
    g = make_game(
        hand0=["AS", "QD"],
        table=["AH"],
        piles=[(10, ["4C", "6C"], [1])],
    )
    assert acts(g, kind=ActionKind.RAISE) == []  # no 11 in hand, 12 unreachable


def test_raise_with_hand_card_only():
    g = make_game(hand0=["AS", "JD"], piles=[(10, ["4C", "6C"], [1])])
    raises = acts(g, card="AS", kind=ActionKind.RAISE)
    assert len(raises) == 1
    a = raises[0]
    assert a.value == 11 and a.raised_from == 10 and a.loose == ()
    g.step(a)
    assert 10 not in g.piles
    assert g.piles[11].creators == frozenset({0})


def test_raise_absorbs_loose_sets_at_new_value():
    g = make_game(
        hand0=["2S", "QC"],
        table=["5H", "7H", "QD"],
        piles=[(10, ["4C", "6C"], [1])],
    )
    raises = acts(g, card="2S", kind=ActionKind.RAISE)
    assert len(raises) == 1
    assert raises[0].value == 12
    assert raises[0].loose == tuple(sorted(C("5H", "7H", "QD")))


def test_cannot_raise_own_pile():
    g = make_game(hand0=["AS", "JD"], piles=[(10, ["4C", "6C"], [0])])
    assert acts(g, kind=ActionKind.RAISE) == []


def test_cannot_raise_doubled_pile():
    g = make_game(hand0=["AS", "JD"], piles=[(10, ["4C", "6C", "10H"], [1])])
    assert g.piles[10].doubled
    assert acts(g, kind=ActionKind.RAISE) == []


def test_raise_requires_holding_target_value():
    g = make_game(hand0=["AS", "KD"], piles=[(10, ["4C", "6C"], [1])])
    # Holding K (13), not J (11): cannot raise 10 to 11 with the ace.
    assert acts(g, card="AS", kind=ActionKind.RAISE) == []


# ----------------------------------------------------------------- throwing


def test_cannot_throw_card_that_can_capture():
    g = make_game(hand0=["9S", "2H"], table=["9D"])
    assert acts(g, card="9S", kind=ActionKind.THROW) == []
    assert len(acts(g, card="2H", kind=ActionKind.THROW)) == 1


def test_cannot_throw_card_that_feeds_own_pile():
    g = make_game(
        hand0=["3S", "QH", "2D"],
        table=["9H"],
        piles=[(12, ["5C", "7C"], [0])],
    )
    # 3♠ could build 12 onto my own pile (3+9): throwing it is forbidden.
    assert acts(g, card="3S", kind=ActionKind.THROW) == []
    assert len(acts(g, card="3S", kind=ActionKind.BUILD, value=12)) == 1
    # 2♦ feeds nothing: free to throw.
    assert len(acts(g, card="2D", kind=ActionKind.THROW)) == 1


def test_card_that_feeds_opponents_pile_may_be_thrown():
    g = make_game(
        hand0=["3S", "QH", "2D"],
        table=["9H"],
        piles=[(12, ["5C", "7C"], [1])],
    )
    assert len(acts(g, card="3S", kind=ActionKind.THROW)) == 1


def test_reserved_card_may_only_capture():
    g = make_game(hand0=["QH", "2D"], piles=[(12, ["5C", "7C"], [0])])
    qh_actions = acts(g, card="QH")
    assert all(a.kind is ActionKind.PICKUP for a in qh_actions)
    assert all(a.takes_pile for a in qh_actions)


def test_reserved_rule_lifts_with_second_copy():
    g = make_game(hand0=["QH", "QD", "2D"], piles=[(12, ["5C", "7C"], [0])])
    kinds = {a.kind for a in acts(g, card="QH")}
    assert ActionKind.PICKUP in kinds and ActionKind.BUILD in kinds
    assert ActionKind.THROW not in kinds  # it can capture, so still no throw


def test_thrown_card_lands_on_table():
    g = make_game(hand0=["2H", "5D"], hand1=["KC"], table=["9D"])
    [throw] = acts(g, card="2H", kind=ActionKind.THROW)
    g.step(throw)
    assert card_from("2H") in g.table
    assert g.turn == 1


# ------------------------------------------------------------------- sweeps


def test_sweep_awarded_for_clearing_table():
    g = make_game(hand0=["9S", "2H"], hand1=["KC"], table=["4H", "5H"])
    [pickup] = acts(g, card="9S", kind=ActionKind.PICKUP)
    assert pickup.sweeps
    g.step(pickup)
    assert g.sweeps[0] == 1


def test_no_sweep_when_pile_remains():
    g = make_game(
        hand0=["9S", "2H"],
        hand1=["KC"],
        table=["4H", "5H"],
        piles=[(12, ["5C", "7C"], [1])],
    )
    [pickup] = acts(g, card="9S", kind=ActionKind.PICKUP)
    assert not pickup.sweeps
    g.step(pickup)
    assert g.sweeps[0] == 0


def test_sweep_on_second_to_last_play_counts():
    g = make_game(hand0=["9S"], hand1=["KC"], table=["4H", "5H"], deck=())
    [pickup] = acts(g, card="9S", kind=ActionKind.PICKUP)
    assert pickup.sweeps
    g.step(pickup)
    assert g.sweeps[0] == 1


def test_no_sweep_on_final_play_of_round():
    g = make_game(
        hand0=["9S"],
        hand1=[],
        table=["4H", "5H"],
        deck=(),
        points=(10, 0),
        captured=(30, 19),
    )
    [pickup] = acts(g, card="9S", kind=ActionKind.PICKUP)
    assert not pickup.sweeps
    g.step(pickup)  # ends the round
    # 10 preset + 9 (9♠) + 4 majority = 23; no 50 sweep bonus.
    assert g.round_history[-1] == (23, 0)


# ------------------------------------------------------------ round scoring


def test_leftovers_go_to_last_capturer():
    g = make_game(
        hand0=["2H"],
        hand1=[],
        table=["KS"],
        deck=(),
        last_capturer=1,
        captured=(26, 23),
    )
    [throw] = acts(g, card="2H", kind=ActionKind.THROW)
    g.step(throw)  # round ends; K♠ and 2♥ left on the table go to player 1
    # p1: 13 (K♠) + 0 (2♥) = 13 card points, 25 cards vs 26 -> p0 majority +4.
    assert g.round_history[-1] == (4, 13)


def test_capture_majority_bonus_and_tie():
    g = make_game(hand0=["2H"], hand1=[], deck=(), last_capturer=1, captured=(26, 25))
    [throw] = acts(g, card="2H", kind=ActionKind.THROW)
    g.step(throw)  # 2♥ goes to p1 as leftover: 26 vs 26 -> +2 each
    assert g.round_history[-1] == (2, 2)


# ----------------------------------------------- declaration & opening move


def test_table_hidden_until_declaration():
    g = Game(seed=7)
    fp = g.first_player
    assert g.awaiting == "declare"
    assert g.table == []
    assert len(g.unseen[fp]) == 48  # everything but their own hand
    options = g.declare_options()
    assert options and all(9 <= v <= 13 for v in options)
    hand_values = {card_value(c) for c in g.hands[fp]}
    assert set(options) <= hand_values
    g.declare(options[0])
    assert len(g.table) == 4
    assert len(g.unseen[fp]) == 44


def test_declaration_must_be_held():
    g = Game(seed=7)
    bad = sorted(set(range(9, 14)) - set(g.declare_options()))
    if bad:
        with pytest.raises(ValueError):
            g.declare(bad[0])


def test_first_player_dealt_a_nine_or_higher():
    for seed in range(40):
        g = Game(seed=seed)
        assert any(card_value(c) >= 9 for c in g.hands[g.first_player])


def test_opening_move_at_declared_value_and_no_throw_if_avoidable():
    for seed in range(40):
        g = Game(seed=seed)
        declared = g.declare_options()[-1]
        g.declare(declared)
        actions = g.legal_actions()
        assert actions
        assert all(a.value == declared for a in actions)
        kinds = {a.kind for a in actions}
        if ActionKind.THROW in kinds:
            assert kinds == {ActionKind.THROW}


def test_deal_counts():
    g = Game(seed=3)
    fp = g.first_player
    assert len(g.hands[fp]) == 4 and len(g.hands[1 - fp]) == 4
    assert len(g.deck) == 40 and len(g._facedown) == 4
    g.declare(g.declare_options()[0])
    g.step(g.legal_actions()[0])
    # After the opening move the first half is fully dealt: 11 + 12 in hand.
    assert len(g.hands[fp]) == 11 and len(g.hands[1 - fp]) == 12
    assert len(g.deck) == 24


def test_second_half_dealt_when_hands_empty():
    g = Game(seed=3)
    rng = random.Random(0)
    g.declare(rng.choice(g.declare_options()))
    plays = 0
    while plays < 24:
        g.step(rng.choice(g.legal_actions()))
        plays += 1
    assert len(g.hands[0]) == 12 and len(g.hands[1]) == 12
    assert g.deck == []


# ----------------------------------------------------- multi-round structure


def test_leader_plays_first_next_round():
    g = make_game(hand0=["2H"], hand1=[], deck=(), last_capturer=0, captured=(40, 11))
    g.differential = 0
    [throw] = acts(g, card="2H", kind=ActionKind.THROW)
    g.step(throw)  # p0 wins the round comfortably
    assert g.differential > 0
    assert g.first_player == 0
    assert g.awaiting == "declare"


def test_tied_differential_keeps_same_dealer():
    g = make_game(hand0=["2H"], hand1=[], deck=(), last_capturer=1, captured=(26, 25), first_player=1)
    g.differential = 0
    [throw] = acts(g, card="2H", kind=ActionKind.THROW)
    g.step(throw)  # 2-2 round: differential stays 0
    assert g.differential == 0
    assert g.first_player == 1  # unchanged


def test_game_ends_at_win_lead():
    g = Game(seed=11, win_lead=1)
    rng = random.Random(11)
    while not g.game_over:
        if g.awaiting == "declare":
            g.declare(rng.choice(g.declare_options()))
        else:
            g.step(rng.choice(g.legal_actions()))
    assert abs(g.differential) >= 1
    assert g.winner == (0 if g.differential > 0 else 1)


# -------------------------------------------------------------------- fuzz


def test_fuzz_invariants_over_random_games():
    for seed in range(12):
        g = Game(seed=seed, win_lead=100)
        rng = random.Random(seed)
        steps = 0
        while not g.game_over:
            if g.awaiting == "declare":
                g.declare(rng.choice(g.declare_options()))
                continue
            total = (
                len(g.deck)
                + sum(len(h) for h in g.hands)
                + len(g.table)
                + sum(len(p.cards) for p in g.piles.values())
                + sum(len(c) for c in g.captured)
            )
            assert total == 52
            for p in (0, 1):
                assert g.unseen[p] == set(g.deck) | set(g.hands[1 - p])
            for hand in g.hands:
                hand_values = [card_value(c) for c in hand]
                assert hand_values == sorted(hand_values), "hands stay value-sorted"
            actions = g.legal_actions()
            assert actions, "a player must always have a legal action"
            played_values = [card_value(a.card) for a in actions]
            assert played_values == sorted(played_values), \
                "legal actions follow the hand's value order"
            action = rng.choice(actions)
            pre_sweeps = g.sweeps[g.turn]
            pre_round = g.round_num
            g.step(action)
            if action.sweeps and g.round_num == pre_round:
                assert g.sweeps[1 - g.turn] == pre_sweeps + 1
            steps += 1
            assert steps < 100_000
        assert abs(g.differential) >= 100
        assert g.winner == (0 if g.differential > 0 else 1)
        for s0, s1 in g.round_history:
            assert s0 + s1 >= 100
            assert (s0 + s1 - 100) % 50 == 0
