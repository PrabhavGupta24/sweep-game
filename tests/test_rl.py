"""RL layer tests: RoundResult, the pure encoders, and the SweepEnv stream."""

import inspect
import random

import numpy as np
import pytest

from sweep import ui
from sweep.agents import RandomAgent
from sweep.cards import card_from
from sweep.engine import ActionKind, Game
from sweep.rl import (
    ACTION_SIZE,
    DECLARE,
    OBS_SIZE,
    SweepEnv,
    encode_action,
    encode_observation,
)
from test_rules import C, make_game

# ------------------------------------------------------------------ helpers


def random_game(seed, win_lead=100):
    """Play a full random game to completion."""
    g = Game(seed=seed, win_lead=win_lead)
    rng = random.Random(seed)
    while not g.game_over:
        if g.awaiting == "declare":
            g.declare(rng.choice(g.declare_options()))
        else:
            g.step(rng.choice(g.legal_actions()))
    return g


def midgame_after_rounds(seed=0, rounds=2):
    """A live position with `rounds` finished rounds behind it."""
    g = Game(seed=seed, win_lead=10**6)
    rng = random.Random(seed)
    while len(g.round_history) < rounds or g.awaiting == "declare":
        if g.awaiting == "declare":
            g.declare(rng.choice(g.declare_options()))
        else:
            g.step(rng.choice(g.legal_actions()))
    return g


def decisions_of_round(seed):
    """(view, awaiting, candidates) for every decision of one random round."""
    g = Game(seed=seed)
    rng = random.Random(seed)
    out = []
    while len(g.round_history) == 0 and not g.game_over:
        if g.awaiting == "declare":
            player = g.first_player
            candidates = [(DECLARE, v) for v in g.declare_options()]
        else:
            player = g.turn
            candidates = g.legal_actions()
        out.append((g.view(player), g.awaiting, candidates))
        if g.awaiting == "declare":
            g.declare(rng.choice(g.declare_options()))
        else:
            g.step(rng.choice(g.legal_actions()))
    return out


# -------------------------------------------------------------- RoundResult


def test_round_results_consistent_over_random_games():
    for seed in range(8):
        g = random_game(seed)
        assert len(g.round_results) == len(g.round_history)
        for rr, scores in zip(g.round_results, g.round_history):
            assert rr.scores == tuple(scores)
            for p in range(2):
                assert (rr.card_points[p] + rr.majority_bonus[p]
                        + 50 * rr.sweeps[p]) == rr.scores[p]
            assert rr.majority_bonus in ((4, 0), (0, 4), (2, 2))
            if rr.leftover_to is not None:
                assert rr.leftover_to in (0, 1)
                assert rr.leftover_cards
                assert sum(rr.captured_counts) == 52
            else:
                # Nothing was awarded as leftovers: every leftover card (if
                # any — only when nobody captured all round) stays unowned.
                assert sum(rr.captured_counts) + len(rr.leftover_cards) == 52
            assert len(set(rr.leftover_cards)) == len(rr.leftover_cards)


def test_round_result_leftovers_are_the_final_table():
    g = make_game(
        hand0=["2H"],
        hand1=[],
        table=["KS"],
        deck=(),
        last_capturer=1,
        captured=(26, 23),
    )
    [throw] = [a for a in g.legal_actions() if a.kind is ActionKind.THROW]
    g.step(throw)  # round ends; K♠ and the thrown 2♥ go to player 1
    rr = g.round_results[-1]
    assert rr.leftover_to == 1
    assert sorted(rr.leftover_cards) == sorted(C("KS", "2H"))
    assert rr.captured_counts == (26, 25)
    assert rr.majority_bonus == (4, 0)
    assert rr.card_points == (0, 13)  # K♠ arrives with the leftovers
    assert rr.scores == g.round_history[-1] == (4, 13)


def test_round_result_matches_ui_reconstruction():
    """The UI's public-info reconstruction was built to recover exactly this
    breakdown: it must agree with RoundResult on every shared field."""
    for seed in (0, 1, 2):
        g = Game(seed=seed, win_lead=200)
        agent = RandomAgent(seed=seed)
        while not g.game_over:
            if g.awaiting == "declare":
                g.declare(agent.declare(g))
                continue
            finished = len(g.round_results)
            snap = ui.public_snapshot(g)
            player = g.turn
            action = agent.act(g)
            g.step(action)
            if len(g.round_results) > finished:
                rr = g.round_results[-1]
                summary = ui.round_summary(snap, action, player, rr.scores)
                assert summary["captured_counts"] == rr.captured_counts
                assert summary["card_points"] == rr.card_points
                assert summary["bonus"] == rr.majority_bonus
                assert summary["sweeps"] == rr.sweeps
                assert summary["scores"] == rr.scores


def test_clone_copies_round_results_as_new_list():
    g = midgame_after_rounds(seed=1, rounds=2)
    c = g.clone()
    assert c.round_results == g.round_results
    assert c.round_results is not g.round_results
    c.round_results.pop()
    assert len(g.round_results) == 2


# ----------------------------------------------------------------- encoders


def test_sizes_dtype_and_range():
    for seed in range(4):
        for view, awaiting, candidates in decisions_of_round(seed):
            obs = encode_observation(view, awaiting)
            assert obs.shape == (OBS_SIZE,)
            assert obs.dtype == np.float32
            assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
            for item in candidates:
                vec = encode_action(item, view)
                assert vec.shape == (ACTION_SIZE,)
                assert vec.dtype == np.float32
                assert np.all(vec >= -1.0) and np.all(vec <= 1.0)


def test_encoding_ignores_hidden_deck_order():
    """Only the unseen SET is public; the deck's order is hidden. Shuffling
    the deck in place must leave every encoding bit-identical."""
    g = Game(seed=5)
    g.declare(g.declare_options()[0])
    rng = random.Random(5)
    for _ in range(4):
        g.step(rng.choice(g.legal_actions()))
    assert g.deck
    p = g.turn
    candidates = g.legal_actions()
    obs_before = encode_observation(g.view(p), g.awaiting)
    acts_before = [encode_action(a, g.view(p)) for a in candidates]

    rng.shuffle(g.deck)

    assert np.array_equal(encode_observation(g.view(p), g.awaiting), obs_before)
    for a, before in zip(g.legal_actions(), acts_before):
        assert np.array_equal(encode_action(a, g.view(p)), before)


def test_encoders_take_views_not_games():
    assert list(inspect.signature(encode_observation).parameters) == \
        ["view", "awaiting", "np_out"]
    assert list(inspect.signature(encode_action).parameters) == \
        ["item", "view", "np_out"]
    with pytest.raises(TypeError):
        encode_observation(Game(seed=0), "declare")


def test_observation_multi_hots_and_pile_block():
    g = make_game(
        hand0=["9S", "2H"],
        table=["4H", "5H"],
        piles=[(12, ["5C", "7C"], [1])],
    )
    obs = encode_observation(g.view(0), "play")
    assert {c for c in range(52) if obs[c]} == set(C("9S", "2H"))
    assert {c for c in range(52) if obs[52 + c]} == set(C("4H", "5H"))
    assert {c for c in range(52) if obs[104 + c]} == set(C("5C", "7C"))
    assert not obs[156:208].any()  # make_game leaves unseen empty

    base = 208 + 6 * (12 - 9)  # the pile-of-12 block
    assert obs[base] == 1.0  # exists
    assert obs[base + 1] == 0.0  # not mine
    assert obs[base + 2] == 1.0  # opponent's
    assert obs[base + 3] == 0.0  # not doubled
    assert obs[base + 4] == pytest.approx(2 / 13)
    assert obs[base + 5] == 0.0  # clubs are worth nothing
    for v in (9, 10, 11, 13):
        assert not obs[208 + 6 * (v - 9):208 + 6 * (v - 9) + 6].any()

    # A few scalars: deck of 2, first half, nothing declared, play phase.
    assert obs[245] == pytest.approx(2 / 40)
    assert obs[246] == 0.0
    assert obs[247] == 0.0
    assert obs[250] == 1.0  # nobody has captured yet
    assert obs[252] == 0.0


def test_pickup_encoding_prices_the_pile():
    g = make_game(hand0=["9S", "2H"], table=["9H"], piles=[(9, ["4C", "5C"], [1])])
    view = g.view(0)
    [pickup] = [a for a in g.legal_actions() if a.kind is ActionKind.PICKUP]
    assert pickup.takes_pile and pickup.sweeps
    vec = encode_action(pickup, view)
    assert vec[0] == 1.0 and not vec[1:5].any()  # kind: pickup
    assert {c for c in range(52) if vec[5 + c]} == {card_from("9S")}
    assert {c for c in range(52) if vec[57 + c]} == set(C("9H"))
    assert vec[109] == pytest.approx(9 / 13)
    assert vec[110] == 0.0
    assert vec[111] == 1.0  # takes the pile
    assert vec[112] == 1.0  # sweeps
    # 9♠ (9) + 9♥/4♣/5♣ (0): pile cards are included in the immediate haul.
    assert vec[113] == pytest.approx(9 / 20)
    assert vec[114] == pytest.approx(4 / 16)


def test_throw_encoding_has_no_capture_features():
    g = make_game(hand0=["2H", "5D"], hand1=["KC"], table=["9D"])
    view = g.view(0)
    [throw] = [a for a in g.legal_actions()
               if a.kind is ActionKind.THROW and a.card == card_from("2H")]
    vec = encode_action(throw, view)
    assert vec[3] == 1.0  # kind: throw
    assert vec[5 + card_from("2H")] == 1.0
    assert not vec[57:109].any()
    assert vec[109] == pytest.approx(2 / 13)
    assert vec[111] == vec[112] == vec[113] == vec[114] == 0.0


def test_declare_candidates_encoding():
    g = Game(seed=3)
    view = g.view(g.first_player)
    for v in g.declare_options():
        vec = encode_action((DECLARE, v), view)
        assert vec[4] == 1.0 and not vec[0:4].any()  # kind: declare
        assert not vec[5:57].any()  # no played card
        assert vec[109] == pytest.approx(v / 13)
        assert not vec[110:].any()
    obs = encode_observation(view, g.awaiting)
    assert obs[252] == 1.0  # awaiting-declaration flag
    with pytest.raises(ValueError):
        encode_action(("bogus", 9), view)


def test_np_out_reuse_matches_fresh_encoding():
    g = midgame_after_rounds(seed=2, rounds=1)
    view = g.view(g.turn)
    out = np.full(OBS_SIZE, 7.0, dtype=np.float32)
    assert encode_observation(view, g.awaiting, np_out=out) is out
    assert np.array_equal(out, encode_observation(view, g.awaiting))
    a = g.legal_actions()[0]
    out_a = np.full(ACTION_SIZE, 7.0, dtype=np.float32)
    assert encode_action(a, view, np_out=out_a) is out_a
    assert np.array_equal(out_a, encode_action(a, view))


# ---------------------------------------------------------------- SweepEnv


def test_env_full_episode_fuzz():
    for seed in range(25):
        env = SweepEnv(seed=seed)
        rng = random.Random(seed)
        decision = env.reset()
        steps = 0
        done = False
        while not done:
            game = env.game
            if game.awaiting == "declare":
                assert decision["player"] == game.first_player
                assert decision["candidates"] == \
                    [(DECLARE, v) for v in game.declare_options()]
            else:
                assert decision["player"] == game.turn
                assert decision["candidates"] == game.legal_actions()
            n = len(decision["candidates"])
            assert decision["obs"].shape == (OBS_SIZE,)
            assert decision["encoded_candidates"].shape == (n, ACTION_SIZE)
            assert decision["encoded_candidates"].dtype == np.float32
            decision, rewards, done = env.step(rng.randrange(n))
            steps += 1
            if not done:
                assert rewards is None
        assert steps == 49  # one declaration + 48 plays, done exactly at the end
        assert decision is None
        rr = env.game.round_results[-1]
        assert rewards == ((rr.scores[0] - rr.scores[1]) / 100.0,
                           (rr.scores[1] - rr.scores[0]) / 100.0)
        assert rewards[0] + rewards[1] == 0.0


def test_env_determinism():
    def run(seed):
        env = SweepEnv(seed=seed)
        rng = random.Random(1234)
        decision = env.reset()
        trace = []
        done = False
        while not done:
            trace.append((
                decision["player"],
                decision["obs"].tobytes(),
                decision["encoded_candidates"].tobytes(),
                tuple(repr(c) for c in decision["candidates"]),
            ))
            decision, rewards, done = env.step(rng.randrange(len(decision["candidates"])))
        trace.append(rewards)
        return trace

    for seed in (0, 7):
        assert run(seed) == run(seed)
    assert run(0) != run(7)


def test_env_continue_same_game_plays_next_round():
    env = SweepEnv(seed=42)
    rng = random.Random(42)

    def play_round(decision):
        steps, done = 0, False
        while not done:
            decision, rewards, done = env.step(rng.randrange(len(decision["candidates"])))
            steps += 1
        return steps, rewards

    steps, rewards = play_round(env.reset())
    assert steps == 49
    assert len(env.game.round_results) == 1
    assert not env.game.game_over  # seed chosen so round 1 never decides it

    steps, rewards = play_round(env.continue_same_game())
    assert steps == 49
    assert len(env.game.round_results) == 2
    rr = env.game.round_results[-1]
    assert rewards == ((rr.scores[0] - rr.scores[1]) / 100.0,
                       (rr.scores[1] - rr.scores[0]) / 100.0)


def test_env_step_requires_pending_decision():
    env = SweepEnv(seed=0)
    with pytest.raises(RuntimeError):
        env.step(0)
    rng = random.Random(0)
    decision, done = env.reset(), False
    while not done:
        decision, _, done = env.step(rng.randrange(len(decision["candidates"])))
    with pytest.raises(RuntimeError):  # episode over: no pending decision
        env.step(0)
