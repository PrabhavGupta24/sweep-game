"""Agent and arena tests."""

import pytest

from sweep.agents import GreedyAgent, HeuristicAgent, RandomAgent
from sweep.arena import play_match, round_robin, wilson_ci
from sweep.cards import card_from
from sweep.engine import ActionKind, Game
from test_rules import make_game

AGENT_CLASSES = [RandomAgent, GreedyAgent, HeuristicAgent]


def drive_checked_agents(agents, seed, win_lead=100):
    """Play a full game with two pre-built agents, asserting every choice
    comes from the legal sets."""
    game = Game(seed=seed, win_lead=win_lead)
    trace = []
    while not game.game_over:
        if game.awaiting == "declare":
            v = agents[game.first_player].declare(game)
            assert v in game.declare_options()
            trace.append(("declare", v))
            game.declare(v)
        else:
            action = agents[game.turn].act(game)
            assert action in game.legal_actions()
            trace.append(action)
            game.step(action)
    return game, trace


def drive_checked(agent_cls, seed, win_lead=100):
    """drive_checked_agents with two same-class agents built from `seed`."""
    agents = [agent_cls(seed=seed + 1), agent_cls(seed=seed + 2)]
    return drive_checked_agents(agents, seed, win_lead)


@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
def test_agents_only_play_legal_actions(agent_cls):
    for seed in range(4):
        game, trace = drive_checked(agent_cls, seed)
        assert game.winner in (0, 1)
        assert trace


@pytest.mark.parametrize("agent_cls", AGENT_CLASSES)
def test_agents_deterministic_given_seeds(agent_cls):
    game1, trace1 = drive_checked(agent_cls, seed=7)
    game2, trace2 = drive_checked(agent_cls, seed=7)
    assert trace1 == trace2
    assert game1.round_history == game2.round_history
    assert game1.winner == game2.winner


def test_greedy_takes_highest_point_capture():
    # K♥ captures K♠ (13 pts) vs 9♠ capturing 9♥ (9 pts): greedy must take K♠.
    g = make_game(hand0=["9S", "KH", "2H"], hand1=["3C"], table=["9H", "KS"])
    action = GreedyAgent(seed=0).act(g)
    assert action.kind is ActionKind.PICKUP
    assert action.card == card_from("KH")
    assert action.loose == (card_from("KS"),)


def test_greedy_prefers_sweep():
    # 9♠ sweeps 4♥+5♥ (+50) even though K♥ on K♠ takes more raw card points.
    g = make_game(hand0=["9S", "KH", "2H"], hand1=["3C"], table=["4H", "5H"])
    action = GreedyAgent(seed=0).act(g)
    assert action.kind is ActionKind.PICKUP
    assert action.sweeps
    assert action.card == card_from("9S")


def test_greedy_throws_lowest_point_card():
    # Nothing captures: greedy throws the zero-point low card, not A♠ or 10♦.
    g = make_game(hand0=["AS", "10D", "3H"], hand1=["3C"], table=["KC"])
    action = GreedyAgent(seed=0).act(g)
    assert action.kind is ActionKind.THROW
    assert action.card == card_from("3H")


def test_heuristic_takes_obvious_capture():
    g = make_game(hand0=["KH", "2H", "3D"], hand1=["3C"], table=["KS", "QC"])
    g.unseen = [set(range(26, 52)) - set(g.hands[0]), set()]
    action = HeuristicAgent(seed=0).act(g)
    assert action.kind is ActionKind.PICKUP
    assert action.card == card_from("KH")


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.5
    lo, hi = wilson_ci(10, 10)
    assert 0.5 < lo < 1.0 and hi == 1.0
    lo, hi = wilson_ci(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_tiny_arena_match_reports_sane_numbers():
    result = play_match(
        lambda s: RandomAgent(seed=s),
        lambda s: GreedyAgent(seed=s),
        n_games=8,
        base_seed=0,
        win_lead=100,
    )
    assert result["games"] == 8
    assert result["wins_a"] + result["wins_b"] == 8
    assert 0.0 <= result["win_rate_a"] <= 1.0
    lo, hi = result["ci95"]
    assert 0.0 <= lo <= result["win_rate_a"] <= hi <= 1.0
    assert result["mean_rounds"] >= 1.0
    # 48 plays per round (each player plays 24 cards), every round completes.
    assert result["total_plays"] > 0
    assert result["seconds"] > 0.0


def test_play_match_deterministic():
    kwargs = dict(n_games=4, base_seed=3, win_lead=100)
    r1 = play_match(lambda s: GreedyAgent(seed=s), lambda s: HeuristicAgent(seed=s), **kwargs)
    r2 = play_match(lambda s: GreedyAgent(seed=s), lambda s: HeuristicAgent(seed=s), **kwargs)
    assert (r1["wins_a"], r1["mean_diff_a"], r1["mean_rounds"]) == (
        r2["wins_a"], r2["mean_diff_a"], r2["mean_rounds"]
    )


def test_round_robin_covers_all_pairs():
    factories = {
        "random": lambda s: RandomAgent(seed=s),
        "greedy": lambda s: GreedyAgent(seed=s),
    }
    results = round_robin(factories, n_games=2, base_seed=0, win_lead=100)
    assert set(results) == {("random", "greedy")}
    assert results[("random", "greedy")]["games"] == 2
