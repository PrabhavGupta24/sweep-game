"""Tests for the play.py CLI, focused on opponent selection (--ai)."""

import pytest

import play
from sweep.agents import GreedyAgent, HeuristicAgent, RandomAgent


def test_registry_maps_names_to_agents():
    assert play.AI_AGENTS == {
        "random": RandomAgent,
        "greedy": GreedyAgent,
        "heuristic": HeuristicAgent,
    }
    for name, cls in play.AI_AGENTS.items():
        assert cls(seed=1).name == name  # the agent knows its own name


def _capture_agent(monkeypatch):
    """Patch run() to record the agent it is handed and exit immediately."""
    seen = {}

    def fake_run(console, game, ai):
        seen["ai"] = ai
        return True  # pretend the game finished cleanly

    monkeypatch.setattr(play, "run", fake_run)
    return seen


@pytest.mark.parametrize(
    "flag,cls",
    [("random", RandomAgent), ("greedy", GreedyAgent), ("heuristic", HeuristicAgent)],
)
def test_ai_flag_selects_agent(monkeypatch, flag, cls):
    seen = _capture_agent(monkeypatch)
    assert play.main(["--ai", flag, "--ai-seed", "1"]) == 0
    assert isinstance(seen["ai"], cls)


def test_default_opponent_is_random(monkeypatch):
    seen = _capture_agent(monkeypatch)
    assert play.main([]) == 0
    assert isinstance(seen["ai"], RandomAgent)


def test_unknown_ai_is_rejected(monkeypatch):
    _capture_agent(monkeypatch)
    with pytest.raises(SystemExit):  # argparse rejects an invalid choice
        play.main(["--ai", "bogus"])


def test_ai_seed_is_reproducible(monkeypatch):
    """Same --ai-seed yields agents that make the same first declaration."""
    from sweep.engine import Game

    seen = _capture_agent(monkeypatch)
    play.main(["--ai", "greedy", "--ai-seed", "7"])
    a = seen["ai"]
    play.main(["--ai", "greedy", "--ai-seed", "7"])
    b = seen["ai"]
    g = Game(seed=3)
    assert a.declare(g) == b.declare(g)
