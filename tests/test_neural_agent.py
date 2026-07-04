"""NeuralAgent tests: legality, determinism, checkpoint loading, and the
evaluate.py / play.py CLI wiring for the neural opponent."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

torch.set_num_threads(1)

import evaluate
import play
from sweep.agents import RandomAgent
from sweep.engine import Game
from sweep.rl import NeuralAgent
from sweep.rl.model import PolicyValueNet
from sweep.rl.ppo import save_checkpoint

REPO_ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ helpers


def make_net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet()


def drive_checked_pair(agents, seed, win_lead=100):
    """drive_checked (test_agents) for two pre-built agents: play a full
    game, asserting every choice comes from the legal sets."""
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


def save_random_ckpt(tmp_path, seed=0):
    path = tmp_path / "net.pt"
    save_checkpoint(path, make_net(seed), None, {"round": 0})
    return path


# ------------------------------------------------------------- legality fuzz


@pytest.mark.parametrize("neural_seat", [0, 1])
def test_neural_plays_legal_full_game_vs_random(neural_seat):
    agents = [None, None]
    agents[neural_seat] = NeuralAgent(net=make_net(neural_seat))
    agents[1 - neural_seat] = RandomAgent(seed=neural_seat + 10)
    game, trace = drive_checked_pair(agents, seed=neural_seat)
    assert game.winner in (0, 1)
    assert trace


# -------------------------------------------------------------- determinism


def test_temperature_zero_is_deterministic():
    net = make_net()
    runs = [
        drive_checked_pair([NeuralAgent(net=net), RandomAgent(seed=5)], seed=3)
        for _ in range(2)
    ]
    assert runs[0][1] == runs[1][1]
    assert runs[0][0].round_history == runs[1][0].round_history


def test_sampling_same_seed_is_deterministic():
    net = make_net()
    runs = [
        drive_checked_pair(
            [NeuralAgent(net=net, seed=11, temperature=1.0), RandomAgent(seed=5)],
            seed=3,
        )
        for _ in range(2)
    ]
    assert runs[0][1] == runs[1][1]


def test_sampling_seed_changes_choices():
    # Repeated act() calls on one frozen mid-round state: 64 samples at
    # temperature 1 from >= 4 near-uniform candidates collide with
    # negligible probability, so different seeds must diverge somewhere.
    net = make_net()
    game = Game(seed=0, win_lead=100)
    game.declare(max(game.declare_options()))
    game.step(game.legal_actions()[0])  # past the opening's restricted move
    assert len(game.legal_actions()) >= 4  # a full 4-card hand to choose from

    def picks(seed):
        agent = NeuralAgent(net=net, seed=seed, temperature=1.0)
        return [agent.act(game) for _ in range(64)]

    assert picks(1) != picks(2)


# -------------------------------------------------------------- checkpoints


def test_checkpoint_load_matches_original_net(tmp_path):
    net = make_net(3)
    path = tmp_path / "net.pt"
    save_checkpoint(path, net, None, {"round": 7})
    original = NeuralAgent(net=net)
    loaded = NeuralAgent(ckpt_path=str(path))

    game = Game(seed=4, win_lead=100)
    while not game.game_over:
        if game.awaiting == "declare":
            v = original.declare(game)
            assert loaded.declare(game) == v
            game.declare(v)
        else:
            action = original.act(game)
            assert loaded.act(game) == action
            game.step(action)


def test_fresh_game_declaration_is_legal():
    agent = NeuralAgent(net=make_net())
    game = Game(seed=9)
    assert agent.declare(game) in game.declare_options()


# ---------------------------------------------------------------------- CLI


def test_evaluate_neural_requires_ckpt(capsys):
    with pytest.raises(SystemExit):
        evaluate.main(["neural", "random", "-n", "2"])
    assert "--ckpt" in capsys.readouterr().err


def test_evaluate_neural_vs_random_match(tmp_path, capsys):
    path = save_random_ckpt(tmp_path)
    evaluate.main(["neural", "random", "-n", "2", "--win-lead", "100",
                   "--ckpt", str(path)])
    out = capsys.readouterr().out
    assert "neural vs random" in out


def test_play_neural_requires_ckpt(capsys):
    with pytest.raises(SystemExit):
        play.main(["--ai", "neural"])
    assert "--ckpt" in capsys.readouterr().err


def test_play_neural_quits_cleanly_on_q(tmp_path):
    path = save_random_ckpt(tmp_path)
    proc = subprocess.run(
        [sys.executable, "play.py", "--ai", "neural", "--ckpt", str(path),
         "--seed", "0", "--ai-seed", "0"],
        input="q\n", capture_output=True, text=True, timeout=120,
        cwd=REPO_ROOT,
    )
    # 'q' quits at the first prompt (or exhausts input right after), which
    # abandons the game cleanly with exit status 0.
    assert proc.returncode == 0, proc.stderr
    assert "abandoned" in proc.stdout
