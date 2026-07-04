"""HybridAgent tests.

Covers the one subtle correctness risk (the value head's perspective sign),
the behavior-preserving ISMCTS refactor (against an inlined copy of the
ORIGINAL search loop as reference), legality/determinism/equivalence fuzz, the
value_only contract, and the evaluate.py / play.py CLI wiring.

Suite budget: kept well under a few minutes by using tiny n_sims everywhere
except the two committed-checkpoint CLI smokes.
"""

import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

import evaluate
import play
from sweep.agents import RandomAgent
from sweep.cards import card_value
from sweep.engine import ActionKind, Game
from sweep.ismcts import (
    ISMCTSAgent,
    _exact_reward,
    _rollout_action,
    determinize,
)
from sweep.rl import HybridAgent
from sweep.rl.model import PolicyValueNet
from sweep.rl.ppo import save_checkpoint
from test_agents import drive_checked_agents
from test_rules import make_game

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "models" / "neural-250k.pt"

# ------------------------------------------------------------------ helpers


def make_net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet()


def live_midround_game(turn=0):
    """A handcrafted mid-round position: 3 cards each + a live deck, so a

    single expansion always lands on a still-live leaf (no round can end
    while any hand or the deck is non-empty)."""
    g = make_game(
        hand0=["2H", "3S", "5S"],
        hand1=["7C", "8C", "9C"],
        table=[],
        deck=["2D", "3D"],
        turn=turn,
    )
    g.unseen[turn] = set(g.hands[1 - turn]) | set(g.deck)
    g.unseen[1 - turn] = set(g.hands[turn]) | set(g.deck)
    return g


# ---------------------------------------------------- perspective-sign (key)


def test_leaf_reward_perspective_sign():
    """The critical test: the value head predicts the *mover's* own swing, so

    HybridAgent must negate it when the leaf's mover is not the search root.
    Stub value_only to a constant +0.7 and spy on _leaf_reward: every live
    leaf where det.turn == root must back up +0.7, and where det.turn != root,
    -0.7. A deliberate sign flip in _leaf_reward makes this FAIL (verified).
    """
    const = 0.7
    game = live_midround_game(turn=0)
    root = game.turn

    agent = HybridAgent(net=make_net(), seed=1, n_sims=6)
    agent.net.value_only = lambda obs: const  # stub: constant estimate

    seen = []  # (is_root_mover, reward) per leaf evaluation
    real_leaf_reward = agent._leaf_reward.__func__  # unbound, call explicitly

    def spy(det, r, hist, rng):
        is_root = det.turn == r
        reward = real_leaf_reward(agent, det, r, hist, rng)
        seen.append((is_root, reward))
        return reward

    agent._leaf_reward = spy
    agent.act(game)

    assert seen, "no leaves were evaluated"
    # Every leaf in this position is live (see live_midround_game), so all
    # rewards come from the value head, not the exact-swing fallback.
    for is_root, reward in seen:
        expected = const if is_root else -const
        assert reward == pytest.approx(expected), (is_root, reward)
    # Both perspectives must actually occur, or the sign logic is untested.
    assert any(is_root for is_root, _ in seen)
    assert any(not is_root for is_root, _ in seen)


# ----------------------------------------- plain-ISMCTS behavior regression


def _reference_act(agent, game):
    """A verbatim copy of the ORIGINAL ISMCTSAgent.act search loop (pre-leaf-

    hook refactor), used as an oracle: the refactored base ISMCTSAgent.act
    must choose identical actions for the same seeds. Kept deliberately close
    to the committed original — inline greedy rollout, negamax backprop, exact
    reward from det.round_history[hist], visit-count move choice.
    """
    from sweep.ismcts import _Child, _Node

    actions = game.legal_actions()
    if len(actions) == 1:
        return actions[0]
    root = game.turn
    hist = len(game.round_history)
    tree = _Node()
    for _ in range(agent.n_sims):
        it_rng = random.Random(agent.rng.getrandbits(64))
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
            action = max(existing, key=lambda a: agent._uct(children[a]))
            child = children[action]
            det.step(action)
            path.append((child, mover))
            node = child.node
        scores = det.round_history[hist]
        reward = (scores[root] - scores[1 - root]) / 100.0
        for child, mover in path:
            child.n += 1
            child.w += reward if mover == root else -reward
    return max(actions, key=lambda a: tree.children[a].n if a in tree.children else -1)


def _refactored_trajectory(seed, n_plays):
    game = Game(seed=seed, win_lead=100)
    agents = [ISMCTSAgent(seed=seed + 1, n_sims=12), RandomAgent(seed=seed + 2)]
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


def _reference_trajectory(seed, n_plays):
    game = Game(seed=seed, win_lead=100)
    agents = [ISMCTSAgent(seed=seed + 1, n_sims=12), RandomAgent(seed=seed + 2)]
    trace = []
    while not game.game_over and len(trace) < n_plays:
        if game.awaiting == "declare":
            v = agents[game.first_player].declare(game)
            trace.append(("declare", v))
            game.declare(v)
        else:
            agent = agents[game.turn]
            if isinstance(agent, ISMCTSAgent):
                a = _reference_act(agent, game)
            else:
                a = agent.act(game)
            trace.append(a)
            game.step(a)
    return trace


@pytest.mark.parametrize("seed", [6, 21])
def test_refactored_ismcts_matches_original_loop(seed):
    """The behavior-preserving refactor: the base ISMCTSAgent chooses exactly

    the same actions as an inlined copy of the original search loop, over two
    seeded games vs RandomAgent (declares and plays)."""
    assert _refactored_trajectory(seed, 40) == _reference_trajectory(seed, 40)


# ------------------------------------------------------------- legality fuzz


@pytest.mark.parametrize("hybrid_seat", [0, 1])
def test_hybrid_plays_legal_full_game_vs_random(hybrid_seat):
    agents = [None, None]
    agents[hybrid_seat] = HybridAgent(net=make_net(hybrid_seat), seed=hybrid_seat, n_sims=8)
    agents[1 - hybrid_seat] = RandomAgent(seed=hybrid_seat + 10)
    game, trace = drive_checked_agents(agents, seed=hybrid_seat)
    assert game.winner in (0, 1)
    assert trace


# -------------------------------------------------------------- determinism


def test_hybrid_same_seed_is_deterministic():
    net = make_net()
    runs = [
        drive_checked_agents(
            [HybridAgent(net=net, seed=7, n_sims=8), RandomAgent(seed=5)], seed=3
        )
        for _ in range(2)
    ]
    assert runs[0][1] == runs[1][1]
    assert runs[0][0].round_history == runs[1][0].round_history


def test_hybrid_ckpt_matches_net_instance():
    """A HybridAgent built from a checkpoint path plays identically to one

    built from the same net instance (same seed => same trajectory)."""
    net = make_net(3)
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "net.pt"
        save_checkpoint(path, net, None, {"round": 0})
        from_net = HybridAgent(net=net, seed=2, n_sims=8)
        from_ckpt = HybridAgent(ckpt_path=str(path), seed=2, n_sims=8)

        game = Game(seed=4, win_lead=100)
        while not game.game_over:
            if game.awaiting == "declare":
                v = from_net.declare(game)
                assert from_ckpt.declare(game) == v
                game.declare(v)
            else:
                a = from_net.act(game)
                assert from_ckpt.act(game) == a
                game.step(a)


# --------------------------------------------------------- value_only contract


def test_value_only_matches_forward():
    net = make_net(1)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.standard_normal(253).astype(np.float32)
        cands = np.zeros((1, 1, 115), dtype=np.float32)
        mask = torch.ones((1, 1), dtype=torch.bool)
        _, values = net(torch.as_tensor(obs).unsqueeze(0), torch.as_tensor(cands), mask)
        vo = net.value_only(obs)
        assert isinstance(vo, float)  # plain Python scalar, not a tensor
        assert not isinstance(vo, np.ndarray)
        assert math.isclose(vo, float(values[0]), rel_tol=0, abs_tol=1e-6)


# ---------------------------------------------------------------------- CLI


def test_evaluate_hybrid_requires_ckpt(capsys):
    with pytest.raises(SystemExit):
        evaluate.main(["hybrid", "random", "-n", "2"])
    assert "--ckpt" in capsys.readouterr().err


@pytest.mark.skipif(not CKPT.exists(), reason="committed checkpoint missing")
def test_evaluate_hybrid_vs_random_match(capsys):
    evaluate.main(["hybrid", "random", "-n", "2", "--win-lead", "100",
                   "--sims", "8", "--ckpt", str(CKPT)])
    out = capsys.readouterr().out
    assert "hybrid vs random" in out


def test_play_hybrid_requires_ckpt(capsys):
    with pytest.raises(SystemExit):
        play.main(["--ai", "hybrid"])
    assert "--ckpt" in capsys.readouterr().err


@pytest.mark.skipif(not CKPT.exists(), reason="committed checkpoint missing")
def test_play_hybrid_quits_cleanly_on_q():
    proc = subprocess.run(
        [sys.executable, "play.py", "--ai", "hybrid", "--ckpt", str(CKPT),
         "--seed", "0", "--ai-seed", "0"],
        input="q\n", capture_output=True, text=True, timeout=180,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "abandoned" in proc.stdout
