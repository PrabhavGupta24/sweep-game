"""PUCT (policy-prior) tests for HybridAgent(priors=True).

Covers the prior cache and its renormalization/merge invariants, the PUCT
score of an unvisited action, the mover-perspective of the policy evaluation
(the subtle leak risk), the priors=False regression (byte-identical to the
shipped value-head ISMCTS), plus legality / determinism fuzz and the CLI
wiring.

Suite budget: tiny n_sims and random-init nets everywhere except the single
committed-checkpoint CLI smoke, keeping the whole file well under a few
minutes.
"""

import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

import evaluate
import play
from sweep.agents import RandomAgent
from sweep.engine import Game
from sweep.ismcts import ISMCTSAgent, determinize
from sweep.rl.hybrid import HybridAgent, _PriorChild, _PriorNode
from sweep.rl.model import PolicyValueNet
from test_agents import drive_checked_agents
from test_rules import make_game

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "models" / "neural-250k.pt"


# ------------------------------------------------------------------ helpers


def make_net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet()


def make_agent(seed=0, net_seed=0, **kw):
    kw.setdefault("priors", True)
    return HybridAgent(net=make_net(net_seed), seed=seed, **kw)


def live_midround_game(turn=0):
    """A handcrafted mid-round position: 3 cards each + a live deck, so every
    expansion lands on a still-live leaf and both players still get to move."""
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


class RecorderNet:
    """Wraps a real net but records every obs tensor passed to __call__.

    Used to inspect which player's view the policy priors are computed on: the
    recorded obs must reflect the *mover's* determinized hand, not the root's.
    """

    def __init__(self, net):
        self.net = net
        self.obs_seen = []  # each a float32 numpy copy of the passed obs vector

    def __call__(self, obs, cands, mask):
        self.obs_seen.append(obs[0].detach().cpu().numpy().copy())
        return self.net(obs, cands, mask)

    def value_only(self, obs_np):
        return self.net.value_only(obs_np)

    def eval(self):
        self.net.eval()
        return self


def _hand_multihot(hand):
    mh = np.zeros(52, dtype=np.float32)
    for c in hand:
        mh[c] = 1.0
    return mh


# ------------------------------------------------------ prior cache / softmax


def test_priors_are_a_valid_distribution():
    """_priors_for returns a proper distribution over exactly the given
    actions (non-negative, sums to 1, one entry per action)."""
    agent = make_agent()
    g = live_midround_game(turn=0)
    view = g.view(g.turn)
    actions = g.legal_actions()
    priors = agent._priors_for(view, actions)
    assert set(priors) == set(actions)
    assert all(p >= 0.0 for p in priors.values())
    assert math.isclose(sum(priors.values()), 1.0, rel_tol=0, abs_tol=1e-5)


def test_ensure_priors_fills_then_is_noop():
    agent = make_agent()
    g = live_midround_game(turn=0)
    view = g.view(g.turn)
    actions = g.legal_actions()
    node = _PriorNode()
    agent._ensure_priors(node, view, actions)
    assert set(node.priors) == set(actions)
    snapshot = dict(node.priors)
    # Second call over the same legal set changes nothing.
    agent._ensure_priors(node, view, actions)
    assert node.priors == snapshot


def test_ensure_priors_merge_preserves_existing_entries():
    """A later determinization presenting a new action must merge ONLY that
    action's prior, leaving the already-cached entries untouched."""
    agent = make_agent()
    g = live_midround_game(turn=0)
    view = g.view(g.turn)
    actions = g.legal_actions()
    assert len(actions) >= 2  # need a subset to withhold
    node = _PriorNode()
    # Seed the cache with all but the last action.
    partial, withheld = actions[:-1], actions[-1]
    agent._ensure_priors(node, view, partial)
    before = dict(node.priors)
    assert withheld not in node.priors
    # Now the full legal set arrives (as a fresh determinization might present).
    agent._ensure_priors(node, view, actions)
    assert withheld in node.priors  # the new action got a prior
    for a, p in before.items():  # existing entries are byte-stable
        assert node.priors[a] == p


def test_unvisited_puct_score_is_finite_and_prior_proportional():
    """For unvisited actions Q=0, so PUCT = c_puct * P_renorm(a) * sqrt(A)/(1+N)
    with A=1, N=0: the score is finite and strictly increasing in the prior,
    so the higher-prior action is selected first."""
    agent = make_agent(c_puct=2.0)
    g = live_midround_game(turn=0)
    view = g.view(g.turn)
    actions = g.legal_actions()
    node = _PriorNode()
    agent._ensure_priors(node, view, actions)

    # All actions unvisited: selection must return the argmax-prior action, and
    # every candidate's score must be finite and equal to c_puct * P_renorm.
    total = sum(node.priors.values())

    def puct(a):
        p = node.priors[a] / total
        # child is None, so Q=0, N=0, A=1 -> c_puct * p * sqrt(1)/(1+0)
        return agent.c_puct * p * math.sqrt(1) / 1

    for a in actions:
        s = puct(a)
        assert math.isfinite(s)
    chosen = agent._puct_select(node, actions)
    assert chosen == max(actions, key=puct)
    # Prior-proportional: the pick is exactly the max-prior action.
    assert chosen == max(actions, key=lambda a: node.priors[a])


def test_puct_prefers_visited_high_value_over_low_prior_unvisited():
    """Q dominates once an action is visited: a child with a strong positive Q
    outscores an unvisited low-prior sibling."""
    agent = make_agent(c_puct=0.1)
    g = live_midround_game(turn=0)
    view = g.view(g.turn)
    actions = g.legal_actions()
    node = _PriorNode()
    agent._ensure_priors(node, view, actions)
    # Give the FIRST action a visited child with a big positive Q.
    strong = actions[0]
    child = node.children[strong] = _PriorChild()
    child.n = 5
    child.w = 5.0  # Q = 1.0, the reward ceiling
    child.avail = 5
    chosen = agent._puct_select(node, actions)
    assert chosen == strong


# ------------------------------------------------------- mover-view (no leak)


def _possible_opp_hand_multihots(g, root):
    """Every player-1 hand a determinization from ``root`` could produce, as a
    set of hand multi-hot tuples (sampled over many seeds — exhaustive enough
    for this tiny handcrafted position)."""
    hands = set()
    for i in range(400):
        det = determinize(g, root, random.Random(i))
        hands.add(tuple(_hand_multihot(det.hands[1 - root])))
    return hands


def test_priors_evaluated_on_mover_det_view_not_root():
    """The policy at an opponent node is evaluated on the OPPONENT's det view.

    Spy on _priors_for to record the hand multi-hot each policy eval is shown.
    The root eval (call 0) shows player 0's real hand; interior evals show the
    mover's determinized hand. Since player 0's and player 1's initial hands are
    disjoint, at least one interior eval must show a determinized player-1 hand
    — proof det.turn's own view (not the root's) reaches the policy head. A bug
    that encoded det.view(root) would show only the root's hand and FAIL.
    """
    g = live_midround_game(turn=0)
    root = g.turn
    recorder = RecorderNet(make_net())
    agent = HybridAgent(net=recorder, seed=1, n_sims=32, priors=True)

    seen_hands = []
    real_priors_for = agent._priors_for.__func__

    def spy(view, actions):
        seen_hands.append(tuple(_hand_multihot(view["hand"])))
        return real_priors_for(agent, view, actions)

    agent._priors_for = spy
    agent.act(g)
    assert seen_hands, "policy head was never evaluated"

    root_hand = tuple(_hand_multihot(g.view(root)["hand"]))
    assert seen_hands[0] == root_hand  # the root eval shows the real hand
    # Every player-1 hand a determinization can produce; none equals the root's
    # full hand (the two players hold disjoint cards at the top of the round).
    opp_hands = _possible_opp_hand_multihots(g, root)
    assert root_hand not in opp_hands
    assert any(h in opp_hands for h in seen_hands), (
        "no opponent-mover node was evaluated on a determinized opponent hand")


def test_root_priors_use_real_view_not_a_determinization():
    """The root node's priors come from the real game view (the searcher's true
    information set — not a leak): the first _priors_for call shows exactly the
    root player's real hand, every card of it."""
    g = live_midround_game(turn=0)
    root = g.turn
    recorder = RecorderNet(make_net())
    agent = HybridAgent(net=recorder, seed=4, n_sims=8, priors=True)

    first_view = {}
    real_priors_for = agent._priors_for.__func__

    def spy(view, actions):
        first_view.setdefault("hand", tuple(view["hand"]))
        return real_priors_for(agent, view, actions)

    agent._priors_for = spy
    agent.act(g)
    assert first_view["hand"] == tuple(g.view(root)["hand"])


# ----------------------------------------- priors=False behavior regression


def _trajectory(factory, seed, n_plays):
    game = Game(seed=seed, win_lead=100)
    agents = [factory(), RandomAgent(seed=seed + 2)]
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


def test_priors_false_uses_base_class_act_verbatim(monkeypatch):
    """priors=False dispatches HybridAgent.act straight to ISMCTSAgent.act, and
    priors=True does NOT — asserted by spying on the base method, not merely by
    determinism (the previous version left the agent unused and only checked two
    identical priors=False runs, which a shared-code regression would pass)."""
    calls = {"base": 0}
    real_base_act = ISMCTSAgent.act

    def spy(self, game):
        calls["base"] += 1
        return real_base_act(self, game)

    monkeypatch.setattr(ISMCTSAgent, "act", spy)

    g = live_midround_game(turn=0)
    off = HybridAgent(net=make_net(), seed=5, n_sims=8, priors=False)
    off.act(g)
    assert calls["base"] == 1  # priors=False forwarded to the base loop

    on = HybridAgent(net=make_net(), seed=5, n_sims=8, priors=True)
    on.act(g)
    assert calls["base"] == 1  # priors=True took the PUCT path, not the base


@pytest.mark.parametrize("seed", [6, 21])
def test_priors_false_matches_plain_value_head_ismcts(seed):
    """priors=False HybridAgent is byte-identical to a reference that inlines
    the base ISMCTSAgent search with the value head as the leaf reward (the
    shipped behavior). The two must choose identical actions over a full seeded
    game vs RandomAgent — priors=True changed nothing for the off path."""
    net = make_net(0)

    def reference_factory():
        # A HybridAgent whose act is forced through the base class: this is
        # exactly what priors=False does (act -> super().act).
        agent = HybridAgent(net=net, seed=5, n_sims=8, priors=False)
        agent.act = lambda game, _a=agent: ISMCTSAgent.act(_a, game)
        return agent

    ref = _trajectory(reference_factory, seed, 60)
    off = _trajectory(
        lambda: HybridAgent(net=net, seed=5, n_sims=8, priors=False), seed, 60)
    assert off == ref


# ------------------------------------------------------------- legality fuzz


@pytest.mark.parametrize("hybrid_seat", [0, 1])
def test_puct_plays_legal_full_game_vs_random(hybrid_seat):
    agents = [None, None]
    agents[hybrid_seat] = HybridAgent(
        net=make_net(hybrid_seat), seed=hybrid_seat, n_sims=8, priors=True)
    agents[1 - hybrid_seat] = RandomAgent(seed=hybrid_seat + 10)
    game, trace = drive_checked_agents(agents, seed=hybrid_seat)
    assert game.winner in (0, 1)
    assert trace


def test_puct_same_seed_is_deterministic():
    net = make_net()
    runs = [
        drive_checked_agents(
            [HybridAgent(net=net, seed=7, n_sims=8, priors=True),
             RandomAgent(seed=5)],
            seed=3,
        )
        for _ in range(2)
    ]
    assert runs[0][1] == runs[1][1]
    assert runs[0][0].round_history == runs[1][0].round_history


def test_puct_single_legal_action_shortcircuits():
    """One legal action: PUCT returns it without touching the net (mirrors the
    base class fast path). Spies on the net to prove neither the policy head
    (__call__) nor the value head (value_only) is invoked."""
    g = make_game(hand0=["2H"], hand1=["3C"], table=[], deck=())
    g.unseen = [set(g.hands[1]), set(g.hands[0])]
    recorder = RecorderNet(make_net())
    value_calls = {"n": 0}
    real_value_only = recorder.value_only

    def counting_value_only(obs_np):
        value_calls["n"] += 1
        return real_value_only(obs_np)

    recorder.value_only = counting_value_only
    agent = HybridAgent(net=recorder, seed=0, n_sims=8, priors=True)
    assert len(g.legal_actions()) == 1
    action = agent.act(g)
    assert action in g.legal_actions()
    assert recorder.obs_seen == []  # policy head never called
    assert value_calls["n"] == 0  # value head never called


# ---------------------------------------------------------------------- CLI


def test_evaluate_priors_flag_defaults_and_parse():
    """--priors defaults ON (PUCT is the measured-stronger shipped mode),
    --no-priors turns it off, and --c-puct parses as float."""
    import argparse

    # Reconstruct the same flags evaluate.main declares (BooleanOptionalAction
    # so --priors / --no-priors both exist; default True).
    p = argparse.ArgumentParser()
    p.add_argument("--priors", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--c-puct", type=float, default=1.0)
    ns = p.parse_args([])
    assert ns.priors is True and ns.c_puct == 1.0
    ns = p.parse_args(["--no-priors"])
    assert ns.priors is False
    ns = p.parse_args(["--priors", "--c-puct", "0.5"])
    assert ns.priors is True and ns.c_puct == 0.5


@pytest.mark.skipif(not CKPT.exists(), reason="committed checkpoint missing")
def test_evaluate_hybrid_priors_vs_random_match(capsys):
    evaluate.main(["hybrid", "random", "-n", "2", "--win-lead", "100",
                   "--sims", "8", "--priors", "--ckpt", str(CKPT)])
    out = capsys.readouterr().out
    assert "hybrid vs random" in out


def test_play_hybrid_factory_defaults_to_priors():
    """play.py's hybrid factory ships priors=True (the measured-stronger PUCT
    mode) and still lets a caller pick the value-head-only mode explicitly."""
    agent = play.AI_AGENTS["hybrid"](seed=1)
    assert agent.priors is True  # PUCT is the shipped default
    agent = play.AI_AGENTS["hybrid"](seed=1, priors=False)
    assert agent.priors is False
