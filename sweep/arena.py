"""Evaluation harness: seat-swapped head-to-head matches and round-robins.

Agent factories are callables taking a seed and returning a fresh Agent, e.g.
``lambda s: GreedyAgent(seed=s)``: every game gets newly seeded agents so runs
are reproducible from ``base_seed`` alone.
"""

import math
import time

from .engine import Game


def wilson_ci(wins, games, z=1.96):
    """Wilson 95% confidence interval for a binomial proportion."""
    if games == 0:
        return (0.0, 1.0)
    phat = wins / games
    denom = 1 + z * z / games
    center = (phat + z * z / (2 * games)) / denom
    half = z * math.sqrt(phat * (1 - phat) / games + z * z / (4 * games * games)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _play_counted(seats, seed, win_lead):
    """Play one game between seats[0] (player 0) and seats[1]; count plays."""
    game = Game(seed=seed, win_lead=win_lead)
    plays = 0
    while not game.game_over:
        if game.awaiting == "declare":
            game.declare(seats[game.first_player].declare(game))
        else:
            game.step(seats[game.turn].act(game))
            plays += 1
    return game, plays


def play_match(make_agent_a, make_agent_b, n_games, base_seed=0, win_lead=200):
    """Play n_games between A and B and return aggregate stats for A.

    Seat-swap design: consecutive games reuse the same game seed (same deals)
    with seats swapped, cancelling any seat advantage. (Here round 1's first
    player is already random from the game seed and the leader plays first in
    later rounds, so seats are nearly symmetric — we swap anyway.)
    """
    start = time.perf_counter()
    wins_a = wins_b = 0
    diff_sum = 0.0
    rounds = 0
    total_plays = 0
    for i in range(n_games):
        game_seed = base_seed + i // 2  # each seed is played twice...
        a_player = i % 2                # ...with seats swapped
        agent_a = make_agent_a(base_seed + 2 * i)
        agent_b = make_agent_b(base_seed + 2 * i + 1)
        seats = [agent_a, agent_b] if a_player == 0 else [agent_b, agent_a]
        game, plays = _play_counted(seats, game_seed, win_lead)
        if game.winner == a_player:
            wins_a += 1
        else:
            wins_b += 1
        diff_sum += game.differential if a_player == 0 else -game.differential
        rounds += len(game.round_history)
        total_plays += plays
    lo, hi = wilson_ci(wins_a, n_games)
    return {
        "games": n_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "win_rate_a": wins_a / n_games if n_games else 0.0,
        "ci95": (lo, hi),
        "mean_diff_a": diff_sum / n_games if n_games else 0.0,
        "mean_rounds": rounds / n_games if n_games else 0.0,
        "total_plays": total_plays,
        "seconds": time.perf_counter() - start,
    }


def round_robin(factories, n_games, base_seed=0, win_lead=200):
    """Play every pair of named factories; returns {(name_a, name_b): result}."""
    names = list(factories)
    results = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            results[(a, b)] = play_match(
                factories[a], factories[b], n_games, base_seed=base_seed, win_lead=win_lead
            )
    return results
