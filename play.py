#!/usr/bin/env python3
"""Play Sweep in the terminal against an AI.

    python3 play.py [--ai {random,greedy,heuristic,ismcts}] [--seed N] [--win-lead N] [--ai-seed N]

The human is always player 0; who plays first each round is decided by the
rules (random in round 1, then the cumulative leader). FAIRNESS: everything
rendered for the human comes from game.view(HUMAN) and game.legal_actions()
plus public scoreboard state — never the opponent's hand or the deck.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from sweep import ui
from sweep.agents import GreedyAgent, HeuristicAgent, RandomAgent
from sweep.engine import Game
from sweep.ismcts import ISMCTSAgent

HUMAN = 0

# Selectable opponents. Each factory takes a `seed` keyword.
AI_AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": HeuristicAgent,
    "ismcts": ISMCTSAgent,
}


def declaration_phase(console: Console, game: Game, ai) -> bool:
    """Handle the start-of-round declaration. Returns False if the human quits."""
    while True:
        view = game.view(HUMAN)
        ui.clear_screen(console)
        ui.render_header(console, view, game.round_num)
        first = "you play" if game.first_player == HUMAN else "the opponent plays"
        console.print(f"[bold]Round {game.round_num} begins — {first} first.[/bold]")
        ui.render_table_area(console, view, hidden=True)
        ui.render_hand(console, view["hand"])

        if game.first_player != HUMAN:
            value = ai.declare(game)
            game.declare(value)
            ui.render_ai_declaration(console, value)
            return ui.wait_enter(console)

        choice = ui.prompt_declare(console, game.declare_options())
        if choice == "quit":
            return False
        if choice == "help":
            ui.clear_screen(console)
            ui.render_help(console)
            if not ui.wait_enter(console):
                return False
            continue
        game.declare(choice)
        return True


def human_turn(console: Console, game: Game, opening: bool):
    """Render the human's screen and apply their chosen action.
    Returns the Action played, or None if the human quits."""
    while True:
        actions = game.legal_actions()
        view = game.view(HUMAN)
        ui.clear_screen(console)
        ui.render_header(console, view, game.round_num)
        ui.render_table_area(console, view)
        ui.render_hand(console, view["hand"])
        if opening:
            ui.render_opening_note(console, view["declared"])
        ui.render_action_menu(console, actions)

        choice = ui.prompt_choice(console, len(actions))
        if choice == "quit":
            return None
        if choice == "help":
            ui.clear_screen(console)
            ui.render_help(console)
            if not ui.wait_enter(console):
                return None
            continue
        action = actions[choice]
        game.step(action)
        return action


def ai_turn(console: Console, game: Game, ai):
    """Play the AI's move, show it, and wait for ENTER.
    Returns the Action played, or None if input runs out."""
    view = game.view(HUMAN)
    ui.clear_screen(console)
    ui.render_header(console, view, game.round_num)
    ui.render_table_area(console, view)

    finished_rounds = len(game.round_history)
    action = ai.act(game)
    game.step(action)
    ui.render_ai_move(console, action)
    if len(game.round_history) == finished_rounds:
        # Only meaningful mid-round; after a round ends the summary screen
        # shows the outcome instead.
        ui.render_table_area(console, game.view(HUMAN),
                             title="Table after opponent's move")
    if not ui.wait_enter(console):
        return None
    return action


def run(console: Console, game: Game, ai) -> bool:
    """Main loop. Returns True if the game finished, False if the human quit."""
    opening = False
    while not game.game_over:
        if game.awaiting == "declare":
            if not declaration_phase(console, game, ai):
                return False
            opening = True  # the next play is the opening move
            continue

        finished_rounds = len(game.round_history)
        snapshot = ui.public_snapshot(game)
        player = game.turn
        if player == HUMAN:
            action = human_turn(console, game, opening)
        else:
            action = ai_turn(console, game, ai)
        if action is None:
            return False
        opening = False

        if len(game.round_history) > finished_rounds:  # the round just ended
            summary = ui.round_summary(snapshot, action, player,
                                       game.round_history[-1])
            ui.clear_screen(console)
            diff = game.differential if HUMAN == 0 else -game.differential
            ui.render_round_summary(console, summary, HUMAN,
                                    len(game.round_history), diff)
            if not game.game_over and not ui.wait_enter(console):
                return False

    diff = game.differential if HUMAN == 0 else -game.differential
    ui.render_game_end(console, game.winner == HUMAN, diff,
                       game.round_history, HUMAN)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Play Sweep in the terminal against an AI.")
    parser.add_argument("--ai", choices=sorted(AI_AGENTS), default="random",
                        help="opponent type (default: random)")
    parser.add_argument("--seed", type=int, default=None,
                        help="game seed (deck shuffles, first player)")
    parser.add_argument("--win-lead", type=int, default=200,
                        help="cumulative lead needed to win (default 200)")
    parser.add_argument("--ai-seed", type=int, default=None,
                        help="seed for the AI's choices")
    args = parser.parse_args(argv)

    console = Console()
    game = Game(seed=args.seed, win_lead=args.win_lead)
    ai = AI_AGENTS[args.ai](seed=args.ai_seed)
    try:
        finished = run(console, game, ai)
    except KeyboardInterrupt:
        finished = False
    if not finished:
        console.print("\n[dim]Game abandoned — thanks for playing.[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
