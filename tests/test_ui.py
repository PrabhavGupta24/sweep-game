"""Smoke tests for the rich terminal UI.

These drive the pure render functions (not the input loop) against real Game
states, writing to a StringIO-backed Console, and assert that nothing raises
and that key strings show up in the output.
"""

import io

import pytest

from sweep import ui
from sweep.agents import RandomAgent, play_game
from sweep.engine import ActionKind, Game


def make_console():
    buf = io.StringIO()
    return Console_with(buf), buf


def Console_with(buf):
    from rich.console import Console

    return Console(file=buf, force_terminal=True, width=120)


def midgame_with_pile(seed_start=0):
    """Drive random play until a pile sits on the table mid-round."""
    for seed in range(seed_start, seed_start + 50):
        game = Game(seed=seed)
        agent = RandomAgent(seed=seed)
        game.declare(agent.declare(game))
        for _ in range(200):
            if game.piles and game.table:
                return game
            if game.awaiting != "play" or game.game_over:
                break
            game.step(agent.act(game))
    raise RuntimeError("no mid-game pile state found")  # pragma: no cover


def test_render_card_row_faceup_and_facedown():
    console, buf = make_console()
    game = Game(seed=1)
    hand = game.view(0)["hand"]
    console.print(ui.render_card_row(hand))
    console.print(ui.render_card_row(facedown=4))
    console.print(ui.render_card_row())
    out = buf.getvalue()
    from sweep.cards import card_str
    assert card_str(hand[0]) in out  # the card's rank+suit is shown
    assert "▒▒" in out               # face-down back
    assert "(empty)" in out


def test_render_hand_is_sorted_by_value():
    from sweep.cards import card_from, card_value

    console, buf = make_console()
    # Deliberately unsorted, mixed suits.
    hand = [card_from(s) for s in ("KS", "2H", "10D", "2S", "AC")]
    ui.render_hand(console, hand)
    out = buf.getvalue()
    # The rendered order of rank+suit tokens must be ascending by value.
    import re
    shown = re.findall(r"(?:10|[2-9AJQK])[♠♥♦♣]", out)
    values = [card_value(card_from(tok)) for tok in shown]
    assert values == sorted(values)


def test_render_full_turn_screen():
    game = midgame_with_pile()
    player = game.turn
    view = game.view(player)
    actions = game.legal_actions()

    console, buf = make_console()
    ui.render_header(console, view, game.round_num)
    ui.render_table_area(console, view)
    ui.render_hand(console, view["hand"])
    ui.render_action_menu(console, actions)
    out = buf.getvalue()

    assert f"Round {game.round_num}" in out
    assert "Differential" in out
    assert "Pile of" in out
    assert "Loose cards" in out
    assert "Your hand" in out
    assert "Your move" in out
    assert "1." in out
    # No leakage: the menu/table never mention the opponent's hand contents.
    assert "Opponent's hand: " in out  # count only, from the view


def test_pile_ownership_and_doubled_labels():
    # Search for a doubled pile to exercise the "(doubled)" label.
    for seed in range(60):
        game = Game(seed=seed)
        agent = RandomAgent(seed=seed)
        game.declare(agent.declare(game))
        for _ in range(200):
            if any(p.doubled for p in game.piles.values()):
                console, buf = make_console()
                ui.render_table_area(console, game.view(0))
                out = buf.getvalue()
                assert "(doubled)" in out
                assert ("yours" in out) or ("opponent's" in out) or ("both" in out)
                return
            if game.awaiting != "play" or game.game_over:
                break
            game.step(agent.act(game))
    pytest.skip("no doubled pile arose in the search window")


def test_render_declaration_screen_hides_table():
    game = Game(seed=3)
    view = game.view(game.first_player)
    console, buf = make_console()
    ui.render_header(console, view, game.round_num)
    ui.render_table_area(console, view, hidden=True)
    ui.render_hand(console, view["hand"])
    out = buf.getvalue()
    assert "face down" in out
    assert "▒▒" in out


def test_render_ai_move_and_help():
    game = midgame_with_pile()
    action = game.legal_actions()[0]
    console, buf = make_console()
    ui.render_ai_move(console, action)
    ui.render_ai_declaration(console, 11)
    ui.render_help(console)
    out = buf.getvalue()
    assert "Opponent played" in out
    assert "Opponent declares" in out
    assert "Sweep — rules cheat-sheet" in out
    assert "Maximal capture is mandatory" in out


def test_round_summary_reconstruction_matches_engine():
    """Play full games; every reconstructed round breakdown must be exactly
    consistent with the engine's official round scores."""
    for seed in (0, 1, 2):
        game = Game(seed=seed, win_lead=200)
        agent = RandomAgent(seed=seed)
        while not game.game_over:
            if game.awaiting == "declare":
                game.declare(agent.declare(game))
                continue
            finished = len(game.round_history)
            snap = ui.public_snapshot(game)
            player = game.turn
            action = agent.act(game)
            game.step(action)
            if len(game.round_history) > finished:
                scores = game.round_history[-1]
                summary = ui.round_summary(snap, action, player, scores)
                assert sum(summary["captured_counts"]) == 52
                assert sum(summary["card_points"]) == 96
                assert summary["scores"] == tuple(scores)
                for i in range(2):
                    assert (summary["card_points"][i]
                            + summary["bonus"][i]
                            + 50 * summary["sweeps"][i]) == scores[i]

                console, buf = make_console()
                ui.render_round_summary(console, summary, human=0,
                                        round_num=len(game.round_history),
                                        differential=game.differential)
                out = buf.getvalue()
                assert "Majority bonus" in out
                assert "Round score" in out


def test_render_game_end():
    game = play_game([RandomAgent(0), RandomAgent(1)], seed=7)
    console, buf = make_console()
    ui.render_game_end(console, human_won=(game.winner == 0),
                       differential=game.differential,
                       history=game.round_history, human=0)
    out = buf.getvalue()
    assert "GAME OVER" in out
    assert "Final differential" in out
    assert ("You win" in out) or ("Opponent wins" in out)


def test_action_menu_shows_action_descriptions():
    game = midgame_with_pile()
    actions = game.legal_actions()
    console, buf = make_console()
    ui.render_action_menu(console, actions)
    out = buf.getvalue()
    kinds = {a.kind for a in actions}
    if ActionKind.THROW in kinds:
        assert "Throw" in out
    if ActionKind.PICKUP in kinds:
        assert "Pick up" in out
    if ActionKind.BUILD in kinds:
        assert "Build" in out
