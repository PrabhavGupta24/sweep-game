"""Terminal UI for Sweep, built on rich.

Layered so it stays testable and fair:

- Pure render functions (``render_*``) take a Console plus plain data taken
  from ``game.view(player)`` / ``game.legal_actions()`` and print rich
  renderables. They never read hidden state and never read input.
- Interaction helpers (``prompt_*``, ``wait_enter``) wrap ``console.input``
  with validation and quit/help handling.
- ``public_snapshot`` / ``round_summary`` reconstruct the end-of-round score
  breakdown from public information only (the engine resets per-round state
  the instant a round ends, so the breakdown is captured on the way out).

The interactive game loop itself lives in ``play.py``.
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .cards import card_str, card_suit, card_value
from .engine import MAJORITY_BONUS, SWEEP_POINTS, TIE_BONUS, ActionKind

RED_SUITS = (1, 2)  # ♥ ♦

# ------------------------------------------------------------- card drawing


def by_value(cards):
    """Cards ordered ascending by game value, suit as a stable tiebreak."""
    return sorted(cards, key=lambda c: (card_value(c), card_suit(c)))


def render_card_row(cards=(), facedown: int = 0) -> Text:
    """Cards drawn side by side as single-line chips; ``facedown`` adds backs.

    One line per row (not 3-line boxes) so a full turn screen — header, table,
    hand, and action menu — fits within a typical terminal height.
    """
    if not cards and not facedown:
        return Text("(empty)", style="dim italic")
    row = Text()
    for card in cards:
        fg = "red" if card_suit(card) in RED_SUITS else "black"
        row.append(f" {card_str(card)} ", style=f"bold {fg} on grey85")
        row.append(" ")
    for _ in range(facedown):
        row.append(" ▒▒ ", style="bold blue on grey30")
        row.append(" ")
    return row


# ------------------------------------------------------------ status header


def render_header(console: Console, view: dict, round_num: int) -> None:
    """Persistent status panel: round, differential, per-player stats."""
    diff = view["differential"]
    if diff > 0:
        lead = f"you lead by {diff}"
    elif diff < 0:
        lead = f"opponent leads by {-diff}"
    else:
        lead = "tied"

    stats = Table(box=None, show_header=True, header_style="bold", pad_edge=False)
    stats.add_column("")
    for col in ("Points", "Sweeps", "Captured"):
        stats.add_column(col, justify="right")
    stats.add_row(
        Text("You", style="bold green"),
        str(view["points"][0]), str(view["sweeps"][0]), str(view["captured_counts"][0]),
    )
    stats.add_row(
        Text("Opponent", style="bold red"),
        str(view["points"][1]), str(view["sweeps"][1]), str(view["captured_counts"][1]),
    )

    info = Text(style="dim")
    info.append(f"Deck: {view['deck_count']} cards")
    info.append(f"   Opponent's hand: {view['opp_hand_count']} cards")
    if view["declared"] is not None:
        info.append(f"   Declared: {view['declared']}")
    if view["second_half"]:
        info.append("   (second half)")

    console.print(
        Panel(
            Group(stats, info),
            title=f"[bold]SWEEP — Round {round_num}[/bold]",
            subtitle=f"Differential: {diff:+d} ({lead})",
            border_style="bright_blue",
        )
    )


# --------------------------------------------------------------- table area


def render_table_area(console: Console, view: dict, hidden: bool = False,
                      title: str = "Table") -> None:
    """The shared table: loose cards vs. piles (or face-down backs)."""
    if hidden:
        body = Group(
            Text("Four cards are dealt face down — revealed after the declaration.",
                 style="dim italic"),
            render_card_row(facedown=4),
        )
    else:
        parts = [Text("Loose cards", style="bold"), render_card_row(by_value(view["table"]))]
        for value in sorted(view["piles"]):
            pile = view["piles"][value]
            if pile["mine"] and pile["opponents"]:
                owner, owner_style = "both", "magenta"
            elif pile["mine"]:
                owner, owner_style = "yours", "green"
            else:
                owner, owner_style = "opponent's", "red"
            head = Text()
            head.append(f"Pile of {value}", style="bold yellow")
            if pile["doubled"]:
                head.append(" (doubled)", style="bold magenta")
            head.append(" — ", style="dim")
            head.append(owner, style=owner_style)
            parts += [head, render_card_row(by_value(pile["cards"]))]
        body = Group(*parts)
    console.print(Panel(body, title=f"[bold]{title}[/bold]", border_style="green"))


def render_hand(console: Console, hand) -> None:
    console.print(Panel(render_card_row(by_value(hand)), title="[bold]Your hand[/bold]",
                        border_style="blue"))


# -------------------------------------------------------------- action menu


def render_action_menu(console: Console, actions) -> None:
    """Numbered legal actions plus the help/quit commands."""
    menu = Table(box=None, show_header=False, padding=(0, 1, 0, 0), pad_edge=False)
    menu.add_column(justify="right", style="bold cyan", no_wrap=True)
    menu.add_column()
    for i, action in enumerate(actions, 1):
        menu.add_row(f"{i}.", Text(str(action)))
    menu.add_row("h.", Text("Help — rules cheat-sheet", style="dim"))
    menu.add_row("q.", Text("Quit the game", style="dim"))
    console.print(Panel(menu, title="[bold]Your move[/bold]", border_style="cyan"))


def render_opening_note(console: Console, declared: int) -> None:
    console.print(Text(
        f"Opening move — it must be made at the declared value of {declared}.",
        style="bold yellow",
    ))


# ----------------------------------------------------------------- AI turns


def render_ai_declaration(console: Console, value: int) -> None:
    line = Text("Opponent declares ", style="bold")
    line.append(str(value), style="bold yellow")
    line.append(". The table is now revealed.", style="bold")
    console.print(Panel(line, title="[bold]Declaration[/bold]", border_style="red"))


def render_ai_move(console: Console, action) -> None:
    line = Text("Opponent played: ", style="bold")
    line.append(str(action), style="bold yellow")
    console.print(Panel(line, title="[bold]Opponent's turn[/bold]", border_style="red"))


# --------------------------------------------------- round / game summaries


def public_snapshot(game) -> dict:
    """Public state captured *before* a play, enough to rebuild the round-end
    breakdown if that play turns out to finish the round. Everything here is
    visible to both players (captures and the table are open information)."""
    return {
        "captured_counts": (len(game.captured[0]), len(game.captured[1])),
        "sweeps": tuple(game.sweeps),
        "pile_sizes": {v: len(p.cards) for v, p in game.piles.items()},
        "last_capturer": game.last_capturer,
    }


def round_summary(snapshot: dict, action, player: int, scores) -> dict:
    """Rebuild card points / majority bonus / sweeps for a finished round from
    the pre-final-play snapshot, the final action, and the engine's official
    round scores. Indices are absolute player indices (0 and 1)."""
    counts = list(snapshot["captured_counts"])
    last_capturer = snapshot["last_capturer"]
    if action.kind is ActionKind.PICKUP:
        gained = 1 + len(action.loose)
        if action.takes_pile:
            gained += snapshot["pile_sizes"][action.value]
        counts[player] += gained
        last_capturer = player
    leftovers = 52 - counts[0] - counts[1]  # the table goes to the last capturer
    if last_capturer is not None:
        counts[last_capturer] += leftovers
    if counts[0] > counts[1]:
        bonus = (MAJORITY_BONUS, 0)
    elif counts[1] > counts[0]:
        bonus = (0, MAJORITY_BONUS)
    else:
        bonus = (TIE_BONUS, TIE_BONUS)
    sweeps = snapshot["sweeps"]  # the final play never scores a sweep
    card_points = tuple(
        scores[i] - SWEEP_POINTS * sweeps[i] - bonus[i] for i in range(2)
    )
    return {
        "captured_counts": tuple(counts),
        "card_points": card_points,
        "bonus": bonus,
        "sweeps": sweeps,
        "scores": tuple(scores),
    }


def render_round_summary(console: Console, summary: dict, human: int,
                         round_num: int, differential: int) -> None:
    """End-of-round breakdown. ``differential`` is from the human's view."""
    me, opp = human, 1 - human
    tbl = Table(box=None, show_header=True, header_style="bold", pad_edge=False)
    tbl.add_column("")
    tbl.add_column("You", justify="right", style="green")
    tbl.add_column("Opponent", justify="right", style="red")

    def row(label, values, bold=False):
        style = "bold" if bold else None
        tbl.add_row(Text(label, style=style), *(Text(str(values[p]), style=style)
                                                for p in (me, opp)))

    row("Cards captured", summary["captured_counts"])
    row("Card points", summary["card_points"])
    row("Majority bonus", [f"+{b}" for b in summary["bonus"]])
    row("Sweeps", [f"{s} (+{SWEEP_POINTS * s})" for s in summary["sweeps"]])
    row("Round score", summary["scores"], bold=True)

    if differential > 0:
        lead = Text(f"New differential: +{differential} — you lead", style="bold green")
    elif differential < 0:
        lead = Text(f"New differential: {differential} — opponent leads", style="bold red")
    else:
        lead = Text("New differential: 0 — all tied up", style="bold")

    console.print(Panel(Group(tbl, Text(), lead),
                        title=f"[bold]Round {round_num} complete[/bold]",
                        border_style="yellow"))


def render_game_end(console: Console, human_won: bool, differential: int,
                    history, human: int) -> None:
    """Winner, final differential, and the per-round history table."""
    me, opp = human, 1 - human
    headline = (Text("You win the game!", style="bold green") if human_won
                else Text("Opponent wins the game.", style="bold red"))
    tbl = Table(show_header=True, header_style="bold", pad_edge=False)
    tbl.add_column("Round", justify="right")
    tbl.add_column("You", justify="right", style="green")
    tbl.add_column("Opponent", justify="right", style="red")
    for i, scores in enumerate(history, 1):
        tbl.add_row(str(i), str(scores[me]), str(scores[opp]))
    final = Text(f"Final differential: {differential:+d}", style="bold")
    console.print(Panel(Group(headline, Text(), tbl, Text(), final),
                        title="[bold]GAME OVER[/bold]", border_style="yellow"))


# ----------------------------------------------------------------- help text

CHEAT_SHEET = """\
[bold]Goal[/bold]  Capture cards for points; first to a cumulative lead of 200 wins.

[bold]Deal[/bold]  4 cards go face down to the table. The first player (4 cards in hand)
declares any value 9–13 they hold; the table is revealed and their opening move
must be made [italic]at that value[/italic] (capture/build there if possible, else throw the
declared card). Hands are then topped up to 12; when both hands empty, 12 more
each are dealt (no new table cards).

[bold]On your turn, play exactly one card:[/bold]
  [cyan]Pick up[/cyan]  Capture every pile of exactly your card's value plus all loose
           cards/groups summing to it. Maximal capture is mandatory. A pile is
           only ever taken by a card of exactly its value.
  [cyan]Build[/cyan]    Form a pile of value 9–13 from your card + table cards. You must
           hold another card of that value. Adding at the same value makes it
           [bold]doubled[/bold] (locked: can only be captured). You may raise an opponent's
           undoubled pile with a single hand card (pile + card = new value);
           the increment must come entirely from your hand.
  [cyan]Throw[/cyan]    Discard face up. Forbidden if the card can pick something up, or
           if it feeds a pile you created, or it's your last card matching
           your own pile (that card is reserved to capture it).

[bold]Scoring (100 pts/round + sweeps)[/bold]
  Spades = face value (91 total) · A♥ A♦ A♣ = 1 each · 10♦ = 2
  Card majority (>26 captured): +4 (26–26 tie: +2 each)
  [bold]Sweep[/bold] — clearing the whole table with a capture: +50 (never on the
  round's last play). Leftover table cards go to whoever captured last.\
"""


def render_help(console: Console) -> None:
    console.print(Panel(CHEAT_SHEET, title="[bold]Sweep — rules cheat-sheet[/bold]",
                        border_style="magenta"))


# ------------------------------------------------------- interaction helpers


def clear_screen(console: Console) -> None:
    """Clear the screen *and scrollback* and home the cursor.

    A plain screen-clear homes the cursor but leaves the shell prompt in
    scrollback, so a frame that's as tall as the window scrolls up by a line
    and the header's top border slides out of view. Clearing scrollback as
    well pins every frame to the terminal's top row.
    """
    if console.is_terminal:
        console.file.write("\033[H\033[2J\033[3J")
        console.file.flush()
    else:  # not a tty (e.g. tests/pipes): nothing to pin
        console.clear()


def wait_enter(console: Console, message: str = "Press ENTER to continue") -> bool:
    """Pause; returns False if input is exhausted (treated as quitting)."""
    try:
        console.input(f"[dim]{message}… [/dim]")
        return True
    except EOFError:
        return False


def prompt_choice(console: Console, n: int):
    """Pick an action: returns an index 0..n-1, or 'quit' / 'help'."""
    while True:
        try:
            raw = console.input(
                f"[bold]Choose an action [1-{n}] ('h' help, 'q' quit): [/bold]"
            ).strip().lower()
        except EOFError:
            return "quit"
        if raw == "q":
            return "quit"
        if raw == "h":
            return "help"
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        console.print(f"[red]Invalid choice — enter a number from 1 to {n}.[/red]")


def prompt_declare(console: Console, options):
    """Pick the opening declaration: returns a value, or 'quit' / 'help'."""
    opts = "/".join(str(v) for v in options)
    while True:
        try:
            raw = console.input(
                f"[bold]Declare a value ({opts}) ('h' help, 'q' quit): [/bold]"
            ).strip().lower()
        except EOFError:
            return "quit"
        if raw == "q":
            return "quit"
        if raw == "h":
            return "help"
        if raw.isdigit() and int(raw) in options:
            return int(raw)
        console.print(f"[red]Invalid declaration — choose one of {opts}.[/red]")
