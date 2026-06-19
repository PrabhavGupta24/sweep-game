# sweep-game

A 2-player card game of Sweep: a pure rules engine (`sweep/`), pluggable agents,
and a terminal UI. The authoritative rules live in [RULES.md](RULES.md).

## Playing the game

Requires Python 3.9+.

```sh
python3 -m pip install -r requirements.txt
python3 play.py
```

You play against a random AI. Options:

```sh
python3 play.py --seed 42        # reproducible deal / first player
python3 play.py --ai-seed 7      # reproducible AI choices
python3 play.py --win-lead 100   # shorter game (default: 200-point lead)
```

In game: pick a numbered action (or type the value at the opening declaration),
press `h` for a one-screen rules cheat-sheet, and `q` to quit. Press ENTER to
acknowledge the opponent's moves.

## Development

```sh
python3 -m pytest tests/ -q
```

- `sweep/engine.py` — headless rules engine (`Game`, `Action`, `Pile`).
- `sweep/agents.py` — the `Agent` protocol and `RandomAgent` baseline.
- `sweep/ui.py` — rich-based rendering and interaction helpers.
- `play.py` — interactive entry point (human vs. AI).
