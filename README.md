# sweep-game

A 2-player card game of Sweep: a pure rules engine (`sweep/`), pluggable agents,
an evaluation harness, and a terminal UI. The authoritative rules live in
[RULES.md](RULES.md).

## Playing the game

Requires Python 3.9+.

```sh
python3 -m pip install -r requirements.txt
python3 play.py
```

You play against an AI opponent. Options:

```sh
python3 play.py --ai heuristic   # opponent: random (default), greedy, or heuristic
python3 play.py --seed 42        # reproducible deal / first player
python3 play.py --ai-seed 7      # reproducible AI choices
python3 play.py --win-lead 100   # shorter game (default: 200-point lead)
```

The opponent types match the agents described under
[Agents & evaluation](#agents--evaluation): `random` plays uniformly, `greedy`
maximizes immediate points, and `heuristic` is the strongest hand-crafted player.

In game: pick a numbered action (or type the value at the opening declaration),
press `h` for a one-screen rules cheat-sheet, and `q` to quit. Press ENTER to
acknowledge the opponent's moves.

## Agents & evaluation

Baseline agents live in `sweep/agents.py`:

- `random` — uniform choice over legal actions.
- `greedy` — one-ply point maximizer: prefers sweep captures, then the highest
  immediate card-point haul, then the most cards; otherwise throws its
  lowest-point card. Never builds.
- `heuristic` — everything greedy does, plus: scores every action by the
  expected points the opponent's next card can take off the resulting table
  (estimated from the unseen cards), builds and raises piles when the opponent
  is unlikely to hold the capture card, denies opponent piles via raises,
  avoids throwing point cards, and avoids leaving the table sweepable.

### ISMCTS search agent

`ismcts` (`sweep/ismcts.py`) is a single-observer Information Set Monte Carlo
Tree Search player: each decision it determinizes the unseen cards (resampling
the opponent's hand and the deck from its own information set, respecting the
pile-reserve rule invariant) and runs UCT tree search over the round, backing
up the round's score swing. In the second half the deck is empty, so the
determinization is the opponent's exact hand and the search plays
perfect-information endgames.

Run a round-robin with the evaluation harness (`sweep/arena.py`):

```
python3 evaluate.py random greedy heuristic -n 200 --win-lead 200 --seed 0
```

`-n` is games per pairing; each game seed is played twice with seats swapped to
cancel any seat advantage. Win rates come with Wilson 95% confidence intervals.

Measured results (200 games per pairing, win lead 200, seed 0):

```
pairing (A vs B)         games A wins B wins  win% A           95% CI diff/g (A) rnds/g    sec
----------------------------------------------------------------------------------------------
random vs greedy           200      0    200    0.0% [  0.0%,  1.9%]     -307.4   2.64    5.0
random vs heuristic        200      0    200    0.0% [  0.0%,  1.9%]     -297.3   2.32   11.2
greedy vs heuristic        200      0    200    0.0% [  0.0%,  1.9%]     -328.0   2.01    9.0
```

So greedy beats random 200/200 and heuristic beats greedy 200/200; the same
ordering holds at `--seed 1000` (heuristic beats greedy 99/100 and random
100/100, greedy beats random 100/100). Mirror matches of each agent against
itself land at ~50%, confirming the harness has no seat bias. The dominance is driven by
sweeps: greedy and random regularly throw onto an empty or near-empty table,
and the heuristic converts those into +50 sweeps while rarely offering one
back.

## Development

```sh
python3 -m pytest tests/ -q
```

- `sweep/engine.py` — headless rules engine (`Game`, `Action`, `Pile`).
- `sweep/agents.py` — the `Agent` protocol and the baseline agents.
- `sweep/arena.py` — match/round-robin evaluation harness.
- `sweep/ui.py` — rich-based rendering and interaction helpers.
- `play.py` — interactive entry point (human vs. AI).
- `evaluate.py` — agent round-robin CLI.
