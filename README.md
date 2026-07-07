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

Note: `ismcts` at its default budget (200 simulations) takes about a second per
decision — roughly 1000x the baselines, ~50s per game. Pass `--sims 50` (or
lower) to `evaluate.py` for quicker, weaker evaluations.

Measured ISMCTS results (200 simulations, c=0.7, win lead 200, seat-swapped):

```
pairing (A vs B)         games A wins B wins  win% A           95% CI diff/g (A) rnds/g
----------------------------------------------------------------------------------------
ismcts vs random            20     20      0  100.0% [ 83.9%,100.0%]     +285.7   2.20
ismcts vs greedy            30     28      2   93.3% [ 78.7%, 98.2%]     +255.5   2.03
ismcts vs heuristic         40      2     38    5.0% [  1.4%, 16.5%]     -236.8   8.18
```

Budget scaling vs heuristic (same 10 deals per row, c=0.7):

```
n_sims                   games A wins B wins  win% A diff/g (A)
---------------------------------------------------------------
50                          10      0     10    0.0%     -258.8
200                         10      1      9   10.0%     -213.6
600                         10      2      8   20.0%     -160.8
```

Search crushes the blind baselines — it never lost to random and dropped only
two of thirty games to greedy — but it does not beat the hand-crafted
heuristic: 2/40 at 200 simulations, and even at 600 simulations only 2/10.
Strength does scale with budget (0/10 → 1/10 → 2/10 on identical deals, mean
deficit shrinking -258.8 → -213.6 → -160.8), and the losses are far closer
than the baselines': matches run ~8 rounds versus ~2 when the heuristic plays
random or greedy, i.e. ismcts concedes roughly 29 points a round where the
baselines concede 130-160 and are blown out immediately. The
second-half perfect-information effect (once the deck is empty every
determinization is the opponent's exact hand, so the search plays exact
endgames) is the most plausible reason the losses stay close, but it is not
enough to overcome the heuristic's stronger first-half play, where the
determinized rollouts are too noisy to reliably punish builds or defend
sweeps. A small exploration-constant sweep on paired deals (c in
{0.35, 0.7, 1.0} at 200 sims vs heuristic, 10 games each: 3/10 with diff
-110.8, vs 1/10 at -213.6, vs 0/10 at -240.6) favored c=0.35, which is now the
default; the tables above were measured at the then-default c=0.7.

Baseline measured results (200 games per pairing, win lead 200, seed 0):

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

### Neural agent (PPO self-play)

`neural` (`sweep/rl/`) is a learned player: a per-action scoring network
(observation trunk + action encoder + value head) trained with PPO over league
self-play — mirror games plus frozen past checkpoints and the scripted
random/greedy/heuristic opponents. One episode is one round; the only reward is
the round score differential at round end. Train your own:

```
python3 -m pip install -r requirements-train.txt
python3 train.py --rounds 250000 --out runs/main --seed 42
python3 evaluate.py neural heuristic --ckpt runs/main/ckpt_latest.pt -n 100
python3 play.py --ai neural --ckpt models/neural-250k.pt
```

A trained checkpoint is committed at `models/neural-250k.pt` (250k rounds,
~5 hours on an Intel MacBook at ~13 rounds/sec). Gauntlet results for that
checkpoint (temperature 0, seat-swapped, win lead 200):

```
pairing (A vs B)         games A wins B wins  win% A           95% CI diff/g (A) rnds/g
----------------------------------------------------------------------------------------
neural vs random            50     50      0  100.0% [ 92.9%,100.0%]     +302.0   2.00
neural vs greedy            50     50      0  100.0% [ 92.9%,100.0%]     +315.7   2.18
neural vs ismcts-50         10      8      2   80.0% [ 49.0%, 94.3%]     +158.4   6.80
neural vs ismcts-200        20     12      8   60.0% [ 38.7%, 78.1%]      +54.6   6.60
neural vs ismcts-600         6      3      3   50.0% [ 18.8%, 81.2%]      -30.7   5.83
neural vs heuristic        100     43     57   43.0% [ 33.7%, 52.8%]      -40.1   8.58
```

The heuristic remains the strongest agent, but its margin over the best
challenger collapsed from -237/game (ismcts-200) to -40/game: the neural agent
is the first to take games off it at all (43/100). Training was still
improving when the 250k budget ran out (mean reward vs heuristic went from
-0.56 to -0.23 per round across the run's quarters), so more compute should
close the remaining gap. Note the non-transitivity: search troubles the net
more than it troubles the heuristic (neural only edges ismcts-200 and ties
ismcts-600 on a small sample, while the heuristic crushes both) — the net and
the search agent have complementary strengths, which is the motivation for a
future net-guided-search hybrid.

## Development

```sh
python3 -m pytest tests/ -q
```

- `sweep/engine.py` — headless rules engine (`Game`, `Action`, `Pile`).
- `sweep/agents.py` — the `Agent` protocol and the baseline agents.
- `sweep/ismcts.py` — the ISMCTS search agent.
- `sweep/rl/` — RL stack: encoders, env, model, PPO, `NeuralAgent`.
- `sweep/arena.py` — match/round-robin evaluation harness.
- `sweep/ui.py` — rich-based rendering and interaction helpers.
- `play.py` — interactive entry point (human vs. AI).
- `evaluate.py` — agent round-robin CLI.
- `train.py` — PPO self-play training CLI.
