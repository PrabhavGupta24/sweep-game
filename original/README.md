# Original implementation

The first version of the project, kept for reference. It was the source the
rules spec (../RULES.md) was reverse-engineered from, and is superseded by the
`sweep/` package:

- `sweep_game.py`, `models.py`, `actions.py` — terminal game with rules logic,
  I/O, and AI hooks intertwined (superseded by `sweep/engine.py` + `play.py`).
- `sweep_ai.py` — early DQN training attempt (to be superseded by the RL stage).
- `ignore/` — untracked scratch drafts (PPO, gym-style env, encoders).

Known rule discrepancies vs RULES.md are listed at the bottom of that file.
This code is not imported by anything and is safe to delete once it has no
remaining reference value.
