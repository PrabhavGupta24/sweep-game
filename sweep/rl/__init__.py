"""RL environment layer: pure encoders plus a decision-stream env.

numpy only — no torch anywhere in this package. Learning code sits on top.
"""

from .encoders import ACTION_SIZE, DECLARE, OBS_SIZE, encode_action, encode_observation
from .env import SweepEnv

__all__ = [
    "ACTION_SIZE",
    "DECLARE",
    "OBS_SIZE",
    "SweepEnv",
    "encode_action",
    "encode_observation",
]
