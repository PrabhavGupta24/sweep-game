"""RL environment layer: pure encoders plus a decision-stream env.

``import sweep.rl`` stays torch-free (numpy only): the eager exports are the
encoders and SweepEnv. NeuralAgent needs torch, so it is re-exported lazily —
the import happens on first attribute access, not at package import.
Learning code imports sweep.rl.model / sweep.rl.ppo directly.
"""

from .encoders import ACTION_SIZE, DECLARE, OBS_SIZE, encode_action, encode_observation
from .env import SweepEnv

__all__ = [
    "ACTION_SIZE",
    "DECLARE",
    "NeuralAgent",
    "OBS_SIZE",
    "SweepEnv",
    "encode_action",
    "encode_observation",
]


def __getattr__(name):
    if name == "NeuralAgent":
        from .agent import NeuralAgent  # deferred: pulls in torch

        return NeuralAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
