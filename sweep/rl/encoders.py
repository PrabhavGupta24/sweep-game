"""Fixed-size numpy encodings of Sweep observations and candidate moves.

Pure functions of *public* data only: they take the plain dict returned by
``game.view(player)`` (plus explicit decision context), never a Game object,
so hidden-information leakage is impossible by construction.

Observation layout (OBS_SIZE = 253 float32 values):

    [0:52)      hand multi-hot
    [52:104)    loose table cards multi-hot
    [104:156)   pile cards multi-hot (all piles pooled)
    [156:208)   unseen cards multi-hot — view['unseen']: opponent's hand + the
                deck as a set, plus, during the declare phase only, the four
                facedown table cards (not yet revealed to either player)
    [208:238)   per pile value 9..13, a block of 6 floats:
                exists, mine, opponents, doubled,
                card count / 13 (clipped to 1), card points / 20 (clipped to 1)
    [238]       my points / 100
    [239]       opponent points / 100
    [240]       my sweeps / 2 (clipped to 1)
    [241]       opponent sweeps / 2 (clipped to 1)
    [242]       my captured count / 52
    [243]       opponent captured count / 52
    [244]       opponent hand count / 13
    [245]       deck count / 40
    [246]       second-half flag
    [247]       declared value / 13 (0 when undeclared)
    [248]       is-first-player flag
    [249]       last capturer is me
    [250]       nobody has captured yet
    [251]       cumulative differential (my perspective), clipped to
                [-200, 200] then / 200 — the only feature that can be negative
    [252]       awaiting-declaration flag

Action layout (ACTION_SIZE = 115 float32 values), where the encoded item is
an engine Action or a ``(DECLARE, value)`` tuple:

    [0:5)       kind one-hot: pickup, build, raise, throw, declare
    [5:57)      played card one-hot (all zero for a declaration)
    [57:109)    loose table cards involved, multi-hot
    [109]       value / 13 (the declared value for declarations)
    [110]       raised-from value / 13 (RAISE only, else 0)
    [111]       takes-pile flag
    [112]       sweep flag
    [113]       immediate captured card points / 20, clipped to 1 (PICKUP
                only: played card + loose cards + any captured pile's cards,
                the latter read from view['piles'][value]['cards'])
    [114]       immediate captured card count / 16, clipped to 1 (PICKUP only)

Every feature lies in [-1, 1]; all but the differential lie in [0, 1].
"""

from __future__ import annotations

import numpy as np

from ..cards import card_points
from ..engine import PILE_MIN, ActionKind

DECLARE = "declare"  # kind tag for declaration candidates: (DECLARE, value)

OBS_SIZE = 253
ACTION_SIZE = 115

# Observation segment offsets.
_OBS_HAND = 0
_OBS_TABLE = 52
_OBS_PILE_CARDS = 104
_OBS_UNSEEN = 156
_OBS_PILE_BLOCKS = 208  # 6 floats per pile value 9..13
_OBS_SCALARS = 238  # 15 scalars

# Action segment offsets.
_ACT_KIND = 0  # 5-way one-hot; engine kinds use their enum value as the index
_ACT_DECLARE_KIND = 4
_ACT_CARD = 5
_ACT_LOOSE = 57
_ACT_SCALARS = 109  # value, raised_from, takes_pile, sweeps, points, count


def encode_observation(view, awaiting, np_out=None):
    """Encode ``game.view(player)`` into a float32 vector of length OBS_SIZE.

    ``awaiting`` is the engine's decision phase ("declare" or "play").
    ``np_out``, when given, must be a float32 array of length OBS_SIZE; it is
    zeroed, filled, and returned (avoids an allocation on the hot path).
    """
    obs = np_out if np_out is not None else np.zeros(OBS_SIZE, dtype=np.float32)
    if np_out is not None:
        obs.fill(0.0)

    for c in view["hand"]:
        obs[_OBS_HAND + c] = 1.0
    for c in view["table"]:
        obs[_OBS_TABLE + c] = 1.0
    for v, pile in view["piles"].items():
        cards = pile["cards"]
        for c in cards:
            obs[_OBS_PILE_CARDS + c] = 1.0
        base = _OBS_PILE_BLOCKS + 6 * (v - PILE_MIN)
        obs[base] = 1.0
        obs[base + 1] = pile["mine"]
        obs[base + 2] = pile["opponents"]
        obs[base + 3] = pile["doubled"]
        obs[base + 4] = min(len(cards) / 13.0, 1.0)
        obs[base + 5] = min(sum(card_points(c) for c in cards) / 20.0, 1.0)
    for c in view["unseen"]:
        obs[_OBS_UNSEEN + c] = 1.0

    s = _OBS_SCALARS
    obs[s] = view["points"][0] / 100.0
    obs[s + 1] = view["points"][1] / 100.0
    obs[s + 2] = min(view["sweeps"][0] / 2.0, 1.0)
    obs[s + 3] = min(view["sweeps"][1] / 2.0, 1.0)
    obs[s + 4] = view["captured_counts"][0] / 52.0
    obs[s + 5] = view["captured_counts"][1] / 52.0
    obs[s + 6] = view["opp_hand_count"] / 13.0
    obs[s + 7] = view["deck_count"] / 40.0
    obs[s + 8] = view["second_half"]
    obs[s + 9] = 0.0 if view["declared"] is None else view["declared"] / 13.0
    obs[s + 10] = view["is_first_player"]
    last = view["last_capturer_is_me"]
    obs[s + 11] = last is True
    obs[s + 12] = last is None
    obs[s + 13] = max(-200, min(200, view["differential"])) / 200.0
    obs[s + 14] = awaiting == DECLARE
    return obs


def encode_action(item, view, np_out=None):
    """Encode a candidate move into a float32 vector of length ACTION_SIZE.

    ``item`` is an engine Action, or a ``(DECLARE, value)`` tuple for the
    round-opening declaration. ``view`` is the acting player's
    ``game.view(player)`` dict — needed to price the pile a PICKUP absorbs.
    ``np_out`` works as in encode_observation.
    """
    vec = np_out if np_out is not None else np.zeros(ACTION_SIZE, dtype=np.float32)
    if np_out is not None:
        vec.fill(0.0)

    if isinstance(item, tuple):
        tag, value = item
        if tag != DECLARE:
            raise ValueError(f"unknown candidate tuple: {item!r}")
        vec[_ACT_KIND + _ACT_DECLARE_KIND] = 1.0
        vec[_ACT_SCALARS] = value / 13.0
        return vec

    vec[_ACT_KIND + item.kind.value] = 1.0
    vec[_ACT_CARD + item.card] = 1.0
    for c in item.loose:
        vec[_ACT_LOOSE + c] = 1.0
    s = _ACT_SCALARS
    vec[s] = item.value / 13.0
    vec[s + 1] = item.raised_from / 13.0
    vec[s + 2] = item.takes_pile
    vec[s + 3] = item.sweeps
    if item.kind is ActionKind.PICKUP:
        cards = (item.card,) + item.loose
        if item.takes_pile:
            cards += tuple(view["piles"][item.value]["cards"])
        vec[s + 4] = min(sum(card_points(c) for c in cards) / 20.0, 1.0)
        vec[s + 5] = min(len(cards) / 16.0, 1.0)
    return vec
