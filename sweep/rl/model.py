"""Policy/value network scoring each candidate move of a Sweep decision.

Decisions carry a variable number of candidates, so the policy is a
per-action scorer rather than a fixed action head: a trunk embeds the
observation, an action head embeds each encoded candidate, and a scorer
turns every (observation, candidate) pair into one logit. Batches pad the
candidate axis to a common length and mask the padding; masked slots are
pinned to MASKED_LOGIT so their softmax probability underflows to exactly 0.

This module is deliberately NOT re-exported from sweep.rl's __init__:
``import sweep.rl`` stays torch-free (encoders + env only), and learning
code imports sweep.rl.model / sweep.rl.ppo directly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoders import ACTION_SIZE, OBS_SIZE

MASKED_LOGIT = -1e9


class PolicyValueNet(nn.Module):
    """Trunk OBS->256->256, action head ACT->128, pairwise scorer, value head."""

    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(OBS_SIZE, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(ACTION_SIZE, 128), nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs, cands, mask):
        """Score a padded batch of decisions.

        obs [B, OBS_SIZE], cands [B, N, ACTION_SIZE], mask [B, N] bool (True
        at real candidates) -> (logits [B, N] with masked slots at exactly
        MASKED_LOGIT, values [B]).
        """
        obs_emb = self.trunk(obs)
        act_emb = self.action_head(cands)
        n = cands.shape[1]
        joint = torch.cat([obs_emb.unsqueeze(1).expand(-1, n, -1), act_emb], dim=-1)
        logits = self.scorer(joint).squeeze(-1)
        logits = logits.masked_fill(~mask, MASKED_LOGIT)
        values = self.value_head(obs_emb).squeeze(-1)
        return logits, values

    @torch.no_grad()
    def act_single(self, obs_np, cands_np, generator, temperature=1.0):
        """Pick a candidate for one decision -> (index, logprob, value).

        ``obs_np`` is the decision's obs vector, ``cands_np`` its
        [n, ACTION_SIZE] encoded candidates (no padding, so nothing is
        masked). Samples from softmax(logits / temperature); the returned
        logprob is under that same sampling distribution, so PPO collection
        should use temperature=1.0 — anything else makes the data
        (mildly) off-policy for an update that recomputes at temperature 1.
        temperature=0 means argmax, a point mass, hence logprob 0.0.
        Deterministic given the passed torch.Generator.
        """
        obs = torch.as_tensor(obs_np).unsqueeze(0)
        cands = torch.as_tensor(cands_np).unsqueeze(0)
        mask = torch.ones(cands.shape[:2], dtype=torch.bool)
        logits, values = self(obs, cands, mask)
        if temperature == 0:
            return int(logits[0].argmax()), 0.0, float(values[0])
        logprobs = F.log_softmax(logits[0] / temperature, dim=-1)
        index = int(torch.multinomial(logprobs.exp(), 1, generator=generator))
        return index, float(logprobs[index]), float(values[0])
