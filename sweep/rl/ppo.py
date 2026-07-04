"""PPO learning stack on the SweepEnv decision stream.

Collection plays one-round episodes (see sweep.rl.env) of two kinds:

    MIRROR  ("self"; also "pool" while the frozen pool is empty): the
            training net plays both seats and BOTH seats' trajectories are
            collected, each with its own terminal reward. Rewards are
            zero-sum, so a mirror episode's two rewards sum to exactly 0.
    LEAGUE  (everything else): the net sits at one random seat and the
            opponent fills the other — a scripted Agent playing the env's
            real Game (fair by the Agent contract), or a frozen pool net
            playing the decision stream. Only the net seat's decisions are
            collected.

Rewards are terminal-only: (score diff)/100 at the round's end, 0 elsewhere.
Advantages use GAE with gamma=1.0, lambda=0.95 per trajectory; they are
normalized inside ppo_update (per update batch), never here.

Everything is deterministic given the caller's random.Random: episode kinds,
seats, env seeds, scripted-agent seeds, and the torch.Generator that drives
action sampling are all derived from it.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.nn import functional as F

from ..agents import GreedyAgent, HeuristicAgent, RandomAgent
from .encoders import ACTION_SIZE, DECLARE
from .env import SweepEnv

GAMMA = 1.0
LAM = 0.95

_SCRIPTED = {"random": RandomAgent, "greedy": GreedyAgent, "heuristic": HeuristicAgent}
OPPONENT_KINDS = ("self", "pool") + tuple(_SCRIPTED)


# ------------------------------------------------------------------ collection


def compute_gae(values, reward, gamma=GAMMA, lam=LAM):
    """(advantages, returns) for one trajectory with a terminal-only reward.

    ``values`` are the critic's per-step estimates; ``reward`` lands on the
    final step. Standard GAE with V(s_T) = 0 at the terminal.
    """
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    adv = np.zeros(n, dtype=np.float32)
    next_value = 0.0
    next_adv = 0.0
    for t in reversed(range(n)):
        r = reward if t == n - 1 else 0.0
        delta = r + gamma * next_value - values[t]
        next_adv = delta + gamma * lam * next_adv
        adv[t] = next_adv
        next_value = values[t]
    return adv, adv + values


def _scripted_index(agent, game, candidates):
    """The candidate index of a scripted agent's move on the env's real Game."""
    if game.awaiting == DECLARE:
        return candidates.index((DECLARE, agent.declare(game)))
    return candidates.index(agent.act(game))


def collect_rounds(net, n_rounds, opponents, rng, pool=(), temperature=1.0):
    """Play n_rounds one-round episodes and return a PPO batch dict.

    ``opponents`` is a weights dict over OPPONENT_KINDS; kinds with weight
    <= 0 are dropped, and with an empty ``pool`` any "pool" weight folds
    into "self". ``pool`` is a sequence of frozen PolicyValueNets.

    Batch tensors (float32 unless noted), T = total collected steps, N = the
    batch's max candidate count:

        obs [T, OBS_SIZE], cands [T, N, ACTION_SIZE] zero-padded,
        mask [T, N] bool, chosen [T] int64, old_logprob [T], value [T],
        advantage [T] (raw GAE, normalized later by ppo_update), ret [T]

    plus metadata: players (np int64 [T], acting seat per step),
    trajectories (list of {kind, episode, seat, reward, start, length};
    steps are laid out trajectory-contiguously), and reward_mean
    ({kind: mean trajectory reward} — exactly 0.0 for "self" since mirror
    rewards are zero-sum by construction).
    """
    weights = {k: w for k, w in opponents.items() if w > 0}
    unknown = set(weights) - set(OPPONENT_KINDS)
    if unknown:
        raise ValueError(f"unknown opponent kinds: {sorted(unknown)}")
    if not pool and "pool" in weights:
        weights["self"] = weights.get("self", 0.0) + weights.pop("pool")
    if not weights:
        raise ValueError("no opponent kind has positive weight")
    kinds = [k for k in OPPONENT_KINDS if k in weights]
    kind_weights = [weights[k] for k in kinds]

    agents = {k: _SCRIPTED[k](seed=rng.getrandbits(64)) for k in kinds if k in _SCRIPTED}
    env = SweepEnv(seed=rng.getrandbits(64))
    gen = torch.Generator()
    gen.manual_seed(rng.getrandbits(64))

    obs_l, cands_l, chosen_l, logprob_l, value_l, players_l = [], [], [], [], [], []
    adv_l, ret_l = [], []
    trajectories = []
    reward_sums = {}
    reward_counts = {}

    for episode in range(n_rounds):
        kind = rng.choices(kinds, weights=kind_weights)[0]
        net_seat = None if kind == "self" else rng.randrange(2)
        opp = None
        if kind == "pool":
            opp = pool[rng.randrange(len(pool))]
        elif kind != "self":
            opp = agents[kind]

        steps = {0: [], 1: []}  # per-seat (obs, encoded, chosen, logprob, value)
        decision = env.reset()
        done = False
        while not done:
            p = decision["player"]
            if net_seat is None or p == net_seat:
                index, logprob, value = net.act_single(
                    decision["obs"], decision["encoded_candidates"], gen, temperature)
                steps[p].append(
                    (decision["obs"], decision["encoded_candidates"], index, logprob, value))
            elif kind == "pool":
                index, _, _ = opp.act_single(
                    decision["obs"], decision["encoded_candidates"], gen, temperature)
            else:
                index = _scripted_index(opp, env.game, decision["candidates"])
            decision, rewards, done = env.step(index)

        for seat in (0, 1) if net_seat is None else (net_seat,):
            traj = steps[seat]
            reward = rewards[seat]
            adv, ret = compute_gae([s[4] for s in traj], reward)
            trajectories.append({
                "kind": kind, "episode": episode, "seat": seat,
                "reward": reward, "start": len(obs_l), "length": len(traj),
            })
            for obs, encoded, index, logprob, value in traj:
                obs_l.append(obs)
                cands_l.append(encoded)
                chosen_l.append(index)
                logprob_l.append(logprob)
                value_l.append(value)
                players_l.append(seat)
            adv_l.append(adv)
            ret_l.append(ret)
            reward_sums[kind] = reward_sums.get(kind, 0.0) + reward
            reward_counts[kind] = reward_counts.get(kind, 0) + 1

    total = len(obs_l)
    n_max = max(c.shape[0] for c in cands_l)
    cands = np.zeros((total, n_max, ACTION_SIZE), dtype=np.float32)
    mask = np.zeros((total, n_max), dtype=bool)
    for i, encoded in enumerate(cands_l):
        cands[i, :encoded.shape[0]] = encoded
        mask[i, :encoded.shape[0]] = True

    return {
        "obs": torch.from_numpy(np.stack(obs_l)),
        "cands": torch.from_numpy(cands),
        "mask": torch.from_numpy(mask),
        "chosen": torch.tensor(chosen_l, dtype=torch.int64),
        "old_logprob": torch.tensor(logprob_l, dtype=torch.float32),
        "value": torch.tensor(value_l, dtype=torch.float32),
        "advantage": torch.from_numpy(np.concatenate(adv_l)),
        "ret": torch.from_numpy(np.concatenate(ret_l)),
        "players": np.array(players_l, dtype=np.int64),
        "trajectories": trajectories,
        "reward_mean": {k: reward_sums[k] / reward_counts[k] for k in reward_sums},
    }


# --------------------------------------------------------------------- update


def evaluate_actions(net, obs, cands, mask, chosen):
    """(logprob [B], entropy [B], values [B]) of chosen actions under net.

    Masked slots have probability exactly 0 and a finite log-probability
    (MASKED_LOGIT - logsumexp), so the entropy sum is NaN-free.
    """
    logits, values = net(obs, cands, mask)
    logprobs = F.log_softmax(logits, dim=-1)
    logprob = logprobs.gather(1, chosen.unsqueeze(1)).squeeze(1)
    entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
    return logprob, entropy, values


def ppo_update(net, optimizer, batch, clip=0.2, epochs=4, minibatch=1024,
               vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
    """One clipped-surrogate PPO update over a collect_rounds batch.

    Advantages are normalized here, over the whole update batch. Minibatch
    order comes from torch.randperm on the global torch RNG. Returns mean
    metrics over all minibatch passes: policy_loss, value_loss, entropy,
    approx_kl ((ratio - 1) - log ratio estimator), clip_frac.
    """
    obs, cands, mask = batch["obs"], batch["cands"], batch["mask"]
    chosen, old_logprob, ret = batch["chosen"], batch["old_logprob"], batch["ret"]
    adv = batch["advantage"]
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "approx_kl": 0.0, "clip_frac": 0.0}
    passes = 0
    n = obs.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, minibatch):
            mb = perm[start:start + minibatch]
            logprob, entropy, values = evaluate_actions(
                net, obs[mb], cands[mb], mask[mb], chosen[mb])
            logratio = logprob - old_logprob[mb]
            ratio = logratio.exp()
            adv_mb = adv[mb]
            surrogate = torch.min(
                ratio * adv_mb,
                torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv_mb)
            policy_loss = -surrogate.mean()
            value_loss = F.mse_loss(values, ret[mb])
            entropy_mean = entropy.mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                metrics["policy_loss"] += float(policy_loss)
                metrics["value_loss"] += float(value_loss)
                metrics["entropy"] += float(entropy_mean)
                metrics["approx_kl"] += float(((ratio - 1.0) - logratio).mean())
                metrics["clip_frac"] += float(((ratio - 1.0).abs() > clip).float().mean())
            passes += 1
    return {k: v / passes for k, v in metrics.items()}


# ---------------------------------------------------------------- checkpoints


def save_checkpoint(path, net, optimizer, meta):
    """Atomically write {net, optimizer, meta} to path (torch.save format)."""
    payload = {
        "net": net.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "meta": meta,
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, net, optimizer=None):
    """Restore net (and optimizer, when given) from path; return meta."""
    payload = torch.load(path, map_location="cpu")
    net.load_state_dict(payload["net"])
    if optimizer is not None and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload["meta"]
