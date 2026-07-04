#!/usr/bin/env python3
"""Train a Sweep PPO agent (CPU): collect one-round episodes, update, repeat.

    python3 train.py --rounds 200000 --out runs/base --seed 0

Each loop iteration collects --batch-rounds episodes with sweep.rl.ppo
.collect_rounds (opponent mix from --opponents: a preset name or a JSON
weights dict over self/pool/greedy/heuristic/random), runs one ppo_update,
and appends a row to <out>/metrics.csv. <out>/ckpt_latest.pt is written
every --save-every rounds and at the end; every --pool-every rounds the
current net is snapshotted into a frozen-opponent pool of up to --pool-size
past selves, sampled by the "pool" opponent weight.

Resume semantics (--resume PATH): the checkpoint stores the net, optimizer,
round counter, update counter, the frozen pool, the collection rng state,
and torch's global RNG state, all of which are restored, so a run split
into chunks CONTINUES the single long run exactly. Restoring the optimizer
includes its learning rate, so a different --lr passed alongside --resume
is ignored (a warning is printed) — there is no cross-chunk lr schedule.
With --threads 1 (the
default) and chunk boundaries that fall on update boundaries (each chunk's
--rounds a multiple of --batch-rounds; note the final update of a run is
truncated to the rounds remaining), the chunked run is bit-identical to the
long run: same game seeds, opponent draws, sampled actions, minibatch
permutations, parameters, and metrics values. NOT bit-identical: the
seconds / rounds_per_sec columns (wall clock), the row layout of
metrics.csv if chunks change --opponents mid-run, and any run with
--threads > 1 (multi-threaded reductions may reorder float ops). numpy's
and python's global RNGs are seeded at startup for hygiene but never
consumed by the training loop, which draws only from the checkpointed
streams.

Crash recovery caveat: metrics.csv gains a row every update, but
ckpt_latest.pt is only written on --save-every crossings and at the end. A
run killed between the two leaves rows newer than the checkpoint; resuming
replays those updates and appends rows repeating the same round counts.
Documented chunked runs (chunk boundaries on update boundaries, ending
normally) are unaffected; otherwise dedupe metrics.csv on the rounds
column, keeping the last occurrence.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sweep.rl.model import PolicyValueNet
from sweep.rl.ppo import (
    OPPONENT_KINDS,
    collect_rounds,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)

DEFAULT_OPPONENTS = {"self": 0.5, "pool": 0.2, "greedy": 0.15, "heuristic": 0.1,
                     "random": 0.05}
PRESETS = {
    "default": DEFAULT_OPPONENTS,
    "self": {"self": 1.0},
    "scripted": {"greedy": 0.4, "heuristic": 0.4, "random": 0.2},
}

CSV_FIELDS = ["update", "rounds", "seconds", "rounds_per_sec", "policy_loss",
              "value_loss", "entropy", "approx_kl", "clip_frac"] + [
              f"reward_{k}" for k in OPPONENT_KINDS]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rounds", type=int, required=True,
                   help="total rounds to have collected when done")
    p.add_argument("--out", required=True, help="output dir, e.g. runs/NAME")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Adam learning rate (ignored with --resume: the "
                        "checkpointed optimizer lr wins)")
    p.add_argument("--batch-rounds", type=int, default=64,
                   help="episodes collected per PPO update")
    p.add_argument("--save-every", type=int, default=5000,
                   help="rounds between ckpt_latest.pt writes (also written at the end)")
    p.add_argument("--pool-size", type=int, default=5,
                   help="max frozen past selves kept as pool opponents")
    p.add_argument("--pool-every", type=int, default=10000,
                   help="rounds between pool snapshots")
    p.add_argument("--opponents", default="default",
                   help=f"preset {sorted(PRESETS)} or a JSON weights dict "
                        f"over {list(OPPONENT_KINDS)}")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="collection sampling temperature; ppo_update "
                        "recomputes logprobs at 1.0, so any other value "
                        "makes the collected data off-policy")
    p.add_argument("--resume", default=None, help="checkpoint to continue from")
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    try:
        args.opponents = resolve_opponents(args.opponents)
    except (KeyError, ValueError) as e:
        p.error(f"bad --opponents: {e}")
    if args.temperature <= 0:
        p.error("--temperature must be > 0: 0 is argmax with old_logprob "
                "fixed at 0.0, which breaks the PPO ratio")
    if args.temperature != 1.0:
        print(f"warning: --temperature {args.temperature:g} collects "
              f"logprobs that are off-policy for the temperature-1 "
              f"ppo_update; use 1.0 for training", file=sys.stderr)
    return args


def resolve_opponents(spec):
    """A preset name or JSON string -> validated weights dict."""
    weights = json.loads(spec) if spec.lstrip().startswith("{") else PRESETS[spec]
    unknown = set(weights) - set(OPPONENT_KINDS)
    if unknown:
        raise ValueError(f"unknown opponent kinds: {sorted(unknown)}")
    if not any(w > 0 for w in weights.values()):
        raise ValueError("no opponent kind has positive weight")
    return dict(weights)


def _frozen_net(state_dict):
    """A no-grad PolicyValueNet loaded from state_dict; leaves the global
    torch RNG untouched (construction runs under fork_rng), which keeps
    resumed runs on the identical RNG stream."""
    with torch.random.fork_rng():
        net = PolicyValueNet()
    net.load_state_dict(state_dict)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    return net


def _crossed(prev, now, every):
    """True when the round counter crossed a multiple of `every`."""
    return every > 0 and now // every > prev // every


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt_latest.pt"
    metrics_path = out / "metrics.csv"

    net = PolicyValueNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    collect_rng = random.Random(args.seed)
    pool_dicts = []  # oldest first, at most --pool-size entries
    rounds_done = 0
    update_idx = 0

    if args.resume:
        meta = load_checkpoint(args.resume, net, optimizer)
        restored_lr = optimizer.param_groups[0]["lr"]
        if restored_lr != args.lr:
            print(f"warning: --resume restored optimizer lr {restored_lr:g}; "
                  f"--lr {args.lr:g} is ignored", file=sys.stderr)
        rounds_done = meta["rounds_done"]
        update_idx = meta["update_idx"]
        pool_dicts = meta["pool"]
        collect_rng.setstate(meta["py_rng_state"])
        torch.set_rng_state(meta["torch_rng_state"])
    pool_nets = [_frozen_net(sd) for sd in pool_dicts]

    def checkpoint():
        save_checkpoint(ckpt_path, net, optimizer, {
            "rounds_done": rounds_done,
            "update_idx": update_idx,
            "pool": pool_dicts,
            "py_rng_state": collect_rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "seed": args.seed,
        })

    new_csv = not metrics_path.exists()
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_csv:
            writer.writeheader()

        while rounds_done < args.rounds:
            n = min(args.batch_rounds, args.rounds - rounds_done)
            t0 = time.perf_counter()
            batch = collect_rounds(net, n, args.opponents, collect_rng,
                                   pool_nets, args.temperature)
            stats = ppo_update(net, optimizer, batch)
            seconds = time.perf_counter() - t0
            prev, rounds_done, update_idx = rounds_done, rounds_done + n, update_idx + 1

            row = {"update": update_idx, "rounds": rounds_done,
                   "seconds": f"{seconds:.3f}",
                   "rounds_per_sec": f"{n / seconds:.3f}"}
            row.update({k: f"{v:.6g}" for k, v in stats.items()})
            for kind in OPPONENT_KINDS:
                mean = batch["reward_mean"].get(kind)
                row[f"reward_{kind}"] = "" if mean is None else f"{mean:.6g}"
            writer.writerow(row)
            f.flush()
            rewards = " ".join(
                f"{k}={v:.3f}" for k, v in sorted(batch["reward_mean"].items()))
            print(f"update {update_idx}  rounds {rounds_done}/{args.rounds}  "
                  f"{n / seconds:.2f} rounds/s  kl {stats['approx_kl']:.4f}  "
                  f"ent {stats['entropy']:.3f}  reward: {rewards}")

            if _crossed(prev, rounds_done, args.pool_every) and args.pool_size > 0:
                pool_dicts.append(
                    {k: v.detach().clone() for k, v in net.state_dict().items()})
                pool_dicts = pool_dicts[-args.pool_size:]
                pool_nets = [_frozen_net(sd) for sd in pool_dicts]
            if _crossed(prev, rounds_done, args.save_every):
                checkpoint()

    checkpoint()
    print(f"done: {rounds_done} rounds, checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
