#!/usr/bin/env python3
"""Round-robin evaluation of Sweep agents.

Usage:
    python3 evaluate.py random greedy heuristic -n 200 --win-lead 200 --seed 0

`-n` is games per pairing, split across seat-swapped pairs (each game seed is
played twice with seats swapped).

`ismcts` at its default budget takes ~1s per decision (~1000x the baselines);
pass e.g. `--sims 50` for quicker, weaker evaluations.

`neural` plays a trained policy network and requires `--ckpt PATH` (a
checkpoint written by train.py / sweep.rl.ppo.save_checkpoint).

`hybrid` is ISMCTS with that trained value head replacing the greedy rollout;
it also requires `--ckpt PATH`, and `--sims` sets its per-decision budget.
"""

import argparse

from sweep.agents import GreedyAgent, HeuristicAgent, RandomAgent
from sweep.arena import round_robin
from sweep.ismcts import ISMCTSAgent

REGISTRY = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": HeuristicAgent,
    "ismcts": ISMCTSAgent,
    "neural": None,  # built lazily in main() so torch loads only when used
    "hybrid": None,  # built lazily in main() so torch loads only when used
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "agents", nargs="+", choices=sorted(REGISTRY), help="agents to round-robin"
    )
    parser.add_argument("-n", dest="n_games", type=int, default=200,
                        help="games per pairing (default 200)")
    parser.add_argument("--win-lead", type=int, default=200,
                        help="cumulative lead that ends a game (default 200)")
    parser.add_argument("--seed", type=int, default=0, help="base seed (default 0)")
    parser.add_argument("--sims", type=int, default=None,
                        help="ismcts/hybrid simulations per decision "
                             "(default: agent default)")
    parser.add_argument("--ckpt", default=None,
                        help="checkpoint path for the neural/hybrid agent "
                             "(required with either)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="neural sampling temperature (default 0 = argmax; "
                             "per-game seeds only affect neural above 0)")
    parser.add_argument("--priors", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="hybrid: guide search with the net's policy head "
                             "(PUCT); on by default (measured stronger). "
                             "Use --no-priors for plain value-head ISMCTS")
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="hybrid PUCT exploration constant (default 1.0; "
                             "only used with --priors)")
    args = parser.parse_args(argv)

    names = list(dict.fromkeys(args.agents))  # dedupe, keep order
    if len(names) < 2:
        parser.error("need at least two distinct agents")
    if "neural" in names and args.ckpt is None:
        parser.error("the neural agent requires --ckpt PATH")
    if "hybrid" in names and args.ckpt is None:
        parser.error("the hybrid agent requires --ckpt PATH")

    def factory(name):
        if name == "neural":
            # Lazy (pulls in torch); load the checkpoint once, not per game.
            from sweep.rl.agent import NeuralAgent
            from sweep.rl.model import PolicyValueNet
            from sweep.rl.ppo import load_checkpoint
            net = PolicyValueNet()
            load_checkpoint(args.ckpt, net)
            return lambda s: NeuralAgent(net=net, seed=s,
                                         temperature=args.temperature)
        if name == "hybrid":
            # Lazy (pulls in torch); load the checkpoint once, not per game.
            from sweep.rl.hybrid import HybridAgent
            from sweep.rl.model import PolicyValueNet
            from sweep.rl.ppo import load_checkpoint
            net = PolicyValueNet()
            load_checkpoint(args.ckpt, net)
            kw = {} if args.sims is None else {"n_sims": args.sims}
            kw["priors"] = args.priors
            kw["c_puct"] = args.c_puct
            return lambda s: HybridAgent(net=net, seed=s, **kw)
        cls = REGISTRY[name]
        if name == "ismcts" and args.sims is not None:
            return lambda s: cls(seed=s, n_sims=args.sims)
        return lambda s: cls(seed=s)

    factories = {name: factory(name) for name in names}
    results = round_robin(factories, args.n_games, base_seed=args.seed,
                          win_lead=args.win_lead)

    header = (
        f"{'pairing (A vs B)':<24} {'games':>5} {'A wins':>6} {'B wins':>6} "
        f"{'win% A':>7} {'95% CI':>16} {'diff/g (A)':>10} {'rnds/g':>6} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))
    for (a, b), r in results.items():
        lo, hi = r["ci95"]
        print(
            f"{a + ' vs ' + b:<24} {r['games']:>5} {r['wins_a']:>6} {r['wins_b']:>6} "
            f"{r['win_rate_a'] * 100:>6.1f}% [{lo * 100:>5.1f}%,{hi * 100:>5.1f}%] "
            f"{r['mean_diff_a']:>+10.1f} {r['mean_rounds']:>6.2f} {r['seconds']:>6.1f}"
        )


if __name__ == "__main__":
    main()
