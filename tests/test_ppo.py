"""PPO stack tests: model masking, GAE, updates, checkpoints, collection,
and a small end-to-end train.py smoke."""

import csv
import math
import random

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

import train
from sweep.rl import ACTION_SIZE, OBS_SIZE
from sweep.rl.model import MASKED_LOGIT, PolicyValueNet
from sweep.rl.ppo import (
    collect_rounds,
    compute_gae,
    evaluate_actions,
    load_checkpoint,
    ppo_update,
    save_checkpoint,
)

# ------------------------------------------------------------------ helpers


def make_net(seed=0):
    torch.manual_seed(seed)
    return PolicyValueNet()


def gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def random_decision(n, seed=0):
    """A plausible single decision: (obs [OBS], cands [n, ACT]) float32."""
    rng = np.random.default_rng(seed)
    obs = rng.random(OBS_SIZE, dtype=np.float32)
    cands = rng.random((n, ACTION_SIZE), dtype=np.float32)
    return obs, cands


def synthetic_batch(net, total=48, n_max=5, seed=0):
    """A shape-correct PPO batch with old_logprob/value taken from the net."""
    g = gen(seed)
    rng = np.random.default_rng(seed)
    obs = torch.rand(total, OBS_SIZE, generator=g)
    cands = torch.rand(total, n_max, ACTION_SIZE, generator=g)
    counts = torch.from_numpy(rng.integers(1, n_max + 1, size=total))
    mask = torch.arange(n_max).unsqueeze(0) < counts.unsqueeze(1)
    chosen = torch.from_numpy(rng.integers(0, counts.numpy()))
    with torch.no_grad():
        old_logprob, _, value = evaluate_actions(net, obs, cands, mask, chosen)
    adv = torch.from_numpy(rng.standard_normal(total).astype(np.float32))
    return {"obs": obs, "cands": cands, "mask": mask, "chosen": chosen,
            "old_logprob": old_logprob, "value": value, "advantage": adv,
            "ret": value + 0.1 * adv}


# -------------------------------------------------------------------- model


def test_forward_masks_padding_to_probability_zero():
    net = make_net()
    g = gen()
    obs = torch.rand(3, OBS_SIZE, generator=g)
    cands = torch.rand(3, 7, ACTION_SIZE, generator=g)
    counts = torch.tensor([1, 4, 7])
    mask = torch.arange(7).unsqueeze(0) < counts.unsqueeze(1)
    logits, values = net(obs, cands, mask)
    assert logits.shape == (3, 7) and values.shape == (3,)
    assert (logits[~mask] == MASKED_LOGIT).all()
    probs = torch.softmax(logits, dim=-1)
    assert (probs[~mask] == 0.0).all()  # exactly zero, not merely tiny
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))
    # The all-but-padding row is a one-candidate decision: probability 1.
    assert probs[0, 0] == 1.0


def test_single_candidate_decision_gives_logprob_zero():
    net = make_net()
    obs, cands = random_decision(n=1)
    index, logprob, value = net.act_single(obs, cands, gen())
    assert index == 0
    assert logprob == 0.0
    assert math.isfinite(value)


def test_act_single_logprob_matches_manual_log_softmax():
    net = make_net()
    obs, cands = random_decision(n=4, seed=1)
    index, logprob, value = net.act_single(obs, cands, gen(3))
    assert 0 <= index < 4
    with torch.no_grad():
        logits, values = net(torch.tensor(obs).unsqueeze(0),
                             torch.tensor(cands).unsqueeze(0),
                             torch.ones(1, 4, dtype=torch.bool))
        manual = torch.log_softmax(logits[0], dim=-1)
    assert logprob == pytest.approx(float(manual[index]), abs=1e-6)
    assert value == pytest.approx(float(values[0]), abs=1e-6)


def test_act_single_temperature_zero_is_argmax_and_generator_determinism():
    net = make_net()
    obs, cands = random_decision(n=6, seed=2)
    index, logprob, _ = net.act_single(obs, cands, gen(), temperature=0)
    with torch.no_grad():
        logits, _ = net(torch.tensor(obs).unsqueeze(0),
                        torch.tensor(cands).unsqueeze(0),
                        torch.ones(1, 6, dtype=torch.bool))
    assert index == int(logits[0].argmax())
    assert logprob == 0.0  # a point mass: certainty
    picks_a = [net.act_single(obs, cands, gen(9))[0] for _ in range(5)]
    picks_b = [net.act_single(obs, cands, gen(9))[0] for _ in range(5)]
    assert picks_a == picks_b


def test_evaluate_actions_matches_manual_on_a_tiny_case():
    net = make_net()
    g = gen(4)
    obs = torch.rand(1, OBS_SIZE, generator=g)
    cands = torch.rand(1, 6, ACTION_SIZE, generator=g)
    mask = torch.tensor([[True, True, True, True, False, False]])
    chosen = torch.tensor([2])
    with torch.no_grad():
        logprob, entropy, values = evaluate_actions(net, obs, cands, mask, chosen)
        # Manual: log_softmax over the 4 real logits only.
        logits, _ = net(obs, cands, mask)
    real = torch.log_softmax(logits[0, :4], dim=-1)
    assert float(logprob) == pytest.approx(float(real[2]), abs=1e-6)
    manual_entropy = float(-(real.exp() * real).sum())
    assert float(entropy) == pytest.approx(manual_entropy, abs=1e-5)
    assert torch.isfinite(entropy).all() and torch.isfinite(values).all()


# ---------------------------------------------------------------------- GAE


def test_gae_three_step_hand_computed():
    values = [0.5, -0.25, 0.1]
    reward = 1.0
    # Terminal-only reward, gamma=1, lambda=0.95, V(terminal)=0:
    #   delta2 = 1.0 - 0.1            = 0.9      adv2 = 0.9
    #   delta1 = 0.1 - (-0.25)        = 0.35     adv1 = 0.35 + .95*0.9   = 1.205
    #   delta0 = -0.25 - 0.5          = -0.75    adv0 = -0.75 + .95*1.205 = 0.39475
    adv, ret = compute_gae(values, reward)
    assert adv == pytest.approx([0.39475, 1.205, 0.9])
    assert ret == pytest.approx([0.89475, 0.955, 1.0])
    assert adv.dtype == ret.dtype == np.float32


def test_gae_single_step_and_lambda_zero():
    adv, ret = compute_gae([0.3], -0.5)
    assert adv == pytest.approx([-0.8]) and ret == pytest.approx([-0.5])
    # lambda=0 degenerates to one-step TD errors.
    adv, ret = compute_gae([0.5, -0.25, 0.1], 1.0, lam=0.0)
    assert adv == pytest.approx([-0.75, 0.35, 0.9])


# ------------------------------------------------------------------- update


def test_ppo_update_runs_and_changes_parameters():
    net = make_net()
    batch = synthetic_batch(net)
    before = {k: v.detach().clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    torch.manual_seed(0)  # drives the minibatch permutation
    metrics = ppo_update(net, optimizer, batch, epochs=2, minibatch=16)
    assert set(metrics) == {"policy_loss", "value_loss", "entropy",
                            "approx_kl", "clip_frac"}
    assert all(math.isfinite(v) for v in metrics.values())
    assert 0.0 <= metrics["clip_frac"] <= 1.0
    assert metrics["entropy"] > 0.0
    after = net.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


# -------------------------------------------------------------- checkpoints


def test_checkpoint_roundtrip_is_exact(tmp_path):
    net = make_net(seed=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = synthetic_batch(net, total=16, seed=1)
    torch.manual_seed(1)
    ppo_update(net, optimizer, batch, epochs=1, minibatch=16)  # give Adam state
    meta = {"rounds_done": 123, "update_idx": 7, "pool": [],
            "note": {"nested": [1, "two", 3.0]}}
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net, optimizer, meta)

    net2 = make_net(seed=2)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr=1e-3)
    assert load_checkpoint(path, net2, optimizer2) == meta
    for k, v in net.state_dict().items():
        assert torch.equal(v, net2.state_dict()[k])
    s1, s2 = optimizer.state_dict(), optimizer2.state_dict()
    assert s1["param_groups"] == s2["param_groups"]
    for pid, state in s1["state"].items():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, s2["state"][pid][k])
            else:
                assert v == s2["state"][pid][k]


# --------------------------------------------------------------- collection


def _traj_slice(batch, traj):
    lo = traj["start"]
    return slice(lo, lo + traj["length"])


def test_collect_rounds_consistency():
    net = make_net()
    batch = collect_rounds(
        net, 6, {"self": 0.5, "greedy": 0.25, "random": 0.25},
        random.Random(3))
    total = batch["obs"].shape[0]
    n_max = batch["cands"].shape[1]
    assert batch["obs"].shape == (total, OBS_SIZE)
    assert batch["cands"].shape == (total, n_max, ACTION_SIZE)
    assert batch["mask"].shape == (total, n_max)
    counts = batch["mask"].sum(dim=1)
    assert (counts >= 1).all()
    assert (batch["chosen"] >= 0).all() and (batch["chosen"] < counts).all()
    assert (batch["old_logprob"] <= 0.0).all()
    # Padding is all-zero beyond each step's candidate count.
    assert (batch["cands"] * ~batch["mask"].unsqueeze(-1) == 0.0).all()
    assert torch.allclose(batch["ret"], batch["advantage"] + batch["value"])

    trajs = batch["trajectories"]
    assert sum(t["length"] for t in trajs) == total
    assert [t["start"] for t in trajs] == \
        [sum(u["length"] for u in trajs[:i]) for i in range(len(trajs))]
    kinds = {t["kind"] for t in trajs}
    assert "self" in kinds and kinds & {"greedy", "random"}  # both episode types
    by_episode = {}
    for t in trajs:
        by_episode.setdefault(t["episode"], []).append(t)
    assert len(by_episode) == 6
    for ep_trajs in by_episode.values():
        if ep_trajs[0]["kind"] == "self":
            # MIRROR: both seats collected; zero-sum episode rewards.
            assert sorted(t["seat"] for t in ep_trajs) == [0, 1]
            assert ep_trajs[0]["reward"] + ep_trajs[1]["reward"] == 0.0
            assert sum(t["length"] for t in ep_trajs) == 49  # every decision
        else:
            # LEAGUE: only the net seat's decisions were collected.
            [t] = ep_trajs
            assert (batch["players"][_traj_slice(batch, t)] == t["seat"]).all()
            assert t["length"] < 49
        for t in ep_trajs:
            # gamma=1 terminal-only reward: the last return IS the reward.
            last = t["start"] + t["length"] - 1
            assert float(batch["ret"][last]) == pytest.approx(t["reward"])

    means = batch["reward_mean"]
    assert set(means) == kinds
    assert means["self"] == 0.0  # exactly, by zero-sum construction
    for kind in kinds - {"self"}:
        rewards = [t["reward"] for t in trajs if t["kind"] == kind]
        assert means[kind] == pytest.approx(sum(rewards) / len(rewards))


def test_collect_rounds_pool_weight_folds_into_self_when_pool_empty():
    net = make_net()
    batch = collect_rounds(net, 2, {"pool": 1.0}, random.Random(0))
    assert {t["kind"] for t in batch["trajectories"]} == {"self"}


def test_collect_rounds_pool_opponent_plays_league():
    net = make_net()
    frozen = make_net(seed=5)
    for p in frozen.parameters():
        p.requires_grad_(False)
    batch = collect_rounds(net, 2, {"pool": 1.0}, random.Random(1), pool=[frozen])
    trajs = batch["trajectories"]
    assert [t["kind"] for t in trajs] == ["pool", "pool"]  # one traj per episode
    for t in trajs:
        assert (batch["players"][_traj_slice(batch, t)] == t["seat"]).all()


def test_collect_rounds_rejects_bad_weights():
    net = make_net()
    with pytest.raises(ValueError):
        collect_rounds(net, 1, {"ismcts": 1.0}, random.Random(0))
    with pytest.raises(ValueError):
        collect_rounds(net, 1, {"self": 0.0}, random.Random(0))


def test_collect_rounds_seeded_determinism():
    net = make_net()
    opponents = {"self": 0.4, "pool": 0.2, "greedy": 0.2, "random": 0.2}

    def run():
        return collect_rounds(net, 4, opponents, random.Random(7))

    a, b = run(), run()
    for key in ("obs", "cands", "mask", "chosen", "old_logprob", "value",
                "advantage", "ret"):
        assert torch.equal(a[key], b[key]), key
    assert (a["players"] == b["players"]).all()
    assert a["trajectories"] == b["trajectories"]
    assert a["reward_mean"] == b["reward_mean"]


# ------------------------------------------------------------------ train.py


def read_metrics(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_train_smoke_end_to_end(tmp_path):
    out = tmp_path / "run"
    train.main(["--rounds", "40", "--batch-rounds", "20",
                "--out", str(out), "--seed", "1"])
    assert (out / "ckpt_latest.pt").exists()
    rows = read_metrics(out / "metrics.csv")
    assert [r["rounds"] for r in rows] == ["20", "40"]
    assert [r["update"] for r in rows] == ["1", "2"]
    for row in rows:
        assert float(row["rounds_per_sec"]) > 0
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl",
                    "clip_frac"):
            assert math.isfinite(float(row[key]))
        assert float(row["reward_self"]) == 0.0  # mirror rewards are zero-sum
        assert row["reward_pool"] == ""  # pool empty: folded into self
    net = PolicyValueNet()
    meta = load_checkpoint(out / "ckpt_latest.pt", net)
    assert meta["rounds_done"] == 40 and meta["update_idx"] == 2
    assert meta["pool"] == []  # --pool-every never hit in 40 rounds


def test_train_resume_continues_bit_identically(tmp_path):
    common = ["--batch-rounds", "4", "--seed", "3", "--opponents",
              '{"self": 1.0}', "--pool-every", "4", "--pool-size", "2"]
    long_out = tmp_path / "long"
    train.main(["--rounds", "8", "--out", str(long_out)] + common)

    chunk_out = tmp_path / "chunked"
    train.main(["--rounds", "4", "--out", str(chunk_out)] + common)
    train.main(["--rounds", "8", "--out", str(chunk_out), "--resume",
                str(chunk_out / "ckpt_latest.pt")] + common)

    net_a, net_b = PolicyValueNet(), PolicyValueNet()
    meta_a = load_checkpoint(long_out / "ckpt_latest.pt", net_a)
    meta_b = load_checkpoint(chunk_out / "ckpt_latest.pt", net_b)
    for k, v in net_a.state_dict().items():
        assert torch.equal(v, net_b.state_dict()[k]), k
    assert meta_a["rounds_done"] == meta_b["rounds_done"] == 8
    assert meta_a["update_idx"] == meta_b["update_idx"] == 2
    assert meta_a["py_rng_state"] == meta_b["py_rng_state"]
    assert torch.equal(meta_a["torch_rng_state"], meta_b["torch_rng_state"])
    assert len(meta_a["pool"]) == len(meta_b["pool"]) == 2
    for sd_a, sd_b in zip(meta_a["pool"], meta_b["pool"]):
        for k in sd_a:
            assert torch.equal(sd_a[k], sd_b[k])
    # The long and chunked runs logged identical training metrics.
    strip = ("seconds", "rounds_per_sec")  # wall clock: documented exception
    rows_a = [{k: v for k, v in r.items() if k not in strip}
              for r in read_metrics(long_out / "metrics.csv")]
    rows_b = [{k: v for k, v in r.items() if k not in strip}
              for r in read_metrics(chunk_out / "metrics.csv")]
    assert rows_a == rows_b


def test_parse_args_temperature_guards(capsys):
    base = ["--rounds", "1", "--out", "x"]
    train.parse_args(base)
    assert capsys.readouterr().err == ""  # default 1.0: silent
    train.parse_args(base + ["--temperature", "0.5"])
    assert "off-policy" in capsys.readouterr().err
    for bad in ("0", "-1"):
        with pytest.raises(SystemExit):
            train.parse_args(base + ["--temperature", bad])


def test_resume_keeps_checkpointed_lr_and_warns(tmp_path, capsys):
    out = tmp_path / "run"
    common = ["--batch-rounds", "2", "--seed", "5", "--opponents", '{"self": 1.0}']
    train.main(["--rounds", "2", "--out", str(out)] + common)
    ckpt = str(out / "ckpt_latest.pt")
    train.main(["--rounds", "4", "--out", str(out), "--resume", ckpt] + common)
    assert capsys.readouterr().err == ""  # same lr: no warning
    train.main(["--rounds", "6", "--out", str(out), "--resume", ckpt,
                "--lr", "1e-5"] + common)
    err = capsys.readouterr().err
    assert "0.0003" in err and "1e-05 is ignored" in err
    payload = torch.load(out / "ckpt_latest.pt", map_location="cpu")
    assert payload["optimizer"]["param_groups"][0]["lr"] == pytest.approx(3e-4)


def test_opponents_presets_and_json():
    assert train.resolve_opponents("default") == train.DEFAULT_OPPONENTS
    assert train.resolve_opponents('{"greedy": 1.0}') == {"greedy": 1.0}
    with pytest.raises(KeyError):
        train.resolve_opponents("nope")
    with pytest.raises(ValueError):
        train.resolve_opponents('{"ismcts": 1.0}')
    with pytest.raises(ValueError):
        train.resolve_opponents('{"self": 0.0}')
