#!/usr/bin/env python3
"""Multi-hop relay: A-B-C must exchange state through B, and must be an exact no-op when off.

    python scripts/15_test_comm_relay.py

Three agents are placed in a straight corridor of free space at spacings that make A-B and B-C
linked and A-C not, and the assertions check the four things the relay is supposed to fix:

    1. comm_group is the transitive closure of comm_mask (symmetric, reflexive)
    2. after ONE step all three hold the SAME map. Before the relay, the in-place pair loop in
       world_warp.fuse_maps handed the far map to the higher-index leaf in the same step and to
       the lower-index one a step later, if the link held at all — an order-dependent half-relay
       nobody designed.
    3. A learns C's TRUE position, staleness clock and offer baseline (IR2 gives in-flock members
       ground truth, env.py:276), and reads contact = 1
    4. comm_mask, comm_duty_cycle and the sync reward stay on the DIRECT link, so the radio
       telemetry and the reward budget are untouched
    5. with comm_relay=False every one of the above reverts, bit for bit

Runs N>1 envs. A batch-dim bug is invisible at P=1 — that is exactly how a gather on dim 1 in
freeze_hypotheses killed v15's first run at iteration 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from env.explorer import GT_FREE, EnvCfg, Explorer  # noqa: E402
from env.maps import load_split  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
N_ENVS, M = 4, 3
SPLIT = "train/difficult"
MAP_IDX = 50


def build(relay: bool) -> Explorer:
    """LOS comm with a range that admits one hop of the chain and not two."""
    cfg = EnvCfg(n_envs=N_ENVS, n_agents=M, n_hops=6, comm_model="los", comm_range_px=90.0,
                 sensor_range_px=80.0, max_episode_steps=500, comm_relay=relay, map_seed=7)
    split = load_split(SPLIT, device=DEV)
    env = Explorer(split, cfg, seed=3)
    for n in range(N_ENVS):
        env.reload_map(env_idx=n, map_idx=MAP_IDX + n)
    return env


def chain_positions(env: Explorer) -> torch.Tensor:
    """A-B-C spaced 70 px apart along a free row: |AB| = |BC| = 70 < 90, |AC| = 140 > 90.

    Anchored on a row of the ground truth that is free across the whole span, so the LOS test
    fails on distance alone and not on a wall that happens to sit between two of them.
    """
    gt = env.world.gt_torch                                     # [N, H, W]
    pos = env.pos.clone()
    span = 140
    for n in range(env.N):
        free = (gt[n] == GT_FREE)                               # GT_FREE == 1, GT_OBST == 0
        found = False
        for y in range(0, env.H, 4):
            row = free[y]
            run = 0
            for x in range(env.W):
                run = run + 1 if bool(row[x]) else 0
                if run > span + 20:
                    x0 = x - span - 10
                    for a in range(M):
                        pos[n, a, 0] = float(x0 + a * 70)
                        pos[n, a, 1] = float(y)
                    found = True
                    break
            if found:
                break
        assert found, f"env {n}: no free row wide enough for the chain"
    return pos


def place(env: Explorer, pos: torch.Tensor) -> None:
    env.pos.copy_(pos)
    env._refresh_obs(None)


def pin(env: Explorer) -> None:
    """Freeze the physics so the chain geometry survives the step.

    _comm_check runs AFTER _move_and_scan, so an unpinned agent takes a lattice hop (up to
    nr*sqrt(2) = 22.6 px) before connectivity is evaluated and a 70 px chain can drift to 115 px —
    outside any comm range that also excludes the 140 px A-C pair. Re-aiming every target at the
    agent's own position keeps the LiDAR scan and drops the displacement.
    """
    orig = env._move_and_scan
    env._move_and_scan = lambda _tgt: orig(env.pos.clone())


def run_case(relay: bool) -> dict:
    env = build(relay)
    pos = chain_positions(env)
    place(env, pos)
    pin(env)
    act = torch.zeros((env.N, env.M), dtype=torch.long, device=env.dev)
    valid = env.obs["action_mask"].bool()
    for n in range(env.N):
        for a in range(env.M):
            ok = valid[n, a].nonzero()
            if ok.numel():
                act[n, a] = int(ok[0])
    _obs, _r, _done, info = env.step(act)

    lo = env.world.occupancy_logodds_torch                       # [N, M, H, W]
    return dict(
        env=env,
        direct=info["comm_mask"].clone(),
        group=info["comm_group"].clone(),
        duty=float(info["metrics"]["comm_duty_cycle"]),
        gduty=float(info["metrics"]["comm_group_duty"]),
        conn=float(info["metrics"]["comm_connected"]),
        sync=info["sync_paid"].clone(),
        lo=lo.clone(),
        lkp=env.last_known_pos.clone(),
        tlc=env.t_last_comm.clone(),
        contact=env.obs["agent_scalars"][..., 3].clone(),        # AGENT_SCALAR_DIM order
        pos=env.pos.clone(),
    )


def main() -> None:
    print(f"device={DEV}  N={N_ENVS} envs  M={M} agents  split={SPLIT}")
    on = run_case(relay=True)
    off = run_case(relay=False)

    eye = torch.eye(M, dtype=torch.bool, device=DEV).view(1, M, M)
    d = on["direct"]
    print(f"\ndirect comm_mask (env 0):\n{d[0].int().cpu().numpy()}")
    print(f"group   comm_mask (env 0):\n{on['group'][0].int().cpu().numpy()}")

    # --- 1. the chain is what we think it is, in EVERY env, and the closure is correct ----------
    assert bool((d[:, 0, 1] & d[:, 1, 2]).all()), "A-B and B-C must be linked; adjust comm_range"
    assert not bool(d[:, 0, 2].any()), "A-C must NOT be linked; adjust comm_range"
    expect = torch.ones((N_ENVS, M, M), dtype=torch.bool, device=DEV)
    assert torch.equal(on["group"], expect), "closure of a connected chain must be all-True"
    assert torch.equal(on["group"], on["group"].transpose(1, 2)), "closure must be symmetric"
    assert bool(on["group"][:, torch.arange(M), torch.arange(M)].all()), "diagonal must be True"
    print("1. closure of A-B-C is all-True, symmetric, reflexive                    OK")

    # --- 2. one step, one map ------------------------------------------------------------------
    lo = on["lo"]
    same_on = torch.allclose(lo[:, 0], lo[:, 1]) and torch.allclose(lo[:, 1], lo[:, 2])
    lo_off = off["lo"]
    ac_off = torch.allclose(lo_off[:, 0], lo_off[:, 2])
    known = lambda t: (t.abs() > 1e-6).flatten(1).sum(-1)   # noqa: E731
    print(f"   known cells/agent  relay ON : {known(lo.view(N_ENVS * M, -1)).view(N_ENVS, M)[0].tolist()}")
    print(f"   known cells/agent  relay OFF: {known(lo_off.view(N_ENVS * M, -1)).view(N_ENVS, M)[0].tolist()}")
    assert same_on, "with the relay all three must hold the same map after ONE step"
    assert not ac_off, "without the relay A and C must NOT agree (the bug being fixed)"
    print("2. one step -> A, B, C hold an identical map (and do not, with relay off) OK")

    # --- 3. A learns C's true position, clock and contact bit -----------------------------------
    true_c = on["pos"][:, 2, :]
    assert torch.allclose(on["lkp"][:, 0, 2, :], true_c), "A must hold C's TRUE position"
    assert bool((on["tlc"][:, 0, 2] == on["tlc"][:, 0, 1]).all()), "A's clock for C must be reset"
    assert bool((on["contact"] > 0.5).all()), "every agent must read contact = 1"
    assert not torch.allclose(off["lkp"][:, 0, 2, :], true_c), "relay off: A must NOT know C"
    print("3. A gets C's true position, staleness clock and contact bit             OK")

    # --- 4. the direct channel and the reward budget are untouched ------------------------------
    assert torch.equal(on["direct"], off["direct"]), "comm_mask must stay the DIRECT link"
    assert on["duty"] == off["duty"], "comm_duty_cycle must stay on the direct link"
    assert torch.equal(on["sync"], off["sync"]), "sync reward must stay on the direct link"
    print(f"4. direct mask, comm_duty ({on['duty']:.3f}) and sync reward unchanged        OK")
    print(f"   new: comm_group_duty {on['gduty']:.3f} (off {off['gduty']:.3f})  "
          f"comm_connected {on['conn']:.3f} (off {off['conn']:.3f})")

    # --- 5. M < 3 is an exact early return ------------------------------------------------------
    e2 = build(relay=True)
    e2.M = 2   # exercise the guard directly; the closure of a 2x2 mask is the identity
    m2 = torch.tensor([[[True, False], [False, True]]], device=DEV)
    assert torch.equal(e2._comm_closure(m2), m2), "M<3 must return the mask unchanged"
    print("5. M<3 closure is an exact no-op                                          OK")

    # --- 6. relay OFF is bit-exact, by construction not by sampling -----------------------------
    # step() does `comm_group = closure(mask) if cfg.comm_relay else mask`, so with the relay off
    # every consumer receives the SAME OBJECT the pairwise check returned. Nothing downstream can
    # differ, which is what lets every pre-v20 checkpoint keep reproducing its own numbers: their
    # saved env cfg has no comm_relay key, and from_ckpt_dict falls back to the dataclass False.
    seen = {}
    e3 = build(relay=False)
    orig_fuse, orig_lkp, orig_ref = e3.world.fuse_maps, e3._update_last_known_pos, e3._refresh_obs
    e3.world.fuse_maps = lambda m: (seen.__setitem__("fuse", m), orig_fuse(m))[1]
    e3._update_last_known_pos = lambda m: (seen.__setitem__("lkp", m), orig_lkp(m))[1]
    e3._refresh_obs = lambda m=None: (seen.__setitem__("obs", m), orig_ref(m))[1]
    e3._comm_check_orig = e3._comm_check
    e3._comm_check = lambda: (lambda r: (seen.__setitem__("check", r), r)[1])(e3._comm_check_orig())
    e3.step(torch.zeros((e3.N, e3.M), dtype=torch.long, device=e3.dev))
    for k in ("fuse", "lkp", "obs"):
        assert seen[k] is seen["check"], f"relay off: {k} got a different object from _comm_check"
    assert EnvCfg.from_ckpt_dict({"n_agents": 4}).comm_relay is False, \
        "a ckpt with no comm_relay key must fall back to False"
    print("6. relay OFF passes the IDENTICAL tensor everywhere; ckpt default False   OK")

    print("\nALL OK")


if __name__ == "__main__":
    main()
