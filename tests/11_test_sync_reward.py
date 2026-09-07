"""The sync-event reward must pay for a real map exchange and for nothing else.

_sync_rewards is the one piece of v11 whose behavior is not obvious from reading it, and every
guard in it exists to close a specific farming exploit. The set-difference maps are supplied
directly (synthetic, exact) so each property is checked in isolation rather than hoping a random
policy happens to produce a separation; a final integration case then confirms the term is really
wired into step()'s reward.

  A. TETHER — a pair that stays in contact is paid on the rising edge and never again. This is the
     exploit that matters: with ss_thresh=-70 the radio reaches 150-310 px while the two 80 px
     LiDAR disks separate at 160 px, so "walk in parallel at the comm boundary" gives continuous
     comm with disjoint sensing. Under any per-step transfer reward that is the optimal policy.
  B. REAL SYNC — the payment equals |M_i \\ M_j| / scan_norm on the pre-fusion maps, and give/recv
     are mirror images across the pair.
  C. MIN GAP — a contact sooner than sync_min_gap pays zero (the maps still fuse: the guard
     suppresses the payment, not the physics), and a spawn contact never pays.
  D. CONSERVATION — syncing at t1 and again at t2 pays exactly what syncing only at t2 pays. This
     is why the reward cannot be farmed by frequency; it can only be forfeited by never syncing.
  E. INTEGRATION — reward_terms["sync"] appears in step() and matches ζ_g·(give + ρ·recv).

    python tests/11_test_sync_reward.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split

ZG, RHO, GAP = 0.25, 0.5, 32
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(name)


def make_env(split, device: str, **over) -> Explorer:
    cfg = EnvCfg(n_envs=1, n_agents=2, n_hops=6, max_episode_steps=4096,
                 sync_give_weight=ZG, sync_recv_ratio=RHO, sync_min_gap=GAP,
                 map_seed=0, **over)
    return Explorer(split, cfg, seed=0)


def comm_on(env: Explorer) -> torch.Tensor:
    return torch.ones((1, 2, 2), dtype=torch.bool, device=env.dev)


def maps(env: Explorer, n0: int, n1: int, overlap: int) -> torch.Tensor:
    """Synthetic per-agent FREE-node masks: agent 0 owns nodes [0, n0), agent 1 owns
    [n0 - overlap, n0 - overlap + n1). So |M_0 \\ M_1| = n0 - overlap and |M_1 \\ M_0| = n1 - overlap."""
    m = torch.zeros((1, 2, env.N_max), dtype=torch.bool, device=env.dev)
    m[0, 0, :n0] = True
    m[0, 1, n0 - overlap:n0 - overlap + n1] = True
    return m


def arm(env: Explorer, t: int) -> None:
    """Put the pair out of contact at step t, far enough past the last paid sync to be payable."""
    env.t = torch.full_like(env.t, t)
    env._comm_prev_sync.zero_()
    env._sync_t_last_paid.zero_()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    split = load_split(args.split, device=args.device)
    sn = None

    # ---- A. Permanent tether pays once, then zero forever -------------------------------------
    print("A. tether (never breaks contact)")
    env = make_env(split, args.device)
    arm(env, GAP + 1)
    paid_steps = 0
    for _ in range(80):
        _, _, p = env._sync_rewards(comm_on(env), maps(env, 400, 400, 0))
        env.t = env.t + 1
        paid_steps += int(p.sum().item() > 0)
    check("paid exactly once over 80 tethered steps", paid_steps == 1, f"paid_steps={paid_steps}")

    # ---- B. A real sync pays the set difference ------------------------------------------------
    print("B. genuine separation then contact")
    env = make_env(split, args.device)
    sn = float(env.cfg.scan_norm_nodes)
    arm(env, GAP + 1)
    m = maps(env, n0=400, n1=300, overlap=100)     # give=300, recv=200 nodes
    g, r, p = env._sync_rewards(comm_on(env), m)
    check("payment fires", float(p[0, 0].item()) == 1.0)
    check("give == |M_0 \\ M_1| / scan_norm", abs(float(g[0, 0]) - 300.0 / sn) < 1e-5,
          f"got {float(g[0,0]):.4f} want {300.0/sn:.4f}")
    check("recv == |M_1 \\ M_0| / scan_norm", abs(float(r[0, 0]) - 200.0 / sn) < 1e-5,
          f"got {float(r[0,0]):.4f} want {200.0/sn:.4f}")
    check("agent 0's give == agent 1's recv (mirror)", abs(float(g[0, 0]) - float(r[0, 1])) < 1e-6)
    print(f"      → pays agent 0 {ZG * (float(g[0,0]) + RHO * float(r[0,0])):+.3f}; a realistic "
          f"200-step surplus (~340 nodes) pays {ZG * (340/sn) * (1 + RHO):+.3f} "
          f"vs a ~1.7-1.9 detour cost")

    # ---- C. Min-gap + spawn guard --------------------------------------------------------------
    print("C. min-gap and spawn guards")
    env = make_env(split, args.device)
    arm(env, GAP + 1)
    env._sync_rewards(comm_on(env), maps(env, 400, 300, 100))          # paid
    env._comm_prev_sync.zero_()                                        # fresh rising edge...
    env.t = env.t + 3                                                  # ...only 3 steps later
    g2, r2, _ = env._sync_rewards(comm_on(env), maps(env, 800, 600, 100))
    check("re-contact inside min_gap pays nothing", float((g2 + r2).sum().item()) == 0.0)
    env3 = make_env(split, args.device)                                # fresh episode, t=0
    g3, r3, _ = env3._sync_rewards(comm_on(env3), maps(env3, 400, 300, 100))
    check("spawn contact pays nothing (they start synced)", float((g3 + r3).sum().item()) == 0.0)
    # The physics is untouched by the guard: fusion is step()'s job, not _sync_rewards'.
    env4 = make_env(split, args.device)
    before = env4.world.occupancy_torch.clone()
    env4.world.fuse_maps(comm_on(env4))
    check("fusion still happens regardless of payment",
          bool(torch.equal(env4.world.occupancy_torch[0, 0], env4.world.occupancy_torch[0, 1]))
          or not torch.equal(before, env4.world.occupancy_torch))

    # ---- D. Conservation: sync twice == sync once at the end -----------------------------------
    print("D. conservation (frequency cannot be farmed)")
    # Agent 0 discovers A = [0,300) by t1 and B = [300,700) by t2; agent 1 discovers nothing.
    A, B = 300, 400
    env = make_env(split, args.device)
    arm(env, GAP + 1)
    m_end = maps(env, n0=A + B, n1=0, overlap=0)
    once = float(env._sync_rewards(comm_on(env), m_end)[0][0, 0].item())
    env = make_env(split, args.device)
    arm(env, GAP + 1)
    g_a = float(env._sync_rewards(comm_on(env), maps(env, n0=A, n1=0, overlap=0))[0][0, 0].item())
    # After that sync both hold A; agent 0 then adds B, so the second delivery is B alone.
    env._comm_prev_sync.zero_()
    env.t = env.t + GAP + 1
    m2 = maps(env, n0=A + B, n1=0, overlap=0)
    m2[0, 1, :A] = True                                    # agent 1 received A at the first sync
    g_b = float(env._sync_rewards(comm_on(env), m2)[0][0, 0].item())
    twice = g_a + g_b
    check("total give identical whether they sync once or twice",
          abs(once - twice) < 1e-5, f"once={once:.4f} twice={twice:.4f} ({g_a:.4f}+{g_b:.4f})")

    # ---- E. Wired into step()'s reward ---------------------------------------------------------
    print("E. integration with step()")
    env = make_env(split, args.device)
    seen_key = False
    for _ in range(5):
        _, _, _, info = env.step(torch.randint(0, 8, (1, 2), device=env.dev))
        seen_key = "sync" in info["reward_terms"]
    check("reward_terms['sync'] present every step", seen_key)
    env = make_env(split, args.device)
    env.t = torch.full_like(env.t, GAP + 1)
    env._comm_prev_sync.zero_()
    env._sync_t_last_paid.zero_()
    env.cfg.force_full_comm = True                          # guarantee a rising edge on this step
    _, rew, _, info = env.step(torch.randint(0, 8, (1, 2), device=env.dev))
    st = float(info["reward_terms"]["sync"].item())
    gd = float(info["reward_terms"]["sync_give_diag"].item())
    rd = float(info["reward_terms"]["sync_recv_diag"].item())
    check("reward_terms['sync'] == ζ_g·(give + ρ·recv)", abs(st - ZG * (gd + RHO * rd)) < 1e-6,
          f"sync={st:.6f} give={gd:.6f} recv={rd:.6f}")
    check("a paid sync was recorded", float(info["reward_terms"]["sync_events"].item()) > 0.0)

    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        sys.exit(1)
    print("all sync-reward properties hold")


if __name__ == "__main__":
    main()
