"""Integration test for the PATHFRONT teammate belief (EnvCfg.belief_mode='pathfront').

Checks, over a real multi-agent rollout with agents that separate (comm breaks → hypotheses freeze):
  [1] no crash, belief_p finite, feat[4] finite
  [2] Σp ≈ 1 on every alive (observer,teammate) row, every step (transit AND bloom phases)
  [3] uniform mode still Σp≈1 (unchanged fallback)
  [4] perf: pathfront vs uniform sps ratio
  [5] hypotheses actually freeze (some rows become alive with weights summing to 1)

    docker exec marlauder python /workspace/MARLauder/scripts/11_test_pathfront_belief.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split


def run(mode: str, steps: int = 80, n_envs: int = 8, device: str = "cuda:0"):
    split = load_split("train/difficult", device=device)
    cfg = EnvCfg.from_ckpt_dict({}, n_envs=n_envs, n_agents=2, max_episode_steps=steps + 2,
                                use_teammate_belief=True, comm_model="los",
                                comm_range_px=35.0, belief_mode=mode)
    env = Explorer(split, cfg, seed=7)
    g = torch.Generator(device=device).manual_seed(0)
    bad = 0
    min_sum = 9.0
    max_alive = 0
    feat_bad = 0
    t0 = time.time()
    for t in range(steps):
        a = torch.randint(0, 8, (n_envs, 2), device=device, generator=g)
        obs, r, d, info = env.step(a)
        bp = env._belief_p                       # [B, M, N_max]
        al = env._belief_alive                   # [B, M]
        if bp is None or not torch.isfinite(bp).all():
            bad += 1
            continue
        if "node_feat" in obs and not torch.isfinite(obs["node_feat"]).all():
            feat_bad += 1
        s = bp.sum(-1)[al]                        # Σp over nodes, alive rows only
        if s.numel():
            min_sum = min(min_sum, float(s.min()))
            max_alive = max(max_alive, int(al.sum()))
    sps = (steps * n_envs) / (time.time() - t0)
    return dict(bad=bad, min_sum=(min_sum if max_alive else 1.0), max_alive=max_alive,
                feat_bad=feat_bad, sps=sps)


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    u = run("uniform", device=dev)
    p = run("pathfront", device=dev)
    print("uniform  ", u)
    print("pathfront", p)
    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"[{'OK' if cond else 'FAIL'}] {name}")
        ok = ok and cond
    check("pathfront no crash / finite belief", p["bad"] == 0)
    check("pathfront feat[4] finite", p["feat_bad"] == 0)
    check("pathfront Σp≈1 on alive rows (all steps)", abs(p["min_sum"] - 1.0) < 1e-3)
    check("pathfront froze hypotheses (alive rows > 0)", p["max_alive"] > 0)
    check("uniform Σp≈1 unchanged", abs(u["min_sum"] - 1.0) < 1e-3)
    print(f"[info] sps uniform={u['sps']:.0f} pathfront={p['sps']:.0f} "
          f"ratio={p['sps']/max(u['sps'],1e-6):.2f}")
    print("\nRESULT:", "ALL OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
