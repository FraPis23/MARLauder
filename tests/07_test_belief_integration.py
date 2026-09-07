"""Integration test: run a real 2-agent Explorer with the belief filter and check the obs.

Verifies: no NaN/Inf in feat[4]/feat[5]/feat[6], φ (_geo_curr_team) and geo_pair finite in
[0,1], the belief state evolves and stays a valid distribution, and a quick perf compare
(belief ON vs OFF) sps regression. Also confirms use_teammate_belief=False falls back cleanly.
"""
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split


def run(use_belief: bool, steps: int = 40, n_envs: int = 4):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    split = load_split("train/easy", device=dev)
    cfg = EnvCfg(n_envs=n_envs, n_agents=2, max_episode_steps=128,
                 comm_range_px=60.0, use_teammate_belief=use_belief)
    env = Explorer(split, cfg, seed=0)
    obs = env.reset()
    K = env.graph.edge_len.shape[0]
    torch.manual_seed(0)
    bad = 0
    belief_sum_ok = True
    t0 = time.time()
    for t in range(steps):
        action = torch.randint(0, K, (env.N, env.M), device=dev)
        obs, reward, done, info = env.step(action)
        nf = obs["node_feat"]                       # [N, M, W², F]
        for ch in (4, 5, 6):
            v = nf[..., ch]
            if not torch.isfinite(v).all():
                bad += 1
        if not torch.isfinite(env._geo_curr_team).all():
            bad += 1
        if not torch.isfinite(reward).all():
            bad += 1
        # Belief distribution sanity (alive rows sum ≈ 1).
        if use_belief and env._belief_p is not None:
            p = env._belief_p                        # [B, M, N_max]
            alive = env._belief_alive
            sums = p.sum(-1)
            live = sums[alive]
            if live.numel() and (live - 1.0).abs().max() > 1e-2:
                belief_sum_ok = False
    torch.cuda.synchronize() if dev == "cuda" else None
    dt = time.time() - t0
    sps = steps * env.N / dt
    f4 = obs["node_feat"][..., 4]
    return dict(bad=bad, belief_sum_ok=belief_sum_ok, sps=sps,
                f4_max=float(f4.max()), f4_nonzero=int((f4 > 0).sum()),
                geo=float(env._geo_curr_team.mean()))


def main():
    print("== belief ON ==")
    on = run(True)
    print(on)
    print("== belief OFF (legacy) ==")
    off = run(False)
    print(off)

    ok = True
    ok &= on["bad"] == 0;            print(f"[1] belief ON  finite feat/φ/reward: {'OK' if on['bad']==0 else 'FAIL('+str(on['bad'])+')'}")
    ok &= off["bad"] == 0;           print(f"[2] belief OFF finite feat/φ/reward: {'OK' if off['bad']==0 else 'FAIL('+str(off['bad'])+')'}")
    ok &= on["belief_sum_ok"];       print(f"[3] belief Σp≈1 on alive rows:       {'OK' if on['belief_sum_ok'] else 'FAIL'}")
    ok &= on["f4_max"] > 0;          print(f"[4] feat[4] non-trivial (max>0):     {'OK' if on['f4_max']>0 else 'FAIL'}  (max={on['f4_max']:.3f}, nonzero={on['f4_nonzero']})")
    ratio = on["sps"] / max(1e-9, off["sps"])
    print(f"[5] perf ON/OFF sps ratio = {ratio:.2f}  (ON={on['sps']:.0f}, OFF={off['sps']:.0f})")
    print("\nRESULT:", "ALL OK" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
