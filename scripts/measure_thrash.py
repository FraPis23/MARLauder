"""Measure the deterministic target-thrash / stationarity pathology on a checkpoint.

This is THE verification for the strategic-head commitment fix (cand_prev_branch_match).
Loads a checkpoint, runs one deterministic AND one stochastic episode on a fixed map, and
reports, for each:
  - mean per-step displacement (px)         — ~hop size = moving; ~0 = frozen
  - target-change rate (per agent per step) — 0.99 = thrashing; ~0.1 = committed
  - final explored fraction                 — net progress

Baseline pathology (pre-fix, iter-9 ckpt): DET target-change=0.99/step, explored=5%
                                            STO target-change=0.12/step, covers map.
Fix target: DET target-change should fall toward STO's (~0.1) and explored should climb.

Usage:
  docker exec marlauder bash -lc 'cd /workspace/MARLauder && \
    python scripts/measure_thrash.py --ckpt runs/run_default/ckpt_100.pt --map-idx 120'
"""
import argparse
import numpy as np
import torch

from env.maps import load_split
from env.explorer import EnvCfg, Explorer
from models.actor_critic import MarlActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, default=120)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ecfg = EnvCfg.from_ckpt_dict(ck["cfg"]["env"], n_envs=1, n_agents=2)
    split = load_split(args.split, device=args.device)
    env = Explorer(split, ecfg, seed=args.map_idx)
    M = ecfg.n_agents

    m = MarlActorCritic(n_agents=M).to(args.device)
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in ck["model"].items()}
    if "path_bias" in sd and "path_bias_learn" not in sd:
        sd["path_bias_learn"] = sd.pop("path_bias")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    m.eval()

    def run(determ: bool):
        env.reload_map(env_idx=0, map_idx=args.map_idx)
        ha, hc = m.init_hidden(1, args.device)
        obs = env.obs
        disp, chg, prev = [], 0, None
        explored = 0.0
        for _ in range(args.steps):
            o = m.act(obs, ha, hc, deterministic=determ)
            ta = o["target_argmax"][0].tolist()
            if prev is not None:
                chg += sum(ta[a] != prev[a] for a in range(M))
            prev = ta
            p0 = env.pos.clone()
            obs, _, d, info = env.step(o["action"], target_choice=o["target_argmax"])
            ha, hc = o["hidden_actor"], o["hidden_critic"]
            disp.append((env.pos - p0).norm(dim=-1).mean().item())
            explored = float(info["explored_rate"][0].item())
            if bool(d[0]):
                break
        return np.mean(disp), chg / (len(disp) * M), explored, len(disp)

    print(f"ckpt={args.ckpt} iter={ck.get('iter')} hop={ecfg.nr}px map={args.map_idx}")
    for determ in (True, False):
        dm, ch, ex, n = run(determ)
        lbl = "DETERMINISTIC" if determ else "STOCHASTIC   "
        print(f"{lbl}: disp/step={dm:5.2f}px  target-change={ch:4.2f}/step  "
              f"explored={ex:5.1%}  steps={n}")
    print("PASS if DETERMINISTIC target-change << 0.99 and explored >> 5%")


if __name__ == "__main__":
    main()
