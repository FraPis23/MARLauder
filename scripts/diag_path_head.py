"""StrategicHead path-consistency + scoreboard diagnostic.

Two purposes (2026-06-15):
  1. VERIFY the user's claim that the committed target-path crosses walls / unknown space.
     Reconstructs the EXACT path the GIF renders (walk bf_parent_from_curr from the head's
     chosen candidate back to curr, identical to eval/rollout.py) and rasterizes every
     segment at ~1px against THAT agent's own occupancy.
  2. Seed the StrategicHead scoreboard (coverage eval/score measures none of the head's job):
       wall_cross_rate     fraction of path segments crossing >=1 KNOWN-OBSTACLE pixel  (Bug A: collision undersampling)
       unknown_cross_rate  fraction of path segments crossing >=1 UNKNOWN pixel         (Bug B: FREE graph allows UNKNOWN)
       path_known_frac     fraction of all path pixels that are known-FREE              (target = 1.0 after the conservative fix)
       target_reachable_rate fraction of steps where the head's target connects back to curr (vs no-op fallback)
       mean_path_len_px    sanity

Decentralized note: all quantities are computed from a single agent's own occupancy + its
own chosen path — nothing privileged. This is a measurement, not a train-time signal.

Usage (inside whatever container has the repo + GPU; do NOT share a GPU with a live sweep):
  PYTHONPATH=$PWD python scripts/diag_path_head.py --ckpt runs/run_div/final.pt
  PYTHONPATH=$PWD python scripts/diag_path_head.py --ckpt runs/run_div/final.pt --split train/difficult
"""
import argparse
import numpy as np
import torch

from env.maps import load_split
from env.explorer import EnvCfg, Explorer
from models.actor_critic import MarlActorCritic

# Occupancy codes (env/explorer.py): per-agent map.
_UNKNOWN = 0
_FREE = 1
_OBSTACLE = 2


def _rasterize(x0: float, y0: float, x1: float, y1: float, H: int, W: int):
    """Integer pixel coords sampled along the segment at ~1px (endpoints included)."""
    n = max(1, int(round(max(abs(x1 - x0), abs(y1 - y0)))))
    xs = np.clip(np.round(np.linspace(x0, x1, n + 1)).astype(np.int64), 0, W - 1)
    ys = np.clip(np.round(np.linspace(y0, y1, n + 1)).astype(np.int64), 0, H - 1)
    return ys, xs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, nargs="+", default=[120, 1543, 2877, 5530, 9904])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ecfg = EnvCfg.from_ckpt_dict(ck["cfg"]["env"], n_envs=1, n_agents=2)
    disable_strategic = bool(ck["cfg"].get("disable_strategic", False))
    if disable_strategic:
        print("WARNING: checkpoint has no StrategicHead (disable_strategic). "
              "Path is the legacy argmax target; head metrics are N/A.")
    split = load_split(args.split, device=args.device)
    env = Explorer(split, ecfg, seed=args.map_idx[0])

    m = MarlActorCritic(n_agents=2, disable_strategic=disable_strategic).to(args.device)
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in ck["model"].items()}
    if "path_bias" in sd and "path_bias_learn" not in sd:
        sd["path_bias_learn"] = sd.pop("path_bias")
    m.load_state_dict(sd, strict=False)
    m.eval()

    M = 2
    H, W = env.H, env.W
    node_xy = env.graph.node_xy                                    # [N_max, 2]

    seg_tot = 0
    seg_wall = 0          # segments crossing >=1 OBSTACLE pixel
    seg_unknown = 0       # segments crossing >=1 UNKNOWN pixel
    px_tot = 0
    px_free = 0
    reach_tot = 0
    reach_ok = 0
    path_len_sum = 0.0
    path_len_n = 0

    for midx in args.map_idx:
        env.reload_map(env_idx=0, map_idx=int(midx))
        ha, hc = m.init_hidden(1, args.device)
        for _ in range(args.steps):
            obs = env.obs
            out = m.act(obs, ha, hc, deterministic=True)
            tgt = out["target_choice"]                            # [N, M]
            cand_idx = obs["cand_idx"]                            # [N, M, K]
            curr_g = obs["curr_idx_global"]                      # [N, M]
            bf_par = obs["bf_parent_from_curr"]                  # [N, M, N_max]
            K_cand = cand_idx.shape[-1]

            for ag in range(M):
                k_slot = int(tgt[0, ag].item())
                if not (0 <= k_slot < K_cand):
                    continue
                cand_global = int(cand_idx[0, ag, k_slot].item())
                curr_global = int(curr_g[0, ag].item())
                if cand_global < 0:
                    continue
                # Walk BF parent cand -> curr (same as eval/rollout.py).
                path_nodes = [cand_global]
                cur_n = cand_global
                reached = (cand_global == curr_global)
                for _ in range(200):
                    par_n = int(bf_par[0, ag, cur_n].item())
                    if par_n < 0 or par_n == cur_n:
                        break
                    path_nodes.append(par_n)
                    if par_n == curr_global:
                        reached = True
                        break
                    cur_n = par_n
                path_nodes.reverse()                              # curr -> ... -> cand
                reach_tot += 1
                reach_ok += int(reached)

                # Rasterize each segment against THIS agent's occupancy.
                occ = env.world.occupancy_torch[0, ag].cpu().numpy()   # [H, W] uint8
                xy = [(float(node_xy[n, 0].item()), float(node_xy[n, 1].item())) for n in path_nodes]
                for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
                    path_len_sum += float(np.hypot(x1 - x0, y1 - y0))
                    path_len_n += 1
                    ys, xs = _rasterize(x0, y0, x1, y1, H, W)
                    vals = occ[ys, xs]
                    seg_tot += 1
                    seg_wall += int((vals == _OBSTACLE).any())
                    seg_unknown += int((vals == _UNKNOWN).any())
                    px_tot += vals.size
                    px_free += int((vals == _FREE).sum())

            obs, _r, d, _info = env.step(out["action"], target_choice=out["target_argmax"])
            ha, hc = out["hidden_actor"], out["hidden_critic"]
            if bool(d[0].item()):
                break

    def rate(a, b):
        return (a / b) if b else float("nan")

    print(f"ckpt={args.ckpt}  split={args.split}  maps={args.map_idx}  segments={seg_tot}")
    print(f"  wall_cross_rate       = {rate(seg_wall, seg_tot):.3f}   (Bug A: path crosses a KNOWN wall; target 0.0)")
    print(f"  unknown_cross_rate    = {rate(seg_unknown, seg_tot):.3f}   (Bug B: path crosses UNKNOWN; target 0.0 after fix)")
    print(f"  path_known_frac       = {rate(px_free, px_tot):.3f}   (path pixels that are known-FREE; target 1.0)")
    print(f"  target_reachable_rate = {rate(reach_ok, reach_tot):.3f}   (head target connects to curr on FREE graph)")
    print(f"  mean_path_len_px      = {rate(path_len_sum, path_len_n):.2f}")


if __name__ == "__main__":
    main()
