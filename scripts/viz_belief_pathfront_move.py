"""Belief pathfront — the OBSERVER MOVES and explores past a frontier (release / chase).

Drives the real env with belief_mode='pathfront'. Agent 0 (observer) explores outward while agent 1
walks off and holds (contact breaks → hypotheses freeze). We render agent 0's belief of agent 1 on
agent 0's own partial map, step by step. The point to watch: mass accumulates on agent 0's frontiers
(reservoirs); as agent 0 EXPLORES PAST one of those frontiers, that frontier becomes interior, its
accumulator RELEASES, and the mass flows on to the new outer frontier — the belief "chases" the
frontier outward. Nothing ever sits on unknown nodes; the known map grows and the belief follows.

    python scripts/viz_belief_pathfront_move.py --split train/difficult --map-idx 30 \
        --steps 140 --comm-range 30 --out test/belief/belief_pathfront_move.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split

_UNK, _FREE, _OBST = 0, 1, 2
_REDHEAT = LinearSegmentedColormap.from_list("redheat", [
    (0.000, "#000000"), (0.003, "#fff2a0"), (0.02, "#ffb020"), (0.10, "#e23010"), (1.00, "#7a0000")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/difficult")
    ap.add_argument("--map-idx", type=int, default=30)
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--comm-range", type=float, default=30.0)
    ap.add_argument("--vmax", type=float, default=0.03)
    ap.add_argument("--frame-ms", type=float, default=160.0)
    ap.add_argument("--out", type=Path, default=Path("test/belief/belief_pathfront_move.gif"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cfg = EnvCfg.from_ckpt_dict({}, n_envs=1, n_agents=2, max_episode_steps=args.steps + 2,
                                use_teammate_belief=True, belief_mode="pathfront",
                                comm_model="los", comm_range_px=float(args.comm_range))
    env = Explorer(load_split(args.split, device=dev), cfg, seed=int(args.map_idx))
    env.store_render_global = True
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))
    obs = env._last_obs
    e, OBS, TM = 0, 0, 1                                              # observer=agent0, teammate=agent1
    K = env.graph.edge_idx_static.shape[1]
    node_xy = env.graph.node_xy.cpu().numpy()
    eidx = env.graph.edge_idx_static
    nfx = node_xy[:, 0].astype(int).clip(0, env.W - 1)
    nfy = node_xy[:, 1].astype(int).clip(0, env.H - 1)
    last_act = [-1, -1]

    def act():
        """Agent0: frontier-seeking (value_field argmax, no backtrack) → explores OUTWARD. Agent1: walk
        directly away from agent0 for the first stretch (to break comm), then hold."""
        vf = obs["value_field"][0].clone(); am = obs["action_mask"][0].bool()
        for a in range(2):
            if last_act[a] >= 0:
                am[a, K - 1 - last_act[a]] = False
        a0 = int(torch.where(am[0], vf[0], torch.full_like(vf[0], -1e9)).argmax())
        # agent1 away from agent0
        c1 = int(env.curr_idx_global[e, TM]); nb = eidx[c1]
        nxy = node_xy[nb.clamp(min=0).cpu().numpy()]
        away = ((torch.tensor(nxy, device=dev) - env.pos[e, OBS]).pow(2).sum(-1))
        v1 = am[1].clone()
        if last_act[1] >= 0:
            v1[K - 1 - last_act[1]] = False
        a1 = int(torch.where(v1, away, torch.full_like(away, -1e9)).argmax())
        last_act[0], last_act[1] = a0, a1
        return torch.tensor([[a0, a1]], device=dev)

    frames = []
    for s in range(args.steps):
        obs, r, d, info = env.step(act())
        cat = env.world.occupancy_torch[e, OBS].cpu().numpy()[nfy, nfx]     # observer's node occupancy
        bp = env._belief_p.view(env.N, env.M, env.M, env.N_max)[e, OBS, TM].cpu().numpy()   # p over nodes
        alive = bool(env._belief_alive.view(env.N, env.M, env.M)[e, OBS, TM])
        a0xy = env.pos[e, OBS].cpu().numpy(); a1xy = env.pos[e, TM].cpu().numpy()
        lkp = env.last_known_pos[e, OBS, TM].cpu().numpy()

        fig, ax = plt.subplots(figsize=(8.8, 8), dpi=100)
        ax.set_facecolor("#07070c")
        col = np.where(cat == _FREE, "#2b2b3d", np.where(cat == _OBST, "#59201c", "#12121c"))
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=9, c=col, zorder=1)
        sc = ax.scatter(node_xy[:, 0], node_xy[:, 1], s=13, c=bp, cmap=_REDHEAT,
                        vmin=0.0, vmax=float(args.vmax), zorder=3)
        # only overlay where belief > 0 (redraw those on top so map shows elsewhere)
        hot = np.where(bp > 1e-6)[0]
        if hot.size:
            ax.scatter(node_xy[hot, 0], node_xy[hot, 1], s=15, c=bp[hot], cmap=_REDHEAT,
                       vmin=0.0, vmax=float(args.vmax), zorder=4)
        ax.scatter(*a1xy, s=70, c="#a3e635", edgecolors="k", linewidths=0.6, zorder=6)   # teammate truth
        ax.scatter(*a0xy, s=130, facecolors="none", edgecolors="#22d3ee", linewidths=2.2, zorder=6)  # observer
        ax.scatter(*lkp, s=110, c="white", marker="x", linewidths=2.0, zorder=7)          # last-known
        expl = float((cat != _UNK).mean())
        ax.set_title(f"PATHFRONT (observer moving)  step={s}  explored={expl:.0%}  "
                     f"Σp={bp.sum():.2f}  {'ALIVE' if alive else '—'}   "
                     f"cyan=observer  lime=teammate  ✕=last-known", color="w", fontsize=9)
        pad = 20
        ax.set_xlim(node_xy[:, 0].min() - pad, node_xy[:, 0].max() + pad)
        ax.set_ylim(node_xy[:, 1].max() + pad, node_xy[:, 1].min() - pad)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(f"prob (fixed 0→{args.vmax})", color="w", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="w", labelsize=7); plt.setp(cb.ax.get_yticklabels(), color="w")
        fig.tight_layout(); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
        plt.close(fig)
        if bool(d[e]) if torch.is_tensor(d) else bool(d):
            break
    frames += [frames[-1]] * 15

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=float(args.frame_ms), loop=0)
    print(f"[save] {args.out}  ({len(frames)} frames @ {args.frame_ms}ms)")


if __name__ == "__main__":
    main()
