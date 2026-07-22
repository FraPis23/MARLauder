"""Minimal belief-EXPANSION GIF — watch the teammate-position belief bloom.

Purpose-built to answer one question: how does the uniform geodesic-ball belief
(env/teammate_belief.py) SPREAD from the last-known teammate node once contact is lost?

Scenario (default): agent 0 is a STATIONARY observer, agent 1 walks away. While they are in
comm the belief is a single point (the truth). The step comm breaks, the belief is BORN at the
last-known node and then EXPANDS one hop per env step over the optimistic (known-free ∪ frontier→
unknown) graph. We render agent 0's belief of agent 1.

Nodes are colored by the STEP THEY ENTERED the zone (a birth-step gradient), so the frame shows
concentric wavefronts — bright = just reached, dark = reached long ago. Unreached nodes stay dim.
Markers: ◯ observer (a0, cyan) · ● teammate truth (a1, lime) · ✕ last-known node (white).

    python scripts/viz_belief_expand.py --split train/difficult --map-idx 30 --steps 160 \
        --comm-range 40 --out test/belief/belief_expand_clean.gif

Env-side only — no checkpoint needed. Any split works; open maps give round rings, corridor maps
show the belief hugging the walls (both correct).
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
import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/difficult")
    ap.add_argument("--map-idx", type=int, default=30)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--comm-range", type=float, default=40.0,
                    help="lower → contact breaks sooner → belief is born + expands earlier")
    ap.add_argument("--observer", type=int, default=0, help="which agent's belief to show")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("test/belief/belief_expand_clean.gif"))
    args = ap.parse_args()

    obs_i, tm_j = args.observer, 1 - args.observer

    cfg = EnvCfg.from_ckpt_dict({}, n_envs=1, n_agents=2, max_episode_steps=args.steps + 1,
                                use_teammate_belief=True, comm_model="los",
                                comm_range_px=float(args.comm_range))
    split = load_split(args.split, device=args.device)
    env = Explorer(split, cfg, seed=int(args.map_idx))
    env.store_render_global = True
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))

    obs = env._last_obs
    e = 0
    K = env.graph.edge_idx_static.shape[1]
    node_xy = env.graph.node_xy.cpu().numpy()                     # [N_max, 2] pixel coords
    eidx_static = env.graph.edge_idx_static
    gt = None
    if hasattr(env.world, "gt_torch"):
        gt = (env.world.gt_torch[e] == 1).cpu().numpy().astype(np.float32)   # free space mask

    birth = np.full(node_xy.shape[0], -1, dtype=np.int32)         # step each node joined the zone
    last_move = -1
    frames = []

    for t in range(args.steps):
        # observer HOLDS (masked action → env keeps it put); teammate walks the valid neighbour
        # farthest from the observer, no immediate backtrack → it reliably loses contact and leaves.
        am = obs["action_mask"][e].bool()                         # [M, K]
        obs_xy = env.pos[e, obs_i]
        inv = (~am[obs_i]).nonzero()
        a_obs = int(inv[0]) if inv.numel() else 0                 # hold
        curr_j = int(env.curr_idx_global[e, tm_j])
        nbrs = eidx_static[curr_j]
        nbr_xy = node_xy[nbrs.clamp(min=0).cpu().numpy()]
        dist_away = ((torch.tensor(nbr_xy, device=obs_xy.device) - obs_xy).pow(2).sum(-1))
        valid_j = am[tm_j].clone()
        if last_move >= 0:
            valid_j[K - 1 - last_move] = False
        score = torch.where(valid_j, dist_away, torch.full_like(dist_away, -1e9))
        a_j = int(score.argmax())
        last_move = a_j
        action = torch.zeros((1, 2), dtype=torch.long, device=am.device)
        action[0, obs_i] = a_obs
        action[0, tm_j] = a_j
        obs, _, done, _ = env.step(action)

        rg = env._render_global
        bp = rg["belief_p"][e].cpu().numpy()                      # [M, M, N_max]
        alive = bool(rg["belief_alive"][e][obs_i, tm_j])
        comm = bool(rg["comm_mask"][e][obs_i, tm_j])
        p = bp[obs_i, tm_j]                                       # observer's belief of teammate
        in_zone = p > 1e-9
        newly = in_zone & (birth < 0)
        birth[newly] = t

        # ---- render -------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 7), dpi=110)
        ax.set_facecolor("#0a0a12")
        if gt is not None:
            ax.imshow(gt, cmap="Greys", alpha=0.10, origin="upper",
                      extent=[0, gt.shape[1], gt.shape[0], 0], zorder=0)
        # dim all lattice nodes for context
        ax.scatter(node_xy[:, 0], node_xy[:, 1], s=3, c="#26263a", zorder=1)
        # belief zone colored by recency of entry (birth step). Recent = bright.
        zi = np.where(in_zone)[0]
        if zi.size:
            age = t - birth[zi]                                  # 0 = just entered
            ax.scatter(node_xy[zi, 0], node_xy[zi, 1], s=14, c=age, cmap="plasma_r",
                       vmin=0, vmax=max(1, t), zorder=2)
        # markers
        oxy = env.pos[e, obs_i].cpu().numpy()
        txy = env.pos[e, tm_j].cpu().numpy()
        lkp = env.last_known_pos[e, obs_i, tm_j].cpu().numpy()
        ax.scatter(*oxy, s=120, facecolors="none", edgecolors="#22d3ee", linewidths=2.2,
                   zorder=4, label=f"observer a{obs_i}")
        ax.scatter(*txy, s=70, c="#a3e635", edgecolors="k", linewidths=0.6,
                   zorder=5, label=f"teammate a{tm_j} (truth)")
        ax.scatter(*lkp, s=110, c="white", marker="x", linewidths=2.0,
                   zorder=6, label="last-known node")
        state = "IN COMM (belief = point)" if comm else (
            f"OUT OF COMM — belief zone: {int(in_zone.sum())} nodes" if alive else "LOST")
        ax.set_title(f"a{obs_i} → a{tm_j} teammate belief   t={t}   {state}",
                     color="w", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.3, labelcolor="w")
        pad = 30
        vx, vy = node_xy[:, 0], node_xy[:, 1]
        ax.set_xlim(vx.min() - pad, vx.max() + pad)
        ax.set_ylim(vy.max() + pad, vy.min() - pad)               # image y-down
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
        plt.close(fig)
        if bool(done[e]) if torch.is_tensor(done) else bool(done):
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=100, loop=0)
    print(f"[save] {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
