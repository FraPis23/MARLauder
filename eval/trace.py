"""Capture a full per-step decision trace of one episode for the web inspector.

Writes  <out_root>/traces/<tag>/trace.json  + frames/*.png, and maintains
<out_root>/traces/index.json (the episode picker the inspector reads). Used by
scripts/trace_episode.py (CLI) and the training driver (per-checkpoint dumps).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from env.explorer import EnvCfg, Explorer
from env.frontier import compute_frontier
from eval.render import shade_occupancy_prob, paint_frontier
from models.actor_critic import K

_REPO = Path(__file__).resolve().parent.parent
_KOFF = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]


def _r(x, nd=4):
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


@torch.no_grad()
def capture_trace(model, split, env_cfg_dict: dict, n_agents: int, map_idx: int,
                  n_steps: int, out_root: Path, tag: str, device: str) -> dict:
    """Run one deterministic episode, dump trace.json + frames under out_root/traces/<tag>/."""
    env_cfg = EnvCfg.from_ckpt_dict(env_cfg_dict, n_envs=1, n_agents=n_agents,
                                    max_episode_steps=n_steps + 1)
    env = Explorer(split, env_cfg, seed=int(map_idx))
    env.store_render_global = True
    env.reload_map(env_idx=0, map_idx=int(map_idx))

    was_training = model.training
    model.eval()

    M = n_agents
    tdir = Path(out_root) / "traces" / tag
    fdir = tdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)

    h_act, h_crit = model.init_hidden(1, device)
    obs = env.obs
    rg = env._render_global
    node_xy = rg["node_xy"].cpu().numpy()
    edge_idx = rg["edge_idx"].cpu().numpy()
    H, W = env.H, env.W
    nr = float(env.cfg.nr)

    steps = []
    for t in range(n_steps):
        gate = model._strategic_gate(obs, M)
        out = model.act(obs, h_act, h_crit, deterministic=True, nr=nr)
        rg = env._render_global
        nf = rg["node_feat"][0].cpu().numpy()
        nv = rg["node_valid"][0].cpu().numpy()
        ev = rg["edge_valid"][0].cpu().numpy()
        curr = rg["curr_idx"][0].cpu().numpy()
        tgt = rg["target"][0].cpu().numpy()
        gd = rg["guidepost_dist"][0].cpu().numpy()
        logits = out["logits"][0].cpu().numpy()
        value = float(out["value"][0].cpu().numpy())
        action = out["action"][0].cpu().numpy()
        gp_bias = obs["guidepost_nbr_bias"][0].cpu().numpy()
        amask = obs["action_mask"][0].cpu().numpy()
        gate_np = gate.cpu().numpy().reshape(M) if gate is not None else np.zeros(M)

        for a in range(M):
            prob = torch.sigmoid(env.world.occupancy_logodds_torch[0, a]).cpu().numpy()
            fr = compute_frontier(env.world.occupancy_torch[0:1, a])[0].cpu().numpy()
            Image.fromarray(paint_frontier(shade_occupancy_prob(prob), fr)).save(
                fdir / f"a{a}_t{t:04d}.png")

        obs, reward, done, info = env.step(out["action"], target_choice=out["target_argmax"])
        dbg = env._dbg_reward or {}

        rec = {"t": t, "agents": []}
        for a in range(M):
            vidx = np.nonzero(nv[a])[0]
            nodes = [{
                "i": int(n), "x": _r(node_xy[n, 0], 1), "y": _r(node_xy[n, 1], 1),
                "util": _r(nf[a, n, 2], 4), "age": _r(nf[a, n, 3], 3),
                "team": _r(nf[a, n, 4], 3), "gp": _r(nf[a, n, 5], 1),
                "bf": _r(gd[a, n] / nr, 2),
            } for n in vidx.tolist()]
            edges = []
            for s in vidx.tolist():
                for k in range(K):
                    if ev[a, s, k]:
                        d = int(edge_idx[s, k])
                        if 0 <= d and s < d:
                            edges.append([int(s), d])
            cur = int(curr[a])
            rew = {kk: _r(v[0, a]) for kk, v in dbg.items()} if dbg else {}
            rec["agents"].append({
                "frame": f"frames/a{a}_t{t:04d}.png", "curr": cur, "target": int(tgt[a]),
                "pos": [_r(env.pos[0, a, 0].item(), 1), _r(env.pos[0, a, 1].item(), 1)],
                "value": _r(value), "gate": int(round(float(gate_np[a]))),
                "action": int(action[a]),
                "logits": [_r(x, 3) for x in logits[a].tolist()],
                "action_mask": [int(b) for b in amask[a].tolist()],
                "guidepost_dir": [_r(x, 2) for x in gp_bias[a].tolist()],
                "curr_nbrs": [int(edge_idx[cur, k]) for k in range(K)],
                "reward": rew, "nodes": nodes, "edges": edges,
            })
        h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
        steps.append(rec)
        if bool(done[0].item()):
            break

    try:
        union = (env.world.occupancy_torch == 1).any(dim=1).view(1, -1).sum().item()
        explored = union / float(env.free_total[0].item())
    except Exception:
        explored = None

    meta = {"tag": tag, "map_idx": map_idx, "n_agents": M, "H": H, "W": W, "nr": nr,
            "n_steps": len(steps), "K_OFFSETS": _KOFF,
            "final_explored": _r(explored) if explored is not None else None,
            "gate_eps": getattr(model, "strategic_gate_eps", 0.0),
            "target_mode": getattr(model, "target_mode", "analytic")}
    (tdir / "trace.json").write_text(json.dumps({"meta": meta, "steps": steps}))

    # maintain the episode picker index + ensure the viewer is present
    idx_path = Path(out_root) / "traces" / "index.json"
    try:
        index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    except Exception:
        index = []
    entry = {"tag": tag, "map_idx": map_idx, "n_steps": len(steps),
             "final_explored": meta["final_explored"]}
    index = [e for e in index if e.get("tag") != tag] + [entry]
    idx_path.write_text(json.dumps(index, indent=1))
    viewer = _REPO / "viz" / "inspector.html"
    if viewer.exists():
        (Path(out_root) / "inspector.html").write_bytes(viewer.read_bytes())

    if was_training:
        model.train()
    print(f"[trace] {tag}  steps={len(steps)}  explored={meta['final_explored']}  → {tdir}")
    return meta
