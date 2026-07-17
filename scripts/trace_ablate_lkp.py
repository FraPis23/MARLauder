"""Reconstruct a saved trace episode identically up to a given step, then continue with the
teammate's last-known-position channel removed from the actor's observation (ablation).

Bit-identical to scripts/eval_ckpt.py's capture_trace() up to --split-step: same seed (=map_idx),
same map, same model, same deterministic policy. At --split-step the env's teammate_obs flag is
flipped off (env/explorer.py EnvCfg.teammate_obs), which zeros node_feat[4] (teammate potential
field) and node_feat[6] (radar-teammate) for the rest of the rollout — the same ablation hook the
training code already uses for teammate_obs=False checkpoints (env/explorer.py:1162-1195). Only
the ACTOR's obs changes; env physics/comm bookkeeping (last_known_pos itself, comm_mask) keep
updating normally underneath, they just stop being read into node_feat.

    python scripts/trace_ablate_lkp.py --ckpt runs/RUN/ckpt_best.pt --split test/complex \\
        --map-idx 499 --steps 512 --split-step 280 --out runs/RUN --tag ckpt_best_complex_m2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from PIL import Image

from env.explorer import EnvCfg, Explorer
from env.frontier import compute_frontier
from eval.ckpt_loader import load_model_from_ckpt
from eval.render import shade_occupancy_prob, paint_frontier
from env.maps import load_split
from models.actor_critic import K

_KOFF = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
F_NAMES = ["x_rel", "y_rel", "utility", "age", "teammate_pot", "radar_util", "radar_teammate"]


def _r(x, nd=4):
    v = float(x)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


def _render_union_rgb(env) -> np.ndarray:
    occ = env.world.occupancy_torch[0]
    free_any = (occ == 1).any(dim=0)
    obst_any = (occ == 2).any(dim=0)
    union = torch.zeros_like(occ[0])
    union[obst_any] = 2
    union[free_any] = 1
    prob = torch.sigmoid(env.world.occupancy_logodds_torch[0].max(dim=0).values).cpu().numpy()
    fr = compute_frontier(union.unsqueeze(0).to(torch.uint8))[0].cpu().numpy()
    return paint_frontier(shade_occupancy_prob(prob), fr)


@torch.no_grad()
def run(model, split, env_cfg_dict: dict, n_agents: int, map_idx: int, n_steps: int,
        split_step: int, out_root: Path, tag: str, device: str) -> dict:
    env_cfg = EnvCfg.from_ckpt_dict(env_cfg_dict, n_envs=1, n_agents=n_agents,
                                    max_episode_steps=n_steps + 1)
    env = Explorer(split, env_cfg, seed=int(map_idx))
    env.store_render_global = True
    env.reload_map(env_idx=0, map_idx=int(map_idx))

    complete_thresh = float(env.cfg.done_explored_thresh)
    env.cfg.done_explored_thresh = 2.0
    env.cfg.max_episode_steps = n_steps + 5

    was_training = model.training
    model.eval()
    _compiled_enc = getattr(model.encoder, "_orig_mod", None)
    if _compiled_enc is not None:
        compiled_wrapper = model.encoder
        model.encoder = _compiled_enc
    model.store_logit_components = True
    model.encoder.store_attn = True

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
    agent_frames = [[] for _ in range(M)]
    explored = None
    completed = False
    gdir = tdir / "gat"
    gdir_ready = False
    has_contrib = False
    for t in range(n_steps):
        if t == split_step:
            env.cfg.teammate_obs = False
            print(f"[trace_ablate_lkp] t={t}: teammate_obs -> False (last_known_pos removed from actor obs)")
        out = model.act(obs, h_act, h_crit, deterministic=True)
        rg = env._render_global
        nf = rg["node_feat"][0].cpu().numpy()
        nv = rg["node_valid"][0].cpu().numpy()
        ub = rg["util_boundary"][0].cpu().numpy()
        uv = rg["util_volume"][0].cpu().numpy()
        ev = rg["edge_valid"][0].cpu().numpy()
        curr = rg["curr_idx"][0].cpu().numpy()
        logits = out["logits"][0].cpu().numpy()
        value = float(out["value"][0].cpu().numpy())
        action = out["action"][0].cpu().numpy()
        amask = obs["action_mask"][0].cpu().numpy()
        pos_all = rg["pos"][0].cpu().numpy()
        lkp = rg["last_known_pos"][0].cpu().numpy()
        cmask = rg["comm_mask"][0].cpu().numpy()

        dbg = model._dbg_logits or {}
        enc_attn = dbg.get("enc_attn")
        enc_contrib = dbg.get("enc_contrib")
        l2g = obs["local_to_global"][0].cpu().numpy()
        ei_loc = obs["edge_idx"][0].cpu().numpy()
        ev_loc = obs["edge_valid"][0].cpu().numpy()
        gat_agents = None
        if enc_attn is not None:
            Ln = len(enc_attn)
            gat_agents = []
            for a in range(M):
                loc = np.array([i for i in range(l2g.shape[1]) if int(l2g[a, i]) >= 0], dtype=np.int64)
                nodes = l2g[a, loc].astype(int).tolist()
                nbr = np.full((loc.size, K), -1, dtype=np.int64)
                for c, i in enumerate(loc.tolist()):
                    for k in range(K):
                        if ev_loc[a, i, k]:
                            nbr[c, k] = int(l2g[a, int(ei_loc[a, i, k])])
                w_layers = []
                c_layers = []
                for l in range(Ln):
                    al = np.round(enc_attn[l][0, a].cpu().numpy()[loc], 3)
                    per_head = [np.concatenate([al[:, K:K + 1, h], al[:, :K, h]], axis=1).tolist()
                                for h in range(al.shape[-1])]
                    w_layers.append(per_head)
                    if enc_contrib is not None:
                        cl = enc_contrib[l][0, a].cpu().numpy()[loc]
                        cl = np.round(np.concatenate([cl[:, K:K + 1], cl[:, :K]], axis=1), 4)
                        c_layers.append(cl.tolist())
                gat_agents.append({"node": nodes, "nbr": nbr.tolist(), "w": w_layers,
                                   "c": (c_layers if enc_contrib is not None else None)})
            if enc_contrib is not None:
                has_contrib = True
            if not gdir_ready:
                gdir.mkdir(parents=True, exist_ok=True); gdir_ready = True
            (gdir / f"{t:04d}.json").write_text(json.dumps(gat_agents))

        for a in range(M):
            prob = torch.sigmoid(env.world.occupancy_logodds_torch[0, a]).cpu().numpy()
            fr = compute_frontier(env.world.occupancy_torch[0:1, a])[0].cpu().numpy()
            agent_frames[a].append(paint_frontier(shade_occupancy_prob(prob), fr))

        obs, reward, done, info = env.step(out["action"])
        dbg = env._dbg_reward or {}

        rec = {"t": t, "agents": []}
        for a in range(M):
            vidx = np.nonzero(nv[a])[0]
            nodes = [{
                "i": int(n), "x": _r(node_xy[n, 0], 1), "y": _r(node_xy[n, 1], 1),
                "util": _r(nf[a, n, 2], 4), "age": _r(nf[a, n, 3], 3),
                "team": _r(nf[a, n, 4], 3),
                "bu": _r(nf[a, n, 5], 3), "bt": _r(nf[a, n, 6], 3),
                "ub": _r(ub[a, n], 3), "uv": _r(uv[a, n], 3),
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
            teammates = [{
                "j": int(j), "comm": int(cmask[a, j]),
                "est": [_r(lkp[a, j, 0], 1), _r(lkp[a, j, 1], 1)],
                "true": [_r(pos_all[j, 0], 1), _r(pos_all[j, 1], 1)],
            } for j in range(M) if j != a]
            curr_nbrs = [int(edge_idx[cur, k]) for k in range(K)]
            rec["agents"].append({
                "frame": f"frames/a{a}.gif", "fi": t, "curr": cur,
                "pos": [_r(pos_all[a, 0], 1), _r(pos_all[a, 1], 1)],
                "value": _r(value),
                "action": int(action[a]),
                "logits": [_r(x, 3) for x in logits[a].tolist()],
                "action_mask": [int(b) for b in amask[a].tolist()],
                "curr_nbrs": curr_nbrs,
                "reward": rew, "nodes": nodes, "edges": edges,
                "teammates": teammates,
                "has_gat": gat_agents is not None,
                "teammate_obs": bool(env.cfg.teammate_obs),
            })
        explored = float(info["explored_rate"][0].item())
        h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
        steps.append(rec)
        if explored >= complete_thresh:
            completed = True
            break

    if completed:
        union_rgb = _render_union_rgb(env)
        for a in range(M):
            agent_frames[a].append(union_rgb)
        term_fi = len(agent_frames[0]) - 1
        pos_now = env.pos[0].cpu().numpy()
        term_agents = []
        for a in range(M):
            term_agents.append({
                "frame": f"frames/a{a}.gif", "fi": term_fi,
                "curr": int(steps[-1]["agents"][a]["curr"]) if steps else 0,
                "pos": [_r(pos_now[a, 0], 1), _r(pos_now[a, 1], 1)],
                "value": None, "action": -1,
                "logits": [None] * K, "action_mask": [0] * K,
                "curr_nbrs": [-1] * K,
                "reward": {}, "nodes": [], "edges": [], "has_gat": False,
                "teammates": [{"j": int(j), "comm": 1,
                               "est":  [_r(pos_now[j, 0], 1), _r(pos_now[j, 1], 1)],
                               "true": [_r(pos_now[j, 0], 1), _r(pos_now[j, 1], 1)]}
                              for j in range(M) if j != a],
            })
        steps.append({"t": len(steps), "agents": term_agents, "terminal": True})

    for a in range(M):
        if not agent_frames[a]:
            continue
        kept, fis = [], []
        for f in agent_frames[a]:
            if kept and np.array_equal(f, kept[-1]):
                fis.append(len(kept) - 1)
            else:
                kept.append(f); fis.append(len(kept) - 1)
        imgs = [Image.fromarray(f) for f in kept]
        imgs[0].save(fdir / f"a{a}.gif", save_all=True, append_images=imgs[1:],
                     duration=120, loop=0, disposal=1, optimize=True)
        for i, st in enumerate(steps):
            st["agents"][a]["fi"] = fis[i]

    meta = {"tag": tag, "map_idx": map_idx, "n_agents": M, "H": H, "W": W, "nr": nr,
            "n_hops": int(env.cfg.n_hops), "win": 2 * int(env.cfg.n_hops) + 3,
            "n_steps": len(steps), "K_OFFSETS": _KOFF,
            "final_explored": _r(explored) if explored is not None else None,
            "completed": bool(completed),
            "gat_layers": len(model.encoder.layers),
            "gat_heads": int(getattr(model.encoder.layers[0], "n_heads", 1)),
            "has_contrib": has_contrib,
            "head_feat_groups": (model.encoder.head_feat_groups()
                                 if hasattr(model.encoder, "head_feat_groups") else None),
            "feat_names": F_NAMES,
            "ablation": {"kind": "teammate_obs_off", "split_step": split_step}}
    (tdir / "trace.json").write_text(json.dumps({"meta": meta, "steps": steps}))

    idx_path = Path(out_root) / "traces" / "index.json"
    try:
        index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    except Exception:
        index = []
    entry = {"tag": tag, "map_idx": map_idx, "n_steps": len(steps),
             "final_explored": meta["final_explored"], "completed": bool(completed)}
    index = [e for e in index if e.get("tag") != tag] + [entry]
    idx_path.write_text(json.dumps(index, indent=1))

    model.encoder.store_attn = False
    model.store_logit_components = False
    if _compiled_enc is not None:
        model.encoder = compiled_wrapper
    if was_training:
        model.train()
    print(f"[trace_ablate_lkp] {tag}  steps={len(steps)}  explored={meta['final_explored']}  -> {tdir}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--map-idx", type=int, required=True)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--split-step", type=int, required=True,
                     help="step at which teammate_obs (last_known_pos into actor obs) is turned off")
    ap.add_argument("--n-agents", type=int, default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    model, env_peek = load_model_from_ckpt(args.ckpt, args.device, n_agents=args.n_agents)
    n_agents = int(getattr(model, "M", args.n_agents or 2))
    split = load_split(args.split, device=args.device)

    run(model, split, env_peek or {}, n_agents, args.map_idx, args.steps, args.split_step,
        args.out, args.tag, args.device)


if __name__ == "__main__":
    main()
