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


def _render_union_rgb(env) -> np.ndarray:
    """RGB of the UNION of all agents' maps (the complete collectively-known map).

    Used for the terminal completion frame: after rendezvous the agents have fused maps, so
    the union is the full explored environment — makes 'episode completed' visible.
    """
    occ = env.world.occupancy_torch[0]                              # [M, H, W] (0 unk,1 free,2 obst)
    free_any = (occ == 1).any(dim=0)
    obst_any = (occ == 2).any(dim=0)
    union = torch.zeros_like(occ[0])
    union[obst_any] = 2
    union[free_any] = 1                                            # free wins (GT consistent)
    prob = torch.sigmoid(env.world.occupancy_logodds_torch[0].max(dim=0).values).cpu().numpy()
    fr = compute_frontier(union.unsqueeze(0).to(torch.uint8))[0].cpu().numpy()
    return paint_frontier(shade_occupancy_prob(prob), fr)


@torch.no_grad()
def capture_trace(model, split, env_cfg_dict: dict, n_agents: int, map_idx: int,
                  n_steps: int, out_root: Path, tag: str, device: str) -> dict:
    """Run one deterministic episode, dump trace.json + frames under out_root/traces/<tag>/."""
    env_cfg = EnvCfg.from_ckpt_dict(env_cfg_dict, n_envs=1, n_agents=n_agents,
                                    max_episode_steps=n_steps + 1)
    env = Explorer(split, env_cfg, seed=int(map_idx))
    env.store_render_global = True
    env.reload_map(env_idx=0, map_idx=int(map_idx))

    # The env auto-resets (wiping occupancy) the moment an episode terminates, so we'd never
    # be able to render the COMPLETED/fused map. Disable termination+truncation for the trace
    # and detect completion ourselves via explored_rate ≥ complete_thresh.
    complete_thresh = float(env.cfg.done_explored_thresh)
    env.cfg.done_explored_thresh = 2.0       # never terminate → never auto-reset
    env.cfg.max_episode_steps = n_steps + 5  # never truncate within the trace

    was_training = model.training
    model.eval()

    # torch.compile(mode="reduce-overhead") wraps model.encoder in CUDA graphs whose output
    # buffers are reused across calls. The trace invokes model.act TWICE per step (deterministic
    # action + grad-attribution forward), so the 2nd call overwrites the 1st's encoder output →
    # "accessing tensor output of CUDAGraphs that has been overwritten". Trace is offline (perf
    # irrelevant) → run the encoder EAGER for the trace, restore the compiled wrapper after.
    _compiled_enc = getattr(model.encoder, "_orig_mod", None)
    if _compiled_enc is not None:
        compiled_wrapper = model.encoder
        model.encoder = _compiled_enc

    # REAL GAT attention capture. store_attn makes every MaskedGATLayer keep the actual per-head
    # neighbor-attention softmax it computed; store_logit_components plumbs it out of act() as
    # _dbg_logits["enc_attn"]. These are the model's own weights — no estimation, no attribution.
    model.store_logit_components = True
    model.encoder.store_attn = True

    # Node input-feature names (for the selected-node panel; these are raw obs values, not any
    # attribution). Order matches node_feat channels 0..5.
    F_NAMES = ["x_rel", "y_rel", "utility", "age", "teammate_pot", "guidepost"]

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
    agent_frames = [[] for _ in range(M)]   # per-agent RGB frames → one animated GIF each
    explored = None          # last true explored_rate from info; set each step
    completed = False        # episode reached complete_thresh coverage
    for t in range(n_steps):
        gate = model._strategic_gate(obs, M)
        out = model.act(obs, h_act, h_crit, deterministic=True)
        rg = env._render_global
        nf = rg["node_feat"][0].cpu().numpy()
        nv = rg["node_valid"][0].cpu().numpy()
        ub = rg["util_boundary"][0].cpu().numpy()   # [M, N_max] boundary-pixel ribbon
        uv = rg["util_volume"][0].cpu().numpy()      # [M, N_max] revealable unknown volume
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
        # Teammate visibility for the inspector (env 0). pos_all = ground-truth xy of every
        # agent; lkp[i,j] = i's believed pos of j; cmask[i,j] = i&j communicating this step.
        pos_all = rg["pos"][0].cpu().numpy()                  # [M, 2]
        lkp = rg["last_known_pos"][0].cpu().numpy()           # [M, M, 2]
        cmask = rg["comm_mask"][0].cpu().numpy()              # [M, M]
        # ---- REAL GAT attention (no estimation, no interpretation) ----
        # act() stashed, per layer, the actual per-head neighbor-attention softmax used in message
        # passing:  enc_attn[layer] = [1, M, W2, K+1, H].  Slot k∈[0,K) is the k-th window-local
        # neighbor (obs["edge_idx"]); slot K is self. These are the EXACT weights the network
        # computed — stepping `layer` in the viewer = stepping one real message-passing hop; the
        # final layer's keys already summarize (L−1) hops, so layer L is the full L-hop result.
        # Window-local node indices are mapped to GLOBAL lattice ids via obs["local_to_global"] so
        # the viewer can colour the exact nodes on the map. Per-head kept SEPARATE: the model uses
        # every head (concatenated → o_proj), so a head-mean is not a value the network uses — we
        # never compute one. The viewer shows one real head at a time.
        dbg = model._dbg_logits or {}
        enc_attn = dbg.get("enc_attn")                       # list[L] of [1, M, W2, K+1, H] or None
        l2g    = obs["local_to_global"][0].cpu().numpy()     # [M, W2] global id (-1 pad)
        ei_loc = obs["edge_idx"][0].cpu().numpy()            # [M, W2, K] window-local nbr idx
        ev_loc = obs["edge_valid"][0].cpu().numpy()          # [M, W2, K] bool
        gat_agents = None
        if enc_attn is not None:
            Ln = len(enc_attn)
            gat_agents = []
            for a in range(M):
                loc = np.array([i for i in range(l2g.shape[1]) if int(l2g[a, i]) >= 0], dtype=np.int64)
                nodes = l2g[a, loc].astype(int).tolist()
                nbr = np.full((loc.size, K), -1, dtype=np.int64)         # neighbor GLOBAL ids (-1 masked)
                for c, i in enumerate(loc.tolist()):
                    for k in range(K):
                        if ev_loc[a, i, k]:
                            nbr[c, k] = int(l2g[a, int(ei_loc[a, i, k])])
                # weights [layer][head][node] = [self, n0..n_{K-1}], self moved to slot 0. All real.
                # NOTE: not `W` — that name holds the map width (H, W = env.H, env.W) above.
                w_layers = []
                for l in range(Ln):
                    al = np.round(enc_attn[l][0, a].cpu().numpy()[loc], 3)   # [V, K+1, H]
                    per_head = [np.concatenate([al[:, K:K + 1, h], al[:, :K, h]], axis=1).tolist()
                                for h in range(al.shape[-1])]
                    w_layers.append(per_head)
                gat_agents.append({"node": nodes, "nbr": nbr.tolist(), "w": w_layers})

        for a in range(M):
            prob = torch.sigmoid(env.world.occupancy_logodds_torch[0, a]).cpu().numpy()
            fr = compute_frontier(env.world.occupancy_torch[0:1, a])[0].cpu().numpy()
            agent_frames[a].append(paint_frontier(shade_occupancy_prob(prob), fr))   # → animated GIF later

        obs, reward, done, info = env.step(out["action"])
        dbg = env._dbg_reward or {}

        rec = {"t": t, "agents": []}
        for a in range(M):
            vidx = np.nonzero(nv[a])[0]
            nodes = [{
                "i": int(n), "x": _r(node_xy[n, 0], 1), "y": _r(node_xy[n, 1], 1),
                "util": _r(nf[a, n, 2], 4), "age": _r(nf[a, n, 3], 3),
                "team": _r(nf[a, n, 4], 3), "gp": _r(nf[a, n, 5], 1),
                "bf": _r(gd[a, n] / nr, 2),
                # utility seed components: boundary-pixel ribbon × revealable volume.
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
            # Per-teammate visibility from a's POV. comm=1 → a knows j's true pos this step
            # (belief == truth); comm=0 → est is a stale guess (drawn differently by the UI).
            teammates = [{
                "j": int(j), "comm": int(cmask[a, j]),
                "est": [_r(lkp[a, j, 0], 1), _r(lkp[a, j, 1], 1)],   # a's believed pos of j
                "true": [_r(pos_all[j, 0], 1), _r(pos_all[j, 1], 1)],  # ground truth (UI shows only if comm)
            } for j in range(M) if j != a]
            # Per move-neighbor GLOBAL node id (viewer maps a logit cell → the node it points to).
            curr_nbrs = [int(edge_idx[cur, k]) for k in range(K)]
            rec["agents"].append({
                # one animated GIF per agent; "fi" = frame index (= step t) to seek to.
                "frame": f"frames/a{a}.gif", "fi": t, "curr": cur, "target": int(tgt[a]),
                # pre-step position (matches the rendered frame + teammate overlay; reading
                # env.pos AFTER step() shows the next/teleported-on-reset spot — the last-step bug).
                "pos": [_r(pos_all[a, 0], 1), _r(pos_all[a, 1], 1)],
                "value": _r(value), "gate": int(round(float(gate_np[a]))),
                "action": int(action[a]),
                "logits": [_r(x, 3) for x in logits[a].tolist()],
                "action_mask": [int(b) for b in amask[a].tolist()],
                "guidepost_dir": [_r(x, 2) for x in gp_bias[a].tolist()],
                "curr_nbrs": curr_nbrs,
                "reward": rew, "nodes": nodes, "edges": edges,
                "teammates": teammates,
                # REAL per-layer, per-head GAT neighbor-attention softmax for this agent's window.
                "gat": (gat_agents[a] if gat_agents is not None else None),
            })
        # True coverage from info (reset is disabled above, so occupancy is intact).
        explored = float(info["explored_rate"][0].item())
        h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
        steps.append(rec)
        if explored >= complete_thresh:        # map complete → natural episode end
            completed = True
            break

    # Terminal completion frame: the UNION of both agents' (now-fused) maps = the full
    # environment. Appended as a final step so it's visible that the episode ended COMPLETE.
    if completed:
        union_rgb = _render_union_rgb(env)
        for a in range(M):
            agent_frames[a].append(union_rgb)      # union = last frame of every agent's GIF
        term_fi = len(agent_frames[0]) - 1
        pos_now = env.pos[0].cpu().numpy()     # [M, 2] final positions (no reset happened)
        term_agents = []
        for a in range(M):
            term_agents.append({
                "frame": f"frames/a{a}.gif", "fi": term_fi,        # union map (last GIF frame)
                "curr": int(steps[-1]["agents"][a]["curr"]) if steps else 0,
                "target": -1,
                "pos": [_r(pos_now[a, 0], 1), _r(pos_now[a, 1], 1)],
                "value": None, "gate": 0, "action": -1,
                "logits": [None] * K, "action_mask": [0] * K,
                "guidepost_dir": [0.0] * K, "curr_nbrs": [-1] * K,
                "reward": {}, "nodes": [], "edges": [], "gat": None,
                # all agents share the complete map now → draw teammates solid at true pos.
                "teammates": [{"j": int(j), "comm": 1,
                               "est":  [_r(pos_now[j, 0], 1), _r(pos_now[j, 1], 1)],
                               "true": [_r(pos_now[j, 0], 1), _r(pos_now[j, 1], 1)]}
                              for j in range(M) if j != a],
            })
        steps.append({"t": len(steps), "agents": term_agents, "terminal": True})

    # One animated GIF per agent (replaces the per-step PNG flood); seekable by index client-side
    # via WebCodecs ImageDecoder. PIL collapses consecutive-IDENTICAL frames, so we dedup first and
    # remap each step's "fi" to the kept GIF frame index (stationary steps just share a frame).
    for a in range(M):
        if not agent_frames[a]:
            continue
        kept, fis = [], []
        for f in agent_frames[a]:
            if kept and np.array_equal(f, kept[-1]):
                fis.append(len(kept) - 1)          # identical to previous → reuse that GIF frame
            else:
                kept.append(f); fis.append(len(kept) - 1)
        imgs = [Image.fromarray(f) for f in kept]
        imgs[0].save(fdir / f"a{a}.gif", save_all=True, append_images=imgs[1:],
                     duration=120, loop=0, disposal=1, optimize=True)
        for i, st in enumerate(steps):             # len(steps) == len(agent_frames[a]) == len(fis)
            st["agents"][a]["fi"] = fis[i]

    meta = {"tag": tag, "map_idx": map_idx, "n_agents": M, "H": H, "W": W, "nr": nr,
            "n_hops": int(env.cfg.n_hops), "win": 2 * int(env.cfg.n_hops) + 3,  # ego-window side (cells)
            "n_steps": len(steps), "K_OFFSETS": _KOFF,
            "final_explored": _r(explored) if explored is not None else None,
            "completed": bool(completed),
            "gate_eps": getattr(model, "strategic_gate_eps", 0.0),
            "target_mode": "analytic",
            # GAT geometry for the attention viewer: n_layers = message-passing hops, n_heads =
            # separate attention heads (all real, all used by the model).
            "gat_layers": len(model.encoder.layers),
            "gat_heads": int(getattr(model.encoder.layers[0], "n_heads", 1)),
            "feat_names": F_NAMES}   # input-feature names for the node panel
    (tdir / "trace.json").write_text(json.dumps({"meta": meta, "steps": steps}))

    # maintain the episode picker index + ensure the viewer is present
    idx_path = Path(out_root) / "traces" / "index.json"
    try:
        index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    except Exception:
        index = []
    entry = {"tag": tag, "map_idx": map_idx, "n_steps": len(steps),
             "final_explored": meta["final_explored"], "completed": bool(completed)}
    index = [e for e in index if e.get("tag") != tag] + [entry]
    idx_path.write_text(json.dumps(index, indent=1))
    viewer = _REPO / "viz" / "inspector.html"
    if viewer.exists():
        (Path(out_root) / "inspector.html").write_bytes(viewer.read_bytes())

    # Turn the attention/debug stash back OFF before restoring the (possibly compiled) encoder —
    # the driver reuses this same model to keep training, and store_attn would tax every forward.
    model.encoder.store_attn = False
    model.store_logit_components = False
    if _compiled_enc is not None:
        model.encoder = compiled_wrapper          # restore compiled encoder for training
    if was_training:
        model.train()
    print(f"[trace] {tag}  steps={len(steps)}  explored={meta['final_explored']}  → {tdir}")
    return meta
