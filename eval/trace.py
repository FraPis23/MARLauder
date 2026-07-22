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
                  n_steps: int, out_root: Path, tag: str, device: str,
                  action_fn=None) -> dict:
    """Run one deterministic episode, dump trace.json + frames under out_root/traces/<tag>/.

    action_fn: optional callable (env, obs) -> action[N,M] long. When given, it OVERRIDES the
    model's chosen action for stepping the env, while the model still runs so the trace keeps real
    logits/value/GAT attention. Used to drive a separation-inducing heuristic (agents spread out)
    so the teammate belief goes out-of-range and its propagation is visible in the inspector even
    when no well-trained policy is available to load."""
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
    # attribution). Order matches node_feat channels 0..6.
    F_NAMES = ["x_rel", "y_rel", "utility", "age", "teammate_pot",
               "radar_util", "radar_teammate"]

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
    # GAT attention (per-layer, per-head, per-window-node) is heavy — ~0.5-0.7MB per agent per
    # step. Embedded inline in trace.json it made long/hard episodes (e.g. 512-step evals on
    # "complex" maps) balloon to 600-700MB, which no browser can fetch/parse → the whole
    # inspector silently fails to render (nothing populates, not even the agent/step picker).
    # Written instead to one small side file per step, fetched lazily by the viewer only for the
    # step currently being looked at (viz/inspector.html::ensureGat). `has_gat` in the inline
    # step record just tells the viewer whether it's worth asking.
    gdir = tdir / "gat"
    gdir_ready = False
    has_contrib = False
    for t in range(n_steps):
        out = model.act(obs, h_act, h_crit, deterministic=True)
        if action_fn is not None:
            out["action"] = action_fn(env, obs)   # heuristic override (agents separate) — see docstring
        rg = env._render_global
        nf = rg["node_feat"][0].cpu().numpy()
        nv = rg["node_valid"][0].cpu().numpy()
        ub = rg["util_boundary"][0].cpu().numpy()   # [M, N_max] boundary-pixel ribbon
        uv = rg["util_volume"][0].cpu().numpy()      # [M, N_max] revealable unknown volume
        ev = rg["edge_valid"][0].cpu().numpy()
        curr = rg["curr_idx"][0].cpu().numpy()
        logits = out["logits"][0].cpu().numpy()
        value = float(out["value"][0].cpu().numpy())
        action = out["action"][0].cpu().numpy()
        amask = obs["action_mask"][0].cpu().numpy()
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
        enc_contrib = dbg.get("enc_contrib")                 # list[L] of [1, M, W2, K+1] or None
        #   `enc_contrib` = the REAL per-neighbor value-contribution magnitude ‖c_j‖ to the output
        #   embedding, combining all H heads exactly as the model does (attn × value → o_proj mix,
        #   NOT a head-mean). This is the honest "global" weight the viewer shows by default; the
        #   per-head softmax (enc_attn) is the drill-down. Both are exact, no estimation.
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
                c_layers = []
                for l in range(Ln):
                    al = np.round(enc_attn[l][0, a].cpu().numpy()[loc], 3)   # [V, K+1, H]
                    per_head = [np.concatenate([al[:, K:K + 1, h], al[:, :K, h]], axis=1).tolist()
                                for h in range(al.shape[-1])]
                    w_layers.append(per_head)
                    # combined REAL value-contribution ‖c_j‖ [self, n0..n_{K-1}]; self moved to slot 0.
                    if enc_contrib is not None:
                        cl = enc_contrib[l][0, a].cpu().numpy()[loc]          # [V, K+1]
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
            agent_frames[a].append(paint_frontier(shade_occupancy_prob(prob), fr))   # → animated GIF later

        obs, reward, done, info = env.step(out["action"])
        dbg = env._dbg_reward or {}

        # RAW teammate belief posterior per node (unmasked → includes UNKNOWN nodes the diffusion
        # spreads onto, which feat[4]/"team" drops). Max over teammates, peak-normalized per agent
        # for a readable ramp. This is the field that shows the belief PROPAGATING across the map.
        bp = rg.get("belief_p")
        if bp is not None:
            bp0 = bp[0].cpu().numpy()                       # [M, M, N_max]
            # RAW posterior p (max over teammates) — NOT peak-normalised, so the web renders it on a
            # FIXED absolute scale (BEL_VMAX) and the frontier accumulation magnitude is comparable
            # across frames. Σ p = 1 per teammate.
            bel_disp = bp0.max(axis=1)                      # [M, N_max]
        else:
            bel_disp = None
        # Pathfront TRANSIT dots (uniform 1.0 markers) — the travelling hypotheses BEFORE they bloom.
        # Rendered as distinct points so the viewer sees each dot depart lkp→frontier (viz only).
        bt_rg = rg.get("belief_transit")
        bel_transit = bt_rg[0].cpu().numpy().max(axis=1) if bt_rg is not None else None   # [M, N_max]

        rec = {"t": t, "agents": []}
        for a in range(M):
            belj = bel_disp[a] if bel_disp is not None else None
            beltj = bel_transit[a] if bel_transit is not None else None
            # Draw valid nodes PLUS any node carrying belief mass or a transit dot.
            has_bel = (belj > 1e-4) if belj is not None else np.zeros(nv[a].shape, dtype=bool)
            if beltj is not None:
                has_bel = has_bel | (beltj > 0.5)
            vidx = np.nonzero(nv[a] | has_bel)[0]
            nodes = [{
                "i": int(n), "x": _r(node_xy[n, 0], 1), "y": _r(node_xy[n, 1], 1),
                "util": _r(nf[a, n, 2], 4), "age": _r(nf[a, n, 3], 3),
                "team": _r(nf[a, n, 4], 3),
                # bel = RAW belief posterior (incl. unknown nodes) → the propagation field.
                "bel": _r(float(belj[n]), 4) if belj is not None else 0.0,   # RAW p (fixed-scale in web)
                # belt = 1 when a still-travelling transit dot sits on this node (pathfront, viz only).
                "belt": 1 if (beltj is not None and beltj[n] > 0.5) else 0,
                # RADAR boundary-summary channels (nonzero only on geodesic horizon nodes):
                # bu = out-of-window utility mass, bt = out-of-window teammate direction.
                "bu": _r(nf[a, n, 5], 3), "bt": _r(nf[a, n, 6], 3),
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
                "frame": f"frames/a{a}.gif", "fi": t, "curr": cur,
                # pre-step position (matches the rendered frame + teammate overlay; reading
                # env.pos AFTER step() shows the next/teleported-on-reset spot — the last-step bug).
                "pos": [_r(pos_all[a, 0], 1), _r(pos_all[a, 1], 1)],
                "value": _r(value),
                "action": int(action[a]),
                "logits": [_r(x, 3) for x in logits[a].tolist()],
                "action_mask": [int(b) for b in amask[a].tolist()],
                "curr_nbrs": curr_nbrs,
                "reward": rew, "nodes": nodes, "edges": edges,
                "teammates": teammates,
                # REAL per-layer, per-head GAT neighbor-attention softmax for this agent's window —
                # NOT embedded here (see gdir above); this just flags whether gat/{t:04d}.json exists.
                "has_gat": gat_agents is not None,
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
                "pos": [_r(pos_now[a, 0], 1), _r(pos_now[a, 1], 1)],
                "value": None, "action": -1,
                "logits": [None] * K, "action_mask": [0] * K,
                "curr_nbrs": [-1] * K,
                "reward": {}, "nodes": [], "edges": [], "has_gat": False,
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
            # GAT geometry for the attention viewer: n_layers = message-passing hops, n_heads =
            # separate attention heads (all real, all used by the model).
            "gat_layers": len(model.encoder.layers),
            "gat_heads": int(getattr(model.encoder.layers[0], "n_heads", 1)),
            "has_contrib": has_contrib,   # whether gat/*.json rows carry the "c" (contribution) field
            # Per-head A2 raw-feature groups → the viewer labels heads by what they specialize on
            # (head0→geometry, head1→utility, head2→teammate, …). Real config, not an inference.
            "head_feat_groups": (model.encoder.head_feat_groups()
                                 if hasattr(model.encoder, "head_feat_groups") else None),
            "feat_names": F_NAMES}   # input-feature names for the node panel
    (tdir / "trace.json").write_text(json.dumps({"meta": meta, "steps": steps}))

    # maintain the episode picker index (the viewer itself is served canonically by
    # viz/web_server.py for any /<run>/inspector.html request — no per-run copy, so it's never
    # stale at whatever version existed when this run's traces happened to be captured)
    idx_path = Path(out_root) / "traces" / "index.json"
    try:
        index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    except Exception:
        index = []
    entry = {"tag": tag, "map_idx": map_idx, "n_steps": len(steps),
             "final_explored": meta["final_explored"], "completed": bool(completed)}
    index = [e for e in index if e.get("tag") != tag] + [entry]
    idx_path.write_text(json.dumps(index, indent=1))

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
