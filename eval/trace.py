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

    # Per-neighbor logit attribution BY INPUT FEATURE. For each move-neighbor of the current
    # node, contribution_f = (∂ neighbor-logit / ∂ feature_f) · feature_f (input×gradient) of
    # that neighbor's own node features → "the values that brought this logit to that number".
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
        out = model.act(obs, h_act, h_crit, deterministic=True, nr=nr)
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
        # Per move-neighbor logit attribution by INPUT FEATURE via Integrated Gradients.
        # The action-logit = pointer score q(h_act)·k(nbr_emb)/√d — deeply NONLINEAR in the
        # input features (GAT message passing → embeddings → GRU → bilinear pointer). IG gives,
        # for the CLICKED neighbor's OWN node features, a value × weight = product decomposition:
        #   weight_f  = ∫ ∂logit/∂feat_f dα   (integrated gradient = multiplier, independent of value)
        #   product_f = weight_f · value_f    (this feature's contribution to the logit)
        # Completeness:  logit = baseline + Σ_f product_f (own node) + other (rest of the window).
        # "other" is the exact remainder (all OTHER window nodes), so it always reconstructs.
        cn_local = obs["curr_nbr"][0].cpu().numpy()          # [M, K] window-local neighbor idx
        Fdim = obs["node_feat"].shape[-1]
        IG_STEPS = 64                                         # Riemann steps; max|recon−logit|≈0.02
        dev = obs["node_feat"].device
        feat_vals  = np.zeros((M, K, Fdim), dtype=np.float32)  # clicked neighbor's own feature values
        feat_wts   = np.zeros((M, K, Fdim), dtype=np.float32)  # integrated gradient (weight) per feature
        feat_prod  = np.zeros((M, K, Fdim), dtype=np.float32)  # weight · value (own-node contribution)
        other_term = np.zeros((M, K), dtype=np.float32)        # contribution of all OTHER window nodes
        nf_full = obs["node_feat"].detach()                   # [1, M, N_max, F]
        Nw = nf_full.shape[2]
        with torch.no_grad():                                 # baseline logit at zeroed features
            obs_b = dict(obs); obs_b["node_feat"] = torch.zeros_like(nf_full)
            base_logit = model.act(obs_b, h_act, h_crit, deterministic=True, nr=nr)["logits"][0].cpu().numpy()
        with torch.enable_grad():
            # Batch the S interpolation points into the env dimension (each alpha = one
            # independent env, actor is per-env) → ONE forward + M·K backward (not M·K·S).
            alphas = torch.linspace(0.5 / IG_STEPS, 1 - 0.5 / IG_STEPS, IG_STEPS,
                                    device=dev).view(IG_STEPS, 1, 1, 1)
            obs_g = {kk: (v.expand(IG_STEPS, *v.shape[1:]) if torch.is_tensor(v) and v.shape[0] == 1 else v)
                     for kk, v in obs.items()}
            x = (nf_full * alphas).clone().requires_grad_(True)          # [S, M, N_max, F]
            obs_g["node_feat"] = x
            ha = h_act.expand(IG_STEPS, *h_act.shape[1:])                # [S, M, d]
            hc = h_crit.expand(IG_STEPS, *h_crit.shape[1:])             # [S, d]
            lg = model.act(obs_g, ha, hc, deterministic=True, nr=nr)["logits"]  # [S, M, K]
            for a in range(M):
                for k in range(K):
                    if not bool(amask[a, k]):
                        continue
                    if x.grad is not None:
                        x.grad = None
                    lg[:, a, k].sum().backward(retain_graph=True)        # ∂Σ_s logit_s / ∂x_s
                    avg_grad = x.grad[:, a].mean(dim=0)                  # mean grad along path [N_max, F]
                    nbl = int(cn_local[a, k])
                    val = nf_full[0, a, nbl]                             # neighbor's own feature values [F]
                    wt = avg_grad[nbl]                                   # integrated grad at own node [F]
                    feat_vals[a, k] = val.cpu().numpy()
                    feat_wts[a, k]  = wt.cpu().numpy()
                    feat_prod[a, k] = (wt * val).cpu().numpy()           # own-node products (value·weight)
                    total_ig = float((nf_full[0, a] * avg_grad).sum())   # IG over ALL window nodes
                    other_term[a, k] = total_ig - float((wt * val).sum())  # remainder = other nodes

        for a in range(M):
            prob = torch.sigmoid(env.world.occupancy_logodds_torch[0, a]).cpu().numpy()
            fr = compute_frontier(env.world.occupancy_torch[0:1, a])[0].cpu().numpy()
            agent_frames[a].append(paint_frontier(shade_occupancy_prob(prob), fr))   # → animated GIF later

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
            # Per move-neighbor (global node id), the by-feature attribution of its logit.
            curr_nbrs = [int(edge_idx[cur, k]) for k in range(K)]
            nbr_attrib = [{
                "node": curr_nbrs[k], "logit": _r(logits[a, k], 3),
                "base": _r(base_logit[a, k], 3),    # logit at zeroed-features baseline (intercept)
                "other": _r(other_term[a, k], 4),   # contribution of all OTHER window nodes (remainder)
                "vals":    [_r(feat_vals[a, k, f], 3) for f in range(min(Fdim, 6))],   # feature value
                "wts":     [_r(feat_wts[a, k, f], 4) for f in range(min(Fdim, 6))],    # weight (∫gradient)
                "prod":    [_r(feat_prod[a, k, f], 4) for f in range(min(Fdim, 6))],   # product = wt·val
            } for k in range(K) if bool(amask[a, k]) and curr_nbrs[k] >= 0]
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
                "nbr_attrib": nbr_attrib,
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
                "reward": {}, "nodes": [], "edges": [], "nbr_attrib": [],
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
            "target_mode": getattr(model, "target_mode", "analytic"),
            "feat_names": F_NAMES}   # input-feature names for the per-neighbor attribution
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

    if _compiled_enc is not None:
        model.encoder = compiled_wrapper          # restore compiled encoder for training
    if was_training:
        model.train()
    print(f"[trace] {tag}  steps={len(steps)}  explored={meta['final_explored']}  → {tdir}")
    return meta
