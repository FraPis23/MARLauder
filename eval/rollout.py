"""Deterministic single-episode rollout for eval. Collects frames for the GIF.

Usage from scripts/run_eval.py:
    rollout = EvalRollout(env, model, cfg)
    frames, stats = rollout.run()
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from env.explorer import Explorer
from env.frontier import compute_frontier
from eval.render import composite_frame, hstack_frames
from models.actor_critic import MarlActorCritic


@dataclass
class EvalCfg:
    max_steps: int = 128
    env_idx: int = 0
    deterministic: bool = True
    draw_edges: bool = True
    trail_len: int = 40


class EvalRollout:
    def __init__(self, env: Explorer, model: MarlActorCritic, cfg: EvalCfg) -> None:
        self.env = env
        self.model = model
        self.cfg = cfg
        # Render full-graph (not just ego window) utility — needs the global stash.
        env.store_render_global = True

    @torch.no_grad()
    def run(self) -> tuple[list[np.ndarray], dict]:
        env = self.env
        N = env.N
        device = env.dev
        h_act, h_crit = self.model.init_hidden(N, device)
        obs = env.obs
        e = self.cfg.env_idx
        M = env.M
        frames: list[np.ndarray] = []
        trails: list[list[tuple[float, float]]] = [[] for _ in range(M)]
        explored_hist: list[float] = []
        gt_np = env.world.gt_torch[e].cpu().numpy()
        for t in range(self.cfg.max_steps):
            out = self.model.act(obs, h_act, h_crit, deterministic=self.cfg.deterministic)
            action = out["action"]
            # The pursued target is the env's analytic guidepost: its world coords +
            # curr→target BF path are shipped in obs. Snapshot BEFORE env.step.
            gp_target_xy = obs["guidepost_target_xy"]                           # [N, M, 2]
            strategic_target_xy: list[tuple[float, float]] = [
                (float(gp_target_xy[e, ag, 0].item()), float(gp_target_xy[e, ag, 1].item()))
                for ag in range(M)
            ]
            strategic_path_xy: list[np.ndarray] = [None] * M                    # use obs path below
            obs, reward, done, info = env.step(action)
            h_act = out["hidden_actor"]
            h_crit = out["hidden_critic"]
            nonterm = (~done).float()
            h_act = h_act * nonterm.view(-1, 1, 1)
            h_crit = h_crit * nonterm.view(-1, 1)
            # Update trails with current positions
            for ag in range(M):
                trails[ag].append((float(env.pos[e, ag, 0]), float(env.pos[e, ag, 1])))

            # Communication links: pairs where comm_mask[e, i, j] is True (i < j)
            cm = obs.get("comm_mask")
            comm_links = []
            if cm is not None and M > 1:
                for i in range(M):
                    for j in range(i + 1, M):
                        if bool(cm[e, i, j].item()):
                            comm_links.append((trails[i][-1], trails[j][-1]))

            explored = float(info["explored_rate"][e].item())
            explored_hist.append(explored)
            step_t = int(env.t[e].item())

            # One panel per agent — each shows their own occupancy map
            agent_frames: list[np.ndarray] = []
            for ag in range(M):
                prob_ag = torch.sigmoid(
                    env.world.occupancy_logodds_torch[e, ag]
                ).cpu().numpy()                                              # [H, W]
                occ_ag = env.world.occupancy_torch[e:e+1, ag]               # [1, H, W]
                frontier_ag = compute_frontier(occ_ag)[0].cpu().numpy()     # [H, W]

                # GLOBAL graph (full map) when the stash is present; else fall back to the
                # ego window shipped in obs.
                rg = env._render_global
                win_mask_ag = None
                win_bbox_ag = None
                if rg is not None:
                    nxy_ag    = rg["node_xy"].cpu().numpy()                 # [N_max, 2]
                    nv_ag     = rg["node_valid"][e, ag].cpu().numpy()       # [N_max]
                    util_ag   = rg["utility"][e, ag].cpu().numpy()          # [N_max]
                    eidx_ag   = rg["edge_idx"].cpu().numpy()               # [N_max, K]
                    evalid_ag = rg["edge_valid"][e, ag].cpu().numpy()      # [N_max, K]
                    curr_ag   = int(rg["curr_idx"][e, ag])                  # GLOBAL node idx
                    # Ego-window highlight: global node indices of the window around curr.
                    win_idx = rg["window_idx_table"][curr_ag].cpu().numpy() # [W²], -1 pad
                    win_valid = win_idx[win_idx >= 0]
                    win_mask_ag = np.zeros(nxy_ag.shape[0], dtype=bool)
                    win_mask_ag[win_valid] = True
                    if win_valid.size:
                        wx, wy = nxy_ag[win_valid, 0], nxy_ag[win_valid, 1]
                        pad = 0.5 * float(env.cfg.nr)
                        win_bbox_ag = (float(wx.min()) - pad, float(wy.min()) - pad,
                                       float(wx.max()) + pad, float(wy.max()) + pad)
                else:
                    nxy_ag   = obs["node_xy"][e, ag].cpu().numpy()
                    nv_ag    = obs["node_valid"][e, ag].cpu().numpy()
                    util_ag  = obs["utility"][e, ag].cpu().numpy()
                    eidx_ag  = obs["edge_idx"][e, ag].cpu().numpy()
                    evalid_ag = obs["edge_valid"][e, ag].cpu().numpy()
                    curr_ag  = int(obs["curr_idx"][e, ag])

                # G.3.a — render STRATEGIC head's pick (captured pre-step above), not env-argmax.
                tgt_xy_ag = strategic_target_xy[ag]
                ax_, ay_ = trails[ag][-1]
                at_target = (abs(tgt_xy_ag[0] - ax_) < env.cfg.nr
                             and abs(tgt_xy_ag[1] - ay_) < env.cfg.nr)
                # G.3.b — BF path from curr to strategic pick (CORRECT path through known-FREE).
                if strategic_path_xy[ag] is not None and len(strategic_path_xy[ag]) > 0:
                    sp = strategic_path_xy[ag]
                    P_max = obs["guidepost_path_xy"].shape[2]
                    path_xy_ag = np.full((P_max, 2), float("nan"), dtype=np.float32)
                    path_valid_ag = np.zeros((P_max,), dtype=bool)
                    L = min(P_max, len(sp))
                    path_xy_ag[:L] = sp[:L]
                    path_valid_ag[:L] = True
                else:
                    path_xy_ag    = obs["guidepost_path_xy"][e, ag].cpu().numpy()
                    path_valid_ag = obs["guidepost_path_valid"][e, ag].cpu().numpy()

                other_ags = [oag for oag in range(M) if oag != ag]
                im_ag = composite_frame(
                    prob=prob_ag, gt=gt_np, frontier=frontier_ag,
                    nxy=nxy_ag, nv=nv_ag, util=util_ag, curr=curr_ag,
                    agent_xy=trails[ag][-1],
                    trail=trails[ag][-self.cfg.trail_len:],
                    step=step_t, explored=explored,
                    draw_edges=self.cfg.draw_edges, eidx=eidx_ag, evalid=evalid_ag,
                    win_node_mask=win_mask_ag, win_bbox=win_bbox_ag,
                    path_xy=path_xy_ag, path_valid=path_valid_ag,
                    target_xy=tgt_xy_ag if not at_target else None,
                    extra_agents_xy=[trails[oag][-1] for oag in other_ags],
                    extra_agents_trails=[trails[oag][-self.cfg.trail_len:] for oag in other_ags],
                    extra_agent_indices=other_ags,
                    comm_links=comm_links if comm_links else None,
                    agent_idx=ag,
                    agent_label=f"A{ag}",
                )
                agent_frames.append(np.array(im_ag))

            frames.append(hstack_frames(agent_frames))
            if bool(done[e].item()):
                break
        stats = {
            "final_explored": explored_hist[-1] if explored_hist else 0.0,
            "n_frames": len(frames),
        }
        return frames, stats
