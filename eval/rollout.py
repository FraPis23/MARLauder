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
            # G.3.a/b — snapshot strategic head's chosen target + BF path BEFORE env.step.
            target_choice_t = out["target_choice"]                              # [N, M] long
            cand_xy_now = obs["cand_xy"]                                        # [N, M, K, 2]
            cand_idx_now = obs["cand_idx"]                                      # [N, M, K] long
            bf_parent_now = obs["bf_parent_from_curr"]                          # [N, M, N_max]
            curr_idx_global_now = obs["curr_idx_global"]                        # [N, M]
            K_cand = cand_xy_now.shape[-2]
            node_xy_global = env.graph.node_xy                                  # [N_max, 2]
            strategic_target_xy: list[tuple[float, float]] = [(0.0, 0.0)] * M
            strategic_path_xy: list[np.ndarray] = [None] * M                    # [P, 2] per ag
            for ag in range(M):
                k_slot = int(target_choice_t[e, ag].item())
                if 0 <= k_slot < K_cand:
                    strategic_target_xy[ag] = (
                        float(cand_xy_now[e, ag, k_slot, 0].item()),
                        float(cand_xy_now[e, ag, k_slot, 1].item()),
                    )
                    # Walk BF parent from cand back to curr to build correct path.
                    cand_global = int(cand_idx_now[e, ag, k_slot].item())
                    curr_global = int(curr_idx_global_now[e, ag].item())
                    if cand_global >= 0:
                        path_nodes = [cand_global]
                        cur_n = cand_global
                        # Walk parent ≤ 200 steps (safety). parent[v]=-1 means unreachable.
                        for _ in range(200):
                            par_n = int(bf_parent_now[e, ag, cur_n].item())
                            if par_n < 0 or par_n == cur_n:
                                break
                            path_nodes.append(par_n)
                            if par_n == curr_global:
                                break
                            cur_n = par_n
                        # Reverse so path runs curr → ... → cand.
                        path_nodes.reverse()
                        xy_arr = np.array(
                            [(float(node_xy_global[n, 0].item()),
                              float(node_xy_global[n, 1].item())) for n in path_nodes],
                            dtype=np.float32,
                        )
                        strategic_path_xy[ag] = xy_arr
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
