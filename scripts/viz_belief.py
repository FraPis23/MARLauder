"""Belief-filter visualization GIF.

Runs a trained 2-agent policy and renders, per agent, a heatmap of that agent's BELIEF of
the teammate's position (env._belief_p) over the graph nodes — so you can literally watch the
teammate belief COLLAPSE at contact, DIFFUSE out-of-range, CARVE where the agent looks and the
teammate isn't, and SNOWPLOW toward the frontier. Overlays each agent's true position (dot) and
its last-known-position of the teammate (target marker) for comparison.

    python scripts/viz_belief.py --ckpt runs/rdv_easy_.../ckpt_best.pt \\
        --split train/easy --map-idx 120 --steps 220 --comm-range 45 --out runs/belief.gif

Lower --comm-range → agents drop out of contact sooner → more diffuse/carve to watch. Any good
2-agent checkpoint works; the belief filter is env-side, independent of what the policy was
trained on (feat[4] the actor sees may be off-distribution, but the belief render is exact).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import imageio.v2 as imageio
import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
from eval.render import composite_frame, hstack_frames
from env.frontier import compute_frontier
from models.actor_critic import MarlActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, default=120)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--comm-range", type=float, default=45.0,
                    help="override comm_range_px (lower → agents separate → belief diffuses more)")
    ap.add_argument("--policy", choices=["model", "greedy", "scripted"], default="greedy",
                    help="greedy = frontier-seeking value_field argmax; model = load --ckpt; "
                         "scripted = agent0 explores (greedy) while agent1 stays put — the cleanest "
                         "check that A0's belief of A1 diffuses/carves/piles-at-frontier correctly.")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--d-hidden", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/belief.gif"))
    args = ap.parse_args()

    env_peek = {}
    model = None
    if args.policy == "model":
        if args.ckpt is None:
            raise SystemExit("--policy model requires --ckpt")
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg_peek = ckpt.get("cfg", {})
        env_peek = cfg_peek.get("env", {}) if isinstance(cfg_peek, dict) else {}

    overrides = dict(n_envs=1, n_agents=2, max_episode_steps=args.steps + 1,
                     use_teammate_belief=True, comm_model="los",
                     comm_range_px=float(args.comm_range))
    split = load_split(args.split, device=args.device)
    env_cfg = EnvCfg.from_ckpt_dict(env_peek, **overrides)
    env = Explorer(split, env_cfg, seed=int(args.map_idx))
    env.store_render_global = True
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))

    if args.policy == "model":
        n_layers = (cfg_peek.get("n_layers") if isinstance(cfg_peek, dict) else None) \
                   or env_peek.get("n_hops") or 2
        model = MarlActorCritic(n_agents=2, d=args.d_hidden, n_heads=args.n_heads,
                                n_layers=int(n_layers)).to(args.device)
        sd = {k.replace("encoder._orig_mod.", "encoder."): (v.to(args.device) if torch.is_tensor(v) else v)
              for k, v in ckpt["model"].items()}
        model.load_state_dict(sd, strict=False)
        if isinstance(cfg_peek, dict):
            model.use_gru = bool(cfg_peek.get("use_gru", True))
        model.eval()
        h_act, h_crit = model.init_hidden(1, args.device)
        print(f"[load] {args.ckpt} iter={ckpt.get('iter','?')}")
    print(f"[policy] {args.policy}  comm_range={args.comm_range}")

    obs = env._last_obs
    K = env.graph.edge_len.shape[0]
    M = 2

    last_act = [-1, -1]

    def greedy_action(obs):
        """Frontier-seeking: pick the neighbour branch with the highest value_field, masked to
        valid actions. Forbids immediately reversing the previous move (NBR_OFFSETS reverse of k is
        7-k) to stop 2-node ping-pong, so the agents actually cover ground and separate → the belief
        goes out-of-range and diffuses. A small per-agent branch bias splits them apart."""
        vf = obs["value_field"][0].clone()            # [M, K]
        am = obs["action_mask"][0].bool()             # [M, K] valid actions
        bias = torch.zeros_like(vf)
        bias[0, : K // 2] += 0.03                      # agent 0 leans to low-index branches
        bias[1, K // 2:] += 0.03                       # agent 1 leans to high-index branches
        for ag in range(M):
            if last_act[ag] >= 0:
                am[ag, K - 1 - last_act[ag]] = False   # no immediate backtrack
        score = torch.where(am, vf + bias, torch.full_like(vf, -1e9))
        act = score.argmax(dim=-1)                     # [M]
        for ag in range(M):
            last_act[ag] = int(act[ag])
        return act.view(1, M)

    node_xy = env.graph.node_xy                        # [N_max, 2]
    eidx_static = env.graph.edge_idx_static            # [N_max, K]

    def scripted_action(obs):
        """agent1 HOLDS (invalid action → env stall-keeps it put); agent0 walks directly AWAY from
        agent1 (valid neighbour maximizing distance to A1), with anti-backtrack, so it reliably
        crosses the map, loses comm, and explores. Isolates A0's belief of the STATIONARY A1: it
        should diffuse from A1's spot (A0 no longer looks there → not carved), carve A0's own swept
        trail, and redistribute onto the new nodes A0 reveals — the exact propagation to verify."""
        am = obs["action_mask"][0].bool()              # [M, K]
        a1_xy = env.pos[0, 1]                           # [2] teammate position
        curr0 = int(env.curr_idx_global[0, 0])
        nbrs = eidx_static[curr0]                       # [K] global neighbour node idx (-1 pad)
        nbr_xy = node_xy[nbrs.clamp(min=0)]            # [K, 2]
        dist_away = (nbr_xy - a1_xy).pow(2).sum(-1)     # [K] farther = better
        valid0 = am[0].clone()
        if last_act[0] >= 0:
            valid0[K - 1 - last_act[0]] = False        # no immediate backtrack
        score0 = torch.where(valid0, dist_away, torch.full_like(dist_away, -1e9))
        a0 = int(score0.argmax())
        last_act[0] = a0
        invalid1 = (~am[1]).nonzero()
        a1 = int(invalid1[0]) if invalid1.numel() else 0   # masked slot → hold
        return torch.tensor([[a0, a1]], device=am.device)
    e = 0
    trails = {ag: [] for ag in range(M)}
    gt_np = (env.world.gt_torch[e] == 1).cpu().numpy().astype(np.uint8) if hasattr(env.world, "gt_torch") else None
    frames = []

    for t in range(args.steps):
        with torch.no_grad():
            if args.policy == "model":
                out = model.act(obs, h_act, h_crit, deterministic=True)
                action = out["action"]
                h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
            elif args.policy == "scripted":
                action = scripted_action(obs)
            else:
                action = greedy_action(obs)
        obs, reward, done, info = env.step(action)
        for ag in range(M):
            trails[ag].append((float(env.pos[e, ag, 0]), float(env.pos[e, ag, 1])))

        rg = env._render_global
        if rg is None or rg.get("belief_p") is None:
            raise SystemExit("belief_p missing — is use_teammate_belief on and M>1?")
        nxy = rg["node_xy"].cpu().numpy()
        eidx = rg["edge_idx"].cpu().numpy()
        cm = rg["comm_mask"][e].cpu().numpy()               # [M, M]
        belief_p = rg["belief_p"][e].cpu().numpy()          # [a, j, N_max]
        alive = rg["belief_alive"][e].cpu().numpy()         # [a, j]

        panels = []
        for ag in range(M):
            other = 1 - ag
            occ_ag = env.world.occupancy_torch[e:e + 1, ag]
            prob_ag = (occ_ag[0].float() / 2.0).clamp(0, 1).cpu().numpy() if occ_ag.dtype != torch.float32 \
                      else occ_ag[0].cpu().numpy()
            # occupancy_torch is categorical {0 unk,1 free,2 obst}; map to a prob-like shade.
            cat = env.world.occupancy_torch[e, ag].cpu().numpy()
            prob_ag = np.where(cat == 1, 0.9, np.where(cat == 2, 0.1, 0.5)).astype(np.float32)
            frontier_ag = compute_frontier(occ_ag)[0].cpu().numpy()
            nv_ag = rg["node_valid"][e, ag].cpu().numpy()
            evalid_ag = rg["edge_valid"][e, ag].cpu().numpy()
            curr_ag = int(rg["curr_idx"][e, ag])
            # Belief heatmap = this agent's posterior over the OTHER agent's node.
            bel = belief_p[ag, other].copy()
            if bel.max() > 0:
                bel = np.sqrt(bel / bel.max())              # peak-normalize + √-gamma so diffuse tails show
            # Show belief on ALL nodes it lives on (not just strict-valid): union valid for drawing.
            nv_draw = nv_ag | (bel > 1e-4)
            lkp = env.last_known_pos[e, ag, other].cpu().numpy()   # stale last-known point
            comm_now = bool(cm[ag, other])
            label = f"A{ag}→A{other} belief" + ("  [IN COMM]" if comm_now else f"  {'ALIVE' if alive[ag,other] else 'LOST'}")
            im = composite_frame(
                prob=prob_ag, gt=gt_np, frontier=frontier_ag,
                nxy=nxy, nv=nv_draw, util=bel, curr=curr_ag,
                agent_xy=trails[ag][-1], trail=trails[ag][-40:],
                step=t, explored=float((cat != 0).mean()),
                draw_edges=False, eidx=eidx, evalid=evalid_ag,
                target_xy=(float(lkp[0]), float(lkp[1])),      # marker at last-known position
                extra_agents_xy=[trails[other][-1]],           # the teammate's TRUE position (dot)
                extra_agents_trails=[trails[other][-40:]],
                extra_agent_indices=[other],
                agent_idx=ag, agent_label=label,
            )
            panels.append(np.array(im))
        frames.append(hstack_frames(panels))
        if bool(done[e]) if torch.is_tensor(done) else bool(done):
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=90, loop=0)
    print(f"[save] {args.out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
