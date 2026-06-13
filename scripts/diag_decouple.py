"""Diagnose strategic↔tactical decoupling, esp. after a map-exchange (comm) event.

User observation: the strategic target (yellow) correctly points at real frontiers, but the
agent's MOVEMENT ignores it and re-explores the teammate's just-received area. This script
quantifies that, deterministically, on a fixed map:

  follow_rate          — steps where the agent's action == its committed target's BF first-hop
                         (does the tactical pointer obey the strategic target at all?)
  unproductive_rate    — steps where the agent found 0 union-new cells (wasted / redundant move)

Both are reported OVERALL and restricted to the W steps AFTER a fusion delivered new cells
(comm_event). If post-comm follow_rate << overall, or post-comm unproductive_rate >> overall,
the failure is the tactical layer not re-planning on new info (→ refresh GRU / boost path_bias /
add an "already-known neighbor" obs feature).

Usage:
  docker exec marlauder bash -lc 'cd /workspace/MARLauder && \
    PYTHONPATH=/workspace/MARLauder python scripts/diag_decouple.py \
    --ckpt runs/run_phase1/final.pt --map-idx 120'
"""
import argparse
import numpy as np
import torch

from env.maps import load_split
from env.explorer import EnvCfg, Explorer
from models.actor_critic import MarlActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, nargs="+", default=[120, 1543, 2877, 5530, 9904])
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--post-comm-window", type=int, default=4, help="steps after a comm_event counted as 'post-comm'")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ecfg = EnvCfg.from_ckpt_dict(ck["cfg"]["env"], n_envs=1, n_agents=2)
    split = load_split(args.split, device=args.device)
    env = Explorer(split, ecfg, seed=args.map_idx[0])
    M = ecfg.n_agents

    m = MarlActorCritic(n_agents=M).to(args.device)
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in ck["model"].items()}
    if "path_bias" in sd and "path_bias_learn" not in sd:
        sd["path_bias_learn"] = sd.pop("path_bias")
    m.load_state_dict(sd, strict=False)
    m.eval()
    W = args.post_comm_window

    # Accumulators (over all maps): [overall, post_comm].
    follow_hit = [0, 0]; follow_tot = [0, 0]
    unprod_hit = [0, 0]; unprod_tot = [0, 0]

    for midx in args.map_idx:
        env.reload_map(env_idx=0, map_idx=int(midx))
        ha, hc = m.init_hidden(1, args.device)
        obs = env.obs
        novel_prev = torch.zeros(M, device=args.device)
        since_comm = [10 ** 9] * M     # steps since last comm_event per agent
        for _ in range(args.steps):
            o = m.act(obs, ha, hc, deterministic=True)
            act = o["action"][0]                                   # [M] pointer slot 0..7
            ta = o["target_argmax"][0]                             # [M] committed cand slot
            cbfh = obs["cand_bf_first_hop"][0]                     # [M, K_cand, 8]
            K_cand = cbfh.shape[0 + 1] if cbfh.dim() == 3 else cbfh.shape[1]
            for ag in range(M):
                slot = int(ta[ag].item())
                fh = cbfh[ag, slot]                                # [8]
                bucket_list = [0] + ([1] if since_comm[ag] < W else [])
                if float(fh.sum().item()) > 0:                     # committed target has a valid first hop
                    hop = int(fh.argmax().item())
                    hit = int(int(act[ag].item()) == hop)
                    for b in bucket_list:
                        follow_hit[b] += hit; follow_tot[b] += 1
            obs, _r, d, info = env.step(o["action"], target_choice=o["target_argmax"])
            ha, hc = o["hidden_actor"], o["hidden_critic"]
            novel_now = info["novel_cells_ep"][0]
            nstep = (novel_now - novel_prev).clamp(min=0.0)        # [M] union-new this step
            novel_prev = novel_now.clone()
            ce = obs["comm_event"][0]                              # [M] bool (post-step)
            for ag in range(M):
                bucket_list = [0] + ([1] if since_comm[ag] < W else [])
                unprod = int(float(nstep[ag].item()) == 0.0)
                for b in bucket_list:
                    unprod_hit[b] += unprod; unprod_tot[b] += 1
                since_comm[ag] = 0 if bool(ce[ag].item()) else since_comm[ag] + 1
            if bool(d[0].item()):
                break

    def rate(h, t):
        return (h / t) if t else float("nan")
    print(f"ckpt={args.ckpt}  maps={args.map_idx}  post_comm_window={W}")
    print(f"follow_rate (action == committed first-hop):  overall={rate(follow_hit[0],follow_tot[0]):.2f} "
          f"(n={follow_tot[0]})   post_comm={rate(follow_hit[1],follow_tot[1]):.2f} (n={follow_tot[1]})")
    print(f"unproductive_rate (0 union-new cells/step):   overall={rate(unprod_hit[0],unprod_tot[0]):.2f} "
          f"(n={unprod_tot[0]})   post_comm={rate(unprod_hit[1],unprod_tot[1]):.2f} (n={unprod_tot[1]})")
    print("READ: post_comm follow << overall, or post_comm unproductive >> overall  ⇒  tactical "
          "layer ignores the (correct) target after a map exchange.")


if __name__ == "__main__":
    main()
