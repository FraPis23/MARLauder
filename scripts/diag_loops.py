"""Diagnose the persistent 2-node limit-cycle at execution — v2 (detailed).

v1 found: in loops the target was far/frozen/outside-window with a confident-but-oscillating
pointer. Fixes applied (BF-first-hop direction + proximal-goal annotation) did NOT remove the
loops. v2 collects the signals that DECIDE between the remaining causes:

  A. head thrash            → target_changed high
  B. BF direction unstable  → dir_changed high (the geodesic first-hop itself flips A↔B)
  C. pointer ignores dir    → action_follows_dir LOW (we feed the right direction, GRU/pointer
                              overrides it → the fix is structurally present but behaviorally inert)
  D. orbiting a dead target → target_utility≈0 and/or dist_to_target small (agent reached the
                              target, head won't drop it → oscillates next to an exhausted frontier)

Per step / per agent it records position, current node, action + which node it leads to, pointer
logit margin, head pick + full candidate logits + margin, the BF-first-hop direction WE FEED +
whether the action follows it + whether that direction flips, target utility + euclid distance to
target, BF path length, target-in-window, proximal nothing, per-node revisit count, reward terms.
Auto-detects 2-node loops and prints a loop-vs-baseline contrast.

Usage:
  docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=/workspace/MARLauder \
    python scripts/diag_loops.py --ckpt runs/run_C_h6/final.pt --split train/difficult \
      --map-idx 120 999 1543 2877 3500 --out runs/loops_C_h6.npz'
"""
import argparse
from collections import Counter
import numpy as np
import torch

from env.maps import load_split
from env.explorer import EnvCfg, Explorer
from models.actor_critic import MarlActorCritic

K = 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train/difficult")
    ap.add_argument("--map-idx", type=int, nargs="+", default=[120, 999, 1543, 2877, 3500])
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--out", default="runs/loops.npz")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    n_agents = int(ck["cfg"].get("n_agents", ck["cfg"]["env"].get("n_agents", 2)))
    ecfg = EnvCfg.from_ckpt_dict(ck["cfg"]["env"], n_envs=1, n_agents=n_agents)
    disable_strategic = bool(ck["cfg"].get("disable_strategic", False))
    n_hops = int(ck["cfg"].get("n_hops", ck["cfg"]["env"].get("n_hops", 6)))
    nr = float(ecfg.nr)
    split = load_split(args.split, device=args.device)
    env = Explorer(split, ecfg, seed=args.map_idx[0])
    m = MarlActorCritic(n_agents=n_agents, n_layers=n_hops, disable_strategic=disable_strategic).to(args.device)
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in ck["model"].items()}
    m.load_state_dict(sd, strict=False)
    m.eval()

    cols = ("map", "t", "agent", "curr_node", "action", "action_node",
            "ptr_logit_margin", "act_entropy",
            "target_kslot", "target_node", "target_changed", "target_utility",
            "tgt_logit_margin", "target_in_window", "bf_path_len", "dist_to_target",
            "dir_slot", "dir_changed", "action_follows_dir",
            "revisit_count", "prev_branch_match",
            "reward", "r_revisit", "r_target_switch", "r_stall", "explored", "in_loop")
    rec = {c: [] for c in cols}

    for midx in args.map_idx:
        env.reload_map(env_idx=0, map_idx=int(midx))
        ha, hc = m.init_hidden(1, args.device)
        hist = [[], []]; prev_tgt = [None, None]; prev_dir = [None, None]
        visits = [Counter(), Counter()]
        for t in range(args.steps):
            obs = env.obs
            out = m.act(obs, ha, hc, deterministic=True)
            l2g = obs["local_to_global"][0]; cand_idx = obs["cand_idx"][0]
            bf_par = obs["bf_parent_from_curr"][0]; curr_g = obs["curr_idx_global"][0]
            cand_feat = obs["cand_feat"][0]; cand_util = obs["cand_utility"][0]
            cand_xy = obs["cand_xy"][0]; cbfh = obs["cand_bf_first_hop"][0]   # [M, Kc, 8]
            nbr_g = obs["curr_nbr_global"][0]                                  # [M, 8]
            tlog = out.get("target_logits"); plog = out["logits"][0]
            Kc = cand_idx.shape[-1]
            for ag in range(2):
                cn = int(curr_g[ag].item()); hist[ag].append(cn); visits[ag][cn] += 1
                h = hist[ag]
                in_loop = (len(h) >= 4 and h[-1] == h[-3] and h[-2] == h[-4] and h[-1] != h[-2])
                a = int(out["action"][0, ag].item())
                anode = int(nbr_g[ag, a].item())
                k = int(out["target_choice"][0, ag].item())
                tnode = int(cand_idx[ag, k].item()) if 0 <= k < Kc else -1
                tutil = float(cand_util[ag, k].item()) if 0 <= k < Kc else float("nan")
                txy = cand_xy[ag, k] if 0 <= k < Kc else torch.tensor([float("nan")] * 2)
                dist_t = float(torch.hypot(env.pos[0, ag, 0] - txy[0], env.pos[0, ag, 1] - txy[1]).item()) / nr
                # head margin
                if tlog is not None:
                    tl = tlog[0, ag].clone(); tl[~obs["cand_valid"][0, ag]] = float("-inf")
                    s = torch.sort(tl, descending=True).values
                    tmar = float(s[0]) - (float(s[1]) if s.numel() > 1 and torch.isfinite(s[1]) else float(s[0]))
                else:
                    tmar = float("nan")
                # pointer margin/entropy
                am = obs["action_mask"][0, ag]; pl = plog[ag].clone(); pl[~am] = float("-inf")
                ps = torch.sort(pl, descending=True).values
                pmar = float(ps[0]) - (float(ps[1]) if ps.numel() > 1 and torch.isfinite(ps[1]) else float(ps[0]))
                pe = float(torch.distributions.Categorical(logits=pl.nan_to_num(neginf=-1e4)).entropy().item())
                # the BF-first-hop direction WE FEED the actor for the chosen target
                dhop = cbfh[ag, k] if 0 <= k < Kc else torch.zeros(K)
                dir_slot = int(dhop.argmax().item()) if float(dhop.sum()) > 0 else -1
                follows = int(dir_slot >= 0 and a == dir_slot)
                dir_changed = int(prev_dir[ag] is not None and dir_slot != prev_dir[ag]); prev_dir[ag] = dir_slot
                in_win = bool((l2g[ag] == tnode).any().item()) if tnode >= 0 else False
                # bf path length target→curr
                plen = 0
                if tnode >= 0:
                    cur = tnode
                    for _ in range(256):
                        if cur == cn:
                            break
                        par = int(bf_par[ag, cur].item())
                        if par < 0 or par == cur:
                            break
                        cur = par; plen += 1
                changed = int(prev_tgt[ag] is not None and tnode != prev_tgt[ag]); prev_tgt[ag] = tnode
                pbm = float(cand_feat[ag, k, -1].item()) if 0 <= k < Kc else float("nan")
                vals = [int(midx), t, ag, cn, a, anode, pmar, pe, k, tnode, changed, tutil,
                        tmar, int(in_win), plen, dist_t, dir_slot, dir_changed, follows,
                        visits[ag][cn], pbm]
                for c, v in zip(cols[:21], vals):
                    rec[c].append(v)
            obs, reward, done, info = env.step(out["action"], target_choice=out["target_argmax"])
            ha, hc = out["hidden_actor"], out["hidden_critic"]
            rt = info["reward_terms"]
            for ag in range(2):
                rec["reward"].append(float(reward[0, ag].item()))
                rec["r_revisit"].append(float(rt["revisit"].item()))
                rec["r_target_switch"].append(float(rt["target_switch"].item()))
                rec["r_stall"].append(float(rt["stall"].item()))
                rec["explored"].append(float(info["explored_rate"][0].item()))
                # in_loop appended last (recompute index): set per the two records just added
            # in_loop flags for the two agents this step:
            for ag in range(2):
                h = hist[ag]
                rec["in_loop"].append(int(len(h) >= 4 and h[-1] == h[-3] and h[-2] == h[-4] and h[-1] != h[-2]))
            if bool(done[0].item()):
                break

    arr = {k: np.array(v) for k, v in rec.items()}
    np.savez_compressed(args.out, **arr)
    print(f"saved {len(arr['t'])} step-records → {args.out}")

    loop = arr["in_loop"] == 1; base = ~loop
    n_loop, n_tot = int(loop.sum()), len(loop)
    print(f"\nloop_rate = {n_loop/max(1,n_tot):.3f}  ({n_loop}/{n_tot} agent-steps)")
    if n_loop == 0:
        print("no 2-node loops on these maps."); return

    def cmp(name, key, fmt="{:.3f}"):
        lv = float(np.nanmean(arr[key][loop])); bv = float(np.nanmean(arr[key][base])) if base.any() else float("nan")
        print(f"  {name:22s} loop={fmt.format(lv):>9}  baseline={fmt.format(bv):>9}")

    print("\nloop vs baseline (the cause = what's most different):")
    cmp("action_follows_dir", "action_follows_dir")   # C: LOW in loops ⇒ pointer IGNORES the BF dir we feed
    cmp("dir_changed", "dir_changed")                  # B: HIGH ⇒ the geodesic first-hop itself flips A↔B
    cmp("target_changed", "target_changed")            # A: HIGH ⇒ head thrash
    cmp("target_utility", "target_utility")            # D: ~0 ⇒ orbiting an exhausted target
    cmp("dist_to_target", "dist_to_target", "{:.2f}")  # D: small ⇒ agent already AT the target
    cmp("target_in_window", "target_in_window")
    cmp("bf_path_len", "bf_path_len", "{:.1f}")
    cmp("tgt_logit_margin", "tgt_logit_margin")
    cmp("ptr_logit_margin", "ptr_logit_margin")
    cmp("revisit_count", "revisit_count", "{:.1f}")
    cmp("prev_branch_match", "prev_branch_match")
    print("\nRead: action_follows_dir↓ ⇒ pointer ignores direction (cause C, deepest). "
          "dir_changed↑ ⇒ BF first-hop oscillates (cause B). target_changed↑ ⇒ head thrash (A). "
          "target_utility≈0 / dist_to_target small ⇒ orbiting a dead target (cause D).")


if __name__ == "__main__":
    main()
