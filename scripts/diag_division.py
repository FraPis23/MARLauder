"""Test whether agents divide exploration DELIBERATELY using the (exact) teammate position,
vs separating by chance (fusion/reward removing overlap reactively).

Method — causal counterfactual on a checkpoint, exact positions shared:
  At sampled states, for the focal agent (0):
    1. baseline: teammate believed FAR away  → record focal's chosen target + action.
    2. perturb:  teammate believed ON the focal's own preferred frontier (its baseline target)
                 → re-run policy.
  If the focal now picks a DIFFERENT frontier / steps a DIFFERENT way → it is yielding the
  contested frontier to the teammate = DELIBERATE, position-driven division.
  If its choice is invariant to where the teammate is → separation is NOT position-driven
  (accidental). The teammate signal reaches the net via last_known_pos → node_feat[5] marker,
  teammate-BF distance, and the strategic head's cand features (min_team_dist, own_minus_team).

  target_yield_rate : fraction of states where the strategic TARGET moves off the contested
                      frontier when the teammate is placed on it (head only; needs a head).
  action_yield_rate : fraction where the chosen ACTION (next hop) changes (works for any arch,
                      incl. single-pointer — its GAT still sees the teammate marker).

Usage:
  docker exec marlauder bash -lc 'cd /workspace/MARLauder && \
    PYTHONPATH=/workspace/MARLauder python scripts/diag_division.py --ckpt runs/run_phase0/final.pt'
"""
import argparse
import torch

from env.maps import load_split
from env.explorer import EnvCfg, Explorer
from models.actor_critic import MarlActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, nargs="+", default=[120, 1543, 2877, 5530, 9904])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    ecfg = EnvCfg.from_ckpt_dict(ck["cfg"]["env"], n_envs=1, n_agents=2)
    disable_strategic = bool(ck["cfg"].get("disable_strategic", False))
    split = load_split(args.split, device=args.device)
    env = Explorer(split, ecfg, seed=args.map_idx[0])

    m = MarlActorCritic(n_agents=2, disable_strategic=disable_strategic).to(args.device)
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in ck["model"].items()}
    if "path_bias" in sd and "path_bias_learn" not in sd:
        sd["path_bias_learn"] = sd.pop("path_bias")
    m.load_state_dict(sd, strict=False)
    m.eval()

    def focal_choice(obs, ha, hc):
        o = m.act(obs, ha, hc, deterministic=True)
        tgt_slot = int(o["target_argmax"][0, 0].item())
        tgt_node = int(obs["cand_idx"][0, 0, tgt_slot].item()) if tgt_slot < obs["cand_idx"].shape[-1] else -1
        act = int(o["action"][0, 0].item())
        return tgt_node, act, o

    tgt_yield = tgt_tot = 0
    act_yield = act_tot = 0
    far_xy = torch.tensor([float(env.W) - 1.0, float(env.H) - 1.0], device=args.device)  # corner

    for midx in args.map_idx:
        env.reload_map(env_idx=0, map_idx=int(midx))
        ha, hc = m.init_hidden(1, args.device)
        obs = env.obs
        for _ in range(args.steps):
            # Baseline with teammate believed FAR (so the contested frontier is "free").
            saved_lkp = env.last_known_pos.clone()
            env.last_known_pos[0, 0, 1, :] = far_xy
            env._refresh_obs(comm_mask=torch.ones((1, 2, 2), dtype=torch.bool, device=args.device))
            base_tgt, base_act, _ = focal_choice(env.obs, ha, hc)

            # Perturb: place teammate ON the focal's own preferred frontier.
            if base_tgt >= 0:
                contested_xy = env.graph.node_xy[base_tgt]
                env.last_known_pos[0, 0, 1, :] = contested_xy
                env._refresh_obs(comm_mask=torch.ones((1, 2, 2), dtype=torch.bool, device=args.device))
                pert_tgt, pert_act, _ = focal_choice(env.obs, ha, hc)
                if not disable_strategic:
                    tgt_tot += 1
                    tgt_yield += int(pert_tgt != base_tgt)
                act_tot += 1
                act_yield += int(pert_act != base_act)

            # Restore real belief, then step the env with the REAL-obs greedy action.
            env.last_known_pos.copy_(saved_lkp)
            env._refresh_obs(comm_mask=torch.ones((1, 2, 2), dtype=torch.bool, device=args.device))
            o = m.act(env.obs, ha, hc, deterministic=True)
            obs, _r, d, _info = env.step(o["action"], target_choice=o["target_argmax"])
            ha, hc = o["hidden_actor"], o["hidden_critic"]
            if bool(d[0].item()):
                break

    def rate(h, t):
        return (h / t) if t else float("nan")
    arch = "single-pointer (no head)" if disable_strategic else "strategic head"
    print(f"ckpt={args.ckpt}  arch={arch}  maps={args.map_idx}")
    if not disable_strategic:
        print(f"target_yield_rate = {rate(tgt_yield, tgt_tot):.2f}  (n={tgt_tot})  "
              "← target moves off the contested frontier when teammate placed on it")
    print(f"action_yield_rate = {rate(act_yield, act_tot):.2f}  (n={act_tot})  "
          "← next-hop changes when teammate placed on the focal's target")
    print("HIGH (≳0.4) ⇒ DELIBERATE position-driven division.  ~0 ⇒ position-blind (accidental).")


if __name__ == "__main__":
    main()
