"""Verify the actor_pre proportionality concern: do the small-dim signals
(next_hop_dir = 8, prev_action = 8) get drowned by the 128-dim embeddings
(curr_emb, strategic_emb) inside actor_pre = Linear(2*d + 2*K -> d)?

Two complementary measures over a real greedy rollout of a checkpoint:

  (1) INJECTION magnitude — actor_pre is linear, so
        actor_in = b + W_curr@curr + W_strat@strat + W_dir@dir + W_prev@prev
      Report mean ||W_block @ x_block|| per block = how much signal each block
      pushes into the 128-dim pre-activation. Blocks that inject ~0 are drowned.
      (Norms don't sum to ||actor_in|| because blocks aren't orthogonal — this is
       a relative-magnitude read, not a variance split.)

  (2) ABLATION KL — zero one block at the input of actor_pre, re-run the full
      actor (GRU + pointer), and measure KL(P_full || P_ablated) on the action
      distribution. This is the causal downstream impact on the POLICY, which is
      what actually matters (a small injection can still swing the argmax).

Block layout in the concat [curr_emb | strategic_emb | next_hop_dir | prev_action]:
  curr  = [0:d)   strat = [d:2d)   dir = [2d:2d+K)   prev = [2d+K:2d+2K)

Usage:
  docker exec marlauder bash -lc 'cd /workspace/MARLauder && \
    PYTHONPATH=/workspace/MARLauder python scripts/diag_actor_pre.py \
      --ckpt runs/v06_perf_learned/final.pt'
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from env.maps import load_split
from env.explorer import EnvCfg, Explorer
from models.actor_critic import MarlActorCritic, K as KDIM


def load_model(ckpt: str, device: str, n_agents: int):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    ecfg = EnvCfg.from_ckpt_dict(ck["cfg"]["env"], n_envs=1, n_agents=n_agents)
    disable_strategic = bool(ck["cfg"].get("disable_strategic", False))
    target_mode = str(ck["cfg"].get("target_mode", "analytic"))
    strategic_gate_eps = float(ck["cfg"].get("strategic_gate_eps", 0.0))
    m = MarlActorCritic(
        n_agents=n_agents,
        disable_strategic=disable_strategic,
        target_mode=target_mode,
        strategic_gate_eps=strategic_gate_eps,
    ).to(device)
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in ck["model"].items()}
    # Drop params whose shape no longer matches (e.g. critic_pre grew by CRITIC_GLOBAL_DIM
    # in later code). The actor path we measure (encoder/actor_pre/gru_actor/pointer) must
    # match exactly or we abort — only non-actor mismatches are tolerated.
    cur = m.state_dict()
    keep, dropped = {}, []
    for k, v in sd.items():
        if k in cur and cur[k].shape == v.shape:
            keep[k] = v
        else:
            dropped.append(k)
    actor_keys = [k for k in dropped if k.startswith(("encoder.", "actor_pre", "gru_actor", "pointer", "strategic_head"))]
    if actor_keys:
        raise SystemExit(f"ABORT: actor-path shape mismatch (ckpt != current code): {actor_keys}")
    if dropped:
        print(f"[warn] skipped {len(dropped)} non-actor params (shape drift): {dropped}")
    m.load_state_dict(keep, strict=False)
    m.eval()
    return m, ecfg, disable_strategic, target_mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, nargs="+", default=[120, 1543, 2877, 5530, 9904])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    m, ecfg, disable_strategic, target_mode = load_model(args.ckpt, args.device, args.n_agents)
    d = m.d
    K = KDIM
    blocks = {
        "curr_emb":  slice(0, d),
        "strat_emb": slice(d, 2 * d),
        "next_dir":  slice(2 * d, 2 * d + K),
        "prev_act":  slice(2 * d + K, 2 * d + 2 * K),
    }

    # Hook state: capture the concat input x, and optionally zero one block before the Linear.
    cap = {"x": None, "zero": None}

    def pre_hook(mod, inp):
        x = inp[0]
        cap["x"] = x.detach().clone()
        if cap["zero"] is not None:
            x = x.clone()
            x[:, blocks[cap["zero"]]] = 0.0
            return (x,)
        return None

    m.actor_pre.register_forward_pre_hook(pre_hook)
    W = m.actor_pre.weight.detach()   # [d, 2d+2K]

    # Accumulators. "*_act" = restricted to agent-rows where the block input is nonzero
    # (||x_block||>0), so the gate zeroing next_dir on local-utility steps does not dilute it.
    inj = {k: 0.0 for k in blocks}          # sum mean ||W_block x_block|| (all rows)
    inj_act = {k: 0.0 for k in blocks}      # sum over active rows
    inj_act_n = {k: 0 for k in blocks}      # active-row count
    inj_n = 0
    actor_in_norm = 0.0
    kl = {k: 0.0 for k in blocks}           # sum mean KL over agents (all rows)
    kl_act = {k: 0.0 for k in blocks}
    kl_act_n = {k: 0 for k in blocks}
    kl_n = 0

    split = load_split(args.split, device=args.device)
    env = Explorer(split, ecfg, seed=args.map_idx[0])

    def dist_from_logits(logits, amask):
        # logits [N,M,K]; mask invalid to -inf for a proper categorical
        lg = logits.masked_fill(~amask.bool(), -1e4)
        return F.softmax(lg, dim=-1)

    for midx in args.map_idx:
        env.reload_map(env_idx=0, map_idx=int(midx))
        ha, hc = m.init_hidden(1, args.device)
        for _ in range(args.steps):
            obs = env.obs
            amask = obs["action_mask"]                       # [1,M,K]

            # --- full forward (captures x via hook) ---
            cap["zero"] = None
            with torch.no_grad():
                out_full = m.act(obs, ha, hc, deterministic=True)
            x = cap["x"]                                      # [M, 2d+2K]
            P_full = dist_from_logits(out_full["logits"], amask)  # [1,M,K]

            # (1) injection magnitude per block
            for name, sl in blocks.items():
                contrib = (x[:, sl] @ W[:, sl].T).norm(dim=-1)   # [M]
                inj[name] += contrib.mean().item()
                active = x[:, sl].norm(dim=-1) > 1e-8            # [M] block actually fed
                if active.any():
                    inj_act[name] += contrib[active].sum().item()
                    inj_act_n[name] += int(active.sum().item())
            actor_in_norm += (x @ W.T).norm(dim=-1).mean().item()
            inj_n += 1

            # (2) ablation KL per block
            for name, sl in blocks.items():
                cap["zero"] = name
                with torch.no_grad():
                    out_abl = m.act(obs, ha, hc, deterministic=True)
                cap["zero"] = None
                P_abl = dist_from_logits(out_abl["logits"], amask)
                kld = (P_full * (P_full.clamp_min(1e-9).log()
                                 - P_abl.clamp_min(1e-9).log())).sum(-1).squeeze(0)  # [M]
                kl[name] += kld.mean().item()
                active = x[:, sl].norm(dim=-1) > 1e-8
                if active.any():
                    kl_act[name] += kld[active].sum().item()
                    kl_act_n[name] += int(active.sum().item())
            kl_n += 1

            # (3) GRU memory ablation — zero the recurrent hidden state this step, re-run,
            # measure KL. High KL ⇒ the recurrent memory actually drives the current decision.
            with torch.no_grad():
                out_h0 = m.act(obs, torch.zeros_like(ha), hc, deterministic=True)
            P_h0 = dist_from_logits(out_h0["logits"], amask)
            kld_h = (P_full * (P_full.clamp_min(1e-9).log()
                               - P_h0.clamp_min(1e-9).log())).sum(-1).mean().item()
            kl.setdefault("_gru_hidden", 0.0)
            kl["_gru_hidden"] += kld_h
            cap["_hn"] = cap.get("_hn", 0) + 1

            # advance with the real greedy action
            o = m.act(obs, ha, hc, deterministic=True)
            obs, _r, done, _info = env.step(o["action"], target_choice=o["target_argmax"])
            ha, hc = o["hidden_actor"], o["hidden_critic"]
            if bool(done[0].item()):
                break

    print(f"ckpt={args.ckpt}")
    print(f"arch: disable_strategic={disable_strategic}  target_mode={target_mode}  "
          f"(strat_emb is ZERO in analytic/ablated archs)")
    print(f"\n(1) INJECTION ||W_block @ x_block|| into actor_in (d={d}); "
          f"mean ||actor_in||={actor_in_norm/max(inj_n,1):.3f}")
    base = max((inj[k] for k in blocks), default=1.0) or 1.0
    print("  block       dim     inj(all)  inj(active)  active%")
    for k, sl in blocks.items():
        dim = sl.stop - sl.start
        v = inj[k] / max(inj_n, 1)
        va = inj_act[k] / inj_act_n[k] if inj_act_n[k] else 0.0
        pct = 100.0 * inj_act_n[k] / (inj_n * args.n_agents) if inj_n else 0.0
        bar = "#" * int(40 * (inj[k] / base))
        print(f"  {k:10s} {dim:4d}  {v:9.3f}  {va:10.3f}  {pct:5.0f}%  {bar}")
    print("\n(2) ABLATION KL(P_full || P_zero_block) on action dist "
          "(higher = block matters more to the POLICY)")
    base2 = max((kl[k] for k in blocks), default=1.0) or 1.0
    print("  block       dim      KL(all)   KL(active)")
    for k, sl in blocks.items():
        dim = sl.stop - sl.start
        v = kl[k] / max(kl_n, 1)
        va = kl_act[k] / kl_act_n[k] if kl_act_n[k] else 0.0
        bar = "#" * int(40 * (kl[k] / base2))
        print(f"  {k:10s} {dim:4d}  {v:9.4f}  {va:10.4f}  {bar}")
    if "_gru_hidden" in kl:
        print(f"\n(3) GRU MEMORY: KL(P_full || P_zero_hidden) = "
              f"{kl['_gru_hidden']/max(cap.get('_hn',1),1):.4f}   "
              "(high ⇒ recurrent state drives the decision; ~0 ⇒ GRU unused / reactive)")
    print("\nREAD: if next_dir/prev_act show inj≈0 AND KL≈0 vs curr_emb ⇒ small-dim "
          "signals are drowned (disproportion confirmed). High KL despite low inj ⇒ "
          "small injection still swings the argmax (no real problem).")


if __name__ == "__main__":
    main()
