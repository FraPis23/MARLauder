#!/usr/bin/env python3
"""What the POLICY actually sees from the teammate belief, on the training split. No training.

    python scripts/pf_obs_diag.py --ckpt runs/<run>/ckpt_best.pt --split train/easy \
        --label v14 --min-unknown 1 --min-util 0.10

v14 ran 60 iterations below v13 on every eval block. Between the two there are a dozen belief
changes AND a new frontier gate, so "which one costs coverage" cannot be answered by staring at
the module. It can be answered by measuring the two OBSERVATION CHANNELS the belief feeds:

    feat[4] teammate_pot   — BF-proximity potential toward the believed teammate position
    feat[6] radar_teammate — the belief FIELD itself, mass-transported beyond the ego window
                             (this is what `--radar-team-source belief` puts in the obs)

A belief that is merely mistaken still gives the policy a usable gradient. A belief that is DEAD
(alive=False → channel zeroed) or FLAT (spread over the whole known map → no direction) does not,
and the policy is then strictly worse off than with the old one. Those two failure modes are
exactly what this session's changes could have introduced — `alive` now goes False when the field
is emptied, and the terminal corner equidistributes over known-but-unheard ground — so they are
what is measured here, per step, over real episodes driven by a real policy.

Run the same checkpoint and seed under different belief settings and compare the columns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from env.explorer import EnvCfg, Explorer  # noqa: E402
from env.maps import load_split  # noqa: E402
from eval.ckpt_loader import load_model_from_ckpt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--maps", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--label", default="cfg")
    ap.add_argument("--min-unknown", type=int, default=None)
    ap.add_argument("--min-util", type=float, default=None)
    ap.add_argument("--impl", type=Path, default=None,
                    help="swap in a different copy of env/teammate_belief_pathfront.py")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.impl is not None:
        # Replace the module IN PLACE in sys.modules before Explorer imports from it, so the whole
        # env uses the alternative copy without touching the working tree.
        import importlib.util
        spec = importlib.util.spec_from_file_location("env.teammate_belief_pathfront", str(args.impl))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["env.teammate_belief_pathfront"] = mod
        import env.explorer as _ex
        _ex.advance_pathfront = mod.advance_pathfront
        _ex.freeze_hypotheses = mod.freeze_hypotheses

    model, env_peek = load_model_from_ckpt(args.ckpt, args.device, n_agents=2)
    split = load_split(args.split, device=args.device)
    cfg_d = dict(env_peek or {})
    cfg_d["use_teammate_belief"] = True
    cfg_d["belief_mode"] = "pathfront"
    cfg_d["radar_team_source"] = "belief"
    if args.min_unknown is not None:
        cfg_d["pf_frontier_min_unknown"] = int(args.min_unknown)
    if args.min_util is not None:
        cfg_d["pf_frontier_min_util"] = float(args.min_util)
    model.eval()

    agg = dict(steps=0, oo=0, alive=0, sump=0.0, onfr=0.0, nfr=0.0, flat=0,
               f4max=0.0, f4nz=0.0, f6max=0.0, f6nz=0.0, explored=0.0, n_ep=0,
               hasfr=0, deadfr=0)
    for mi in args.maps:
        cfg = EnvCfg.from_ckpt_dict(cfg_d, n_envs=1, n_agents=2, max_episode_steps=args.steps + 1)
        env = Explorer(split, cfg, seed=int(mi))
        env.store_render_global = True
        env.reload_map(env_idx=0, map_idx=int(mi))
        env.cfg.done_explored_thresh = 2.0
        env.cfg.max_travel_px = 0.0
        env.cfg.max_travel_frac = 0.0
        env.cfg.max_episode_steps = args.steps + 5
        h_act, h_crit = model.init_hidden(1, args.device)
        obs = env.obs
        with torch.no_grad():
            for _ in range(args.steps):
                out = model.act(obs, h_act, h_crit, deterministic=True)
                rg = env._render_global
                bp, bal, cm = rg.get("belief_p"), rg.get("belief_alive"), rg.get("comm_mask")
                nf = rg["node_feat"][0, 0]                       # [N_max, F] agent 0, global
                nv = rg["node_valid"][0, 0].bool()
                agg["steps"] += 1
                if bp is not None and not bool(cm[0, 0, 1]):
                    agg["oo"] += 1                                # out of comm: the belief matters
                    a_ok = bool(bal[0, 0, 1])
                    agg["alive"] += int(a_ok)
                    p = bp[0, 0, 1]
                    tot = float(p.sum())
                    agg["sump"] += tot
                    fr = rg["belief_frontier"][0, 0].bool()
                    agg["nfr"] += float(fr.sum())
                    # A DEAD CHANNEL AND AN EXHAUSTED MAP ARE NOT THE SAME THING, and reading
                    # `alive` alone confuses them: on train/easy the episodes finish at explored
                    # 1.0000, so a large share of the out-of-comm steps happen with NO FRONTIER
                    # ANYWHERE, where an empty belief is the honest answer (that is exactly what
                    # 08_no_target_left encodes). Only "empty WHILE openings existed" is a fault.
                    agg["hasfr"] += int(bool(fr.any()))
                    agg["deadfr"] += int(bool(fr.any()) and tot <= 1e-6)
                    agg["onfr"] += float((p * fr.float()).sum())
                    # FLAT = the field carries no direction: its peak is within 3x the mean over
                    # known ground. A uniform spread over the known map scores 1.0 here and is
                    # worth exactly nothing to a policy trying to decide WHERE the teammate is.
                    k = int(nv.sum())
                    if k > 0 and tot > 1e-6:
                        agg["flat"] += int(float(p.max()) < 3.0 * tot / k)
                    f4, f6 = nf[:, 4], nf[:, 6]
                    agg["f4max"] += float(f4.max())
                    agg["f4nz"] += float((f4[nv] > 1e-4).float().mean()) if k else 0.0
                    agg["f6max"] += float(f6.max())
                    agg["f6nz"] += float((f6[nv] > 1e-4).float().mean()) if k else 0.0
                obs, _, done, info = env.step(out["action"])
                if bool(done[0]):
                    break
        agg["explored"] += float(info["explored_rate"][0])
        agg["n_ep"] += 1

    o = max(1, agg["oo"])
    print(f"\n=== {args.label} · {args.split} · maps {args.maps} · {args.ckpt.name} ===")
    print(f"  steps {agg['steps']}   out-of-comm {agg['oo']} ({100*agg['oo']/max(1,agg['steps']):.0f}%)"
          f"   explored (mean) {agg['explored']/max(1,agg['n_ep']):.4f}")
    print(f"  belief ALIVE while out of comm : {100*agg['alive']/o:5.1f}%   "
          f"(of which {100*(o-agg['hasfr'])/o:.0f}% of steps had NO frontier on the map at all)")
    print(f"  belief EMPTY *while* openings exist: {100*agg['deadfr']/max(1,agg['hasfr']):5.1f}%"
          f"   <- THIS is the dead-channel number; `alive` alone confuses it with a finished map")
    print(f"  mean Sigma p                   : {agg['sump']/o:.4f}")
    print(f"  mean fraction on frontiers     : {agg['onfr']/o:.4f}")
    print(f"  mean #frontier nodes           : {agg['nfr']/o:.1f}")
    print(f"  FLAT field (peak < 3x mean)    : {100*agg['flat']/o:5.1f}%   <- high here = no direction")
    print(f"  feat[4] teammate_pot   max {agg['f4max']/o:.4f}   nonzero {100*agg['f4nz']/o:5.1f}%")
    print(f"  feat[6] radar_teammate max {agg['f6max']/o:.4f}   nonzero {100*agg['f6nz']/o:5.1f}%")


if __name__ == "__main__":
    main()
