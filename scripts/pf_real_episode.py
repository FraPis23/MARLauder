#!/usr/bin/env python3
"""The pathfront belief on a REAL map, real policy, two agents — same page, same metrics as the bench.

    python scripts/pf_real_episode.py --ckpt runs/<run>/ckpt_best.pt --split test/hybrid --map-idx 1

`scripts/pf_scenarios.py` settles arguments about the model on 26x11 hand-drawn maps where every
step can be reasoned about by eye. This runs the SAME reporting on a 500x500 map driven by the
trained policy, so the properties that were verified there can be checked where they actually have
to hold. Nothing here is scripted: the agents choose their own moves, comm breaks when it breaks.

The page is written next to the scenario pages and linked from the same index, because the point is
to read them together — a claim that holds on 03_bifurcation and not here is not a claim.

Only the belief of agent A about agent B is shown (`--agent` / `--teammate`), and only while it is
ALIVE: while the two are in comm the belief is a delta on the true position and there is nothing to
check. Steps are sampled (`--every`) because a 120-step episode at one SVG per step is a 40 MB page.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from env.explorer import EnvCfg, Explorer  # noqa: E402
from env.maps import load_split  # noqa: E402
from eval.ckpt_loader import load_model_from_ckpt  # noqa: E402
from scripts.pf_scenarios import PAGE_CSS  # noqa: E402

_UNK, _FREE, _OBST = 0, 1, 2


def render(nxy, occ, valid, p, seen, front, transit, obs_xy, tm_xy, lkp_xy, nr, W, H, cell=7):
    """Same visual grammar as the scenario pages, on the lattice of a real map."""
    sx = cell / nr
    out = [f'<svg width="{W * sx:.0f}" height="{H * sx:.0f}" viewBox="0 0 {W * sx:.0f} '
           f'{H * sx:.0f}" class="grid" style="background:#0d1117">']
    pm = max(float(p.max()), 1e-9)
    for i in range(nxy.shape[0]):
        if not bool(valid[i]):
            continue
        x, y = float(nxy[i, 0]) * sx - cell / 2, float(nxy[i, 1]) * sx - cell / 2
        o = int(occ[i])
        fill = "#000000" if o == _OBST else ("#161b22" if o == _UNK
                                             else ("#123047" if bool(seen[i]) else "#2b3138"))
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" fill="{fill}"/>')
        v = float(p[i])
        if v > 1e-7:
            a = min(1.0, 0.15 + 0.85 * (v / pm) ** 0.5)
            out.append(f'<rect x="{x + 0.5:.1f}" y="{y + 0.5:.1f}" width="{cell - 1}" '
                       f'height="{cell - 1}" fill="#ff4d4d" fill-opacity="{a:.3f}"/>')
        if bool(front[i]):
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" '
                       f'fill="none" stroke="#f0c674" stroke-width="0.8"/>')
        if bool(transit[i]):
            out.append(f'<circle cx="{x + cell / 2:.1f}" cy="{y + cell / 2:.1f}" r="{cell * 0.5:.1f}" '
                       f'fill="none" stroke="#ffd866" stroke-width="1.6"/>')
    for xy, col, r in ((lkp_xy, "#ffffff", 2.5), (tm_xy, "#4dd0e1", 3.5), (obs_xy, "#7bd88f", 3.5)):
        if xy is None:
            continue
        out.append(f'<circle cx="{float(xy[0]) * sx:.1f}" cy="{float(xy[1]) * sx:.1f}" r="{r}" '
                   f'fill="{col}"/>')
    out.append("</svg>")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/hybrid")
    ap.add_argument("--map-idx", type=int, default=1)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--agent", type=int, default=0, help="whose belief")
    ap.add_argument("--teammate", type=int, default=1, help="belief ABOUT whom")
    ap.add_argument("--every", type=int, default=2, help="render one step in N (stats cover all)")
    ap.add_argument("--out", type=Path, default=_REPO / "runs" / "pf_scenarios")
    ap.add_argument("--name", default=None, help="page basename (default: real_<split>_m<idx>)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    a, j = args.agent, args.teammate
    name = args.name or f"09_real_{args.split.replace('/', '-')}_m{args.map_idx}"

    model, env_peek = load_model_from_ckpt(args.ckpt, args.device, n_agents=2)
    split = load_split(args.split, device=args.device)
    cfg_d = dict(env_peek or {})
    cfg_d["use_teammate_belief"] = True
    cfg_d["belief_mode"] = "pathfront"
    cfg = EnvCfg.from_ckpt_dict(cfg_d, n_envs=1, n_agents=2, max_episode_steps=args.steps + 1)
    env = Explorer(split, cfg, seed=int(args.map_idx))
    env.store_render_global = True
    env.reload_map(env_idx=0, map_idx=int(args.map_idx))
    # Same reason as eval/trace.py: the env auto-resets on termination and would wipe the map
    # mid-page. Every stop condition is disabled here and the loop just runs its budget.
    env.cfg.done_explored_thresh = 2.0
    env.cfg.max_travel_px = 0.0
    env.cfg.max_travel_frac = 0.0
    env.cfg.max_episode_steps = args.steps + 5
    model.eval()

    h_act, h_crit = model.init_hidden(1, args.device)
    obs = env.obs
    nxy = env.graph.node_xy.cpu()
    nr, Wpx, Hpx = float(env.cfg.nr), env.W, env.H

    frames, rows, worst = [], [], (1e9, -1e9)
    last_comm = [-1]                                 # last step the two were in comm
    with torch.no_grad():
        for t in range(args.steps):
            out = model.act(obs, h_act, h_crit, deterministic=True)
            rg = env._render_global
            bp, bal = rg.get("belief_p"), rg.get("belief_alive")
            # `belief_alive` is ALSO true while the two are in comm: on collapse the explorer
            # overwrites the field with a delta on the true position and sets alive. That delta is
            # inside comm and is not a frontier, so it scores `seen & not frontier` = 1.0 and
            # drowns every real reading. In comm there is no belief to check — skip those steps.
            in_comm = bool(rg["comm_mask"][0, a, j]) if rg.get("comm_mask") is not None else False
            if in_comm:
                last_comm[0] = t
            if bp is not None and bool(bal[0, a, j]) and not in_comm:
                p = bp[0, a, j].cpu()
                seen = rg["belief_seen"][0, a].cpu()
                front = rg["belief_frontier"][0, a].cpu()
                tv = rg["belief_transit"][0, a, j].cpu()
                valid = rg["node_valid"][0, a].cpu().bool()
                occ = torch.full((p.shape[0],), _UNK, dtype=torch.long)
                nf = rg["node_feat"][0, a].cpu()
                occ[valid] = _FREE                       # node_valid = known-free on this lattice
                tot = float(p.sum())
                worst = (min(worst[0], tot), max(worst[1], tot))
                st = {
                    "Σp": f"{tot:.4f}",
                    "on frontiers": f"{float((p * front.float()).sum()):.4f}",
                    "in `seen`": f"{float((p * seen.float()).sum()):.4f}",
                    "seen & not frontier": f"{float((p * seen.float() * (~front).float()).sum()):.4f}",
                    "still travelling": str(int(tv.sum())),
                    "peak in comm": "YES" if bool(seen[int(p.argmax())]) else "no",
                    "peak": f"{float(p.max()):.4f}{'F' if bool(front[int(p.argmax())]) else '-'}",
                    "frontier nodes": str(int(front.sum())),
                    "known nodes": str(int(valid.sum())),
                    "true tm on p>0": f"{float(p[int(rg['curr_idx'][0, j])]):.5f}",
                    # the freeze step is the one interesting piece of provenance: the transit dot
                    # starts AT the last-known node, which is by construction the last place comm
                    # did fire, so it can legitimately still be inside the comm blob for one step.
                    "steps since comm": str(t - last_comm[0]),
                }
                rows.append((t, st))
                if t % args.every == 0:
                    frames.append((t, render(
                        nxy, occ, valid, p, seen, front, tv,
                        rg["pos"][0, a].cpu(), rg["pos"][0, j].cpu(),
                        rg["last_known_pos"][0, a, j].cpu(), nr, Wpx, Hpx), st))
            obs, _, done, info = env.step(out["action"])
            if bool(done[0]):
                break

    if not rows:
        print("[pf-real] belief never went alive (the two agents stayed in comm) — try another map")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = f"Real map — {args.split} #{args.map_idx}, trained policy, 2 agents"
    why = (f"Agent {a}'s belief of agent {j} on a {Hpx}x{Wpx} map, driven by {args.ckpt}. "
           f"Steps shown are the ones where the belief is ALIVE (out of comm); in comm it is a "
           f"delta on the true position and there is nothing to check. Green = the observer, "
           f"cyan = the teammate's TRUE position (the model never sees it), white = last known. "
           f"`seen & not frontier` is the number that matters: probability inside comm range on "
           f"ground that is not an opening.")
    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<p class='why' style='color:#6e7681'>rebuilt {stamp} &middot; alive on "
             f"{len(rows)} steps &middot; &Sigma;p&isin;[{worst[0]:.4f},{worst[1]:.4f}]</p>",
             f'<p class="why">{html.escape(why)}</p>',
             '<p><a href="index.html">&larr; all scenarios</a></p>', '<div class="steps">']
    for t, s, st in frames:
        rh = "".join(f'<tr><td class="k">{html.escape(k)}</td>'
                     f'<td class="{"bad" if k == "Σp" and abs(float(v) - 1) > 5e-4 else ""}">'
                     f'{html.escape(v)}</td></tr>' for k, v in st.items())
        parts.append(f'<div class="step"><h3>step {t}</h3>{s}<table>{rh}</table></div>')
    parts.append("</div>")
    (args.out / f"{name}.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{PAGE_CSS}</style>" + "".join(parts), encoding="utf-8")

    bad = [(t, s) for t, s in rows if abs(float(s["Σp"]) - 1) > 5e-4]
    print(f"[pf-real] {name}: alive {len(rows)} steps  Σp∈[{worst[0]:.6f},{worst[1]:.6f}]  "
          f"{'MASS LEAK on ' + str(len(bad)) + ' steps' if bad else 'ok'}")
    # The whole point of the page: every step where probability sits inside comm range on ground
    # that is not an opening, listed, not summarised. A max alone cannot be acted on.
    bad_stray = [(t, s) for t, s in rows if float(s["seen & not frontier"]) > 1e-3]
    print(f"[pf-real] `seen & not frontier` > 1e-3 on {len(bad_stray)}/{len(rows)} alive steps")
    for t, s in bad_stray[:25]:
        print(f"           t={t:4d}  stray={s['seen & not frontier']}  Σp={s['Σp']}  "
              f"onFr={s['on frontiers']}  peak={s['peak']}  nFrontier={s['frontier nodes']}  "
              f"travelling={s['still travelling']}  sinceComm={s['steps since comm']}")
    print(f"[pf-real] {args.out}/{name}.html")


if __name__ == "__main__":
    main()
