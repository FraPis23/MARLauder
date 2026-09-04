#!/usr/bin/env python3
"""Replay the episodes that FAILED under protocol v2 and draw what the agents actually did.

    python scripts/replay_failed.py --ckpt runs/v20_.../ckpt_best.pt --split hybrid --agents 4 \
        --ir2-dir /workspace/IR2-Multi-Robot-RL-Exploration/comparison/results

Fidelity is the whole point: a failure that does not reproduce is not the failure being explained.
The rollout therefore repeats eval_comparison's chunk EXACTLY — same 25-map chunking, same
`reseed_channel_noise` call site, same per-map `np.random.default_rng(map_seed + map_idx)`, same
IR2 start override, same per-map travel budget, and the same per-cell step cap (which is not just
a cap: explorer.py normalises the episode-time observation by max_episode_steps, so a different
value feeds the actor a different clock and produces a different trajectory).
Only the chunks containing a requested episode are run.

Per failed episode it writes one PNG with two panels — the paths over the ground truth, and the
per-agent own-map coverage against the distance budget — plus these numbers, which is what
separates "it looped" from "it ran out of road":

    revisit    share of steps landing on a lattice node the SAME agent held in the last W steps
    cos(turn)  mean cosine between consecutive displacements; ~1 = committed walk, <=0 = thrashing
    path/net   path walked / straight-line start-to-end; large = wandering, ~1 = a transect
    own_gap    union coverage minus the WEAKEST agent's own coverage at the end. This is the one
               that says the map was found but never delivered.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import math  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.explorer import EnvCfg, Explorer  # noqa: E402
from env.maps import load_split  # noqa: E402
from eval.ckpt_loader import load_model_from_ckpt  # noqa: E402
from scripts.eval_comparison import _COMPARISON_DIR, _load_ir2_reference  # noqa: E402

COLORS = ["#D64545", "#2F7DBF", "#3F9B54", "#C58B1F"]


def loop_stats(path: np.ndarray, nr: float, window: int = 16) -> dict:
    """path [T, 2] px for ONE agent, already trimmed to the episode's real length."""
    d = np.diff(path, axis=0)
    step = np.linalg.norm(d, axis=1)
    moved = step > 1e-3
    if moved.sum() < 2:
        return {"revisit": 0.0, "cos": float("nan"), "path_net": float("nan")}
    u = d[moved] / step[moved, None]
    cos = float((u[1:] * u[:-1]).sum(1).mean()) if len(u) > 1 else float("nan")
    node = np.floor(path / nr).astype(np.int64)
    key = node[:, 1] * 100000 + node[:, 0]
    rev = 0
    for t in range(1, len(key)):
        if key[t] in key[max(0, t - window):t]:
            rev += 1
    net = float(np.linalg.norm(path[-1] - path[0]))
    return {"revisit": rev / max(1, len(key) - 1), "cos": cos,
            "path_net": float(step.sum() / max(1.0, net))}


def draw(out_png: Path, gt: np.ndarray, paths: np.ndarray, own: np.ndarray,
         travel: np.ndarray, budget: float, title: str, n_steps: int) -> None:
    M = paths.shape[1]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 7),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    ax.imshow(gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for m in range(M):
        p = paths[:n_steps + 1, m]
        ax.plot(p[:, 0], p[:, 1], lw=1.4, color=COLORS[m % len(COLORS)], alpha=0.9,
                label=f"agent {m}  own {own[n_steps, m]:.3f}")
        ax.plot(p[0, 0], p[0, 1], "o", ms=9, mfc="none", mew=2, color=COLORS[m % len(COLORS)])
        ax.plot(p[-1, 0], p[-1, 1], "s", ms=7, color=COLORS[m % len(COLORS)])
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])

    t = np.arange(n_steps + 1)
    for m in range(M):
        bx.plot(t, own[:n_steps + 1, m], color=COLORS[m % len(COLORS)], lw=1.6,
                label=f"agent {m} own coverage")
    bx.axhline(0.99, ls="--", lw=1, color="#666")
    bx.text(0, 0.9905, " success needs EVERY agent above this", fontsize=8, color="#666")
    bx.set_ylim(0, 1.02); bx.set_xlabel("step"); bx.set_ylabel("own-map coverage")
    cx = bx.twinx()
    cx.plot(t, travel[:n_steps + 1].max(-1) / max(1.0, budget), color="#111", lw=1.2, ls=":")
    cx.set_ylabel("budget spent (max over agents)", color="#111")
    cx.set_ylim(0, 1.05)
    bx.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", required=True, choices=["hybrid", "corridor", "complex"])
    ap.add_argument("--agents", type=int, required=True)
    ap.add_argument("--ir2-dir", type=Path, required=True)
    ap.add_argument("--eps", type=int, nargs="*", default=None,
                    help="episode indices (rows of map_indices). Default: read the published CSV "
                         "and take every episode whose success was 0")
    ap.add_argument("--max-render", type=int, default=6)
    ap.add_argument("--batch", type=int, default=25, help="MUST match the run being reproduced")
    ap.add_argument("--noise-seed", type=int, default=777)
    ap.add_argument("--map-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/failed_replay"))
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    entries = json.loads((_COMPARISON_DIR / f"map_indices_{args.split}.json").read_text())["entries"]
    pack_idxs = [int(e["pack_idx"]) for e in entries]
    budgets, starts = _load_ir2_reference(args.ir2_dir, args.split, args.agents, len(pack_idxs))
    cap = int(math.ceil(max(budgets) / 16.0)) * 2

    want = args.eps
    if want is None:
        import csv
        f = _COMPARISON_DIR / "results" / f"marlauder_{args.split}_M{args.agents}_v20_v2.csv"
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
        want = [i for i, r in enumerate(rows) if str(r["success"]).strip() in ("0", "False", "0.0")]
        print(f"[replay] {len(want)} failed episodes in {f.name}: {want}")
    want = want[:args.max_render]
    chunks = sorted({i // args.batch for i in want})
    print(f"[replay] running chunk(s) {chunks} of {len(pack_idxs) // args.batch}, "
          f"cap={cap}, budget mean {sum(budgets) / len(budgets):.0f}px")

    split = load_split(f"test/{args.split}", device=args.device)
    peek = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    penv = dict((peek.get("cfg", {}) or {}).get("env", {}) or {})
    penv["comm_relay"] = True
    args.out.mkdir(parents=True, exist_ok=True)

    for ch in chunks:
        lo = ch * args.batch
        chunk = pack_idxs[lo:lo + args.batch]
        cfg = EnvCfg.from_ckpt_dict(penv, n_envs=len(chunk), n_agents=args.agents,
                                    max_episode_steps=cap, max_travel_px=0.0,
                                    max_travel_frac=0.0, done_mode="own", map_seed=args.map_seed)
        env = Explorer(split, cfg, seed=0)
        model, _ = load_model_from_ckpt(args.ckpt, args.device, n_agents=args.agents, verbose=False)
        model.eval()
        env.reseed_channel_noise(args.noise_seed)
        for i, midx in enumerate(chunk):
            env.rng = np.random.default_rng(args.map_seed + int(midx))
            env.reload_map(env_idx=i, map_idx=int(midx), start_override=starts[lo + i])
        env.travel_budget_px = torch.tensor([max(1.0, float(b)) for b in budgets[lo:lo + len(chunk)]],
                                            dtype=torch.float32, device=args.device)
        K, M = len(chunk), args.agents
        h_a, h_c = model.init_hidden(K, args.device)
        obs = env.obs
        alive = torch.ones(K, dtype=torch.bool, device=args.device)
        P = [env.pos.clone().cpu()]
        OW = [torch.zeros(K, M)]
        TR = [torch.zeros(K, M)]
        ep_len = torch.zeros(K, dtype=torch.long)
        term = torch.zeros(K)
        with torch.no_grad():
            for t in range(cap):
                out = model.act(obs, h_a, h_c, deterministic=True)
                obs, _r, done, info = env.step(out["action"])
                h_a, h_c = out["hidden_actor"], out["hidden_critic"]
                P.append(info["pos"].cpu()); OW.append(info["own_cov"].cpu())
                TR.append(info["travel_px"].cpu())
                newly = alive & done
                if bool(newly.any()):
                    ep_len[newly.cpu()] = t + 1
                    term[newly.cpu()] = info["terminated"].float().cpu()[newly.cpu()]
                alive = alive & ~done
                if not bool(alive.any()):
                    break
        P = torch.stack(P).numpy(); OW = torch.stack(OW).numpy(); TR = torch.stack(TR).numpy()
        gt_all = env.world.gt_torch.cpu().numpy()
        nr = float(env.graph.NR)
        for e in [x for x in want if lo <= x < lo + len(chunk)]:
            i = e - lo
            n = int(ep_len[i]) or (P.shape[0] - 1)
            own_end = OW[n, i]
            print(f"\n=== episode {e}  map {chunk[i]} ({entries[e]['file']})  "
                  f"steps {n}  success {int(term[i])}  budget {budgets[e]:.0f}px  "
                  f"spent {TR[n, i].max():.0f}px ({TR[n, i].max() / budgets[e] * 100:.0f}%)")
            print(f"    own coverage per agent: " + "  ".join(f"{v:.4f}" for v in own_end)
                  + f"   -> weakest {own_end.min():.4f}")
            for m in range(M):
                st = loop_stats(P[:n + 1, i, m], nr)
                print(f"    agent {m}: revisit {st['revisit']:.3f}  cos(turn) {st['cos']:+.3f}  "
                      f"path/net {st['path_net']:.2f}  travel {TR[n, i, m]:.0f}px")
            png = args.out / f"{args.split}_M{args.agents}_ep{e:03d}_map{chunk[i]}.png"
            draw(png, gt_all[i], P[:, i], OW[:, i], TR[:, i], budgets[e],
                 f"{args.split} M={args.agents} ep {e} (map {entries[e]['file']}) — FAILED\n"
                 f"{n} steps, {TR[n, i].max():.0f}/{budgets[e]:.0f} px, "
                 f"weakest own coverage {own_end.min():.3f}", n)
            print(f"    -> {png}")
        del model, env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
