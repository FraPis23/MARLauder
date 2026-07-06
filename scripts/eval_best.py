"""Rank every checkpoint of a run by the deterministic eval-suite score and copy the winner
to <run>/ckpt_best.pt.

Works on stopped/done runs (the in-training best-ckpt tracker only helps runs launched AFTER
that code landed; this recovers the best iterate from any run's saved milestone ckpts).

Each ckpt is loaded with its OWN architecture (auto-detected from the state dict — same path as
eval_ckpt.py, so a 6-layer ckpt is scored at 6 layers, never silently at 2) and scored on N>=32
evenly-spaced maps. The highest eval/score ckpt is copied to <run>/ckpt_best.pt.

FAST path (default): all N maps run as ONE batched rollout (n_envs = N), ~N× faster than the
sequential in-training suite. The per-map score mirrors train.driver._run_eval_suite exactly
(auc − w_imb·imbN − w_ov·overlap − w_idle·idle). --slow falls back to the sequential suite
(single env, map-by-map) for validation / tiny splits.

    python scripts/eval_best.py --run runs/guidetemp_difficult --split train/difficult --n-maps 32
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
from eval.ckpt_loader import load_model_from_ckpt
from train import driver as drv


def _discover_ckpts(run: Path) -> list[Path]:
    """All scoreable checkpoints in a run dir, milestones first then stop/final, de-duplicated.
    Skips web/on-demand ckpts (ckpt_web*) — transient dashboard renders — and ckpt_best itself."""
    seen: set[str] = set()
    out: list[Path] = []
    for pat in ("ckpt_[0-9]*.pt", "ckpt_stop.pt", "final.pt"):
        for p in sorted(run.glob(pat)):
            if p.name.startswith("ckpt_web") or p.name == "ckpt_best.pt":
                continue
            if p.name not in seen:
                seen.add(p.name)
                out.append(p)
    return out


@torch.no_grad()
def _score_batched(model, env: Explorer, cfg, map_idxs: tuple[int, ...]) -> dict:
    """Score all maps in ONE batched rollout (n_envs == len(map_idxs)). Per-map score mirrors
    driver._run_eval_suite. Finished envs are frozen (env auto-resets them, so we stop recording
    once an env is done and pad its coverage-AUC with the final explored_rate)."""
    dev = cfg.device
    K = len(map_idxs)
    assert env.N == K, f"env n_envs ({env.N}) must equal n_maps ({K}) for the batched path"
    M = cfg.n_agents
    T = env.cfg.max_episode_steps
    for i, midx in enumerate(map_idxs):
        env.reload_map(env_idx=i, map_idx=int(midx))
    h_act, h_crit = model.init_hidden(K, dev)
    obs = env.obs

    active = torch.ones(K, dtype=torch.bool, device=dev)
    er_sum = torch.zeros(K, device=dev)
    steps_rec = torch.zeros(K, device=dev)
    last_er = torch.zeros(K, device=dev)
    ov_sum = torch.zeros(K, device=dev)
    s90 = torch.full((K,), float(T), device=dev)
    succ = torch.zeros(K, device=dev)
    novel_prev = torch.zeros(K, M, device=dev)
    act_count = torch.zeros(K, M, device=dev)          # per-agent steps that found ≥1 novel cell
    novel_final = torch.zeros(K, M, device=dev)
    SR2 = 2.0 * float(env.cfg.sensor_range_px)
    if M > 1:
        triu = torch.triu(torch.ones(M, M, device=dev), diagonal=1).bool()

    was_training = model.training
    model.eval()
    for t in range(T):
        out = model.act(obs, h_act, h_crit, deterministic=True)
        obs, _r, done, info = env.step(out["action"])
        h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
        er = info["explored_rate"].to(dev).float()                          # [K] pre-reset
        nf = info["novel_cells_ep"].to(dev).float()                         # [K, M] pre-reset cumulative
        a = active.float()
        if M > 1:
            pd = torch.cdist(env.pos, env.pos)                             # [K, M, M]
            ov_step = (pd[:, triu] < SR2).float().mean(dim=-1)             # [K]
        else:
            ov_step = torch.zeros(K, device=dev)
        novel_step = (nf - novel_prev).clamp(min=0.0)                      # [K, M]
        er_sum += er * a
        steps_rec += a
        last_er = torch.where(active, er, last_er)
        ov_sum += ov_step * a
        act_count += (novel_step > 0).float() * a.unsqueeze(-1)
        novel_final = torch.where(active.unsqueeze(-1), nf, novel_final)
        novel_prev = nf
        hit = active & (er >= 0.9) & (s90 >= float(T))
        s90 = torch.where(hit, float(t + 1), s90)
        newly = active & done.to(dev)
        succ = torch.where(newly, info["terminated"].to(dev).float(), succ)
        active = active & ~done.to(dev)
        if not bool(active.any().item()):
            break
    if was_training:
        model.train()

    steps_c = steps_rec.clamp(min=1.0)
    auc = (er_sum + last_er * (T - steps_rec)) / T                          # [K]
    idle = (1.0 - act_count / steps_c.unsqueeze(-1)).max(dim=-1).values     # [K] laziest agent
    ov = ov_sum / steps_c                                                   # [K]
    total_novel = novel_final.sum(dim=-1)                                   # [K]
    imb = torch.where(total_novel > 0,
                      novel_final.max(dim=-1).values / total_novel.clamp(min=1e-9) - 1.0 / M,
                      torch.zeros_like(total_novel))
    imb_denom = (1.0 - 1.0 / M) if M > 1 else 1.0
    imb_norm = imb / imb_denom
    score_map = (auc
                 - cfg.score_w_imbalance * imb_norm
                 - cfg.score_w_overlap * ov
                 - cfg.score_w_idle * idle)                                 # [K]
    return {
        "eval/score":             float(score_map.mean().item()),
        "eval/score_std":         float(score_map.std(unbiased=False).item()),
        "eval/coverage_auc":      float(auc.mean().item()),
        "eval/success_rate":      float(succ.mean().item()),
        "eval/idle_rate_max":     float(idle.mean().item()),
        "eval/contrib_imbalance_norm": float(imb_norm.mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir holding the ckpts")
    ap.add_argument("--split", default=None,
                    help="eval split (default: the run's own training split from its ckpt cfg)")
    ap.add_argument("--n-maps", type=int, default=32, help="evenly-spaced maps in the eval suite (>=32)")
    ap.add_argument("--steps", type=int, default=None,
                    help="episode length (default: the ckpt's max_episode_steps)")
    ap.add_argument("--n-agents", type=int, default=None, help="default: from ckpt cfg")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--slow", action="store_true",
                    help="sequential single-env suite (driver._run_eval_suite) instead of batched")
    ap.add_argument("--no-copy", action="store_true", help="rank only, do not write ckpt_best.pt")
    args = ap.parse_args()

    run: Path = args.run
    ckpts = _discover_ckpts(run)
    if not ckpts:
        print(f"[eval_best] no checkpoints found in {run}")
        sys.exit(1)

    peek = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    pcfg = peek.get("cfg", {}) if isinstance(peek, dict) else {}
    penv = pcfg.get("env", {}) if isinstance(pcfg, dict) else {}
    split_name = args.split or pcfg.get("split") or "train/difficult"
    n_agents = args.n_agents or int(pcfg.get("n_agents", 2))
    steps = args.steps or int(penv.get("max_episode_steps", 384))
    n_maps = max(32, int(args.n_maps))

    split = load_split(split_name, device=args.device)
    n = int(getattr(split, "n", 0)) or 1
    k = max(1, min(n_maps, n))
    map_idxs = tuple(int(i) for i in np.linspace(0, n - 1, k).round().astype(int))

    n_envs = 1 if args.slow else k
    env_cfg = EnvCfg.from_ckpt_dict(penv or {}, n_envs=n_envs, n_agents=n_agents,
                                    max_episode_steps=steps + 1)
    env = Explorer(split, env_cfg, seed=0)

    ecfg = drv.TrainCfg(
        n_agents=n_agents, device=args.device,
        score_w_imbalance=float(pcfg.get("score_w_imbalance", 0.5)),
        score_w_overlap=float(pcfg.get("score_w_overlap", 0.25)),
        score_w_idle=float(pcfg.get("score_w_idle", 0.25)),
    )
    score_fn = (lambda m: drv._run_eval_suite(m, env, ecfg, map_idxs=map_idxs)) if args.slow \
        else (lambda m: _score_batched(m, env, ecfg, map_idxs))

    print(f"[eval_best] run={run.name} split={split_name} maps={k} steps={steps} agents={n_agents} "
          f"mode={'slow' if args.slow else 'batched'} ckpts={[p.name for p in ckpts]}", flush=True)
    print(f"{'ckpt':<16}{'score':>9}{'±std':>8}{'auc':>7}{'succ':>7}{'idle':>7}{'imbN':>7}", flush=True)

    results: list[tuple[str, float]] = []
    for p in ckpts:
        model, _peek = load_model_from_ckpt(p, args.device, n_agents=n_agents)
        s = score_fn(model)
        results.append((p.name, s["eval/score"]))
        print(f"{p.name:<16}{s['eval/score']:>+9.3f}{s['eval/score_std']:>8.3f}"
              f"{s['eval/coverage_auc']:>7.3f}{s['eval/success_rate']:>7.2f}"
              f"{s['eval/idle_rate_max']:>7.2f}{s['eval/contrib_imbalance_norm']:>7.3f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda r: r[1], reverse=True)
    best_name, best_score = results[0]
    print(f"\n[eval_best] WINNER: {best_name}  score={best_score:+.3f}", flush=True)
    if not args.no_copy:
        dst = run / "ckpt_best.pt"
        shutil.copyfile(run / best_name, dst)
        print(f"[eval_best] copied {best_name} → {dst}", flush=True)


if __name__ == "__main__":
    main()
