"""Per-step training profiler — where does a v0.6 training iteration spend time?

Profiles the REAL code path (driver.collect_rollout + mappo.ppo_update) by monkeypatching
CUDA-synchronized timers onto the heavy functions, so the breakdown reflects what training
actually runs (not a reimplementation). Two layers:

  1. PHASE timing      — rollout (env.step vs model.act) vs PPO update, + sps.
  2. COMPONENT timing  — GAT encode, attention head, belief, pointer, BF, Warp kernels,
                         candidate extraction, diversity loss, belief NLL, backward.

CAVEAT: the timers call torch.cuda.synchronize() around each wrapped function. That serializes
the GPU, so ABSOLUTE sps here is lower than a real run — but the RELATIVE attribution (which
component dominates) is what we want, and that is accurate. Nested wraps are reported
hierarchically (e.g. model.act ⊇ _encode), so don't sum siblings blindly; read the tree.

RUN ALONE ON THE GPU (kill any other train/eval first) or the numbers are contention noise:
    docker exec marlauder pkill -f run_train          # if duplicates are running
    docker exec marlauder bash -lc 'cd /workspace/MARLauder && PYTHONPATH=. \
        python scripts/profile_step.py --n-agents 2 --n-envs 32 --rollout-len 64 \
        --max-episode-steps 64 --belief-mode learned --div-coef 0.1'

Add --torch-profiler for an op-level table + a chrome trace (runs/profile/trace.json).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg
from train.buffer import Rollout
from train.driver import TrainCfg, make_env_model, collect_rollout, _normalize_cfg
from train.mappo import ppo_update
from models.value_normalizer import ValueNormalizer
import train.mappo as mappo_mod


# --------------------------------------------------------------------------- #
# CUDA-synchronized accumulating timers (monkeypatched onto hot functions)    #
# --------------------------------------------------------------------------- #
class Timers:
    def __init__(self, device: str) -> None:
        self.t: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)
        self.cuda = device.startswith("cuda")
        self._patched: list[tuple] = []

    def _sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()

    def wrap(self, obj, attr: str, label: str) -> None:
        """Wrap a bound method / module __call__ to accumulate synced wall-time under `label`."""
        if obj is None or not hasattr(obj, attr):
            return
        orig = getattr(obj, attr)

        def wrapped(*a, _orig=orig, _label=label, **k):
            self._sync(); t0 = time.perf_counter()
            r = _orig(*a, **k)
            self._sync(); self.t[_label] += time.perf_counter() - t0; self.n[_label] += 1
            return r

        setattr(obj, attr, wrapped)
        self._patched.append((obj, attr, orig))

    def wrap_module_fn(self, module, fn_name: str, label: str) -> None:
        """Wrap a module-level function (e.g. mappo._diversity_loss)."""
        orig = getattr(module, fn_name)

        def wrapped(*a, _orig=orig, _label=label, **k):
            self._sync(); t0 = time.perf_counter()
            r = _orig(*a, **k)
            self._sync(); self.t[_label] += time.perf_counter() - t0; self.n[_label] += 1
            return r

        setattr(module, fn_name, wrapped)
        self._patched.append((module, fn_name, orig))

    def reset(self) -> None:
        self.t.clear(); self.n.clear()

    def report(self, total_iter_s: float, n_iters: int) -> None:
        rows = sorted(self.t.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n{'component':32s} {'total(s)':>9s} {'calls':>7s} {'ms/call':>9s} {'%iter':>7s}")
        print("-" * 70)
        denom = max(1e-9, total_iter_s)
        for label, tot in rows:
            calls = self.n[label]
            mspc = 1000.0 * tot / max(1, calls)
            print(f"{label:32s} {tot:9.3f} {calls:7d} {mspc:9.3f} {100.0*tot/denom:6.1f}%")


def install_timers(env, model, T: Timers) -> None:
    # ---- rollout: top-level per-step calls ----
    T.wrap(model, "act", "rollout/model.act")
    T.wrap(env, "step", "rollout/env.step")
    # ---- model.act internals (subset of model.act) ----
    T.wrap(model, "_encode", "  GAT encode (_encode)")
    T.wrap(model.strategic_head, "forward", "  attention head (StrategicHead)")
    T.wrap(model, "_belief_spread", "  belief spread (act)")
    T.wrap(model, "_pool_map", "  pool map feats")
    T.wrap(model.pointer, "forward", "  pointer head")
    T.wrap(model, "_annotate_path", "  annotate path (ch6/7)")
    T.wrap(model, "_target_dist_field", "  target-dist BF (ch10)")
    # ---- env.step internals (subset of env.step) ----
    T.wrap(env, "_refresh_obs", "  env._refresh_obs")
    T.wrap(env, "_comm_check", "  env._comm_check")
    T.wrap(env.world, "scan", "  warp LiDAR scan")
    T.wrap(env.world, "fuse_maps", "  warp fuse_maps")
    T.wrap(env.world, "expected_gain", "  warp expected_gain")
    T.wrap(env.graph, "build", "  graph.build")
    T.wrap(env.graph, "bf_from_target", "  graph.bf_from_target (BF)")
    T.wrap(env.graph, "build_guidepost_v2", "  graph.build_guidepost_v2")
    T.wrap(env.graph, "extract_topk_candidates", "  graph.extract_topk")
    T.wrap(env.graph, "extract_local_window", "  graph.extract_window")
    # ---- PPO update internals ----
    T.wrap(model, "encode_chunk", "update/encode_chunk")
    T.wrap(model, "evaluate_step_from_enc", "update/evaluate_step (per tt)")
    T.wrap(model, "belief_nll", "update/belief_nll (per tt)")
    T.wrap_module_fn(mappo_mod, "_diversity_loss", "update/_diversity_loss (per tt)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train/difficult")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-envs", type=int, default=32)
    ap.add_argument("--n-agents", type=int, default=2)
    ap.add_argument("--n-hops", type=int, default=2)
    ap.add_argument("--rollout-len", type=int, default=64)
    ap.add_argument("--max-episode-steps", type=int, default=64)
    ap.add_argument("--belief-mode", default="learned")
    ap.add_argument("--div-coef", type=float, default=0.1)
    ap.add_argument("--k-epochs", type=int, default=4)
    ap.add_argument("--tbptt-steps", type=int, default=16)
    ap.add_argument("--warmup-iters", type=int, default=2)
    ap.add_argument("--measure-iters", type=int, default=3)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--torch-profiler", action="store_true",
                    help="also emit an op-level table + chrome trace (runs/profile/trace.json)")
    args = ap.parse_args()

    cfg = TrainCfg(
        split=args.split, n_envs=args.n_envs, n_agents=args.n_agents, n_hops=args.n_hops,
        rollout_len=args.rollout_len, device=args.device, compile=args.compile,
        belief_mode=args.belief_mode,
        env=EnvCfg(n_envs=args.n_envs, n_agents=args.n_agents, n_hops=args.n_hops,
                   max_episode_steps=args.max_episode_steps),
    )
    cfg.ppo.div_coef = args.div_coef
    cfg.ppo.k_epochs = args.k_epochs
    cfg.ppo.tbptt_steps = args.tbptt_steps
    env, model = make_env_model(cfg)
    vnorm = ValueNormalizer().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_actor)
    h_act, h_crit = model.init_hidden(cfg.n_envs, cfg.device)
    buf = Rollout(env.obs, T=cfg.rollout_len, N=cfg.n_envs, M=cfg.n_agents,
                  d_hidden=cfg.d_hidden, device=cfg.device)

    cuda = cfg.device.startswith("cuda")

    def one_iter():
        nonlocal h_act, h_crit
        buf.h_actor_init.copy_(h_act.detach())
        buf.h_critic_init.copy_(h_crit.detach())
        h_act, h_crit, *_ = collect_rollout(env, model, buf, h_act, h_crit, vnorm)
        ppo_update(model, optimizer, vnorm, buf, cfg.ppo, cfg.device)

    print(f"[profile] split={args.split} n_envs={args.n_envs} M={args.n_agents} "
          f"n_hops={args.n_hops} T={args.rollout_len} k_epochs={args.k_epochs} "
          f"belief={args.belief_mode} compile={args.compile}")
    print(f"[profile] warmup {args.warmup_iters} iters ...")
    for _ in range(args.warmup_iters):
        one_iter()
    if cuda:
        torch.cuda.synchronize()

    # ---- phase timing (rollout vs update) WITHOUT timer overhead first ----
    t_roll = t_upd = 0.0
    for _ in range(args.measure_iters):
        buf.h_actor_init.copy_(h_act.detach()); buf.h_critic_init.copy_(h_crit.detach())
        if cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        h_act, h_crit, *_ = collect_rollout(env, model, buf, h_act, h_crit, vnorm)
        if cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()
        ppo_update(model, optimizer, vnorm, buf, cfg.ppo, cfg.device)
        if cuda: torch.cuda.synchronize()
        t2 = time.perf_counter()
        t_roll += t1 - t0; t_upd += t2 - t1
    n = args.measure_iters
    iter_s = (t_roll + t_upd) / n
    steps_per_iter = cfg.n_envs * cfg.rollout_len
    print("\n==================== PHASE TIMING (no per-fn sync) ====================")
    print(f"rollout/iter : {t_roll/n*1000:8.1f} ms  ({100*t_roll/(t_roll+t_upd):.1f}%)")
    print(f"update /iter : {t_upd/n*1000:8.1f} ms  ({100*t_upd/(t_roll+t_upd):.1f}%)")
    print(f"total  /iter : {iter_s*1000:8.1f} ms   → sps={steps_per_iter/iter_s:.0f} "
          f"(serialized-sync upper bound; real run is faster)")

    # ---- component timing (synced per-fn) ----
    T = Timers(cfg.device)
    install_timers(env, model, T)
    print(f"\n[profile] component pass ({args.measure_iters} iters, per-fn sync) ...")
    t0 = time.perf_counter()
    for _ in range(args.measure_iters):
        one_iter()
    if cuda: torch.cuda.synchronize()
    comp_total = time.perf_counter() - t0
    print("\n==================== COMPONENT TIMING (per-fn synced; tree — don't sum siblings) ====")
    print("rollout: model.act and env.step are top-level; indented rows are subsets of them.")
    T.report(comp_total, args.measure_iters)

    # ---- optional torch.profiler op-level ----
    if args.torch_profiler:
        from torch.profiler import profile, ProfilerActivity
        out = _REPO / "runs/profile"; out.mkdir(parents=True, exist_ok=True)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        print("\n[profile] torch.profiler over 1 iter ...")
        with profile(activities=acts, record_shapes=False, with_stack=False) as prof:
            one_iter()
            if cuda: torch.cuda.synchronize()
        key = "self_cuda_time_total" if cuda else "self_cpu_time_total"
        print(prof.key_averages().table(sort_by=key, row_limit=25))
        trace = out / "trace.json"
        prof.export_chrome_trace(str(trace))
        print(f"[profile] chrome trace → {trace}  (open in chrome://tracing)")


if __name__ == "__main__":
    main()
