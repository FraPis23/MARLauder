"""MAPPO training driver. Collects rollouts, runs PPO updates, checkpoints at
milestones (25/50/75/100%), logs metrics to stdout.
"""
from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import MultiSplit, load_split
from models.actor_critic import MarlActorCritic
from models.value_normalizer import ValueNormalizer
from train.buffer import Rollout
from train.mappo import AMP_DTYPE, MAPPOCfg, ppo_update


def _dump_status(out_dir: Path, status: dict) -> None:
    """Atomically write status.json (event-driven heartbeat for the web inspector).

    Called only on training start, milestone checkpoints, and exit — never per-iter,
    so it costs nothing in the hot loop. Liveness between events is inferred by the web
    server from the recorded PID (see viz/web_server.py), which catches hard kills that
    never reach the exit handler. Temp-file + replace = the server never reads a partial.
    """
    import time
    status["timestamp"] = time.time()
    tmp = out_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status))
    tmp.replace(out_dir / "status.json")


def _read_control(out_dir: Path) -> dict | None:
    """Web control channel: the dashboard drops `control.json` into the run dir; we read and
    DELETE it at the next iteration boundary so the command fires exactly once. Cheap existence
    check per iter (no command = one stat() call). Malformed JSON is dropped with a warning."""
    p = out_dir / "control.json"
    if not p.exists():
        return None
    cmd: dict | None = None
    try:
        cmd = json.loads(p.read_text())
    except Exception as exc:
        print(f"[control] ignoring malformed control.json ({exc})")
    try:
        p.unlink()
    except OSError:
        pass
    return cmd


def _save_ckpt(out_dir: Path, model, vnorm, cfg, it: int, tag: str) -> Path:
    """Write a checkpoint named <tag>.pt. Shared by milestone, on-demand, and final saves."""
    ckpt_path = out_dir / f"{tag}.pt"
    torch.save({
        "iter": it,
        "model": model.state_dict(),
        "vnorm": vnorm.state_dict(),
        "cfg": cfg.__dict__ | {"env": cfg.env.__dict__, "ppo": cfg.ppo.__dict__,
                               "out_dir": str(out_dir), "n_agents": cfg.n_agents},
    }, ckpt_path)
    print(f"[ckpt] {ckpt_path}")
    return ckpt_path


@dataclass
class TrainCfg:
    split: str = "train/easy"
    out_dir: Path = Path("/workspace/MARLauder/runs/train_default")
    total_steps: int = 500_000
    n_envs: int = 8
    n_agents: int = 1
    rollout_len: int = 128
    d_hidden: int = 128
    n_heads: int = 4
    n_hops: int = 6          # ego-centric encoder window radius; n_layers tied to this
    n_layers: int = 6        # GAT layers; default tied to n_hops in make_env_model (6 hops / 6 layers)
    use_gru: bool = False    # default OFF (feed-forward actor/critic, no temporal memory). Opt in via --gru
    gat_actor: bool = True   # False = VF-only ablation: actor steers from the analytic value-field
                             # (+ prev_action/scalars) only, GAT reserved for the critic. --no-gat-actor
    gat_critic: bool = True  # False (with gat_actor False = --no-gat) = encoder never run; critic
                             # falls back to raw-feature mean⊕max projection.
    lr_actor: float = 3e-4
    lr_critic: float = 5e-4   # faster than actor (non-stationary value) but below paper's 1e-3 (oscillation risk)
    device: str = "cuda:0"
    seed: int = 0
    compile: bool = False
    init_ckpt: str | None = None   # warm-start: load model + value-norm from this .pt at startup (optimizer fresh). For stage-split relaunch (easy→difficult) at a different n_envs.
    # auto-eval GIF on each milestone ckpt
    eval_on_ckpt: bool = False
    eval_split: str = "train/easy"
    eval_map_idx: int = -1          # -1 = random each time
    eval_steps: int = 256
    eval_n_maps: int = 2            # GIFs per milestone
    env: EnvCfg = field(default_factory=EnvCfg)
    ppo: MAPPOCfg = field(default_factory=MAPPOCfg)
    # H.5 — curriculum: ramp from easy → easy+difficult mix. When True, train_split uses
    # MultiSplit({easy, difficult}, weights). Weights updated per iter:
    #   0–30%: (1.0, 0.0).  30–60%: (0.7, 0.3).  60–100%: (0.5, 0.5).
    curriculum: bool = False
    curriculum_splits: tuple = ("train/easy", "train/difficult")
    # Performance-GATED curriculum (split-SWAP, not MultiSplit weighting). Trains on one split at
    # a time and ADVANCES to the next split only when the eval suite score clears
    # `curriculum_gate_score` (after a `curriculum_min_stage_iters` dwell on the current split).
    # Split-swap (vs the legacy MultiSplit ramp) is required because easy (480×640) and difficult
    # (1000×1000) have DIFFERENT canvases — a MultiSplit asserts a common canvas, so the easy→
    # difficult mix is impossible. The model is canvas-agnostic (ego-window GAT) so it transfers;
    # only the env + rollout buffer + hidden state are rebuilt on advance. Standalone — does NOT
    # require `curriculum=True` (which drives the incompatible MultiSplit path).
    curriculum_gated: bool = False
    curriculum_gate_score: float = 0.5
    curriculum_min_stage_iters: int = 20
    curriculum_stage_splits: tuple = ("train/easy", "train/difficult")
    # Per-stage max episode length (IR2-style: bigger maps need longer episodes to cover, 196→384).
    # Empty → keep cfg.env.max_episode_steps for every stage. Else len must match stage_splits.
    curriculum_stage_steps: tuple = ()
    # Multi-split eval suite: extra map splits to evaluate at each eval tick (besides the
    # training split). Each split gets its own 1-env Explorer (canvases differ) and is logged
    # under eval/<split>/*. eval/score = mean over all suites (also the curriculum gate signal).
    # Empty → single-suite legacy behavior on the training split.
    eval_suite_splits: tuple = ()
    # Weights & Biases (off by default → no network unless --wandb). For sweeps the agent
    # passes hyperparameters as CLI flags; everything here is logged as the run config.
    wandb: bool = False
    wandb_project: str = "marlauder"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "online"          # online | offline | disabled
    wandb_tags: tuple = ()
    # Composite efficiency score (legacy, rollout-based): ep_end − w_red·redundancy − w_stall·stall.
    eff_w_redundancy: float = 0.5
    eff_w_stall: float = 0.5
    # v2 — fixed eval suite (sweep scoring): every eval_every iters run the policy
    # DETERMINISTICALLY on the fixed EVAL_MAP_IDX maps and log eval/*. The sweep maximizes
    # eval/score = coverage_auc − w_imb·contrib_imbalance − w_ov·sensing_overlap − w_idle·idle_rate_max.
    eval_every: int = 10
    score_w_imbalance: float = 0.5
    score_w_overlap: float = 0.25
    # w_idle penalizes the laziest agent's idle-step fraction → the sweep selects policies where
    # BOTH agents keep opening new area (user requirement: no idle, no turn-taking). idle_rate_max
    # ∈ [0,1] already, so it sits on the same footing as the normalized terms.
    score_w_idle: float = 0.25


def _normalize_cfg(cfg: TrainCfg) -> None:
    cfg.env.n_envs = cfg.n_envs
    cfg.env.n_agents = cfg.n_agents
    cfg.env.n_hops = cfg.n_hops
    cfg.env.max_episode_steps = max(cfg.env.max_episode_steps, cfg.rollout_len)
    # Tie n_layers to n_hops so the GAT receptive field uses the full window.
    cfg.n_layers = cfg.n_hops


def _curriculum_weights(progress: float) -> list[float]:
    """H.5 — Ramp easy/difficult mix from 100% easy to 50/50 over training."""
    if progress < 0.3:
        return [1.0, 0.0]
    if progress < 0.6:
        return [0.7, 0.3]
    return [0.5, 0.5]


def make_env_model(cfg: TrainCfg) -> tuple[Explorer, MarlActorCritic]:
    _normalize_cfg(cfg)
    if cfg.curriculum_gated:
        # Gated curriculum: start on the FIRST stage split (single split; swapped on advance).
        if cfg.curriculum_stage_steps and \
                len(cfg.curriculum_stage_steps) != len(cfg.curriculum_stage_splits):
            raise ValueError(
                f"curriculum_stage_steps {cfg.curriculum_stage_steps} length must match "
                f"curriculum_stage_splits {cfg.curriculum_stage_splits}")
        if cfg.curriculum_stage_steps:
            cfg.env.max_episode_steps = max(int(cfg.curriculum_stage_steps[0]), cfg.rollout_len)
        split = load_split(cfg.curriculum_stage_splits[0], device=cfg.device)
        env = Explorer(split, cfg.env, seed=cfg.seed)
        print(f"[curriculum] GATED enabled, stages={list(cfg.curriculum_stage_splits)} "
              f"start='{cfg.curriculum_stage_splits[0]}' max_steps={cfg.env.max_episode_steps}")
    elif cfg.curriculum:
        splits = [load_split(name, device=cfg.device) for name in cfg.curriculum_splits]
        ms = MultiSplit(splits, weights=_curriculum_weights(0.0))
        env = Explorer(ms, cfg.env, seed=cfg.seed)
        print(f"[curriculum] enabled: {cfg.curriculum_splits} initial weights={ms.weights}")
    else:
        split = load_split(cfg.split, device=cfg.device)
        env = Explorer(split, cfg.env, seed=cfg.seed)
    model = MarlActorCritic(n_agents=cfg.n_agents, d=cfg.d_hidden,
                            n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                            use_gru=cfg.use_gru, gat_actor=cfg.gat_actor,
                            gat_critic=cfg.gat_critic).to(cfg.device)
    if cfg.compile and torch.cuda.is_available():
        try:
            model.encoder = torch.compile(model.encoder, mode="reduce-overhead", dynamic=False)
            print("[compile] model.encoder compiled")
        except Exception as exc:
            print(f"[compile] failed ({exc}), continuing uncompiled")
    return env, model


def collect_rollout(env: Explorer, model: MarlActorCritic, buf: Rollout, h_act, h_crit, vnorm: ValueNormalizer) -> tuple:
    """Collect a T-step rollout. Returns (h_act, h_crit, ep_end_mean, ep_end_n).

    H.1 — ep_end_mean = mean explored_rate at terminal step of each episode that ENDED
    during this rollout. If no episode ends (rollout_len < max_episode_steps and no early
    99%-explored termination), returns NaN. Configure --rollout-len ≥ max-episode-steps
    to ensure completions.
    """
    obs = env.obs
    N = buf.N
    dev = env.dev
    ep_end_explored: list[float] = []
    # Per-step metric/reward-term running sums (averaged at the end).
    term_sums: dict[str, float] = {}
    metric_sums: dict[str, float] = {}
    # First-crossing step per env for steps_to_X (−1 = not yet crossed this rollout).
    # free_at_* records that env's free-cell count at the crossing, to normalize for map
    # size (bigger maps need more steps to cover the same FRACTION).
    steps_to_50 = torch.full((N,), -1.0, device=dev)
    steps_to_90 = torch.full((N,), -1.0, device=dev)
    free_at_50  = torch.ones((N,), device=dev)
    free_at_90  = torch.ones((N,), device=dev)
    # Autocast the policy forward during collection (the update loop already runs AMP): the
    # encoder is the collect-time hot spot and runs ~2× faster in low precision. log_softmax
    # stays fp32 under autocast, so the stored logp matches the update-side numerics.
    use_amp = str(dev).startswith("cuda")
    for t in range(buf.T):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
            out = model.act(obs, h_act, h_crit, deterministic=False)
        action = out["action"]
        logp = out["logp"]
        v_norm = out["value"]
        value = vnorm.denormalize(v_norm)
        obs_next, reward, done, info = env.step(action)
        buf.store(t, obs, action, logp, value, reward, done)
        er = info["explored_rate"]
        stp = info["step"].float()
        # Accumulate reward-term + metric scalars.
        for k, v in info["reward_terms"].items():
            term_sums[k] = term_sums.get(k, 0.0) + float(v.item())
        for k, v in info["metrics"].items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v.item())
        # steps_to_X — first time each env crosses the coverage threshold this rollout.
        free_now = env.free_total.clamp(min=1.0)
        newly50 = (er >= 0.5) & (steps_to_50 < 0)
        steps_to_50 = torch.where(newly50, stp, steps_to_50)
        free_at_50  = torch.where(newly50, free_now, free_at_50)
        newly90 = (er >= 0.9) & (steps_to_90 < 0)
        steps_to_90 = torch.where(newly90, stp, steps_to_90)
        free_at_90  = torch.where(newly90, free_now, free_at_90)
        # Capture explored_rate at end-of-episode moments BEFORE auto-reset.
        if bool(done.any().item()):
            done_idx = done.nonzero(as_tuple=True)[0]
            for e_done in done_idx.tolist():
                ep_end_explored.append(float(er[e_done].item()))
        h_act = out["hidden_actor"]
        h_crit = out["hidden_critic"]
        nonterm = (~done).float()
        h_act = h_act * nonterm.view(-1, 1, 1)
        h_crit = h_crit * nonterm.view(-1, 1)
        obs = obs_next
    # Bootstrap V(s_T) under final obs.
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
        out = model.act(obs, h_act, h_crit, deterministic=True)
        v_last = vnorm.denormalize(out["value"])
    buf.last_value.copy_(v_last)
    ep_end_mean = sum(ep_end_explored) / len(ep_end_explored) if ep_end_explored else float("nan")

    T = buf.T
    agg = {f"reward/{k}": v / T for k, v in term_sums.items()}
    for k, v in metric_sums.items():
        if k in ("team_delta_sum", "step_disp_sum"):
            continue   # combined below
        agg[f"metric/{k}"] = v / T
    # coverage_per_dist = Σ Δunion-frac / Σ displacement-px (exploration efficiency).
    agg["metric/coverage_per_dist"] = metric_sums.get("team_delta_sum", 0.0) / max(
        1e-6, metric_sums.get("step_disp_sum", 0.0))

    def _masked_mean(x: torch.Tensor) -> float:
        m = x >= 0
        return float(x[m].mean().item()) if bool(m.any()) else float("nan")
    agg["metric/steps_to_50"] = _masked_mean(steps_to_50)
    agg["metric/steps_to_90"] = _masked_mean(steps_to_90)
    # Size-normalized speed: steps per 1k free cells (map-size invariant → comparable
    # across maps of different free area). Lower = faster exploration.
    agg["metric/steps_to_50_per_kfree"] = _masked_mean(
        torch.where(steps_to_50 >= 0, steps_to_50 * 1000.0 / free_at_50, steps_to_50))
    agg["metric/steps_to_90_per_kfree"] = _masked_mean(
        torch.where(steps_to_90 >= 0, steps_to_90 * 1000.0 / free_at_90, steps_to_90))
    return h_act, h_crit, ep_end_mean, len(ep_end_explored), agg


def _emit_eval_gif(model: "MarlActorCritic", cfg: "TrainCfg", out: Path, map_idx: int,
                   split_name: str | None = None) -> None:
    """Run a deterministic episode on (split_name|eval_split, map_idx) and save a GIF."""
    import imageio.v2 as imageio
    import numpy as np
    from env.maps import sample_batch
    from eval.rollout import EvalCfg, EvalRollout
    from env.explorer import EnvCfg as _EnvCfg, Explorer as _Explorer
    from env.maps import load_split as _load

    split = _load(split_name or cfg.eval_split, device=cfg.device)
    # I.2 — mirror the FULL training env cfg (force flags, top_k, n_hops, ...) so the
    # eval render reflects the same comm/sharing behavior used in training.
    env_cfg = _EnvCfg.from_ckpt_dict(
        cfg.env.__dict__,
        n_envs=1, n_agents=cfg.n_agents,
        max_episode_steps=cfg.eval_steps + 1,
    )
    env = _Explorer(split, env_cfg, seed=map_idx)
    # Full reset for the specific map (clears all stale caches; correct adjacent spawn).
    env.reload_map(env_idx=0, map_idx=int(map_idx))

    was_training = model.training
    model.eval()
    rollout = EvalRollout(env, model, EvalCfg(max_steps=cfg.eval_steps, env_idx=0,
                                              deterministic=True, draw_edges=True))
    frames, stats = rollout.run()
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=80, loop=0)
    if was_training:
        model.train()
    print(f"[eval] {out}  map={map_idx}  final_explored={stats['final_explored']*100:.1f}%  frames={stats['n_frames']}")


# v2 — fixed validation maps for the eval suite. Same 8 indices for every run/trial on
# every machine → cross-trial scores are comparable ("same exam"). These become validation
# maps: final reporting must use fresh random maps / test splits, not these.
# 32 evenly-spaced map indices across the big train splits (idx up to ~9904). Enlarged from 8 →
# eval/score sampling variance shrinks ~2× (succ resolution 1/32 vs 1/8), so best-checkpoint
# selection and run-to-run comparison stop being dominated by map luck. _eval_map_idxs re-derives
# evenly-spaced indices per split from len(this), so smaller splits stay in-range.
EVAL_MAP_IDX = tuple(int(round(i * 9904 / 31)) for i in range(32))


def _eval_map_idxs(eval_env: Explorer, k: int) -> tuple[int, ...]:
    """Evenly-spaced, in-range, deterministic map indices for a split's eval suite. The fixed
    EVAL_MAP_IDX tuple targets the big train splits (idx up to ~9904); smaller test splits
    (corridor=400) need indices clamped to their own size, else reload_map runs out of range."""
    n = int(getattr(eval_env.split, "n", 0)) or len(EVAL_MAP_IDX)
    k = max(1, min(k, n))
    import numpy as _np
    return tuple(int(i) for i in _np.linspace(0, n - 1, k).round().astype(int))


@torch.no_grad()
def _run_eval_suite(model: MarlActorCritic, eval_env: Explorer, cfg: TrainCfg,
                    map_idxs: tuple[int, ...] | None = None) -> dict:
    """Deterministic episodes on `map_idxs` (default EVAL_MAP_IDX) → eval/* metrics + eval/score.

    Coverage AUC pads an early (successful) finish with its final explored_rate so finishing
    sooner scores strictly higher. contrib_imbalance = max agent share − 1/M of union-new
    cells. sensing_overlap / comm_duty averaged over realized steps.
    """
    eval_map_idx = tuple(map_idxs) if map_idxs is not None else EVAL_MAP_IDX
    was_training = model.training
    model.eval()
    M = cfg.n_agents
    T = eval_env.cfg.max_episode_steps
    aucs, succ, imbs, ovs, duties, s90s = [], [], [], [], [], []
    fairs, score_maps = [], []          # D2: Jain fairness + per-map score (map-luck/noise)
    concs, idles = [], []               # temporal co-activity + per-agent idle rate
    imb_denom = (1.0 - 1.0 / M) if M > 1 else 1.0   # max possible imbalance → normalize to [0,1]
    # Concurrency window: an agent is "active" if it found ≥1 union-new cell in the last
    # W_CONC steps. concurrency_map = fraction of (post-warmup) steps where ALL agents are
    # active-in-window. Jain fairness catches one agent idle the WHOLE episode; concurrency
    # catches agents that ALTERNATE (Jain-fair but never working at the same time).
    W_CONC = 10
    for midx in eval_map_idx:
        eval_env.reload_map(env_idx=0, map_idx=int(midx))
        h_act, h_crit = model.init_hidden(1, cfg.device)
        obs = eval_env.obs
        er_curve: list[float] = []
        ov_sum = 0.0
        duty_sum = 0.0
        novel_final = None
        novel_prev = torch.zeros(M, device=cfg.device)   # per-agent cumulative at t-1
        active_hist: list[torch.Tensor] = []             # per step: [M] bool, found novel this step
        success = False
        s90 = float(T)
        for t in range(T):
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=cfg.device.startswith("cuda")):
                out = model.act(obs, h_act, h_crit, deterministic=True)
            obs, _r, done, info = eval_env.step(out["action"])
            h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
            er = float(info["explored_rate"][0].item())
            er_curve.append(er)
            ov_sum += float(info["metrics"]["sensing_overlap"].item())
            duty_sum += float(info["metrics"]["comm_duty_cycle"].item())
            novel_final = info["novel_cells_ep"][0]
            # Per-step per-agent novel = Δ cumulative; "active" this step if it found anything.
            novel_step = (novel_final - novel_prev).clamp(min=0.0)
            active_hist.append(novel_step > 0.0)
            novel_prev = novel_final.clone()
            if er >= 0.9 and s90 >= float(T):
                s90 = float(t + 1)
            if bool(done[0].item()):
                success = bool(info["terminated"][0].item())
                break
        steps = len(er_curve)
        aucs.append((sum(er_curve) + er_curve[-1] * (T - steps)) / T)
        succ.append(1.0 if success else 0.0)
        ovs.append(ov_sum / max(1, steps))
        duties.append(duty_sum / max(1, steps))
        s90s.append(s90)
        total_novel = float(novel_final.sum().item())
        if total_novel > 0:
            imb_m = float(novel_final.max().item()) / total_novel - 1.0 / M
            # Jain fairness over per-agent novel shares ∈ [1/M, 1]; 1 = perfect equity.
            ss = float((novel_final.float() ** 2).sum().item())
            fair_m = (total_novel ** 2) / (M * ss) if ss > 0 else 1.0
        else:
            imb_m, fair_m = 0.0, 1.0
        imbs.append(imb_m)
        fairs.append(fair_m)
        # Temporal concurrency: stack [steps, M], window-OR over the last W_CONC steps,
        # then require ALL agents active-in-window at each post-warmup step.
        if M > 1 and steps > W_CONC:
            act = torch.stack(active_hist, dim=0).float()                 # [steps, M]
            csum = act.cumsum(dim=0)
            win = csum.clone()
            win[W_CONC:] = csum[W_CONC:] - csum[:-W_CONC]                 # rolling-window novel count
            active_in_win = win[W_CONC:] > 0                              # [steps-W, M]
            conc_m = float(active_in_win.all(dim=1).float().mean().item())
            # Per-agent idle rate: fraction of steps an agent found nothing (max over agents
            # = the laziest agent's idle fraction; 1.0 = one agent never explored).
            idle_m = float((1.0 - act.mean(dim=0)).max().item())
        else:
            conc_m, idle_m = 1.0, 0.0
        concs.append(conc_m)
        idles.append(idle_m)
        # D2: per-map score on the NORMALIZED imbalance so equity is on the same [0,1]
        # footing as coverage_auc (raw imb spans only ~[0, 1−1/M] → was a near-free rider).
        score_maps.append(aucs[-1]
                          - cfg.score_w_imbalance * (imb_m / imb_denom)
                          - cfg.score_w_overlap * ovs[-1]
                          - cfg.score_w_idle * idle_m)
    if was_training:
        model.train()
    n = float(len(eval_map_idx))
    mean_imb = sum(imbs) / n
    mean_score = sum(score_maps) / n
    var_score = sum((s - mean_score) ** 2 for s in score_maps) / n
    out = {
        "eval/coverage_auc":      sum(aucs) / n,
        "eval/contrib_imbalance": mean_imb,                       # raw (kept for continuity)
        "eval/contrib_imbalance_norm": mean_imb / imb_denom,      # [0,1] — drives the score
        "eval/fairness":          sum(fairs) / n,                 # Jain index, 1 = equal (idle catcher)
        "eval/concurrency":       sum(concs) / n,                 # both active in same window (alternation catcher)
        "eval/idle_rate_max":     sum(idles) / n,                 # laziest-agent idle-step fraction
        "eval/sensing_overlap":   sum(ovs) / n,
        "eval/comm_duty":         sum(duties) / n,
        "eval/success_rate":      sum(succ) / n,
        "eval/steps_to_90":       sum(s90s) / n,
        "eval/score_std":         var_score ** 0.5,               # cross-map spread = map-luck
    }
    # Score = mean of per-map scores (equity normalized to [0,1] inside each map).
    out["eval/score"] = mean_score
    return out


def train(cfg: TrainCfg, log_every: int = 1, ckpt_pct: tuple[int, ...] = (20, 40, 60, 80, 100)) -> None:
    import time
    import numpy as np
    cfg.out_dir = Path(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    # Eval RNG always fresh entropy → eval-on-ckpt maps random each run, independent of cfg.seed.
    eval_rng = np.random.default_rng()
    if torch.cuda.is_available():
        # TF32 fastpath for f32 matmuls (Ampere+); inductor warning is silenced by this.
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    env, model = make_env_model(cfg)
    vnorm = ValueNormalizer().to(cfg.device)
    # Warm-start (optional): load model + value-norm from a prior checkpoint. Used to relaunch a
    # new stage (easy→difficult) at a DIFFERENT n_envs in a fresh process, avoiding the in-process
    # curriculum swap (which coexists two envs + recaptures CUDA graphs). Optimizer stays fresh.
    if cfg.init_ckpt:
        ck = torch.load(cfg.init_ckpt, map_location=cfg.device, weights_only=False)
        ck_sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # Reconcile the torch.compile "_orig_mod." key prefix between checkpoint and live model.
        tgt = model.state_dict()
        tgt_has_om = any("_orig_mod." in k for k in tgt)
        ck_has_om = any("_orig_mod." in k for k in ck_sd)
        if tgt_has_om and not ck_has_om:
            ck_sd = {k.replace("encoder.", "encoder._orig_mod.", 1) if k.startswith("encoder.") else k: v
                     for k, v in ck_sd.items()}
        elif not tgt_has_om and ck_has_om:
            ck_sd = {k.replace("._orig_mod.", ".", 1): v for k, v in ck_sd.items()}
        missing, unexpected = model.load_state_dict(ck_sd, strict=False)
        if isinstance(ck, dict) and "vnorm" in ck:
            vnorm.load_state_dict(ck["vnorm"])
        print(f"[init-ckpt] warm-started from {cfg.init_ckpt} "
              f"(iter={ck.get('iter','?') if isinstance(ck,dict) else '?'}, "
              f"missing={len(missing)} unexpected={len(unexpected)})")
    # Separate critic param group at a faster lr: the value target (coverage-left + time-left)
    # is highly non-stationary in this task, so the critic needs to track faster than the actor.
    # Paper Tab.7 uses Adam eps 1e-5 (vs torch default 1e-8). CTDE value-only modules.
    _critic_mods = (model.critic_pre, model.gru_critic, model.critic_head)
    _critic_params = [p for m in _critic_mods for p in m.parameters()]
    _critic_ids = {id(p) for p in _critic_params}
    _actor_params = [p for p in model.parameters() if id(p) not in _critic_ids]
    optimizer = torch.optim.Adam(
        [{"params": _actor_params, "lr": cfg.lr_actor},
         {"params": _critic_params, "lr": cfg.lr_critic}],
        eps=1e-5,
    )
    h_act, h_crit = model.init_hidden(cfg.n_envs, cfg.device)

    # Weights & Biases (optional). Guarded import → absent package / --wandb off = no-op.
    wb = None
    if cfg.wandb:
        try:
            import wandb as wb
            flat_cfg = cfg.__dict__ | {"env": cfg.env.__dict__, "ppo": cfg.ppo.__dict__}
            wb.init(project=cfg.wandb_project, entity=cfg.wandb_entity,
                    group=cfg.wandb_group, name=cfg.wandb_run_name,
                    mode=cfg.wandb_mode, tags=list(cfg.wandb_tags),
                    config=flat_cfg)
        except Exception as exc:
            print(f"[wandb] disabled ({exc})")
            wb = None

    sample_obs = env.obs
    buf = Rollout(sample_obs, T=cfg.rollout_len, N=cfg.n_envs, M=cfg.n_agents,
                  d_hidden=cfg.d_hidden, device=cfg.device)

    # v2 — persistent 1-env eval environment(s) for the fixed eval suite. Mirrors the full
    # training env cfg (force flags, top_k, n_hops, ...) so eval behavior matches training.
    eval_env_cfg = EnvCfg.from_ckpt_dict(cfg.env.__dict__, n_envs=1, n_agents=cfg.n_agents)
    train_eval_name = (cfg.curriculum_stage_splits[0] if cfg.curriculum_gated
                       else cfg.curriculum_splits[0] if cfg.curriculum else cfg.split)
    eval_env = Explorer(load_split(train_eval_name, device=cfg.device), eval_env_cfg, seed=0)
    # Multi-split eval: one persistent 1-env Explorer per requested split (canvases differ →
    # cannot reuse a single graph lattice). Falls back to the single training-split suite when
    # eval_suite_splits is empty. Each entry: (short_name, Explorer, map_idxs).
    K_EVAL = len(EVAL_MAP_IDX)
    if cfg.eval_suite_splits:
        eval_suites = []
        for name in cfg.eval_suite_splits:
            ev = Explorer(load_split(name, device=cfg.device), eval_env_cfg, seed=0)
            short = name.split("/")[-1]
            eval_suites.append((short, ev, _eval_map_idxs(ev, K_EVAL)))
        print(f"[evalsuite] multi-split: {[s[0] for s in eval_suites]}")
    else:
        eval_suites = [("", eval_env, _eval_map_idxs(eval_env, K_EVAL))]

    steps_per_iter = cfg.n_envs * cfg.rollout_len
    n_iters = max(1, cfg.total_steps // steps_per_iter)
    milestones = {int(round(n_iters * p / 100.0)): p for p in ckpt_pct}

    print(f"[train] iters={n_iters}  steps/iter={steps_per_iter}  total≈{n_iters * steps_per_iter:,}")
    # Gated-curriculum stage state: index into cfg.curriculum_stage_splits, advanced on eval
    # score. env already starts on stage 0 (make_env_model). Advance SWAPS the split (rebuilds
    # env + buffer) since stages can have different canvases.
    cur_stage = 0
    iters_on_stage = 0
    # Best-checkpoint tracking: keep the iterate with the highest eval/score, not just the last.
    # The eval curve is non-monotonic (PPO + high-variance dense reward) → final.pt is a random
    # iterate that may be far below the run's actual peak. ckpt_best.pt is overwritten whenever a
    # new eval tick beats the running max, so the run always leaves behind its best model.
    best_eval_score = float("-inf")
    best_eval_iter = -1
    if cfg.curriculum_gated:
        print(f"[curriculum] GATED stage 0='{cfg.curriculum_stage_splits[0]}' "
              f"gate_score={cfg.curriculum_gate_score} min_iters={cfg.curriculum_min_stage_iters}")
    total_train_time = 0.0      # cumulative collect+update time (eval-free)
    total_env_steps = 0

    # Event-driven status for the web inspector. Written ONCE here, then only at milestone
    # ckpts and on exit — zero cost in the hot loop. pid/host let the server detect a hard
    # kill (which never reaches the exit handler) by checking the process is still alive.
    status = {
        "state": "training", "pid": os.getpid(), "host": socket.gethostname(),
        "start_time": time.time(), "total_iters": n_iters,
        "split": cfg.split, "n_agents": cfg.n_agents, "n_envs": cfg.n_envs,
        "iter": 0, "progress_pct": 0.0, "last_ckpt": None, "ep_end_pct": None,
    }
    _dump_status(cfg.out_dir, status)

    # Exit events: normal finish sets state="done" below; any abnormal exit
    # (exception, Ctrl-C, docker stop/SIGTERM) lands here and records "stopped".
    # kill -9 leaves no event → server falls back to the PID liveness check.
    import atexit
    import signal

    def _finalize() -> None:
        if status["state"] == "training":   # not already marked done
            # Stop requested (web Stop button / docker stop / Ctrl-C) → SAVE THE POLICY before
            # exiting so no training is ever lost. status["iter"] tracks the loop in-memory.
            try:
                _save_ckpt(cfg.out_dir, model, vnorm, cfg, int(status.get("iter", 0)), "ckpt_stop")
                status["last_ckpt"] = "ckpt_stop"
            except Exception as exc:
                print(f"[finalize] policy save failed ({exc})")
            status["state"] = "stopped"
            _dump_status(cfg.out_dir, status)
    atexit.register(_finalize)

    def _on_sigterm(*_):
        raise KeyboardInterrupt   # unwind the loop → atexit fires → "stopped"
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)   # docker stop / kill
    except ValueError:
        pass   # not main thread (e.g. some sweep agents) — atexit still covers normal exit

    def _ondemand_eval(evals: list, tag: str, env_steps: int) -> None:
        """Run one or more eval suites on demand (web button). `evals` = list of
        {"split": <name>, "n": <count>}; empty → default eval_split/eval_n_maps. Each suite
        spins a fresh 1-env Explorer on its split, picks n evenly-spaced maps, logs eval/*,
        AND emits a GIF + step-through trace per map → unlocks the web inspector (capture_trace
        also copies inspector.html into the run dir)."""
        from eval.trace import capture_trace
        if not evals:
            evals = [{"split": cfg.eval_split, "n": cfg.eval_n_maps}]
        for spec in evals:
            split_name = spec.get("split", cfg.eval_split)
            n = int(spec.get("n", 1))
            try:
                split_obj = load_split(split_name, device=cfg.device)
                ev = Explorer(split_obj, eval_env_cfg, seed=0)
            except Exception as exc:
                print(f"[control] eval split '{split_name}' load failed ({exc})")
                continue
            idxs = _eval_map_idxs(ev, n)
            s = _run_eval_suite(model, ev, cfg, map_idxs=idxs)
            short = split_name.split("/")[-1]
            print(f"[control eval tag={tag}] {short}(n={len(idxs)}): "
                  f"score={s['eval/score']:+.3f} auc={s['eval/coverage_auc']:.3f} "
                  f"succ={s['eval/success_rate']:.2f} idle={s['eval/idle_rate_max']:.2f}")
            if wb is not None:
                wb.log({k.replace("eval/", f"eval/ondemand/{short}/"): v for k, v in s.items()},
                       step=env_steps)
            # GIF + trace per map → renders + inspector. capture_trace writes traces/<tag>/ +
            # traces/index.json + inspector.html into the run dir, so the inspector unlocks.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()       # free fragmentation headroom for the render envs
            for gi, midx in enumerate(idxs):
                mtag = f"{tag}_{short}_m{gi}"
                gif_path = cfg.out_dir / f"eval_{mtag}.gif"
                try:
                    _emit_eval_gif(model, cfg, gif_path, int(midx), split_name=split_name)
                except Exception as exc:
                    print(f"[control] gif '{mtag}' skipped ({exc})")
                try:
                    capture_trace(model, split_obj, cfg.env.__dict__, cfg.n_agents,
                                  int(midx), cfg.eval_steps, cfg.out_dir, mtag, cfg.device)
                except Exception as exc:
                    print(f"[control] trace '{mtag}' skipped ({exc})")
                if wb is not None and gif_path.exists():
                    wb.log({f"behavior/ondemand_{short}_m{gi}": wb.Video(str(gif_path), fps=12,
                                                                         format="gif")},
                           step=env_steps)

    for it in range(1, n_iters + 1):
        status["iter"] = it          # in-memory only (no disk write) → _finalize saves at the right iter
        # Web control channel — the dashboard drops control.json to (a) save a checkpoint + run
        # the requested eval suites on demand, or (b) hand-switch the curriculum stage. Acted on
        # at the iteration boundary, then consumed. model/optimizer/vnorm persist across a switch.
        ctl = _read_control(cfg.out_dir)
        if ctl:
            kind = ctl.get("cmd")
            if kind == "ckpt_eval":
                tag = str(ctl.get("tag") or f"it{it:04d}")
                _save_ckpt(cfg.out_dir, model, vnorm, cfg, it, f"ckpt_{tag}")
                _ondemand_eval(ctl.get("evals", []), tag, total_env_steps)
                status.update(iter=it, progress_pct=round(100.0 * it / n_iters, 1),
                              last_ckpt=f"ckpt_{tag}", env_steps=total_env_steps)
                _dump_status(cfg.out_dir, status)
            elif kind == "switch_stage":
                # Manual split swap — same rebuild as the gated advance (canvas/N_max may differ).
                _STAGE = {"easy": ("train/easy", 196), "difficult": ("train/difficult", 384)}
                if ctl.get("split"):
                    new_split = str(ctl["split"])
                    new_steps = int(ctl.get("max_steps", cfg.env.max_episode_steps))
                elif ctl.get("stage") in _STAGE:
                    new_split, new_steps = _STAGE[ctl["stage"]]
                else:
                    new_split = None
                    print(f"[control] switch_stage: unknown stage/split ({ctl})")
                if new_split:
                    cfg.env.max_episode_steps = max(int(new_steps), cfg.rollout_len)
                    env = Explorer(load_split(new_split, device=cfg.device), cfg.env, seed=cfg.seed)
                    buf = Rollout(env.obs, T=cfg.rollout_len, N=cfg.n_envs, M=cfg.n_agents,
                                  d_hidden=cfg.d_hidden, device=cfg.device)
                    h_act, h_crit = model.init_hidden(cfg.n_envs, cfg.device)
                    cfg.split = new_split
                    iters_on_stage = 0
                    status.update(split=new_split)
                    _dump_status(cfg.out_dir, status)
                    print(f"[control] MANUAL stage switch → '{new_split}' "
                          f"max_steps={cfg.env.max_episode_steps}; env+buffer rebuilt "
                          f"({env.H}×{env.W})")
            else:
                print(f"[control] unknown cmd ({ctl})")

        # H.5 — update curriculum weights each iter. GATED mode advances on eval score (in the
        # eval block below), so here only the legacy fixed time-ramp touches the weights.
        if cfg.curriculum and not cfg.curriculum_gated and hasattr(env.split, "set_weights"):
            new_w = _curriculum_weights(it / max(1, n_iters))
            if new_w != env.split.weights:
                env.split.set_weights(new_w)
                print(f"[curriculum] iter={it} weights={new_w}")
        iters_on_stage += 1
        buf.h_actor_init.copy_(h_act.detach())
        buf.h_critic_init.copy_(h_crit.detach())
        t_collect = time.time()
        h_act, h_crit, ep_end_mean, ep_end_n, agg = collect_rollout(env, model, buf, h_act, h_crit, vnorm)
        t_update = time.time()
        stats = ppo_update(model, optimizer, vnorm, buf, cfg.ppo, cfg.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()
        # SPS measures only collect + update — milestone eval excluded.
        iter_time = t_end - t_collect
        total_train_time += iter_time
        total_env_steps += steps_per_iter
        sps_iter = steps_per_iter / max(1e-6, iter_time)
        sps_all  = total_env_steps / max(1e-6, total_train_time)
        coll_pct = 100.0 * (t_update - t_collect) / max(1e-6, iter_time)
        coll_sps = steps_per_iter / max(1e-6, t_update - t_collect)
        upd_sps  = steps_per_iter / max(1e-6, t_end - t_update)
        # Composite efficiency (sweep target): coverage − overlap − standing-still.
        # Uses ep_end when available, else current-rollout mean explored proxy via redundancy.
        ep_for_eff = ep_end_mean if ep_end_n > 0 else float("nan")
        redundancy = agg.get("metric/redundancy", 0.0)
        stall_rate = agg.get("metric/stall_rate", 0.0)
        efficiency = (ep_for_eff - cfg.eff_w_redundancy * redundancy
                      - cfg.eff_w_stall * stall_rate)
        if it % log_every == 0:
            # H.1 — ep_end: mean explored_rate at terminal step over episodes that ENDED.
            ep_str = f"ep_end={ep_end_mean*100:5.1f}%(ended={ep_end_n:3d})" if ep_end_n > 0 else "ep_end=   n/a       "
            print(f"[it {it:4d}/{n_iters}] "
                  f"{ep_str}  "
                  f"pg={stats['pg_loss']:+.4f}  v={stats['v_loss']:.4f}  "
                  f"ent={stats['entropy']:.3f}  kl={stats['kl']:+.4f}  "
                  f"clip={stats['clipfrac']*100:.1f}%  "
                  f"redun={redundancy:.2f} stall={stall_rate*100:.0f}% "
                  f"pair={agg.get('metric/mean_pair_dist', 0.0):.2f} "
                  f"sps={sps_iter:.0f}({sps_all:.0f}avg)")
        # Live heartbeat for the web dashboard: refresh iter/progress/sps EVERY iter (status.json
        # is a tiny atomic write; iters are seconds apart so this is free). Previously written only
        # at milestones, so the card's iter + progress bar sat frozen between them.
        status.update(iter=it, progress_pct=round(100.0 * it / n_iters, 1),
                      env_steps=total_env_steps, sps=round(sps_iter),
                      ep_end_pct=(round(ep_end_mean * 100, 1) if ep_end_n > 0
                                  else status.get("ep_end_pct")))
        _dump_status(cfg.out_dir, status)
        # v2 — fixed eval suite: deterministic episodes on per-split fixed maps. The sweep
        # maximizes eval/score (comparable across trials — same maps, no sampling noise).
        # Multi-split: run each suite, log eval/<split>/* + aggregate eval/* = mean over splits.
        eval_stats: dict = {}
        if it % cfg.eval_every == 0 or it == n_iters:
            per_suite = []
            for short, ev, midxs in eval_suites:
                s = _run_eval_suite(model, ev, cfg, map_idxs=midxs)
                per_suite.append((short, s))
                if short:   # multi-split: also expose each split's metrics
                    for k, v in s.items():
                        eval_stats[k.replace("eval/", f"eval/{short}/")] = v
            # Aggregate = mean over suites → primary eval/* (and the curriculum gate signal).
            keys = per_suite[0][1].keys()
            for k in keys:
                eval_stats[k] = sum(s[k] for _, s in per_suite) / len(per_suite)
            # Inspector is mandatory for every run: on the FIRST eval tick (no traces yet) capture
            # one trace so the inspector unlocks automatically — no button press needed. capture_trace
            # also copies inspector.html into the run dir.
            if not (cfg.out_dir / "traces" / "index.json").exists():
                try:
                    from eval.trace import capture_trace
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()   # free fragmentation headroom for the render env
                    _short, _ev, _midxs = eval_suites[0]
                    capture_trace(model, _ev.split, cfg.env.__dict__, cfg.n_agents,
                                  int(_midxs[0]), cfg.eval_steps, cfg.out_dir,
                                  f"auto_it{it:04d}", cfg.device)
                    print(f"[inspector] auto trace captured (it={it}) → inspector unlocked")
                except Exception as exc:
                    print(f"[inspector] auto trace skipped ({exc})")
            split_str = ("  [" + " ".join(
                f"{sh or 'train'}={s['eval/score']:+.2f}" for sh, s in per_suite) + "]") \
                if len(per_suite) > 1 else ""
            print(f"[evalsuite it={it:4d}] score={eval_stats['eval/score']:+.3f}"
                  f"±{eval_stats['eval/score_std']:.3f}  "
                  f"auc={eval_stats['eval/coverage_auc']:.3f}  "
                  f"imbN={eval_stats['eval/contrib_imbalance_norm']:.3f}  "
                  f"fair={eval_stats['eval/fairness']:.3f}  "
                  f"conc={eval_stats['eval/concurrency']:.2f}  "
                  f"idle={eval_stats['eval/idle_rate_max']:.2f}  "
                  f"ov={eval_stats['eval/sensing_overlap']:.2f}  "
                  f"duty={eval_stats['eval/comm_duty']:.2f}  "
                  f"succ={eval_stats['eval/success_rate']:.2f}  "
                  f"s90={eval_stats['eval/steps_to_90']:.0f}{split_str}")
            # Best-checkpoint selection: overwrite ckpt_best.pt whenever this eval beats the peak.
            if eval_stats["eval/score"] > best_eval_score:
                best_eval_score = eval_stats["eval/score"]
                best_eval_iter = it
                _save_ckpt(cfg.out_dir, model, vnorm, cfg, it, "ckpt_best")
                print(f"[best] new best eval/score={best_eval_score:+.3f} at it={it} → ckpt_best.pt")
            # GATED curriculum: advance to the next split when the suite score clears the gate
            # AND the current stage has had its minimum dwell (avoids advancing on a noisy early
            # spike). Advance SWAPS the training split → rebuild env + buffer (canvas/N_max may
            # differ) and zero the hidden state; the model + optimizer + value-norm persist.
            if (cfg.curriculum_gated
                    and cur_stage < len(cfg.curriculum_stage_splits) - 1
                    and eval_stats["eval/score"] >= cfg.curriculum_gate_score
                    and iters_on_stage >= cfg.curriculum_min_stage_iters):
                cur_stage += 1
                iters_on_stage = 0
                next_name = cfg.curriculum_stage_splits[cur_stage]
                if cfg.curriculum_stage_steps:
                    cfg.env.max_episode_steps = max(
                        int(cfg.curriculum_stage_steps[cur_stage]), cfg.rollout_len)
                env = Explorer(load_split(next_name, device=cfg.device), cfg.env, seed=cfg.seed)
                buf = Rollout(env.obs, T=cfg.rollout_len, N=cfg.n_envs, M=cfg.n_agents,
                              d_hidden=cfg.d_hidden, device=cfg.device)
                h_act, h_crit = model.init_hidden(cfg.n_envs, cfg.device)
                print(f"[curriculum] GATED advance → stage {cur_stage} split='{next_name}' "
                      f"(score {eval_stats['eval/score']:+.3f} ≥ {cfg.curriculum_gate_score}); "
                      f"env+buffer rebuilt (canvas {env.H}×{env.W}, max_steps={env.cfg.max_episode_steps})")
        if wb is not None:
            log = {
                "train/pg_loss": stats["pg_loss"], "train/v_loss": stats["v_loss"],
                "train/entropy": stats["entropy"], "train/kl": stats["kl"],
                "train/clipfrac": stats["clipfrac"], "train/nan_skips": stats.get("nan_skips", 0),
                "perf/sps": sps_iter,
                "perf/coll_sps": coll_sps, "perf/upd_sps": upd_sps,
                "explore/ep_end": ep_end_mean, "explore/ep_end_n": ep_end_n,
                "explore/efficiency": efficiency, "iter": it,
            }
            log.update(agg)
            log.update(eval_stats)
            wb.log(log, step=total_env_steps)
        if it in milestones:
            pct = milestones[it]
            ckpt_path = cfg.out_dir / f"ckpt_{pct:03d}.pt"
            torch.save({
                "iter": it,
                "model": model.state_dict(),
                "vnorm": vnorm.state_dict(),
                "cfg": cfg.__dict__ | {"env": cfg.env.__dict__, "ppo": cfg.ppo.__dict__,
                                       "out_dir": str(cfg.out_dir),
                                       "n_agents": cfg.n_agents},
            }, ckpt_path)
            print(f"[ckpt] {ckpt_path}")
            # Milestone status update (rare event — piggybacks the ckpt's own disk write).
            status.update(iter=it, progress_pct=round(100.0 * it / n_iters, 1),
                          last_ckpt=f"ckpt_{pct:03d}", env_steps=total_env_steps,
                          ep_end_pct=round(ep_end_mean * 100, 1) if ep_end_n > 0
                          else status["ep_end_pct"])
            _dump_status(cfg.out_dir, status)
            if cfg.eval_on_ckpt:
                from env.maps import load_split as _ls
                _eval_split = _ls(cfg.eval_split, device=cfg.device)
                n_maps = _eval_split.n
                if cfg.eval_map_idx >= 0:
                    map_indices = [cfg.eval_map_idx] * cfg.eval_n_maps
                else:
                    map_indices = eval_rng.integers(0, n_maps, size=cfg.eval_n_maps).tolist()
                from eval.trace import capture_trace
                for gi, midx in enumerate(map_indices):
                    gif_path = cfg.out_dir / f"eval_ckpt_{pct:03d}_m{gi}.gif"
                    _emit_eval_gif(model, cfg, gif_path, int(midx))
                    # Step-through decision trace for the web inspector (same episode/map).
                    try:
                        capture_trace(model, _eval_split, cfg.env.__dict__, cfg.n_agents,
                                      int(midx), cfg.eval_steps, cfg.out_dir,
                                      f"ckpt_{pct:03d}_m{gi}", cfg.device)
                    except Exception as e:
                        print(f"[trace] skipped ({e})")
                    # Surface behavior in the W&B dashboard (not just on-disk files).
                    if wb is not None:
                        wb.log({f"behavior/rollout_m{gi}": wb.Video(str(gif_path), fps=12,
                                                                    format="gif")},
                               step=total_env_steps)
    # final — include cfg so eval can mirror the env config (I.2).
    torch.save({
        "iter": n_iters,
        "model": model.state_dict(),
        "vnorm": vnorm.state_dict(),
        "cfg": cfg.__dict__ | {"env": cfg.env.__dict__, "ppo": cfg.ppo.__dict__,
                               "out_dir": str(cfg.out_dir),
                               "n_agents": cfg.n_agents},
    }, cfg.out_dir / "final.pt")
    print(f"[done] {cfg.out_dir/'final.pt'}")
    # Clean-completion event: marks DONE so the atexit handler won't downgrade to "stopped".
    status.update(state="done", iter=n_iters, progress_pct=100.0,
                  env_steps=total_env_steps)
    _dump_status(cfg.out_dir, status)
    if wb is not None:
        wb.finish()
