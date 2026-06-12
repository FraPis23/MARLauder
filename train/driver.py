"""MAPPO training driver. Collects rollouts, runs PPO updates, checkpoints at
milestones (25/50/75/100%), logs metrics to stdout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import MultiSplit, load_split
from models.actor_critic import MarlActorCritic
from models.value_normalizer import ValueNormalizer
from train.buffer import Rollout
from train.mappo import MAPPOCfg, ppo_update


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
    n_hops: int = 2          # ego-centric encoder window radius; n_layers tied to this
    n_layers: int = 2        # GAT layers; default tied to n_hops in make_env_model
    path_bias_floor: float = 1.5   # I.3 — fixed floor on target-following bias (actor logits)
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    device: str = "cuda:0"
    seed: int = 0
    compile: bool = False
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
    # eval/score = coverage_auc − w_imb·contrib_imbalance − w_ov·sensing_overlap.
    eval_every: int = 10
    score_w_imbalance: float = 0.5
    score_w_overlap: float = 0.25


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
    if cfg.curriculum:
        splits = [load_split(name, device=cfg.device) for name in cfg.curriculum_splits]
        ms = MultiSplit(splits, weights=_curriculum_weights(0.0))
        env = Explorer(ms, cfg.env, seed=cfg.seed)
        print(f"[curriculum] enabled: {cfg.curriculum_splits} initial weights={ms.weights}")
    else:
        split = load_split(cfg.split, device=cfg.device)
        env = Explorer(split, cfg.env, seed=cfg.seed)
    model = MarlActorCritic(n_agents=cfg.n_agents, d=cfg.d_hidden,
                            n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                            path_bias_floor=cfg.path_bias_floor).to(cfg.device)
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
    for t in range(buf.T):
        with torch.no_grad():
            out = model.act(obs, h_act, h_crit, deterministic=False)
        action = out["action"]
        logp = out["logp"]
        v_norm = out["value"]
        value = vnorm.denormalize(v_norm)
        obs_next, reward, done, info = env.step(action, target_choice=out["target_argmax"])
        buf.store(t, obs, action, logp, value, reward, done, out["target_choice"])
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
    with torch.no_grad():
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


def _emit_eval_gif(model: "MarlActorCritic", cfg: "TrainCfg", out: Path, map_idx: int) -> None:
    """Run a deterministic episode on (eval_split, map_idx) and save a GIF."""
    import imageio.v2 as imageio
    import numpy as np
    from env.maps import sample_batch
    from eval.rollout import EvalCfg, EvalRollout
    from env.explorer import EnvCfg as _EnvCfg, Explorer as _Explorer
    from env.maps import load_split as _load

    split = _load(cfg.eval_split, device=cfg.device)
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
EVAL_MAP_IDX = (120, 1543, 2877, 4012, 5530, 7211, 8650, 9904)


@torch.no_grad()
def _run_eval_suite(model: MarlActorCritic, eval_env: Explorer, cfg: TrainCfg) -> dict:
    """Deterministic episodes on the fixed EVAL_MAP_IDX maps → eval/* metrics + eval/score.

    Coverage AUC pads an early (successful) finish with its final explored_rate so finishing
    sooner scores strictly higher. contrib_imbalance = max agent share − 1/M of union-new
    cells. sensing_overlap / comm_duty averaged over realized steps.
    """
    was_training = model.training
    model.eval()
    M = cfg.n_agents
    T = eval_env.cfg.max_episode_steps
    aucs, succ, imbs, ovs, duties, s90s = [], [], [], [], [], []
    fairs, score_maps = [], []          # D2: Jain fairness + per-map score (map-luck/noise)
    imb_denom = (1.0 - 1.0 / M) if M > 1 else 1.0   # max possible imbalance → normalize to [0,1]
    for midx in EVAL_MAP_IDX:
        eval_env.reload_map(env_idx=0, map_idx=int(midx))
        h_act, h_crit = model.init_hidden(1, cfg.device)
        obs = eval_env.obs
        er_curve: list[float] = []
        ov_sum = 0.0
        duty_sum = 0.0
        novel_final = None
        success = False
        s90 = float(T)
        for t in range(T):
            out = model.act(obs, h_act, h_crit, deterministic=True)
            # No target_choice → target_switch penalty off at eval (reward unused anyway).
            obs, _r, done, info = eval_env.step(out["action"])
            h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
            er = float(info["explored_rate"][0].item())
            er_curve.append(er)
            ov_sum += float(info["metrics"]["sensing_overlap"].item())
            duty_sum += float(info["metrics"]["comm_duty_cycle"].item())
            novel_final = info["novel_cells_ep"][0]
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
        # D2: per-map score on the NORMALIZED imbalance so equity is on the same [0,1]
        # footing as coverage_auc (raw imb spans only ~[0, 1−1/M] → was a near-free rider).
        score_maps.append(aucs[-1]
                          - cfg.score_w_imbalance * (imb_m / imb_denom)
                          - cfg.score_w_overlap * ovs[-1])
    if was_training:
        model.train()
    n = float(len(EVAL_MAP_IDX))
    mean_imb = sum(imbs) / n
    mean_score = sum(score_maps) / n
    var_score = sum((s - mean_score) ** 2 for s in score_maps) / n
    out = {
        "eval/coverage_auc":      sum(aucs) / n,
        "eval/contrib_imbalance": mean_imb,                       # raw (kept for continuity)
        "eval/contrib_imbalance_norm": mean_imb / imb_denom,      # [0,1] — drives the score
        "eval/fairness":          sum(fairs) / n,                 # Jain index, 1 = equal
        "eval/sensing_overlap":   sum(ovs) / n,
        "eval/comm_duty":         sum(duties) / n,
        "eval/success_rate":      sum(succ) / n,
        "eval/steps_to_90":       sum(s90s) / n,
        "eval/score_std":         var_score ** 0.5,               # cross-map spread = map-luck
    }
    # Score = mean of per-map scores (equity normalized to [0,1] inside each map).
    out["eval/score"] = mean_score
    return out


def train(cfg: TrainCfg, log_every: int = 1, ckpt_pct: tuple[int, ...] = (25, 50, 75, 100)) -> None:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_actor)
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

    # v2 — persistent 1-env eval environment for the fixed eval suite. Mirrors the full
    # training env cfg (force flags, top_k, n_hops, ...) so eval behavior matches training.
    eval_env_cfg = EnvCfg.from_ckpt_dict(cfg.env.__dict__, n_envs=1, n_agents=cfg.n_agents)
    eval_suite_split = load_split(
        cfg.curriculum_splits[0] if cfg.curriculum else cfg.split, device=cfg.device)
    eval_env = Explorer(eval_suite_split, eval_env_cfg, seed=0)

    steps_per_iter = cfg.n_envs * cfg.rollout_len
    n_iters = max(1, cfg.total_steps // steps_per_iter)
    milestones = {int(round(n_iters * p / 100.0)): p for p in ckpt_pct}

    print(f"[train] iters={n_iters}  steps/iter={steps_per_iter}  total≈{n_iters * steps_per_iter:,}")
    total_train_time = 0.0      # cumulative collect+update time (eval-free)
    total_env_steps = 0
    for it in range(1, n_iters + 1):
        # H.5 — update curriculum weights each iter.
        if cfg.curriculum and hasattr(env.split, "set_weights"):
            new_w = _curriculum_weights(it / max(1, n_iters))
            if new_w != env.split.weights:
                env.split.set_weights(new_w)
                print(f"[curriculum] iter={it} weights={new_w}")
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
        # v2 — fixed eval suite: deterministic episodes on EVAL_MAP_IDX. The sweep
        # maximizes eval/score (comparable across trials — same maps, no sampling noise).
        eval_stats: dict = {}
        if it % cfg.eval_every == 0 or it == n_iters:
            eval_stats = _run_eval_suite(model, eval_env, cfg)
            print(f"[evalsuite it={it:4d}] score={eval_stats['eval/score']:+.3f}"
                  f"±{eval_stats['eval/score_std']:.3f}  "
                  f"auc={eval_stats['eval/coverage_auc']:.3f}  "
                  f"imbN={eval_stats['eval/contrib_imbalance_norm']:.3f}  "
                  f"fair={eval_stats['eval/fairness']:.3f}  "
                  f"ov={eval_stats['eval/sensing_overlap']:.2f}  "
                  f"duty={eval_stats['eval/comm_duty']:.2f}  "
                  f"succ={eval_stats['eval/success_rate']:.2f}  "
                  f"s90={eval_stats['eval/steps_to_90']:.0f}")
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
            if cfg.eval_on_ckpt:
                from env.maps import load_split as _ls
                _eval_split = _ls(cfg.eval_split, device=cfg.device)
                n_maps = _eval_split.n
                if cfg.eval_map_idx >= 0:
                    map_indices = [cfg.eval_map_idx] * cfg.eval_n_maps
                else:
                    map_indices = eval_rng.integers(0, n_maps, size=cfg.eval_n_maps).tolist()
                for gi, midx in enumerate(map_indices):
                    gif_path = cfg.out_dir / f"eval_ckpt_{pct:03d}_m{gi}.gif"
                    _emit_eval_gif(model, cfg, gif_path, int(midx))
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
    if wb is not None:
        wb.finish()
