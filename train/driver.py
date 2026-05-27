"""MAPPO training driver. Collects rollouts, runs PPO updates, checkpoints at
milestones (25/50/75/100%), logs metrics to stdout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
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
    n_layers: int = 2
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


def _normalize_cfg(cfg: TrainCfg) -> None:
    cfg.env.n_envs = cfg.n_envs
    cfg.env.n_agents = cfg.n_agents
    cfg.env.max_episode_steps = max(cfg.env.max_episode_steps, cfg.rollout_len)


def make_env_model(cfg: TrainCfg) -> tuple[Explorer, MarlActorCritic]:
    _normalize_cfg(cfg)
    split = load_split(cfg.split, device=cfg.device)
    env = Explorer(split, cfg.env, seed=cfg.seed)
    model = MarlActorCritic(n_agents=cfg.n_agents, d=cfg.d_hidden,
                            n_heads=cfg.n_heads, n_layers=cfg.n_layers).to(cfg.device)
    if cfg.compile and torch.cuda.is_available():
        try:
            model.encoder = torch.compile(model.encoder, mode="reduce-overhead", dynamic=False)
            print("[compile] model.encoder compiled")
        except Exception as exc:
            print(f"[compile] failed ({exc}), continuing uncompiled")
    return env, model


def collect_rollout(env: Explorer, model: MarlActorCritic, buf: Rollout, h_act, h_crit, vnorm: ValueNormalizer) -> tuple:
    obs = env.obs
    explored_acc = 0.0
    explored_final = 0.0
    for t in range(buf.T):
        with torch.no_grad():
            out = model.act(obs, h_act, h_crit, deterministic=False)
        action = out["action"]
        logp = out["logp"]
        v_norm = out["value"]
        value = vnorm.denormalize(v_norm)
        obs_next, reward, done, info = env.step(action)
        buf.store(t, obs, action, logp, value, reward, done)
        er = info["explored_rate"]
        explored_acc += float(er.mean().item())
        # peak per-env (env auto-resets on done — capture the value at done==True step,
        # otherwise the running mean of the final step).
        explored_final = float(er.mean().item())
        # Update hidden states; zero where done at this step.
        h_act = out["hidden_actor"]
        h_crit = out["hidden_critic"]
        # done is recorded at step t — for next step's hidden, mask.
        nonterm = (~done).float()
        h_act = h_act * nonterm.view(-1, 1, 1)
        h_crit = h_crit * nonterm.view(-1, 1)
        obs = obs_next
    # Bootstrap V(s_T) under final obs.
    with torch.no_grad():
        out = model.act(obs, h_act, h_crit, deterministic=True)
        v_last = vnorm.denormalize(out["value"])
    buf.last_value.copy_(v_last)
    return h_act, h_crit, explored_acc / buf.T, explored_final


def _emit_eval_gif(model: "MarlActorCritic", cfg: "TrainCfg", out: Path, map_idx: int) -> None:
    """Run a deterministic episode on (eval_split, map_idx) and save a GIF."""
    import imageio.v2 as imageio
    import numpy as np
    from env.maps import sample_batch
    from eval.rollout import EvalCfg, EvalRollout
    from env.explorer import EnvCfg as _EnvCfg, Explorer as _Explorer
    from env.maps import load_split as _load

    split = _load(cfg.eval_split, device=cfg.device)
    env_cfg = _EnvCfg(
        n_envs=1, n_agents=cfg.n_agents,
        nr=cfg.env.nr, sensor_range_px=cfg.env.sensor_range_px,
        comm_range_px=cfg.env.comm_range_px,
        max_episode_steps=cfg.eval_steps + 1,
    )
    env = _Explorer(split, env_cfg, seed=map_idx)
    gt_new, starts_new, fc_new = sample_batch(
        split, 1, indices=np.array([map_idx]), seed=0, device=cfg.device,
    )
    # Load specific map and reset properly using _spread_starts_graph
    # so multi-agent starts are on distinct nodes (not same pixel).
    env.world.gt_torch.copy_(gt_new)
    env.world.occupancy_torch.zero_()
    env.world.occupancy_logodds_torch.zero_()
    env.starts.copy_(starts_new)
    env.free_total.copy_(fc_new.float())
    env.visited_step.fill_(-1)
    env.t.zero_()
    row0, col0 = int(starts_new[0, 0]), int(starts_new[0, 1])
    agent_pos = env._spread_starts_graph(row0, col0)                    # [M, 2] on GPU
    env.pos[0] = agent_pos
    for ag in range(env.M):
        env.last_known_pos[0, :, ag] = agent_pos[ag]
    env.world.set_positions(env.pos)
    env.world.scan()
    env.last_union.copy_((env.world.occupancy_torch == 1).any(dim=1).view(1, -1).float().sum(dim=-1))
    env._refresh_obs()

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


def train(cfg: TrainCfg, log_every: int = 1, ckpt_pct: tuple[int, ...] = (25, 50, 75, 100)) -> None:
    import time
    import numpy as np
    cfg.out_dir = Path(cfg.out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    eval_rng = np.random.default_rng(cfg.seed + 99)
    if torch.cuda.is_available():
        # TF32 fastpath for f32 matmuls (Ampere+); inductor warning is silenced by this.
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    env, model = make_env_model(cfg)
    vnorm = ValueNormalizer().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr_actor)
    h_act, h_crit = model.init_hidden(cfg.n_envs, cfg.device)

    sample_obs = env.obs
    buf = Rollout(sample_obs, T=cfg.rollout_len, N=cfg.n_envs, M=cfg.n_agents,
                  d_hidden=cfg.d_hidden, device=cfg.device)

    steps_per_iter = cfg.n_envs * cfg.rollout_len
    n_iters = max(1, cfg.total_steps // steps_per_iter)
    milestones = {int(round(n_iters * p / 100.0)): p for p in ckpt_pct}

    print(f"[train] iters={n_iters}  steps/iter={steps_per_iter}  total≈{n_iters * steps_per_iter:,}")
    t_start = time.time()
    t_prev = t_start
    total_env_steps = 0
    for it in range(1, n_iters + 1):
        buf.h_actor_init.copy_(h_act.detach())
        buf.h_critic_init.copy_(h_crit.detach())
        t_collect = time.time()
        h_act, h_crit, explored_avg, explored_final = collect_rollout(env, model, buf, h_act, h_crit, vnorm)
        t_update = time.time()
        stats = ppo_update(model, optimizer, vnorm, buf, cfg.ppo, cfg.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.time()
        total_env_steps += steps_per_iter
        sps_iter = steps_per_iter / max(1e-6, t_end - t_prev)
        sps_all = total_env_steps / max(1e-6, t_end - t_start)
        coll_pct = 100.0 * (t_update - t_collect) / max(1e-6, t_end - t_collect)
        coll_sps = steps_per_iter / max(1e-6, t_update - t_collect)
        upd_sps = steps_per_iter / max(1e-6, t_end - t_update)
        t_prev = t_end
        if it % log_every == 0:
            print(f"[it {it:4d}/{n_iters}] "
                  f"explored avg={explored_avg*100:5.1f}% end={explored_final*100:5.1f}%  "
                  f"pg={stats['pg_loss']:+.4f}  v={stats['v_loss']:.4f}  "
                  f"ent={stats['entropy']:.3f}  kl={stats['kl']:+.4f}  "
                  f"clip={stats['clipfrac']*100:.1f}%  "
                  f"sps={sps_iter:.0f}({sps_all:.0f}avg) coll={coll_sps:.0f} upd={upd_sps:.0f}")
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
    # final
    torch.save({
        "iter": n_iters,
        "model": model.state_dict(),
        "vnorm": vnorm.state_dict(),
    }, cfg.out_dir / "final.pt")
    print(f"[done] {cfg.out_dir/'final.pt'}")
