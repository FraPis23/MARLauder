"""MAPPO training driver.

    python scripts/run_train.py --n-envs 128 --total-steps 5_000_000 --out runs/run_001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg
from train.driver import TrainCfg, train
from train.mappo import MAPPOCfg


def main() -> None:
    ap = argparse.ArgumentParser()
    # --- what to train on ---
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--out", type=Path, default=Path("/workspace/MARLauder/runs/run_default"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    # --- scale ---
    ap.add_argument("--total-steps", type=int, default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-agents", type=int, default=1,
                    help="Number of cooperative agents per env")
    ap.add_argument("--comm-range", type=float, default=120.0,
                    help="Communication range in pixels (0 = agents never communicate)")
    ap.add_argument("--rollout-len", type=int, default=128)
    ap.add_argument("--max-episode-steps", type=int, default=512)
    ap.add_argument("--minibatches", type=int, default=1,
                    help="PPO minibatches per epoch (must divide n-envs)")
    ap.add_argument("--force-full-comm", action="store_true",
                    help="A2 debug: bypass dist/LOS check; every pair communicates every step")
    ap.add_argument("--force-full-pos-sharing", action="store_true",
                    help="Debug: persistent teammate-position awareness (positions only, maps still comm-gated)")
    ap.add_argument("--force-full-occupancy-sharing", action="store_true",
                    help="H.4 debug: persistent map fusion every step (occupancy synced across agents)")
    ap.add_argument("--curriculum", action="store_true",
                    help="H.5: train on easy + difficult with ramping mix (0-30%% all-easy, 30-60%% 70/30, 60-100%% 50/50)")
    ap.add_argument("--eval-split", default=None,
                    help="H.5: eval split for eval-on-ckpt (default = --split or test/complex when curriculum)")
    ap.add_argument("--top-k", type=int, default=16,
                    help="Phase A v2: top-K frontier candidates per agent for strategic head")
    # Phase D reward shaping (lattice-level set ops, baselined at last comm).
    ap.add_argument("--scan-weight",     type=float, default=1.0,  help="(diagnostic only since v2; scan_self no longer in reward)")
    ap.add_argument("--novel-scan-weight", type=float, default=1.0, help="α_novel: privileged team-union novel-scan credit (v2 core reward)")
    ap.add_argument("--team-weight",     type=float, default=0.3,  help="β: shared Δunion_free coef")
    ap.add_argument("--give-bonus",      type=float, default=1.5,  help="ζ_give: NEW cells brought to teammate")
    ap.add_argument("--recv-bonus",      type=float, default=0.5,  help="ζ_recv: NEW cells received at rendezvous")
    ap.add_argument("--overlap-pen",     type=float, default=3.0,  help="η_lap: redundant parallel scan penalty")
    ap.add_argument("--yield-scale",     type=float, default=3.0,  help="G.4.a: scale on cand_own_minus_team yield feature")
    ap.add_argument("--proximity-pen",   type=float, default=0.05, help="G.4.b: per-step penalty for being within sensor_range of teammate (gated by comm)")
    ap.add_argument("--path-bias-floor", type=float, default=1.5,  help="I.3: fixed floor on target-following bias (actor logits)")
    ap.add_argument("--revisit-pen",     type=float, default=0.05, help="γ: revisit penalty per step (graduated by recency)")
    ap.add_argument("--revisit-window",  type=int,   default=8,    help="W: revisit lookback steps")
    ap.add_argument("--target-switch-pen", type=float, default=0.01, help="δ_obj: objective second-guessing penalty (argmax intent; v2 default 0.01 — v1's 0.05+sampled pick dominated the dense reward)")
    ap.add_argument("--stall-pen",       type=float, default=0.1,  help="δ_stall: heavy penalty for standing still (no net displacement this step)")
    ap.add_argument("--n-hops", type=int, default=2,
                    help="Ego-centric encoder window radius. Window side = 2·n_hops + 3 "
                         "(49 nodes at 2, 121 at 4, 225 at 6). GAT n_layers tied to n_hops.")
    # --- learning (MAPPO knobs — exposed for W&B sweeps) ---
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip ε")
    ap.add_argument("--k-epochs", type=int, default=4, help="PPO epochs per rollout")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAE λ")
    ap.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    ap.add_argument("--vf-coef", type=float, default=0.5, help="value loss weight")
    ap.add_argument("--tbptt-steps", type=int, default=16, help="TBPTT chunk length")
    # --- flags ---
    ap.add_argument("--compile", action="store_true", help="torch.compile encoder (CUDA only)")
    ap.add_argument("--eval-on-ckpt", action="store_true",
                    help="Emit 2 eval GIFs at each milestone (25/50/75/100%%)")
    ap.add_argument("--eval-steps", type=int, default=-1,
                    help="G.2: episode length for eval-on-ckpt GIFs. -1 = same as --max-episode-steps")
    # --- Weights & Biases ---
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-project", default="marlauder")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-tags", nargs="*", default=[])
    args = ap.parse_args()

    cfg = TrainCfg(
        split=args.split,
        out_dir=args.out,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        rollout_len=args.rollout_len,
        n_hops=args.n_hops,
        lr_actor=args.lr,
        path_bias_floor=args.path_bias_floor,
        device=args.device,
        seed=args.seed,
        compile=args.compile,
        eval_on_ckpt=args.eval_on_ckpt,
        eval_split=(
            args.eval_split if args.eval_split is not None
            else ("test/complex" if args.curriculum else args.split)
        ),
        eval_steps=(args.max_episode_steps if args.eval_steps < 0 else args.eval_steps),
        curriculum=args.curriculum,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=tuple(args.wandb_tags),
        env=EnvCfg(
            n_envs=args.n_envs,
            n_agents=args.n_agents,
            nr=16,                              # lattice spacing — 16px → N_max≈1200 nodes
            max_episode_steps=args.max_episode_steps,
            comm_range_px=args.comm_range,
            n_hops=args.n_hops,
            force_full_comm=args.force_full_comm,
            force_full_pos_sharing=args.force_full_pos_sharing,
            force_full_occupancy_sharing=args.force_full_occupancy_sharing,
            top_k_candidates=args.top_k,
            scan_reward_weight=args.scan_weight,
            novel_scan_weight=args.novel_scan_weight,
            team_reward_weight=args.team_weight,
            give_bonus_coef=args.give_bonus,
            recv_bonus_coef=args.recv_bonus,
            overlap_penalty_coef=args.overlap_pen,
            cand_own_minus_team_scale=args.yield_scale,
            proximity_penalty_coef=args.proximity_pen,
            revisit_penalty_coef=args.revisit_pen,
            revisit_window=args.revisit_window,
            target_switch_penalty_coef=args.target_switch_pen,
            stall_penalty_coef=args.stall_pen,
        ),
        ppo=MAPPOCfg(
            ent_coef=args.ent_coef,
            n_minibatches=args.minibatches,
            clip_eps=args.clip_eps,
            k_epochs=args.k_epochs,
            lam=args.gae_lambda,
            gamma=args.gamma,
            vf_coef=args.vf_coef,
            tbptt_steps=args.tbptt_steps,
        ),
    )
    train(cfg, log_every=1)


if __name__ == "__main__":
    main()
