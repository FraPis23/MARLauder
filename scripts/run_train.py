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
from scripts.train_args import build_parser
from train.driver import TrainCfg, train
from train.mappo import MAPPOCfg


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()

    # Auto-create a fresh run dir when --out is omitted: runs/<run-name|run>_<timestamp>.
    # Keeps every training isolated without typing a unique --out each launch.
    if args.out is None:
        import time as _time
        _stamp = _time.strftime("%Y%m%d_%H%M%S")
        _base = (args.wandb_run_name or "run").strip().replace("/", "_") or "run"
        args.out = Path("/workspace/MARLauder/runs") / f"{_base}_{_stamp}"
        print(f"[out] auto run dir → {args.out}")

    # IR2-style MANUAL curriculum: --stage picks a single split + its episode length and
    # disables any auto-advance. One stage per run; relaunch with the next --stage to advance.
    if args.stage is not None:
        if args.curriculum or args.curriculum_gated:
            ap.error("--stage (manual curriculum) is mutually exclusive with "
                     "--curriculum / --curriculum-gated (automatic). Pick one.")
        _STAGE_STEPS = {"easy": 196, "difficult": 384}      # IR2 values
        args.split = f"train/{args.stage}"
        args.max_episode_steps = _STAGE_STEPS[args.stage]
        print(f"[stage] MANUAL curriculum: stage='{args.stage}' "
              f"split='{args.split}' max_episode_steps={args.max_episode_steps}")

    cfg = TrainCfg(
        split=args.split,
        out_dir=args.out,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        rollout_len=args.rollout_len,
        n_hops=args.n_hops,
        lr_actor=args.lr,
        strategic_gate_eps=args.strategic_gate_eps,
        use_gru=not args.no_gru,
        device=args.device,
        seed=args.seed,
        compile=args.compile,
        init_ckpt=args.init_ckpt,
        eval_on_ckpt=args.eval_on_ckpt,
        eval_n_maps=args.eval_n_maps,
        eval_map_idx=args.eval_map_idx,
        eval_split=(
            args.eval_split if args.eval_split is not None
            else ("test/complex" if args.curriculum else args.split)
        ),
        eval_steps=(args.max_episode_steps if args.eval_steps < 0 else args.eval_steps),
        curriculum=args.curriculum,
        curriculum_gated=args.curriculum_gated,
        curriculum_stage_splits=tuple(
            s.strip() for s in args.curriculum_stage_splits.split(",") if s.strip()
        ),
        curriculum_stage_steps=tuple(
            int(s) for s in args.curriculum_stage_steps.split(",") if s.strip()
        ),
        curriculum_gate_score=args.curriculum_gate_score,
        curriculum_min_stage_iters=args.curriculum_min_stage_iters,
        eval_suite_splits=tuple(
            s.strip() for s in args.eval_suite_splits.split(",") if s.strip()
        ),
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=tuple(args.wandb_tags),
        score_w_imbalance=args.score_w_imbalance,
        score_w_overlap=args.score_w_overlap,
        score_w_idle=args.score_w_idle,
        env=EnvCfg(
            n_envs=args.n_envs,
            n_agents=args.n_agents,
            nr=16,                              # lattice spacing — 16px → N_max≈1200 nodes
            max_episode_steps=args.max_episode_steps,
            sensor_range_px=args.sensor_range,
            comm_range_px=args.comm_range,
            comm_model=args.comm_model,
            ss_thresh=args.ss_thresh,
            n_hops=args.n_hops,
            force_full_comm=args.force_full_comm,
            force_full_pos_sharing=args.force_full_pos_sharing,
            force_full_occupancy_sharing=args.force_full_occupancy_sharing,
            scan_reward_weight=args.scan_weight,
            novel_scan_weight=args.novel_scan_weight,
            team_reward_weight=args.team_weight,
            give_bonus_coef=args.give_bonus,
            recv_bonus_coef=args.recv_bonus,
            overlap_penalty_coef=args.overlap_pen,
            proximity_penalty_coef=args.proximity_pen,
            revisit_penalty_coef=args.revisit_pen,
            revisit_window=args.revisit_window,
            stall_penalty_coef=args.stall_pen,
            analytic_target=True,   # env always owns the global target (StrategicHead removed)
            target_kind=("nearest" if args.target_mode == "nearest" else "analytic"),
            target_beta=args.target_beta,
            target_lambda=args.target_lambda,
            rdv_offer_frac=args.rdv_offer_frac,
            target_keep_margin=args.target_keep_margin,
            target_sep_weight=args.target_sep_weight,
            target_sep_from_offer=args.target_sep_from_offer,
        ),
        ppo=MAPPOCfg(
            ent_coef=args.ent_coef,
            n_minibatches=args.minibatches,
            clip_eps=args.clip_eps,
            k_epochs=args.k_epochs,
            max_grad_norm=args.max_grad_norm,
            lam=args.gae_lambda,
            gamma=args.gamma,
            vf_coef=args.vf_coef,
            tbptt_steps=args.tbptt_steps,
        ),
    )
    train(cfg, log_every=1,
          ckpt_pct=() if args.no_milestone_ckpt else (20, 40, 60, 80, 100))


if __name__ == "__main__":
    main()
