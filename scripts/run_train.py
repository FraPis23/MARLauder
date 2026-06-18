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
    ap.add_argument("--target-yield-weight", type=float, default=0.0, help="J.1: α_yield — reward for CHOOSING a frontier you're closer to than the (live) teammate. One-sided pull (≥0, no repulsion); reactive position-driven division without ping-pong/idle. 0 = off")
    ap.add_argument("--proximity-pen",   type=float, default=0.0,  help="G.4.b: per-step raw-distance penalty (ELIMINATED by default — it caused the ping-pong/deadlock; novel_scan handles anti-chase). >0 only for ablation")
    ap.add_argument("--path-bias-floor", type=float, default=1.5,  help="I.3: fixed floor on target-following bias (actor logits)")
    ap.add_argument("--revisit-pen",     type=float, default=0.05, help="γ: revisit penalty per step (graduated by recency)")
    ap.add_argument("--revisit-window",  type=int,   default=8,    help="W: revisit lookback steps")
    ap.add_argument("--target-switch-pen", type=float, default=0.05, help="δ_obj: graph-tree branch-flip commitment penalty = the deliberation cost; now fires only on genuine (margin-gated) switches")
    ap.add_argument("--switch-margin", type=float, default=1.0, help="Phase 1: keep committed strategic target unless an alt beats it by > this (logit units). Higher = more commitment")
    ap.add_argument("--max-steps-on-option", type=int, default=24, help="Phase 1: horizon cap forcing a strategic re-pick (escape an unreachable-but-still-candidate target)")
    ap.add_argument("--no-strategic-head", action="store_true", help="Phase 3 ablation: bypass the StrategicHead; pointer decides directly, biased by the guidepost (nearest-frontier first-hop). Tests whether the head earns its place")
    ap.add_argument("--strategic-gate-eps", type=float, default=0.0, help="High-level gate: the StrategicHead/guidepost + BF path-bias steer the actor ONLY on steps where max utility in the ego window < this. 0 = gate off (always influences). The high-level chooser is invoked only when local exploration is exhausted")
    ap.add_argument("--target-mode", choices=["analytic", "learned"], default="analytic", help="Global-target source. analytic: env's deterministic rendezvous-aware guidepost (StrategicHead bypassed). learned: the StrategicHead picks (legacy)")
    ap.add_argument("--target-beta", type=float, default=1.0, help="Analytic target: distance discount β in util/(1+β·d/NR). Higher = prefer nearer frontiers")
    ap.add_argument("--target-lambda", type=float, default=1.0, help="Analytic target: rendezvous pull strength λ. 0 = pure exploration; 1 = a full-offer teammate can double a frontier's score")
    ap.add_argument("--rdv-offer-frac", type=float, default=0.15, help="Analytic target: offer saturates (w→1) when map gained since last sync reaches this fraction of total cells")
    ap.add_argument("--target-keep-margin", type=float, default=0.2, help="Analytic target commitment: keep last target unless a new frontier beats it by >this fraction (hysteresis vs ping-pong)")
    ap.add_argument("--progress-reward-coef", type=float, default=0.3, help="PBRS-style shaping: reward per node-unit of Euclidean progress toward the committed target. The missing 'follow utility' gradient. 0 = off")
    ap.add_argument("--stall-pen",       type=float, default=0.1,  help="δ_stall: heavy penalty for standing still (no net displacement this step)")
    ap.add_argument("--score-w-imbalance", type=float, default=0.5, help="eval/score weight on NORMALIZED contrib_imbalance (equity; D2: now on [0,1] imb so equity is a first-class term, not a free rider)")
    ap.add_argument("--score-w-overlap",   type=float, default=0.25, help="eval/score weight on sensing_overlap (redundant sensing)")
    ap.add_argument("--score-w-idle",      type=float, default=0.25, help="eval/score weight on idle_rate_max (laziest agent idle-step fraction) → selects for BOTH agents actively exploring (no idle/turn-taking)")
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
    ap.add_argument("--div-coef", type=float, default=0.0, help="J.3: cross-agent target-diversity loss weight. Penalizes both agents choosing the SAME frontier node, gated by frontier spread (no penalty when only one cluster exists → both push it, no idle). 0 = off")
    ap.add_argument("--div-spread-center", type=float, default=0.20, help="J.3: normalized candidate-spread at which the availability gate centers (below = one cluster = gate off)")
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
        switch_margin=args.switch_margin,
        max_steps_on_option=args.max_steps_on_option,
        disable_strategic=args.no_strategic_head,
        strategic_gate_eps=args.strategic_gate_eps,
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
        score_w_imbalance=args.score_w_imbalance,
        score_w_overlap=args.score_w_overlap,
        score_w_idle=args.score_w_idle,
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
            target_yield_weight=args.target_yield_weight,
            proximity_penalty_coef=args.proximity_pen,
            revisit_penalty_coef=args.revisit_pen,
            revisit_window=args.revisit_window,
            target_switch_penalty_coef=args.target_switch_pen,
            stall_penalty_coef=args.stall_pen,
            analytic_target=(args.target_mode == "analytic"),
            target_beta=args.target_beta,
            target_lambda=args.target_lambda,
            rdv_offer_frac=args.rdv_offer_frac,
            target_keep_margin=args.target_keep_margin,
            progress_reward_coef=args.progress_reward_coef,
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
            div_coef=args.div_coef,
            div_spread_center=args.div_spread_center,
        ),
    )
    train(cfg, log_every=1)


if __name__ == "__main__":
    main()
