"""Shared argument parser for training.

Kept torch-free on purpose: the web dashboard introspects this parser to auto-build the
launch form (every flag, its default, choices, and help → tooltip) WITHOUT importing torch
or the env/model packages (which would allocate GPU in the web-server process). run_train.py
imports build_parser() too, so the CLI and the web form never drift apart.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    # --- what to train on ---
    ap.add_argument("--split", default="train/easy", help="map split to train on (when --stage is not used)")
    ap.add_argument("--stage", choices=["easy", "difficult"], default=None,
                    help="IR2-style MANUAL curriculum: pick one stage and train only on it (no auto-advance). "
                         "Overrides --split and --max-episode-steps to the IR2 coupling "
                         "(easy=train/easy@196 steps, difficult=train/difficult@384 steps). "
                         "Relaunch with the next --stage to advance by hand. Ignored if --curriculum-gated is set.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output run dir. Omit → auto-create runs/<run-name|run>_<timestamp> so "
                         "every training gets its own fresh folder (no manual --out each time).")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--device", default="cuda:0", help="torch device (cuda:0 or cpu)")
    # --- scale ---
    ap.add_argument("--total-steps", type=int, default=5_000_000, help="total env steps to train for")
    ap.add_argument("--n-envs", type=int, default=16, help="parallel environments")
    ap.add_argument("--n-agents", type=int, default=1,
                    help="Number of cooperative agents per env")
    ap.add_argument("--comm-range", type=float, default=120.0,
                    help="[comm-model=los] hard Euclidean comm cutoff in pixels (0 = agents never communicate)")
    ap.add_argument("--comm-model", choices=["signal_strength", "los"], default="signal_strength",
                    help="Comm model: 'signal_strength' = realistic path-loss radio (walls attenuate, per-episode noise); 'los' = legacy hard range+LOS")
    ap.add_argument("--sensor-range", type=float, default=80.0,
                    help="LiDAR sensor range in pixels (realistic 2D-LiDAR reach)")
    ap.add_argument("--ss-thresh", type=float, default=-70.0,
                    help="[comm-model=signal_strength] rx sensitivity (dBm): connect iff received power > this. Lower = longer comm range")
    ap.add_argument("--rollout-len", type=int, default=128, help="rollout length per PPO iteration")
    ap.add_argument("--max-episode-steps", type=int, default=512, help="max steps per episode")
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
    ap.add_argument("--curriculum-gated", action="store_true",
                    help="Performance-gated curriculum (split-SWAP): train on --curriculum-stage-splits one at a time, advancing to the next only when the eval suite score clears --curriculum-gate-score (after --curriculum-min-stage-iters dwell). Standalone — does NOT need --curriculum. Works across different canvases (easy→difficult)")
    ap.add_argument("--curriculum-stage-splits", default="train/easy,train/difficult",
                    help="comma-separated split sequence for gated curriculum (easy→hard). Env+buffer rebuilt on each advance")
    ap.add_argument("--curriculum-stage-steps", default="196,384",
                    help="comma-separated per-stage max episode length (IR2 values: easy=196, difficult=384; bigger maps need longer episodes). Empty = same --max-episode-steps for all stages. Must match --curriculum-stage-splits length")
    ap.add_argument("--no-milestone-ckpt", action="store_true",
                    help="Disable the automatic 20/40/60/80/100%% checkpoints. Use with the web "
                         "dashboard's on-demand 'checkpoint + eval' button to avoid useless ckpts.")
    ap.add_argument("--curriculum-gate-score", type=float, default=0.5,
                    help="eval/score threshold to advance to the next curriculum stage")
    ap.add_argument("--curriculum-min-stage-iters", type=int, default=20,
                    help="min iters on a stage before a gated advance is allowed (anti-noise dwell)")
    ap.add_argument("--eval-split", default=None,
                    help="H.5: eval split for eval-on-ckpt (default = --split or test/complex when curriculum)")
    ap.add_argument("--eval-suite-splits", default="",
                    help="comma-separated extra splits for the multi-split eval suite (e.g. test/corridor,test/complex,test/hybrid). Empty = single suite on the training split")
    # Phase D reward shaping (lattice-level set ops, baselined at last comm).
    ap.add_argument("--scan-weight",     type=float, default=1.0,  help="(diagnostic only since v2; scan_self no longer in reward)")
    ap.add_argument("--novel-scan-weight", type=float, default=1.0, help="α_novel: privileged team-union novel-scan credit (v2 core reward)")
    ap.add_argument("--team-weight",     type=float, default=0.0,  help="β: REMOVED from reward (double-counted novel cells). Dead knob, kept for back-compat")
    ap.add_argument("--give-bonus",      type=float, default=0.06, help="ζ_give: NEW cells brought to teammate (scan_norm units, map-independent)")
    ap.add_argument("--recv-bonus",      type=float, default=0.02, help="ζ_recv: NEW cells received at rendezvous (scan_norm units)")
    ap.add_argument("--overlap-pen",     type=float, default=0.12, help="η_lap: redundant parallel scan penalty (scan_norm units)")
    ap.add_argument("--proximity-pen",   type=float, default=0.0,  help="G.4.b: per-step raw-distance penalty (ELIMINATED by default — it caused the ping-pong/deadlock; novel_scan handles anti-chase). >0 only for ablation")
    ap.add_argument("--revisit-pen",     type=float, default=0.05, help="γ: revisit penalty per step (graduated by recency)")
    ap.add_argument("--revisit-window",  type=int,   default=8,    help="W: revisit lookback steps")
    ap.add_argument("--strategic-gate-eps", type=float, default=0.0, help="High-level gate: the analytic guidepost next-hop direction steers the actor ONLY on steps where max utility in the ego window < this. 0 = gate off (always influences). The global target is consulted only when local exploration is exhausted")
    ap.add_argument("--no-gru", action="store_true", help="GRU ablation: run actor+critic feed-forward with NO temporal memory (bypass both GRUCells). Trains a separate model to measure whether recurrence matters")
    ap.add_argument("--init-ckpt", default=None, help="Warm-start: load model + value-norm from this .pt at startup (optimizer stays fresh). Use to relaunch a new stage (easy→difficult) at a different --n-envs in a fresh process (avoids the in-process curriculum swap + CUDA-graph recapture)")
    ap.add_argument("--target-mode", choices=["analytic", "nearest"], default="analytic", help="Env-owned global-target rule. analytic: deterministic rendezvous-aware scored frontier (util/(1+β·d)·rdv·separation). nearest: the closest reachable frontier by BF distance (no scoring/rendezvous)")
    ap.add_argument("--target-beta", type=float, default=1.0, help="Analytic target: distance discount β in util/(1+β·d/NR). Higher = prefer nearer frontiers")
    ap.add_argument("--target-lambda", type=float, default=1.0, help="Analytic target: rendezvous pull strength λ. 0 = pure exploration; 1 = a full-offer teammate can double a frontier's score")
    ap.add_argument("--rdv-offer-frac", type=float, default=0.15, help="Analytic target: offer saturates (w→1) when map gained since last sync reaches this fraction of total cells")
    ap.add_argument("--target-keep-margin", type=float, default=0.2, help="Analytic target commitment: keep last target unless a new frontier beats it by >this fraction (hysteresis vs ping-pong)")
    ap.add_argument("--target-sep-weight", type=float, default=0.5, help="Analytic target separation strength ∈[0,1]: down-weight frontiers a teammate owns (division of labor). 0 = off")
    ap.add_argument("--target-sep-from-offer", action="store_true", help="Tie separation to map-debt: sep_w = (1−w_offer) per-env. Separate when I have nothing fresh to give, converge when I do. Overrides --target-sep-weight. M>1 only")
    ap.add_argument("--stall-pen",       type=float, default=0.1,  help="δ_stall: heavy penalty for standing still (no net displacement this step)")
    ap.add_argument("--score-w-imbalance", type=float, default=0.5, help="eval/score weight on NORMALIZED contrib_imbalance (equity; D2: now on [0,1] imb so equity is a first-class term, not a free rider)")
    ap.add_argument("--score-w-overlap",   type=float, default=0.25, help="eval/score weight on sensing_overlap (redundant sensing)")
    ap.add_argument("--score-w-idle",      type=float, default=0.25, help="eval/score weight on idle_rate_max (laziest agent idle-step fraction) → selects for BOTH agents actively exploring (no idle/turn-taking)")
    ap.add_argument("--n-hops", type=int, default=6,
                    help="Ego-centric encoder window radius. Window side = 2·n_hops + 3 "
                         "(49 nodes at 2, 121 at 4, 225 at 6). GAT n_layers tied to n_hops "
                         "(default 6 = 6-layer GAT, 6-hop receptive field).")
    # --- learning (MAPPO knobs — exposed for W&B sweeps) ---
    ap.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    ap.add_argument("--ent-coef", type=float, default=0.01, help="entropy bonus coefficient")
    ap.add_argument("--clip-eps", type=float, default=0.15, help="PPO clip ε (≤0.2; 0.15 default — this task is more non-stationary than the paper's benchmarks)")
    ap.add_argument("--k-epochs", type=int, default=4, help="PPO epochs per rollout (keep low: intra-episode obs shift + dense shaping = high non-stationarity)")
    ap.add_argument("--max-grad-norm", type=float, default=2.0, help="gradient clip norm (paper 10.0; 2.0 here — dense shaping spikes gradients)")
    ap.add_argument("--gae-lambda", type=float, default=0.95, help="GAE λ")
    ap.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    ap.add_argument("--vf-coef", type=float, default=0.5, help="value loss weight")
    ap.add_argument("--tbptt-steps", type=int, default=16, help="TBPTT chunk length")
    # --- flags ---
    ap.add_argument("--compile", action="store_true", help="torch.compile encoder (CUDA only)")
    ap.add_argument("--eval-on-ckpt", action="store_true",
                    help="Emit 2 eval GIFs at each milestone (25/50/75/100%%)")
    ap.add_argument("--eval-steps", type=int, default=-1,
                    help="G.2: episode length for eval-on-ckpt GIFs/traces. -1 = same as --max-episode-steps")
    ap.add_argument("--eval-n-maps", type=int, default=2, help="GIFs + decision traces per milestone")
    ap.add_argument("--eval-map-idx", type=int, default=-1, help="fixed eval map (-1 = random each milestone)")
    # --- Weights & Biases ---
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-project", default="marlauder", help="W&B project")
    ap.add_argument("--wandb-entity", default=None, help="W&B entity")
    ap.add_argument("--wandb-group", default=None, help="W&B group")
    ap.add_argument("--wandb-run-name", default=None, help="W&B run name (also seeds the auto run-dir name)")
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"], help="W&B mode")
    ap.add_argument("--wandb-tags", nargs="*", default=[], help="W&B tags")
    return ap


def schema() -> list[dict]:
    """Introspect the parser → JSON-able field list for the web form. One entry per optional
    flag: {flag, dest, kind: 'bool'|'choice'|'int'|'float'|'str', default, choices, help}."""
    ap = build_parser()
    out: list[dict] = []
    for a in ap._actions:
        if not a.option_strings or a.dest in ("help",):
            continue
        flag = a.option_strings[0]
        if a.__class__.__name__ in ("_StoreTrueAction", "_StoreFalseAction"):
            kind = "bool"
        elif a.choices:
            kind = "choice"
        elif a.type in (int,):
            kind = "int"
        elif a.type in (float,):
            kind = "float"
        else:
            kind = "str"
        default = a.default
        if isinstance(default, Path):
            default = str(default)
        out.append({
            "flag":    flag,
            "dest":    a.dest,
            "kind":    kind,
            "default": default,
            "choices": list(a.choices) if a.choices else None,
            "help":    (a.help or "").replace("%%", "%"),
        })
    return out
