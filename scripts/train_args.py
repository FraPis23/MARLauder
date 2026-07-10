"""Shared argument parser for training.

Kept torch-free on purpose: the web dashboard introspects this parser to auto-build the
launch form (every flag, its default, choices, help → tooltip, and CATEGORY → collapsible
section) WITHOUT importing torch or the env/model packages (which would allocate GPU in the
web-server process). run_train.py imports build_parser() too, so the CLI and the web form
never drift apart.

Flags are grouped with argparse's own add_argument_group() — the group title IS the launch
form's section label, read back via schema()'s `category` field. No separate category map to
keep in sync.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    g_run = ap.add_argument_group("Run")
    g_run.add_argument("--split", default="train/easy", help="map split to train on (when --stage is not used)")
    g_run.add_argument("--stage", choices=["easy", "difficult"], default=None,
                    help="IR2-style MANUAL curriculum: pick one stage and train only on it (no auto-advance). "
                         "Overrides --split and --max-episode-steps to the IR2 coupling "
                         "(easy=train/easy@196 steps, difficult=train/difficult@384 steps). "
                         "Relaunch with the next --stage to advance by hand. Ignored if --curriculum-gated is set.")
    g_run.add_argument("--out", type=Path, default=None,
                    help="output run dir. Omit → auto-create runs/<run-name|run>_<timestamp> so "
                         "every training gets its own fresh folder (no manual --out each time).")
    g_run.add_argument("--force", action="store_true",
                    help="overwrite an existing --out directory without asking. Only matters when --out "
                         "names an existing dir; auto-named runs never collide.")
    g_run.add_argument("--seed", type=int, default=0, help="random seed")
    g_run.add_argument("--device", default="cuda:0", help="torch device (cuda:0 or cpu)")

    g_scale = ap.add_argument_group("Scale & episode")
    g_scale.add_argument("--total-steps", type=int, default=5_000_000, help="total env steps to train for")
    g_scale.add_argument("--n-envs", type=int, default=16, help="parallel environments")
    g_scale.add_argument("--n-agents", type=int, default=1,
                    help="Number of cooperative agents per env")
    g_scale.add_argument("--rollout-len", type=int, default=128, help="rollout length per PPO iteration")
    g_scale.add_argument("--max-episode-steps", type=int, default=512, help="max steps per episode")
    g_scale.add_argument("--minibatches", type=int, default=1,
                    help="PPO minibatches per epoch (must divide n-envs)")
    g_scale.add_argument("--n-hops", type=int, default=6,
                    help="Ego-centric encoder window radius. Window side = 2·n_hops + 3 "
                         "(49 nodes at 2, 121 at 4, 225 at 6). GAT n_layers tied to n_hops "
                         "(default 6 = 6-layer GAT, 6-hop receptive field).")

    g_sense = ap.add_argument_group("Sensing & communication")
    g_sense.add_argument("--comm-range", type=float, default=120.0,
                    help="[comm-model=los] hard Euclidean comm cutoff in pixels (0 = agents never communicate)")
    g_sense.add_argument("--comm-model", choices=["signal_strength", "los"], default="signal_strength",
                    help="Comm model: 'signal_strength' = realistic path-loss radio (walls attenuate, per-episode noise); 'los' = legacy hard range+LOS")
    g_sense.add_argument("--sensor-range", type=float, default=80.0,
                    help="LiDAR sensor range in pixels (realistic 2D-LiDAR reach)")
    g_sense.add_argument("--ss-thresh", type=float, default=-70.0,
                    help="[comm-model=signal_strength] rx sensitivity (dBm): connect iff received power > this. Lower = longer comm range")
    g_sense.add_argument("--force-full-comm", action="store_true",
                    help="A2 debug: bypass dist/LOS check; every pair communicates every step")
    g_sense.add_argument("--force-full-pos-sharing", action="store_true",
                    help="Debug: persistent teammate-position awareness (positions only, maps still comm-gated)")
    g_sense.add_argument("--force-full-occupancy-sharing", action="store_true",
                    help="H.4 debug: persistent map fusion every step (occupancy synced across agents)")
    g_sense.add_argument("--no-teammate-obs", action="store_true",
                    help="ABLATION: blind the actor to teammates — zeroes agent_scalars [∆M-gate, staleness], feat[4] teammate-proximity potential and feat[6] radar-teammate. Map fusion at comm, rdv reward gate and privileged critic (geo_pair) unchanged. Pure-exploration test (pair with --rdv-weight 0)")

    g_curr = ap.add_argument_group("Curriculum")
    g_curr.add_argument("--curriculum", action="store_true",
                    help="H.5: train on easy + difficult with ramping mix (0-30%% all-easy, 30-60%% 70/30, 60-100%% 50/50)")
    g_curr.add_argument("--curriculum-gated", action="store_true",
                    help="Performance-gated curriculum (split-SWAP): train on --curriculum-stage-splits one at a time, advancing to the next only when the eval suite score clears --curriculum-gate-score (after --curriculum-min-stage-iters dwell). Standalone — does NOT need --curriculum. Works across different canvases (easy→difficult)")
    g_curr.add_argument("--curriculum-stage-splits", default="train/easy,train/difficult",
                    help="comma-separated split sequence for gated curriculum (easy→hard). Env+buffer rebuilt on each advance")
    g_curr.add_argument("--curriculum-stage-steps", default="196,384",
                    help="comma-separated per-stage max episode length (IR2 values: easy=196, difficult=384; bigger maps need longer episodes). Empty = same --max-episode-steps for all stages. Must match --curriculum-stage-splits length")
    g_curr.add_argument("--curriculum-gate-score", type=float, default=0.5,
                    help="eval/score threshold to advance to the next curriculum stage")
    g_curr.add_argument("--curriculum-min-stage-iters", type=int, default=20,
                    help="min iters on a stage before a gated advance is allowed (anti-noise dwell)")
    g_curr.add_argument("--eval-split", default=None,
                    help="H.5: eval split for eval-on-ckpt (default = --split or test/complex when curriculum)")
    g_curr.add_argument("--eval-suite-splits", default="",
                    help="comma-separated extra splits for the multi-split eval suite (e.g. test/corridor,test/complex,test/hybrid). Empty = single suite on the training split")

    g_reward = ap.add_argument_group("Reward shaping")
    g_reward.add_argument("--novel-scan-weight", type=float, default=1.0, help="α_novel: privileged team-union novel-scan credit (v2 core reward)")
    g_reward.add_argument("--rdv-weight",      type=float, default=0.10, help="w: dense RENDEZVOUS reward = w·g·(φ_prev−φ_now), g=surplus gate. Rewards net geodesic approach toward the owed teammate. 0 disables. M>1 only")
    g_reward.add_argument("--rdv-offer-frac",  type=float, default=0.15, help="Rendezvous gate saturates (g→1) when the map gained since last sync reaches this fraction of the OWN map size AT that sync (relative growth, floored by scan_norm_nodes); also normalizes the ∆M actor obs")
    g_reward.add_argument("--revisit-pen",     type=float, default=0.05, help="γ: revisit penalty per step (graduated by recency)")
    g_reward.add_argument("--revisit-window",  type=int,   default=8,    help="W: revisit lookback steps")
    g_reward.add_argument("--stall-pen",       type=float, default=0.1,  help="δ_stall: heavy penalty for standing still (no net displacement this step)")
    g_reward.add_argument("--radar-gamma",     type=float, default=0.92, help="RADAR feat[5/6] per-hop discount beyond the ego-window horizon. 0.92 mutes frontiers ~45+ hops out (0.4%%/node); 0.97 keeps them visible (~8%% with --radar-util-norm 3)")
    g_reward.add_argument("--radar-util-norm", type=float, default=8.0,  help="RADAR b_util normalization divisor (lower = far frontier mass squashed less)")

    g_target = ap.add_argument_group("Model ablations & warm-start")
    g_target.add_argument("--gru", action="store_true", help="Enable GRU temporal memory in actor+critic. Default OFF: the model runs feed-forward (both GRUCells bypassed)")
    g_target.add_argument("--no-gru", action="store_true", help="Force GRU OFF (redundant with the default; kept for back-compat / explicitness). Overrides --gru")
    g_target.add_argument("--init-ckpt", default=None, help="Warm-start: load model + value-norm from this .pt at startup (optimizer stays fresh). Use to relaunch a new stage (easy→difficult) at a different --n-envs in a fresh process (avoids the in-process curriculum swap + CUDA-graph recapture)")

    g_score = ap.add_argument_group("Eval scoring weights")
    g_score.add_argument("--score-w-imbalance", type=float, default=0.5, help="eval/score weight on NORMALIZED contrib_imbalance (equity; D2: now on [0,1] imb so equity is a first-class term, not a free rider)")
    g_score.add_argument("--score-w-overlap",   type=float, default=0.25, help="eval/score weight on sensing_overlap (redundant sensing)")
    g_score.add_argument("--score-w-idle",      type=float, default=0.25, help="eval/score weight on idle_rate_max (laziest agent idle-step fraction) → selects for BOTH agents actively exploring (no idle/turn-taking)")

    g_ppo = ap.add_argument_group("PPO / learning")
    g_ppo.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    g_ppo.add_argument("--ent-coef", type=float, default=0.01, help="entropy bonus coefficient")
    g_ppo.add_argument("--clip-eps", type=float, default=0.15, help="PPO clip ε (≤0.2; 0.15 default — this task is more non-stationary than the paper's benchmarks)")
    g_ppo.add_argument("--k-epochs", type=int, default=4, help="PPO epochs per rollout (keep low: intra-episode obs shift + dense shaping = high non-stationarity)")
    g_ppo.add_argument("--max-grad-norm", type=float, default=2.0, help="gradient clip norm (paper 10.0; 2.0 here — dense shaping spikes gradients)")
    g_ppo.add_argument("--gae-lambda", type=float, default=0.95, help="GAE λ")
    g_ppo.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    g_ppo.add_argument("--vf-coef", type=float, default=0.5, help="value loss weight")
    g_ppo.add_argument("--tbptt-steps", type=int, default=16, help="TBPTT chunk length")

    g_flags = ap.add_argument_group("Runtime & checkpointing")
    g_flags.add_argument("--compile", action="store_true", help="torch.compile encoder (CUDA only)")
    g_flags.add_argument("--no-milestone-ckpt", action="store_true",
                    help="Disable the automatic 20/40/60/80/100%% checkpoints. Use with the web "
                         "dashboard's on-demand 'checkpoint + eval' button to avoid useless ckpts.")
    g_flags.add_argument("--eval-on-ckpt", action="store_true",
                    help="Emit 2 eval GIFs at each milestone (25/50/75/100%%)")
    g_flags.add_argument("--eval-steps", type=int, default=-1,
                    help="G.2: episode length for eval-on-ckpt GIFs/traces. -1 = same as --max-episode-steps")
    g_flags.add_argument("--eval-n-maps", type=int, default=2, help="GIFs + decision traces per milestone")
    g_flags.add_argument("--eval-map-idx", type=int, default=-1, help="fixed eval map (-1 = random each milestone)")

    g_wandb = ap.add_argument_group("Weights & Biases")
    g_wandb.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    g_wandb.add_argument("--wandb-project", default="marlauder", help="W&B project")
    g_wandb.add_argument("--wandb-entity", default=None, help="W&B entity")
    g_wandb.add_argument("--wandb-group", default=None, help="W&B group")
    g_wandb.add_argument("--wandb-run-name", default=None, help="W&B run name (also seeds the auto run-dir name)")
    g_wandb.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"], help="W&B mode")
    g_wandb.add_argument("--wandb-tags", nargs="*", default=[], help="W&B tags")
    return ap


def schema() -> list[dict]:
    """Introspect the parser → JSON-able field list for the web form. One entry per optional
    flag: {flag, dest, kind: 'bool'|'choice'|'int'|'float'|'str', default, choices, help,
    category}. `category` = the add_argument_group() title it was defined under — the web
    form's collapsible section label, so CLI and web form categorization never drift apart."""
    ap = build_parser()
    out: list[dict] = []
    for group in ap._action_groups:
        for a in group._group_actions:
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
                "flag":     flag,
                "dest":     a.dest,
                "kind":     kind,
                "default":  default,
                "choices":  list(a.choices) if a.choices else None,
                "help":     (a.help or "").replace("%%", "%"),
                "category": group.title,
            })
    return out
