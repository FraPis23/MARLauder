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
    g_run.add_argument("--seed", type=int, default=0, help="random seed (torch: action sampling, init)")
    g_run.add_argument("--map-seed", type=int, default=None,
                    help="Seed the MAP stream too. Default None = fresh OS entropy every run (map diversity), which means two runs with the same --seed still see different maps — fine for training, fatal for an A/B. Pass an int so two configs differ only by the thing under test")
    g_run.add_argument("--device", default="cuda:0", help="torch device (cuda:0 or cpu)")

    g_scale = ap.add_argument_group("Scale & episode")
    g_scale.add_argument("--total-steps", type=int, default=5_000_000, help="total env steps to train for")
    g_scale.add_argument("--n-envs", type=int, default=16, help="parallel environments")
    g_scale.add_argument("--n-agents", type=int, default=1,
                    help="Number of cooperative agents per env")
    g_scale.add_argument("--rollout-len", type=int, default=128, help="rollout length per PPO iteration")
    g_scale.add_argument("--max-episode-steps", type=int, default=512, help="max steps per episode")
    g_scale.add_argument("--max-travel-px", type=float, default=0.0,
                    help="Episode travel budget in px (0 = off, step cap only). Truncates once the "
                         "FARTHEST-travelled robot has covered this much ground. One of our steps is "
                         "a single lattice hop (<=22.63px) while an IR2 step is a waypoint teleport of "
                         "arbitrary length, so an equal step cap is NOT an equal budget — at IR2's "
                         "native caps we get ~43%% of their travel on complex. Distance is the unit "
                         "that means the same thing on both sides (and is IR2's own headline metric). "
                         "The step cap stays active as a safety net for a stalling policy.")
    g_scale.add_argument("--max-travel-frac", type=float, default=0.0,
                    help="Travel budget PER MAP, as px travelled per GT-free-pixel (0 = off; "
                         "overrides --max-travel-px when both are set). Use this for TRAINING: "
                         "train/difficult spans 3.9x in free area p50->p90, so a flat px budget "
                         "starves exactly the large maps where the completion bonus must fire. "
                         "Measured spend of the v12 policy: 0.0217 (hybrid), 0.0311 (complex), "
                         "0.0374 (corridor) px per free-px.")
    g_scale.add_argument("--done-mode", choices=["union", "own"], default="union",
                    help="What ends an episode. 'union' = the TEAM union map hits 99%% (legacy MARLauder). "
                         "'own' = EVERY agent's OWN map hits 99%% — the IR2 rule (their env.check_done), "
                         "which makes sharing part of the task and is what their `success` column measures. "
                         "Under 'own' the completion_bonus almost never fires on hard maps, so it is a real "
                         "change to the reward structure, not just to the stopping rule")
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
    g_sense.add_argument("--no-comm-relay", dest="comm_relay", action="store_false",
                    help="Exchange state only over a DIRECT link. Default is multi-hop relay: with "
                         "A-B-C, A and C share maps, positions and staleness through B, as IR2 does "
                         "(connected components of the comm graph). Pre-v20 behaviour, for a control "
                         "arm. The sync REWARD is on the direct link either way.")
    g_sense.set_defaults(comm_relay=True)
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
                    help="comma-separated splits for the eval suite (e.g. train/difficult,test/complex). "
                         "NOT 'extra': this REPLACES the default single suite on the training split, so "
                         "listing only test splits leaves the run with NO deterministic episodic eval on "
                         "the split it is training on — list the training split explicitly if you want it. "
                         "v19 passed only test/complex and was therefore blind to a policy that degraded on "
                         "BOTH splits (ckpt_020: success 0.41 on train/difficult, 0.09 on test/complex) while "
                         "every per-step training metric stayed flat. Those metrics are rollout MEANS that "
                         "include early-episode states, so they cannot represent episode outcomes and cannot "
                         "show this. Each extra split costs a full suite per eval tick — pair with --eval-every. "
                         "Empty = single suite on the training split")

    g_reward = ap.add_argument_group("Reward shaping")
    g_reward.add_argument("--novel-scan-weight", type=float, default=1.0, help="α_novel: privileged team-union novel-scan credit (v2 core reward)")
    g_reward.add_argument("--rdv-weight",      type=float, default=1.0, help="w: dense RENDEZVOUS reward = w·g·(φ_prev−φ_now), g=surplus gate. At w=1.0 a full-gate approach hop pays 1.0·0.02=0.020 against a 0.015-0.021 step_penalty, i.e. it exactly REBATES the travel cost and leaves the meet-vs-explore decision to --sync-weight. Above ~2 it becomes a chase term. 0 disables. M>1 only")
    g_reward.add_argument("--rdv-offer-frac",  type=float, default=0.15, help="Rendezvous gate saturates (g→1) when the map gained since last sync reaches this fraction of the OWN map size AT that sync (relative growth, floored by scan_norm_nodes); also normalizes the ∆M actor obs")
    g_reward.add_argument("--rdv-clamp-pos", action="store_true",
                    help="Pay only the APPROACH half of the rdv term: Δφ clamped to ≥0, so moving AWAY from the teammate is never taxed. Measured on v15, reward/rdv was −0.20/episode — a standing tax on exactly the divergence exploration requires. This does break the telescoping property, but that property was already gone: g·(φ_prev−φ_now) with a state-dependent gate and no γ was never potential-based shaping. Safe while --rdv-weight < step_penalty_coef·scan_norm_nodes = 0.75, above which approach→retreat→approach becomes free money")
    g_reward.add_argument("--div-weight", type=float, default=0.0,
                help="FRONTIER-DIVERSITY auxiliary ACTOR loss (0 = off, exact no-op). Prices two "
                     "agents committing to the same work: E[shared discounted frontier mass down "
                     "the exits their policies pick], averaged over agent pairs. Not a reward — "
                     "return, advantage and critic are untouched (v18 died moving the reward "
                     "budget at M=4). Not on the local logits — two agents 300px apart share no "
                     "action index, which is why the v17/J.1/J.2 port was zero exactly when the "
                     "duplication was decided. Self-extinguishing: disjoint frontier sets give "
                     "overlap 0 and no gradient. Turning it on makes the env emit a [M,M,K,K] "
                     "tensor per step (33 MB of rollout buffer at N=32,T=256,M=4).")
    g_reward.add_argument("--comm-idle-pen", type=float, default=0.0,
                    help="Cost of STAYING in radio contact on a step that delivered no map. Free on the step a sync is actually PAID, and free near the deadline (same budget ramp as --rdv-urgency-mode budget) so the terminal rendezvous is never taxed. Measured on v16: after the first sync the pair closes 472->342 px and sensor overlap goes 0.089->0.247, because contact fuses the maps, identical maps give identical utility fields, and identical fields pick the same frontier — a self-reinforcing loop that nothing priced. 81.8%% of contact steps are sustained rather than paying. Gated on sync_paid and NOT on the g gate: `offer` is computed after fusion resets its baseline, so g is ~0 on every contact step by construction and would punish the legitimate meeting just as hard. Training-time shaping only — the radio physics stays exactly IR2's. 0 disables")
    g_reward.add_argument("--rdv-urgency-mode", choices=["time", "budget"], default="time",
                    help="What makes a rendezvous urgent. 'time' (legacy) ramps on steps-since-last-sync, so the gate opens merely because the two have been apart — pulling them together mid-episode, when they should still be splitting. 'budget' ramps on the fraction of the episode budget spent (the same travel_frac the actor observes), so the pull appears only near the deadline. Under --done-mode own the terminal meeting is what completes both maps, and there is no reason to pay for it early. Changes what `g` means, and `g` is both agent_scalars[0] and the rdv reward gate — switch it only at a phase boundary")
    g_reward.add_argument("--rdv-urgency-start", type=float, default=0.5,
                    help="budget mode only: fraction of the episode budget spent before urgency starts ramping (0.5 = explore for the first half, then meeting becomes progressively worth more). Ignored when --rdv-urgency-mode time")
    g_reward.add_argument("--rdv-urgency-T", type=float, default=200.0,
                    help="Steps of separation at which the rendezvous urgency nudge saturates. ALSO the normalizer of the staleness ACTOR OBS (was max_episode_steps, which made one step worth 0.0005 at T=2048 and silently rescaled the input across the easy→difficult warm start). Set it to a real physical scale: 'how long apart before meeting is urgent'")
    g_reward.add_argument("--completion-bonus", type=float, default=10.0,
                    help="Terminal payout when the done criterion fires. Under --done-mode own this is the ONLY term paying for the actual objective, and it had no flag at all before v16. Raise it if reward/completion stays flat while coverage plateaus — but check --gamma first: at γ=0.99 a 2048-step episode discounts a bonus of 10 to ≈0 seen from t=0, so the term is invisible no matter how large it is")
    g_reward.add_argument("--step-penalty", type=float, default=0.015,
                    help="Per-axial-step movement cost (diagonal costs ·√2), charged per lattice-edge length. The direct price of hesitation: raise it to buy directness, at the risk of the agent preferring to stop exploring. Distinct from --stall-pen, which prices standing still")
    g_reward.add_argument("--sync-weight",     type=float, default=0.0,
                    help="ζ_g: SYNC-EVENT reward = ζ_g·(give + ρ·recv)/scan_norm_nodes, paid on the RISING EDGE of comm only, and only ≥ --sync-min-gap steps after the last paid sync. give = |my map \\ his map| pre-fusion. This is the OBJECTIVE term for rendezvous (rdv-weight is only shaping). 0.25 ≈ 2.55 per sync after 200 steps apart vs a measured 1.7-1.9 detour cost. 0 disables (pre-v11 behavior)")
    g_reward.add_argument("--sync-recv-ratio", type=float, default=0.5,
                    help="ρ: recv is paid at ρ·ζ_g so BOTH agents gain from meeting (else the map-poor one evades while the rich one chases), while give stays dominant so free-riding on recv doesn't pay")
    g_reward.add_argument("--sync-min-gap",    type=int,   default=32,
                    help="Steps since the last PAID sync required for a contact to pay again. Kills comm-boundary flicker; the contact still FUSES, only the payment is suppressed")
    g_reward.add_argument("--revisit-pen",     type=float, default=0.05, help="γ: revisit penalty per step (graduated by recency)")
    g_reward.add_argument("--revisit-window",  type=int,   default=16,   help="W: revisit lookback steps (8→16 2026-07-15: freshly-scanned trail stays hot longer)")
    g_reward.add_argument("--stall-pen",       type=float, default=0.1,  help="δ_stall: heavy penalty for standing still (no net displacement this step)")
    g_reward.add_argument("--stall-streak-beta", type=float, default=0.5,
                    help="v0.9 cumulative stall: consecutive stalls multiply δ_stall by 1+β·(streak−1), clamped to --stall-streak-cap. 0 disables")
    g_reward.add_argument("--stall-streak-cap",  type=float, default=4.0,
                    help="v0.9: max multiplier on δ_stall for consecutive stalls")
    g_reward.add_argument("--revisit-streak-beta", type=float, default=0.5,
                    help="v0.9 cumulative revisit: landings on recent (age<W) nodes multiply the graduated revisit penalty by 1+β_rev·(streak−1), UNCAPPED. 0 disables")
    g_reward.add_argument("--revisit-streak-decay", type=float, default=0.5,
                    help="v0.9.1: a NON-recent landing subtracts this from the revisit streak instead of zeroing it — one high-age hop can't launder the debt; working it off takes a sustained run on new/old ground")
    g_reward.add_argument("--revisit-streak-cap", type=float, default=float("inf"),
                    help="Max multiplier on the graduated revisit penalty (mirrors --stall-streak-cap). Default inf = legacy uncapped. Measured on v10: streak peaks at 89 → ×45 → 4.05 reward/step and a −44 per-episode tail vs novel +17, i.e. pure return variance that can bury the sync reward. Try 4.0 if reward/revisit p95 dominates")
    g_reward.add_argument("--radar-gamma",     type=float, default=0.92, help="RADAR feat[5/6] per-hop discount beyond the ego-window horizon. 0.92 mutes frontiers ~45+ hops out (0.4%%/node); 0.97 keeps them visible (~8%% with --radar-util-norm 3)")
    g_reward.add_argument("--radar-util-norm", type=float, default=8.0,  help="RADAR b_util normalization divisor (lower = far frontier mass squashed less)")
    g_reward.add_argument("--belief-mode",     choices=["uniform", "pathfront"], default="uniform",
                    help="teammate-position belief model used post-comm-break: 'uniform' geodesic ball (old default) vs 'pathfront' two-phase hypothesis model")
    # What the pathfront belief calls an OPENING. These were EnvCfg-only defaults, so a run's
    # params.json recorded nothing about them and two runs with different frontier semantics were
    # indistinguishable after the fact — exactly the config drift this project has already been
    # bitten by. Defaults here mirror EnvCfg; passing them explicitly is what puts them on record.
    g_reward.add_argument("--pf-frontier-min-unknown", type=int, default=1,
                    help="pathfront: minimum UNKNOWN 8-neighbours for a node to count as an opening. "
                         "Guard only — the real gate is --pf-frontier-min-util. Was 4, which dropped "
                         "large openings the observer had merely approached")
    g_reward.add_argument("--pf-frontier-min-util", type=float, default=1e-6,
                    help="pathfront: minimum PRE-diffusion utility seed (util_raw = ribbon x volume) "
                         "for a node to count as an opening. This is what removes wall-adjacent "
                         "nodes whose 'unknown' is unreachable, and room-corner tips")
    g_reward.add_argument("--radar-team-source", choices=["lkp", "belief"], default="lkp",
                    help="feat[6] RADAR teammate source beyond the ego window: 'lkp' (old default) decays "
                         "a point at each teammate's last-known node; 'belief' mass-transports the belief "
                         "FIELD itself (same gamma_r travel-cost decay as feat[5] b_util) so a belief that "
                         "moved off the lkp (e.g. --belief-mode pathfront) still shows up correctly. "
                         "Silently falls back to lkp when use_teammate_belief is off")

    g_target = ap.add_argument_group("Model ablations & warm-start")
    g_target.add_argument("--gru", action="store_true", help="Enable GRU temporal memory in actor+critic. Default OFF: the model runs feed-forward (both GRUCells bypassed)")
    g_target.add_argument("--no-gru", action="store_true", help="Force GRU OFF (redundant with the default; kept for back-compat / explicitness). Overrides --gru")
    g_target.add_argument("--no-gat-actor", action="store_true",
                    help="ABLATION: VF-only actor — steers from the analytic value-field (+prev_action/agent_scalars) only; curr_emb zeroed, pointer replaced by actor_head(h)+w_vf·vf. GAT still runs for the CTDE critic")
    g_target.add_argument("--no-gat", action="store_true",
                    help="ABLATION: NO GAT AT ALL — encoder never run. Actor as --no-gat-actor (VF-only); critic embedding = masked mean⊕max of raw window node features projected to d (+ critic_global). Big speed/VRAM win. Implies --no-gat-actor")
    g_target.add_argument("--vf-gamma", type=float, default=0.97,
                    help="Value-field per-hop discount: V_k = Σ γ^hops·utility over the BF branch leaving through neighbor k (max-normalized to [0,1], actor obs + pointer logit bias)")
    g_target.add_argument("--init-ckpt", default=None, help="Warm-start: load model + value-norm from this .pt at startup (optimizer stays fresh). Use to relaunch a new stage (easy→difficult) at a different --n-envs in a fresh process (avoids the in-process curriculum swap + CUDA-graph recapture)")

    g_score = ap.add_argument_group("Eval scoring weights")
    g_score.add_argument("--score-w-imbalance", type=float, default=0.5, help="eval/score weight on NORMALIZED contrib_imbalance (equity; D2: now on [0,1] imb so equity is a first-class term, not a free rider)")
    g_score.add_argument("--score-w-overlap",   type=float, default=0.25, help="eval/score weight on sensing_overlap (redundant sensing)")
    g_score.add_argument("--score-w-idle",      type=float, default=0.25, help="eval/score weight on idle_rate_max (laziest agent idle-step fraction) → selects for BOTH agents actively exploring (no idle/turn-taking)")

    g_ppo = ap.add_argument_group("PPO / learning")
    g_ppo.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    g_ppo.add_argument("--sync-weight-m-scale", type=float, default=0.0,
                    help="Scale the sync bonus by (2/M)^THIS. 0 = off (exact no-op at every M). "
                         "Identically 1 at M=2 for any exponent, so M=2 history cannot move. "
                         "MEASURED motivation: v16 M=2 vs v19 M=4 realized reward shares put sync at "
                         "5.5%% vs 9.8%% (x1.78) and sync_rate at 0.0068 vs 0.0206 (x3.03) — meetings "
                         "are ~3x more frequent with 4 robots, so the same per-event bonus buys a "
                         "much bigger slice of the budget. novel needs NO such scaling (already at "
                         "parity, 27.2%% vs 27.3%%, via --novel-scan-weight) and rdv is 0.6-0.7%% at "
                         "both M, i.e. nothing to scale. 1.0 gives 0.5 at M=4 (share-match target "
                         "0.56). WATCH eval/own_coverage_final and eval/sync_gap: under done_mode=own "
                         "every robot needs the others' maps, so under-paying sync breaks completion "
                         "before it shows up in eval/score.")
    g_ppo.add_argument("--ent-coef", type=float, default=0.01, help="entropy bonus coefficient")
    g_ppo.add_argument("--diag-grad", action="store_true",
                    help="Log train/g_pg, train/g_ent and their ratio: ||grad|| of the "
                         "policy-gradient term vs of the entropy bonus, measured with two extra "
                         "backward passes on one chunk per iteration. Comparing the two LOSS "
                         "values cannot settle which one drives the actor (pg is a clipped "
                         "surrogate whose value can be near zero while its gradient is not); on "
                         "v19 the entropy bonus exceeded |pg_loss| in 94%% of iterations, which is "
                         "suggestive but not conclusive without this.")
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
    g_flags.add_argument("--eval-every", type=int, default=10,
                    help="Iterations between eval-suite ticks. The suite is 32 maps x full episodes "
                         "on ONE env and renders nothing: measured on v19 (M=4, 768 steps) it costs "
                         "~19 min, i.e. 27%% of wall time at 10. It is also the only writer of "
                         "ckpt_best.pt and the only place a gated curriculum can advance, so a "
                         "larger value coarsens best-checkpoint resolution and lengthens dwell.")
    g_flags.add_argument("--eval-steps", type=int, default=-1,
                    help="G.2: episode length for eval-on-ckpt GIFs/traces. -1 = same as --max-episode-steps")
    g_flags.add_argument("--trace-steps", type=int, default=512,
                    help="HARD CAP on the episode length of the milestone GIF + inspector trace "
                         "(NOT the eval suite, which still runs full --eval-steps episodes and is "
                         "what picks ckpt_best). eval/trace.py builds the whole episode in Python "
                         "objects before serialising: a 1813-step difficult episode is 3.1 GB of "
                         "JSON on disk and ~27 GB live, which OOM-killed v14 and v15 at it=98 on a "
                         "30 GB host. 512 is v10's value, which ran a full phase 2 without dying.")
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
