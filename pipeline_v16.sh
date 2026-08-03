#!/bin/bash
# v1.6 — EXPLORE FAST AGAIN, SEE THE DEADLINE, PAY FOR THE EXCHANGE.
#
# WHY THIS EXISTS. The policy got slower and less decisive after v12, and it is measurable on the
# same maps with the same metric:
#     v10  union 90% at 193 steps (evalsuite it=488)   union 99% on test/complex map 0 at 335 steps
#     v15  union 90% at 416 steps (metric/steps_to_90) eval/steps_to_90 = 790 on test/complex
# ~2.1x slower. (v10 ran --done-mode union — the argparse default, never passed in pipeline_v10.sh
# — so it solved the easier problem. The slowdown is real; it is just not confound-free.)
# Meanwhile the OBJECTIVE regressed too: metric/comm_duty_cycle 0.155 -> 0.144 -> 0.078 across
# v13/v14/v15, eval/n_syncs 8.84 -> 3.44 -> 2.97, and metric/own_cov_min sat at 0.52 — the union map
# is finished (explore/ep_end 0.9987) but the weakest robot holds half of it.
#
# THREE CAUSES, ALL QUANTIFIED, ALL ADDRESSED HERE.
#
# 1. THE BUDGET STOPPED BINDING. scripts/train_args.py already records the measurement: the v12
#    policy SPENDS 0.0217 (hybrid) / 0.0311 (complex) / 0.0374 (corridor) px per free-px. v13-v15
#    trained at --max-travel-frac 0.06 — 1.6x to 2.8x that. On test/complex 0.06 buys ~1750 hops
#    where v10 finished in 335. And env/explorer.py's own truncation comment says the step cap is
#    only a safety net *because* "a policy that stalls forever burns no distance". So after v13
#    nothing priced hesitation at all. v16 inverts the roles: the STEP CAP is the primary pressure
#    (short, so every hesitant step costs a real fraction of the episode) and --max-travel-frac
#    sits just above measured spend as the per-map FLOOR that keeps the small maps short.
#    (train/difficult spans 3.9x in free area p50->p90, which is why a flat cap alone is wrong.)
#
# 2. gamma 0.99 = a 100-STEP EFFECTIVE HORIZON. v12@384 was ~4 horizons; v15@1856 was ~18. The
#    completion bonus (10.0) seen from t=0 is worth 10*0.99^1856 ~ 0. The policy could not represent
#    FINISHING, so it behaved as a myopic greedy scanner. Short episodes alone do not fix this —
#    at the 768 cap below, gamma=0.99 still discounts the bonus to 0.004:
#          cap    gamma=0.99   gamma=0.998
#          256      0.77         6.0
#          768      0.004        2.15
#         1856      ~0           0.24
#    ONE gamma for all phases: changing it across a warm start invalidates the value head and the
#    loaded vnorm.
#
# 3. THE ACTOR COULD NOT SEE ITS OWN DEADLINE. agent_scalars was [g, staleness]; critic_global had
#    t_frac but the actor had nothing, and under a travel budget t_frac does not even predict the
#    end (a difficult-p50 episode truncates at t_frac~0.23, a p90 one at ~0.90). agent_scalars is
#    now [g, staleness, travel_frac, contact, offer_frac] — see AGENT_SCALAR_DIM in
#    models/actor_critic.py. travel_frac = max(travel/budget, t/T_max) = progress toward whichever
#    criterion binds FIRST.
#
# ALSO CHANGED, each for a measured reason:
#   * --sync-weight 0.25. v13's header records v11 as proof the sync bonus backfires (nSync
#     2.6->2.2, auc 0.605->0.556) — but v11 moved --sync-weight 0->0.25 AND --rdv-weight 0.10->1.0
#     together, and its own header calls w~2 "a chase term". The attribution is confounded. Holding
#     rdv at 0.10 unconfounds it. Sized from v15: sync_give_diag 0.01389/step -> the term is worth
#     ~21% of novel. It is the ONLY reward paying for the exchange, and the exchange is the objective.
#   * --rdv-clamp-pos. v15 measured reward/rdv at -0.20/episode: a standing TAX on the divergence
#     exploration requires. Safe while --rdv-weight < step_penalty_coef*scan_norm_nodes = 0.75.
#   * --rdv-urgency-T 256. Also the staleness OBS normalizer now; it used to be max_episode_steps,
#     which made one step worth 0.0005 at T=2048 and silently rescaled the input across the
#     easy->difficult warm start.
#
# --eval-steps EQUALS THE CAP in every phase. Getting this wrong is how v12 hid its own failure: a
# measurement cap below the training budget reintroduces starvation inside the measurement.
#
# FROM SCRATCH, NOT WARM-STARTED FROM v15. agent_scalars went 2 -> 5; the widening path in
# train/driver.py copies a narrower weight into the LEADING columns, so a v13/v14/v15 checkpoint
# would land its value_field block on the new scalar columns. Phase 1 must start cold.
#
# PHASES ARE GATED, so run them one at a time and check the gate before spending the next block of
# hours. Defaults to all three; override to resume or to stop after a phase:
#     PHASES=1 ./pipeline_v16.sh
#     PHASES="2 3" EASY_OUT=runs/v16_easy_20260803_1200 ./pipeline_v16.sh
#   phase 1 -> 2 : eval/success_rate >= 0.60 on hybrid AND explore/ep_end >= 0.99
#   phase 2 -> 3 : eval/own_coverage_final >= 0.85 AND eval/steps_to_90 below v15's 790
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
PHASES="${PHASES:-1 2 3}"
EASY_OUT=${EASY_OUT:-runs/v16_easy_${TS}}
DIFF_OUT=${DIFF_OUT:-runs/v16_difficult_${TS}}
RDV_OUT=${RDV_OUT:-runs/v16_rdv_${TS}}
has_phase() { [[ " $PHASES " == *" $1 "* ]]; }

COMMON="--n-envs 32 --n-agents 2 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 \
  --done-mode own \
  --gamma 0.998 \
  --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
  --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 \
  --trace-steps 512"

# frac 0.040 is MEASURED, not picked. scripts/probe_reachability.py on the v15 policy (ported into
# the v16 actor layout by scripts/port_ckpt_v16.py, so actor_pre actually loads — dropping it gives
# a random trunk and the sweep returns noise: 0.16 then 0.00 for identical settings):
#     train/easy, cap 256    frac 0.020 -> 0.28   0.025 -> 0.31   0.030 -> 0.38
#                            frac 0.040 -> 0.69   0.060 -> 0.62      (termination rate, own-99%)
# The rate SATURATES at 0.040, so anything above buys longer episodes, not more learning — the same
# shape v13 measured at 0.10 with a weaker policy. 0.030 sits past the knee at 0.38 and would risk
# starving a from-scratch phase 1 of the completion gradient entirely. At 0.040 the episode averages
# 169 steps (vs ~480 under v15's 0.06), so this is still a 2.8x tightening of the real horizon.
if has_phase 1; then
echo "############ PHASE 1: EASY (from scratch, 256-step cap, 2M steps) ############"
python scripts/run_train.py --split train/easy --max-episode-steps 256 --max-travel-frac 0.040 \
  --total-steps 2000000 \
  $COMMON --eval-suite-splits test/hybrid \
  --eval-on-ckpt --eval-split test/hybrid --eval-steps 256 \
  --out ${EASY_OUT}
echo "PHASE1_DONE easy_final=${EASY_OUT}/final.pt"
fi

# Same probe on train/difficult at cap 768: frac 0.030 -> 0.47, 0.040 -> 0.72, 0.050 -> 0.78,
# 0.060 -> 0.75. Saturates at 0.040-0.050 again, so 0.040 is the knee here too. Episodes average
# 321 steps, and the max spend of 15.6k px (~870 hops) is where the 768 cap bites — i.e. exactly on
# the p90 maps it is meant to bound, and nowhere else.
#
# TWO CHANGES vs phase 1, both deliberate and both made HERE because a phase boundary is the only
# place a gate's semantics may move (the split, cap and budget all change here anyway):
#  * --total-steps 4M -> 6M. The extra compute goes into GRADIENT UPDATES (488 -> 732 at the same
#    8192 steps/iter), NOT into --rollout-len. With gamma 0.998 and lambda 0.95, GAE's own credit
#    horizon is 1/(1-0.9481) ≈ 19 steps, so the 256-step rollout already covers 13x it; long-horizon
#    credit travels through V(s), and the rollout boundary bootstraps correctly (nonterm=1). Raising
#    rollout to 768 would triple steps/iter and CUT updates to 163 for the same budget.
#  * --rdv-urgency-mode budget. Legacy "time" ramps the gate on steps-since-last-sync, so it opens
#    merely because the two have been apart — pulling them together in the MIDDLE of an episode, when
#    they should still be splitting. Budget mode ramps on travel_frac (the deadline the actor now
#    observes) starting at half the budget, so the pull appears only near the end. That is the stated
#    objective: explore apart, meet once at the end to complete both maps.
# NOT changed: --revisit-pen stays 0.05. Measured at the end of phase 1 it is -0.009/step against
# novel +0.025 — a healthy ratio — and raising it fights the terminal rendezvous, which REQUIRES
# backtracking. Revisit is per-agent, so walking over the teammate's ground is already free.
if has_phase 2; then
echo "############ PHASE 2: DIFFICULT (768-step cap, 6M steps, warm-start) ############"
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.040 \
  --total-steps 6000000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 \
  $COMMON --eval-suite-splits test/complex \
  --eval-on-ckpt --eval-split test/complex --eval-steps 768 \
  --init-ckpt ${EASY_OUT}/final.pt \
  --out ${DIFF_OUT}
echo "PHASE2_DONE diff_final=${DIFF_OUT}/final.pt"
fi

# own-99% needs a TERMINAL MEETING, and the meeting costs distance on top of exploration. Only once
# exploration is fast is it worth buying headroom (0.040 -> 0.050) and raising the price of the
# exchange (0.25 -> 0.5) — do it earlier and the extra budget is spent wandering, not meeting.
if has_phase 3; then
echo "############ PHASE 3: RENDEZVOUS POLISH (1.5M steps, warm-start) ############"
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.050 \
  --total-steps 1500000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 \
  $COMMON --sync-weight 0.5 --eval-suite-splits test/complex \
  --eval-on-ckpt --eval-split test/complex --eval-steps 768 \
  --init-ckpt ${DIFF_OUT}/final.pt \
  --out ${RDV_OUT}
echo "PHASE3_DONE rdv_final=${RDV_OUT}/final.pt"
fi
echo "PIPELINE_DONE best=${RDV_OUT}/ckpt_best.pt"

# WATCH (scripts/analyze_run.py <run> --reward-budget, and --compare against v15):
#   eval/steps_to_90 + metric/steps_to_90 — THE SPEED OBJECTIVE. v15 = 790 / 416, v10 = 193.
#   metric/comm_duty_cycle (v15 0.078), metric/own_cov_min (v15 0.52) — the exchange objective.
#   reward/completion (v15 56.0/ep) — must rise. If it does not, --completion-bonus now exists.
#   train/kl (v15 0.036) + train/v_loss — gamma 0.998 multiplies value targets ~5x. Sustained
#     KL > 0.05 or a v_loss blow-up means back off to gamma 0.995.
#   reward/revisit — v15 -37.6/ep vs novel +50.3/ep, and analyze_run already prints the >0.7x
#     warning. The streak stays UNCAPPED by an explicit past decision, but --sync-weight now rewards
#     the backtracking that penalty punishes. --revisit-streak-cap 4.0 is the first knob if phase 2
#     stalls.
#   reward/rdv — with the clamp it can no longer go negative. If it grows large the clamp is being
#     farmed and --rdv-weight must stay well under 0.75.
