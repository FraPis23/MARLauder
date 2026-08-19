#!/bin/bash
# v1.8 — FIRST 4-AGENT TRAINING. Same recipe as v16 phase 2, warm started from ITS checkpoint.
#
# WHY THIS IS NOT A NEW RECIPE. The three fixes below are EXACT no-ops at M=2 (measured, see
# below), so v16 phase 2 remains the control and the only deliberate change in this run is
# `--n-agents 4`. v17's `--comm-idle-pen` is NOT here: it was refuted (syncs/ep 5.06→2.25,
# own_cov 0.974→0.889 — pricing contact under done_mode=own buys contact-avoidance, not
# "touch, exchange, leave").
#
# THE THREE FIXES (env/explorer.py, eval/trace.py, viz/inspector.html). All three were latent at
# M=2 because a 2-agent team has exactly ONE teammate, so every "which teammate?" reduction is
# trivially correct. Measured on 60 steps, 16 maps, test/complex, v16 phase-2 policy:
#
#   metric                                   M=2        M=4
#   max |Sigma p - 1| per teammate           9.5e-07    1.8e-06     <- was already correct
#   feat[4]  old vs new                      0.0000     0.9751
#   phi      old vs new                      0.0000     0.8111
#   j_star != nearest teammate               0.0%       78.0%
#
#   1. feat[4] PEAK-NORMALIZED PER TEAMMATE, BEFORE the max over teammates. `_belief_p` is Sigma=1
#      per teammate, so each row has its own scale: freshly lost = sharp peak, lost 200 steps ago =
#      wide shallow plateau. Normalizing AFTER the max divided every teammate by the sharpest one's
#      peak, erasing exactly the teammates that are hardest to find.
#   2. phi RE-AIMED ONTO j_star. The rdv term is w*g*(phi_prev - phi_now); `g` is built entirely
#      from the surplus owed to j_star (the teammate owed the most map), but phi was reduced by
#      min-over-teammates = the NEAREST one. At M=4 those are different robots 78% of the time, so
#      the gate opened because of A3 while the dense term paid for walking toward A1.
#      `geo_pair` (CTDE critic) deliberately STAYS on the min — there "how spread is the team" is
#      the intended reading.
#   3. VIZ: the belief heat is now HUE-LOCKED to the teammate owning the mass (trace serializes
#      `belw` = argmax over teammates; the inspector tints with the same `acol` palette as the
#      agent dots). The merged single-scalar field was unreadable at M>2 — you could not tell
#      whether the mass on a door meant A1 went through it or A3 did.
#   Plus: the console `redun=` is now divided by (M-1). The raw metric saturates at M-1 when the
#   maps are identical, so an M=4 number (~2.3) was not comparable to an M=2 one (~0.85). The
#   LOGGED `metric/redundancy` stays raw so every historical run stays comparable.
#
# COST, MEASURED (4 iterations each, same recipe, from scratch): M=2 54 sps, M=4 39 sps. That is
# 1.4x slower per ENV-step but FASTER per AGENT-step (156 vs 108), because the O(M^2) part (belief
# filter + per-teammate Bellman-Ford) is not what dominates — the GAT encoder, the Warp LiDAR and
# the MAPPO update all scale linearly in M. 6M env-steps ~ 43h.
#
# BUDGET IS DELIBERATELY UNCHANGED AT 0.040 px/free-px PER AGENT. Four agents therefore get 2x the
# team distance two agents had. The alternative (halving it to hold team distance constant) would
# be a budget never measured on any policy, and the knee was measured at 0.040. Keeping the
# per-agent allowance is the conservative reading of "we may put ourselves in a WORSE position,
# never a better one": each robot must still reach 99% of the map on the same allowance it had at
# M=2, while now owing map to three teammates instead of one. The IR2 comparison cells are
# governed separately by eval_comparison.py, which applies IR2's own per-cell max_dist.
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/MARLauder
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
INIT_CKPT=${INIT_CKPT:-runs/v16_difficult_20260803_135950/final.pt}
OUT=${OUT:-runs/v18_m4_${TS}}

if [ ! -f "${INIT_CKPT}" ]; then
  echo "ERRORE: checkpoint di warm start non trovato: ${INIT_CKPT}"; exit 1
fi

# The actor/critic weights contain NO agent-count dimension (the critic pools mean+max over agents
# into a fixed 2*d, see models/actor_critic.py:164), so an M=2 checkpoint loads into an M=4 model
# unchanged. Expect `missing=0 unexpected=0` and NO "DROPPED" line — anything else means a weight
# was silently randomized and the run must be killed.
python scripts/run_train.py --split train/difficult --max-episode-steps 768 --max-travel-frac 0.040 \
  --n-agents 4 \
  --total-steps 6000000 \
  --rdv-urgency-mode budget --rdv-urgency-start 0.5 \
  --n-envs 32 --rollout-len 256 --n-hops 6 --tbptt-steps 8 --minibatches 1 \
  --k-epochs 4 --radar-gamma 0.97 --radar-util-norm 3 --belief-mode pathfront \
  --radar-team-source belief --map-seed 0 \
  --pf-frontier-min-unknown 1 --pf-frontier-min-util 1e-6 \
  --done-mode own \
  --gamma 0.998 \
  --rdv-weight 0.10 --rdv-clamp-pos --rdv-urgency-T 256 \
  --sync-weight 0.25 --sync-recv-ratio 0.5 --sync-min-gap 32 \
  --trace-steps 512 \
  --eval-suite-splits test/complex \
  --eval-on-ckpt --eval-split test/complex --eval-steps 768 \
  --init-ckpt ${INIT_CKPT} \
  --out ${OUT}
echo "V18_DONE final=${OUT}/final.pt best=${OUT}/ckpt_best.pt"

# DA GUARDARE. The M=2 reference is runs/v16_difficult_20260803_135950, but note that several of
# these are NOT directly comparable across agent counts — the ones that are are marked (cmp).
#   metric/own_cov_gap    v16 0.173 (last 50 it)  -> THE number for done_mode=own. Four robots all
#                         reaching 99% is a strictly harder problem than two; if this climbs past
#                         ~0.35 and stays there, the completion bonus is unreachable at M=4 and the
#                         run is the v12 failure mode again (check reward/completion -> 0).
#   eval/own_coverage_final  v16 0.968 (last 6 evals)  -> must not collapse. (cmp)
#   eval/success_rate     v16 0.677  -> expect LOWER at M=4: it is an AND over four robots. (cmp)
#   eval/n_syncs          v16 4.68   -> expect HIGHER: six pairs instead of one, not a regression.
#   redun= in console     now /(M-1), so ~0.85 at M=2 and ~0.75 at M=4 are comparable. (cmp)
#   feat[4] presence      fix 1 should RAISE it at M=4 vs what the old code would have produced —
#                         scripts/pf_obs_diag.py is the tool, run it on ckpt_020 and compare against
#                         the same probe on a v16 M=2 checkpoint.
#   [init-ckpt] line      missing=0 unexpected=0, NO "DROPPED". Kill the run otherwise.
