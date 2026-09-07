#!/usr/bin/env python3
"""WHY an agent finds nothing on a step. Splits idle_frac into its two disjoint regimes. No training.

    python tools/idle_diag.py --ckpt runs/v19_m4_20260811_122528/ckpt_stop.pt \
        --split test/complex --n-agents 4 --label v19-M4
    python tools/idle_diag.py --ckpt runs/v16_difficult_20260803_135950/ckpt_best.pt \
        --split test/complex --n-agents 2 --label v16-M2

v19 closed at metric/idle_frac 0.774 (M=4) against v16's 0.662 (M=2), and the reflex reading of
that number — "the average agent stands still 78% of steps" — is wrong. metric/stall_rate, which
is the real immobility detector (step_disp < nr/2), reads 0.0036 and 0.00043 on the same two runs:
the agents MOVE on 99.6% of steps. What idle_frac actually says is "scanned nothing the TEAM did
not already have", and that collapses two failure modes with opposite fixes:

    TRANSIT    my_new == 0                 walking through space I already mapped myself.
                                           An assignment/routing problem. Nothing to do with
                                           teammates.
    REDUNDANT  my_new > 0, novel == 0      I scanned real ground a teammate already held.
                                           An INFORMATION problem — and out of comm the agent
                                           has no physical way to know, which is why the split
                                           is also crossed with contact below.

Two further cautions this script is built around. First, novel is measured against the PRIVILEGED
team union, which holds more of the map at any t the more agents there are, so idle_frac is not
behaviourally comparable across M even though it is a plain mean (coverage_per_dist is). Compare
the SHARES here, not the totals. Second, for TRANSIT steps the question "was there anywhere to
go" is answered by the BF distance from the agent's current node to its own nearest frontier —
reported at the bottom, and the difference between "commuting to a far frontier" (a structural
cost of larger M) and "circling next to frontiers that exist" (a policy failure).

Runs N envs in parallel (never P=1: a batch-dim bug in the env is invisible at P=1 — that is how
a gather-on-dim-1 killed v15 run 1 at it=0). Contributions from an env stop at its FIRST done, so
the auto-reset never mixes two episodes into one bucket.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

import numpy as np  # noqa: E402

from env.explorer import EnvCfg, Explorer  # noqa: E402
from env.frontier import compute_frontier  # noqa: E402
from env.maps import load_split  # noqa: E402
from eval.ckpt_loader import load_model_from_ckpt  # noqa: E402
# The v2 protocol loader lives with the generator that froze it; importing rather than copying it
# keeps ONE definition of "IR2's budget on map k" and of the CSV/starts pairing rules.
from scripts.eval_comparison import _load_ir2_reference  # noqa: E402

# info["idle_bucket"] codes (env/explorer.py::_compute_metrics).
PRODUCTIVE, REDUNDANT, TRANSIT = 0, 1, 2
BUCKETS = ("productive", "redundant", "transit")


class Acc:
    """Agent-step counters. Every tally is masked by `alive` so a finished episode contributes
    nothing after its done step."""

    def __init__(self, M: int, n_q: int, dev: str) -> None:
        z = lambda *s: torch.zeros(s, dtype=torch.float64, device=dev)  # noqa: E731
        self.total = z()
        self.bucket = z(3)                 # productive / redundant / transit
        self.by_q = z(n_q, 3)              # x budget quartile
        self.by_contact = z(2, 3)          # 0 = out of comm, 1 = in comm
        self.by_agent = z(M, 3)            # x agent slot
        self.idle_stalled = z()
        self.idle_revisit = z()
        self.stalled = z()
        # TRANSIT steps only: distance to own nearest frontier (finite) / no frontier reachable.
        self.transit_d: list[torch.Tensor] = []
        self.transit_nofront = z()
        self.transit_n = z()
        # Of the no-REACHABLE-frontier transit steps, how many have no frontier AT ALL in the
        # agent's own map (genuinely finished locally) versus frontiers that exist but sit in a
        # component of the known-free graph disconnected from where the agent stands — the latter
        # is what map fusion produces: a teammate's far region arrives across an unexplored gap.
        self.stranded_none = z()          # no frontier at all in own map
        self.stranded_discon = z()        # frontiers exist, none reachable
        self.stranded_owncov = z()        # Σ own_cov over those steps (are they actually done?)
        # Is transit even GOING anywhere? progress = -Δ(distance to own nearest frontier) / step
        # displacement. 1 = the step went straight down the path to the frontier, 0 = sideways,
        # <0 = walked away from it. "Moving" and "closing on something worth reaching" are not the
        # same claim, and only the second one makes transit a cost of exploring rather than waste.
        self.prog_n = z()
        self.prog_toward = z()            # progress > +0.5
        self.prog_flat = z()              # |progress| <= 0.5
        self.prog_away = z()              # progress < -0.5
        self.prog_sum = z()
        # Progress split by whether the frontier is inside the ego window at all. The actor's node
        # features cover a (2*n_hops+3)^2 lattice window; beyond it the only signal is the coarse
        # radar channel. If progress collapses outside the window, the failure is observational
        # (the agent cannot see where to go), not a matter of which frontier it picked.
        self.prog_by_d_n = z(3)
        self.prog_by_d_sum = z(3)
        # How often the transitive closure actually adds an edge. A relay only pays when the team
        # forms a CHAIN (A-B linked, B-C linked, A-C not); if pairs are either together or far
        # apart the closure is the identity and the fix is correctness, not performance.
        self.duty_direct = z()
        self.duty_group = z()
        self.connected = z()
        self.steps = z()
        # Heading persistence on transit steps. Two different failures produce the same near-zero
        # net progress: a policy that cannot SEE its target (the median frontier sits ~2x outside
        # the ego window and these checkpoints run use_gru=False, so there is no memory to hold a
        # commitment either) walks a consistent but wrong direction; a policy that keeps switching
        # target oscillates. cos(turn) between consecutive displacement vectors separates them —
        # a committed walk stays near +1 whatever its target, oscillation drives it toward 0 or below.
        self.turn_n = z()
        self.turn_sum = z()
        self.turn_rev = z()          # cos < 0, i.e. the step reversed on the previous one
        self.turn_back = z()         # cos < -0.7, near-perfect backtrack
        # TRANSIT RUNS: a maximal stretch of consecutive transit steps, i.e. one uninterrupted
        # commute between two pieces of real work. Length says how much travel a unit of work
        # costs; straightness = |net displacement| / path walked says whether that travel was a
        # route or a wander. Neither is confounded by WHICH frontier the agent chose, unlike
        # progress-toward-the-nearest-frontier above, and both are directly comparable across M.
        self.run_steps: list[torch.Tensor] = []
        self.run_straight: list[torch.Tensor] = []
        # ---- CONTRIBUTION RANK. contrib_imbalance is Sigma_i |share_i - 1/M| over union-new cells,
        # and it is a per-EPISODE quantity, so a per-step bucket mean cannot explain it: `by_agent`
        # above is keyed by SLOT, and a slot is not a role — the agent that carries the episode is
        # a different index on every map, so averaging by slot washes the effect out. Ranking the
        # agents WITHIN each episode by their own share and then averaging by RANK keeps it.
        # rank 0 = the episode's top contributor, rank M-1 = its lowest.
        self.rank_n = z()          # scalar: episodes closed (rank bins share it)
        self.rank_share = z(M)
        self.rank_bucket = z(M, 3)
        self.rank_steps = z(M)
        self.imb_sum = z()
        self.ep_n = z()
        # Episode headline, so this script can be gated against the comparison CSVs it is meant
        # to explain: if max_dist and steps do not land on the CSV, it is running other episodes.
        self.ep_dist = z()
        self.ep_steps = z()
        self.ep_bound = z()
        # Per-rank RADIO and DELIVERY, to tell a relay apart from a straggler: a relay should sit
        # in contact MORE than the agents it serves, and both roles must still end with the map.
        self.rank_contact = z(M)
        self.rank_owncov = z(M)
        # Paid syncs per rank. `contact` says two radios could hear each other; `sync_paid` is the
        # rising edge that actually DELIVERED map. A courier shows up here, a passenger does not.
        self.rank_sync = z(M)
        # WAS THE LONG WALK NECESSARY, OR CHOSEN? Two explanations for the M=4 relocation tail have
        # opposite fixes, so they must be told apart before anything is trained:
        #   FORCED  d0 (distance to the agent's own nearest frontier at the moment the run STARTS)
        #           is already large -> there was nothing near to do; the fix is spatial allocation
        #           so agents stop exhausting each other's neighbourhoods.
        #   CHOSEN  d0 is small yet the agent still walked for many steps -> it passed up nearby
        #           work for a distant target; the fix is the value field's horizon (vf_gamma),
        #           which has a precedent: at 0.92 a frontier 45 hops out was worth 0.4%/node.
        # path/d0 separates them further: ~1 means it went straight to the nearest thing, >>1 means
        # it went somewhere else entirely.
        self.run_d0: list[torch.Tensor] = []
        self.run_path: list[torch.Tensor] = []
        self.run_dend: list[torch.Tensor] = []
        # The confound that would make a long walk RATIONAL: the nearby frontier is a crack and
        # the distant one is a room. graph_lattice's utility is exactly that judgement
        # (fr_frac x revealable volume), so compare the utility of the NEAREST frontier at run
        # start against the BEST one, and how far that best one was. If the agent is simply
        # walking to the best-utility target, the lever is how utility trades off against
        # distance (vf_gamma), not the policy.
        self.run_u_near: list[torch.Tensor] = []
        self.run_u_best: list[torch.Tensor] = []
        self.run_d_best: list[torch.Tensor] = []


@torch.no_grad()
def run(model, env: Explorer, cfg_steps: int, n_q: int, acc: Acc,
        autocast: bool = True) -> int:
    """One batch of episodes (all N envs already loaded with their maps). Returns steps run."""
    dev = env.dev
    N, M, B = env.N, env.M, env.N * env.M
    h_act, h_crit = model.init_hidden(N, str(dev))
    obs = env.obs
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    prev = None            # (d_start, disp, is_transit) of the previous step, for the progress test
    prev_vec = None        # (step_vec, live) of the previous step, for the heading test
    # Open transit run per (env, agent): step count, path length walked, start position.
    run_n = torch.zeros((N, M), dtype=torch.float32, device=dev)
    run_path = torch.zeros((N, M), dtype=torch.float32, device=dev)
    run_p0 = env.pos.clone()
    run_d0 = torch.full((N, M), float("inf"), device=dev)
    run_un = torch.zeros((N, M), device=dev)
    run_ub = torch.zeros((N, M), device=dev)
    run_db = torch.full((N, M), float("inf"), device=dev)
    # Per-episode, per-agent bucket tally, closed at the agent's own done step (see acc.rank_*).
    ep_bucket = torch.zeros((N, M, 3), dtype=torch.float64, device=dev)
    ep_contact = torch.zeros((N, M), dtype=torch.float64, device=dev)
    ep_sync = torch.zeros((N, M), dtype=torch.float64, device=dev)
    # Half-width of the actor's ego window in pixels: the node features stop here, and anything
    # further is only visible through the coarse radar channel.
    win_px = float(env.cfg.n_hops * env.graph.NR)
    t = 0
    for t in range(cfg_steps):
        # eval_comparison runs the actor in FULL precision. Under --ir2-dir this script has to
        # reproduce those exact episodes, and bf16 autocast changes the argmax on near-ties often
        # enough to move the trajectory (measured: 73.2 vs 78.0 steps/episode on the same 8 maps),
        # which would diagnose a policy the CSVs never ran.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=autocast and str(dev).startswith("cuda")):
            out = model.act(obs, h_act, h_crit, deterministic=True)
        h_act, h_crit = out["hidden_actor"], out["hidden_critic"]

        # BF distance from curr to every node, and the agent's OWN frontier nodes, BEFORE the step
        # advances them — this is the field the policy just acted on. _dist_curr_prev is the
        # persistent [N, M, N_max] copy _refresh_obs writes every step (explorer.py: bf_from_curr).
        d_curr = env._dist_curr_prev.view(B, -1)                                    # [B, N_max]
        occ_b = env.world.occupancy_torch.reshape(B, env.H, env.W)
        # A node counts as a frontier node exactly as the graph does it: graph_lattice's
        # util_boundary is a box count of frontier PIXELS within radius NR of the node centre
        # (fr_frac), not the centre pixel itself. Point-sampling the frontier map at node centres
        # undercounts massively — a max-pool of the same radius reproduces `fr_frac > 0`.
        r = max(2, int(env.graph.NR))
        fr_px = compute_frontier(occ_b).float().unsqueeze(1)                        # [B, 1, H, W]
        fr_box = torch.nn.functional.max_pool2d(fr_px, 2 * r + 1, stride=1, padding=r).squeeze(1)
        fr_node = fr_box.view(B, -1)[:, env._node_flat_idx] > 0                     # [B, N_max]
        d_masked = torch.where(fr_node, d_curr, torch.full_like(d_curr, float("inf")))
        d_front = d_masked.amin(-1).view(N, M)                                      # [N, M] px
        n_front = fr_node.sum(-1).view(N, M)                                        # [N, M] nodes
        # Utility of the NEAREST reachable frontier, and of the BEST one plus how far it is.
        util = env._utility_global.view(B, -1)                                      # [B, N_max]
        reach_fr = fr_node & torch.isfinite(d_curr)
        u_masked = torch.where(reach_fr, util, torch.full_like(util, -1.0))
        u_best, i_best = u_masked.max(-1)
        u_near = util.gather(1, d_masked.argmin(-1, keepdim=True)).squeeze(1)
        d_best = d_curr.gather(1, i_best.unsqueeze(1)).squeeze(1)
        u_near, u_best, d_best = u_near.view(N, M), u_best.view(N, M), d_best.view(N, M)

        # The step that CAUSED the change from the previous d_front to this one has already been
        # taken, so score it now: its displacement and its bucket were stashed last iteration.
        if prev is not None:
            d_prev, disp_prev, tr_prev = prev
            ok = tr_prev & torch.isfinite(d_front) & torch.isfinite(d_prev) & (disp_prev > 1e-3)
            if bool(ok.any()):
                progress = (-(d_front - d_prev) / disp_prev.clamp(min=1e-3))[ok]
                acc.prog_n += ok.sum()
                acc.prog_toward += (progress > 0.5).sum()
                acc.prog_flat += ((progress >= -0.5) & (progress <= 0.5)).sum()
                acc.prog_away += (progress < -0.5).sum()
                acc.prog_sum += progress.double().sum()
                dd = d_prev[ok]
                for bi, (lo, hi) in enumerate(((0.0, win_px), (win_px, 2 * win_px),
                                               (2 * win_px, float("inf")))):
                    sel = (dd >= lo) & (dd < hi)
                    acc.prog_by_d_n[bi] += sel.sum()
                    acc.prog_by_d_sum[bi] += progress[sel].double().sum()

        tp_before = env.travel_px.clone()
        pos_before = env.pos.clone()                                                # [N, M, 2]
        obs, _r, done, info = env.step(out["action"])
        disp = info["travel_px"] - tp_before                                        # [N, M] px
        step_vec = env.pos - pos_before                                             # [N, M, 2]

        bucket = info["idle_bucket"].long()                                         # [N, M]
        flags = info["idle_flags"].long()                                           # [N, M]
        alive_m = alive.view(N, 1).expand(N, M)                                     # [N, M]
        w = alive_m.to(torch.float64)
        acc.total += w.sum()

        onehot = torch.zeros((N, M, 3), dtype=torch.float64, device=dev)
        onehot.scatter_(2, bucket.unsqueeze(-1), 1.0)
        onehot = onehot * w.unsqueeze(-1)
        acc.bucket += onehot.sum((0, 1))
        acc.by_agent += onehot.sum(0)
        ep_bucket += onehot

        stalled = (flags & 1) > 0
        revisit = (flags & 2) > 0
        contact = ((flags & 4) > 0).long()                                          # [N, M]
        ep_contact += (contact == 1).to(torch.float64) * w
        if "sync_paid" in info:
            ep_sync += info["sync_paid"].to(torch.float64) * w
        for c in (0, 1):
            acc.by_contact[c] += (onehot * (contact == c).unsqueeze(-1)).sum((0, 1))

        # Budget quartile — travel_frac is what actually truncates the episode. Falls back to the
        # step fraction when no travel budget is configured (then travel_frac is identically 0).
        # budget_px() is the SINGLE source of truth the env itself truncates on and feeds to the
        # actor as travel_frac; re-deriving it here from cfg would silently ignore an explicit
        # per-map budget (protocol v2 sets env.travel_budget_px, which no cfg field records).
        bud_v = env.budget_px()
        if bud_v is not None:
            frac = (info["travel_px"] / bud_v.unsqueeze(1).clamp(min=1.0)).clamp(0.0, 1.0)
        else:
            frac = torch.full((N, M), t / max(1, cfg_steps), device=dev)
        q = (frac * n_q).long().clamp(0, n_q - 1)                                   # [N, M]
        for qi in range(n_q):
            acc.by_q[qi] += (onehot * (q == qi).unsqueeze(-1)).sum((0, 1))

        met = info["metrics"]
        acc.steps += 1
        acc.duty_direct += float(met["comm_duty_cycle"])
        acc.duty_group += float(met["comm_group_duty"])
        acc.connected += float(met["comm_connected"])

        idle = bucket > 0
        acc.stalled += (stalled & alive_m).sum()
        acc.idle_stalled += (idle & stalled & alive_m).sum()
        acc.idle_revisit += (idle & revisit & alive_m).sum()

        is_transit = (bucket == TRANSIT) & alive_m
        acc.transit_n += is_transit.sum()
        finite = torch.isfinite(d_front) & is_transit
        stranded = is_transit & ~torch.isfinite(d_front)
        acc.transit_nofront += stranded.sum()
        acc.stranded_none += (stranded & (n_front == 0)).sum()
        acc.stranded_discon += (stranded & (n_front > 0)).sum()
        acc.stranded_owncov += (info["own_cov"] * stranded).sum()
        if bool(finite.any()):
            acc.transit_d.append(d_front[finite].detach().float().cpu())

        # Heading persistence, scored on consecutive transit steps of the same live episode.
        live = is_transit & alive.view(N, 1).expand(N, M) & ~done.view(N, 1)
        if prev_vec is not None:
            pv, plive = prev_vec
            ok = live & plive & (disp > 1e-3) & (pv.norm(dim=-1) > 1e-3)
            if bool(ok.any()):
                a_ = step_vec / step_vec.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                b_ = pv / pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                cos = (a_ * b_).sum(-1)[ok]
                acc.turn_n += ok.sum()
                acc.turn_sum += cos.double().sum()
                acc.turn_rev += (cos < 0).sum()
                acc.turn_back += (cos < -0.7).sum()
        prev_vec = (step_vec, live)

        # Transit runs. A run extends while `live`; it CLOSES on the first non-transit step (real
        # work done) and is DISCARDED on done, because a run cut by the episode ending has no
        # terminating piece of work and would bias the length downward.
        closing = (~live) & (run_n > 0) & alive_m & ~done.view(N, 1)
        if bool(closing.any()):
            euclid = (env.pos - run_p0).norm(dim=-1)                                # [N, M] px
            straight = (euclid / run_path.clamp(min=1e-3))[closing]
            acc.run_steps.append(run_n[closing].detach().cpu())
            acc.run_straight.append(straight.clamp(0, 1).detach().cpu())
            # d_front here is measured from the position the agent occupies at the START of this
            # step, i.e. exactly where the run ended.
            acc.run_d0.append(run_d0[closing].detach().float().cpu())
            acc.run_path.append(run_path[closing].detach().float().cpu())
            acc.run_dend.append(d_front[closing].detach().float().cpu())
            acc.run_u_near.append(run_un[closing].detach().float().cpu())
            acc.run_u_best.append(run_ub[closing].detach().float().cpu())
            acc.run_d_best.append(run_db[closing].detach().float().cpu())
        starting = live & (run_n == 0)
        run_p0 = torch.where(starting.unsqueeze(-1), pos_before, run_p0)
        run_d0 = torch.where(starting, d_front, run_d0)
        run_un = torch.where(starting, u_near, run_un)
        run_ub = torch.where(starting, u_best, run_ub)
        run_db = torch.where(starting, d_best, run_db)
        run_n = torch.where(live, run_n + 1, torch.zeros_like(run_n))
        run_path = torch.where(live, run_path + disp.float(), torch.zeros_like(run_path))

        # Stash for the next iteration's progress test; a reset env must not be scored across the
        # episode boundary, so carry `alive` into the transit mask itself.
        prev = (d_front, disp, live)

        # ---- close the finished episodes and bin their agents by contribution rank.
        # info is pre-auto-reset (explorer.py), so novel_cells_ep here is the episode's final
        # attribution, the same tensor eval_comparison turns into contrib_imbalance.
        newly = alive & done
        if bool(newly.any()):
            nov = info["novel_cells_ep"].to(torch.float64)                          # [N, M]
            share = nov / nov.sum(-1, keepdim=True).clamp(min=1.0)                  # [N, M]
            idx = share.argsort(dim=-1, descending=True)                            # rank -> slot
            sel = newly.nonzero(as_tuple=True)[0]
            sh_r = share[sel].gather(1, idx[sel])                                   # [n, M] by rank
            bk_r = ep_bucket[sel].gather(1, idx[sel].unsqueeze(-1).expand(-1, -1, 3))
            ct_r = ep_contact[sel].gather(1, idx[sel])
            oc_r = info["own_cov"].to(torch.float64)[sel].gather(1, idx[sel])
            acc.rank_contact += ct_r.sum(0)
            acc.rank_owncov += oc_r.sum(0)
            acc.rank_sync += ep_sync[sel].gather(1, idx[sel]).sum(0)
            acc.rank_n += float(sel.numel())
            acc.rank_share += sh_r.sum(0)
            acc.rank_bucket += bk_r.sum(0)
            acc.rank_steps += bk_r.sum(-1).sum(0)
            tp = info["travel_px"].to(torch.float64)[sel]                           # [n, M]
            acc.ep_dist += tp.amax(-1).sum()
            acc.ep_steps += bk_r.sum(-1).sum()
            bud_e = env.budget_px()
            if bud_e is not None:
                acc.ep_bound += (tp.amax(-1) >= bud_e.to(torch.float64)[sel] - 22.63).sum()
            if M > 1:
                acc.imb_sum += (share[sel] - 1.0 / M).abs().sum(-1).sum()
                acc.ep_n += float(sel.numel())
        ep_bucket = torch.where(done.view(N, 1, 1), torch.zeros_like(ep_bucket), ep_bucket)
        ep_contact = torch.where(done.view(N, 1), torch.zeros_like(ep_contact), ep_contact)
        ep_sync = torch.where(done.view(N, 1), torch.zeros_like(ep_sync), ep_sync)

        alive = alive & ~done
        if not bool(alive.any()):
            break
    return t + 1


def pct(x: torch.Tensor, tot: torch.Tensor) -> list[str]:
    t = float(tot)
    return [f"{float(v) / t:6.3f}" if t > 0 else "     -" for v in x]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--n-agents", type=int, required=True,
                    help="MUST match the checkpoint's training M (v19=4, v16=2)")
    ap.add_argument("--n-envs", type=int, default=8, help="parallel episodes; never 1")
    ap.add_argument("--n-batches", type=int, default=4, help="episode batches of --n-envs maps")
    ap.add_argument("--steps", type=int, default=0,
                    help="episode step cap; 0 = the checkpoint's own max_episode_steps")
    ap.add_argument("--quartiles", type=int, default=4)
    ap.add_argument("--label", default="run")
    ap.add_argument("--comm-relay", dest="comm_relay", action="store_true", default=None,
                    help="force multi-hop relay ON (a pre-v20 ckpt carries comm_relay=False)")
    ap.add_argument("--no-comm-relay", dest="comm_relay", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ir2-dir", type=Path, default=None,
                    help="PROTOCOL v2 conditions: run on the 100 comparison maps of --split, "
                         "pinned to IR2's start positions and to IR2's OWN per-map distance as the "
                         "budget. Use it when the question is about numbers produced by "
                         "eval_comparison --ir2-dir; the budget drives travel_frac, travel_frac "
                         "drives the rendezvous urgency, and urgency is what pulls the agents "
                         "together — diagnosing under a different budget diagnoses a different "
                         "policy. --split must then be one of hybrid/corridor/complex.")
    ap.add_argument("--noise-seed", type=int, default=777,
                    help="radio-shadowing stream; 777 is what the comparison CSVs were made with")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    assert args.n_envs > 1, "run with N>1: a batch-dim bug in the env is invisible at P=1"

    model, env_peek = load_model_from_ckpt(args.ckpt, args.device, n_agents=args.n_agents)
    model.eval()
    split = load_split(args.split, device=args.device)
    # from_ckpt_dict restores the CHECKPOINT's env cfg — code changes do NOT apply to it unless
    # overridden here, which is exactly the trap that made an old ckpt silently pin old semantics.
    overrides = dict(n_envs=args.n_envs, n_agents=args.n_agents, map_seed=args.seed)
    if args.comm_relay is not None:
        overrides["comm_relay"] = bool(args.comm_relay)
    v2_budgets = v2_starts = v2_maps = None
    if args.ir2_dir is not None:
        cmp_split = args.split.split("/")[-1]
        idx_f = _REPO / "eval" / "comparison" / f"map_indices_{cmp_split}.json"
        if not idx_f.is_file():
            sys.exit(f"[idle_diag] --ir2-dir needs {idx_f}; --split must be hybrid/corridor/complex")
        v2_maps = [int(e["pack_idx"]) for e in json.loads(idx_f.read_text())["entries"]]
        v2_budgets, v2_starts = _load_ir2_reference(args.ir2_dir, cmp_split, args.n_agents,
                                                    len(v2_maps))
        # Same trap eval_comparison documents: from_ckpt_dict keeps every valid field found in the
        # checkpoint, so the TRAINING budget (v20: max_travel_frac=0.03) comes back and wins over
        # the explicit per-map budget at the truncation site. Must be zeroed.
        overrides["max_travel_frac"] = 0.0
        overrides["max_travel_px"] = 0.0
        # max_episode_steps is NOT only a cap: explorer.py normalises the episode-time observation
        # by it (T_max), so leaving the checkpoint's 768 here while eval_comparison ran the cell at
        # its own safety-net cap feeds the actor a different clock and produces different
        # trajectories. Measured on hybrid_M4: 72.1 steps / 1408 px against the CSV's 74.5 / 1456.
        # Formula copied from eval_comparison (max over the WHOLE cell, not the chunk).
        overrides["max_episode_steps"] = int(math.ceil(max(v2_budgets) / 16.0)) * 2
    cfg = EnvCfg.from_ckpt_dict(dict(env_peek or {}), **overrides)
    steps = args.steps or int(cfg.max_episode_steps)
    env = Explorer(split, cfg, seed=args.seed)
    env_cfg, nr = cfg, env.graph.NR

    print(f"\n=== idle_diag  label={args.label}  M={args.n_agents}  N={args.n_envs} envs  "
          f"split={args.split}  steps<={steps}  batches={args.n_batches}")
    print(f"    ckpt={args.ckpt}")
    print(f"    max_travel_frac={cfg.max_travel_frac}  comm_model={cfg.comm_model}  "
          f"comm_relay={getattr(cfg, 'comm_relay', 'n/a')}  done_mode={cfg.done_mode}")

    acc = Acc(args.n_agents, args.quartiles, args.device)
    n_maps = int(getattr(split, "n", 0)) or args.n_envs

    budgets, starts, map_idxs = v2_budgets, v2_starts, v2_maps
    if args.ir2_dir is not None:
        n_take = args.n_envs * args.n_batches
        if n_take > len(map_idxs):
            sys.exit(f"[idle_diag] asked for {n_take} episodes but the comparison cell has "
                     f"{len(map_idxs)} maps — lower --n-batches")
        print(f"    PROTOCOL v2 — {n_take}/{len(map_idxs)} comparison maps, IR2 starts, per-map "
              f"budget mean {sum(budgets[:n_take]) / n_take:.0f} px "
              f"[{min(budgets[:n_take]):.0f}, {max(budgets[:n_take]):.0f}]")

    off_all = []
    for b in range(args.n_batches):
        if args.ir2_dir is not None:
            # Per CHUNK, exactly where eval_comparison reseeds it: the shadowing stream is drawn
            # per env, so the realisation a given map meets depends on the chunk size and on where
            # the stream stood. Same reseed point + same --n-envs as --batch = the same channel.
            env.reseed_channel_noise(args.noise_seed)
        for i in range(args.n_envs):
            if map_idxs is None:
                env.reload_map(env_idx=i, map_idx=(b * args.n_envs + i) % n_maps)
                continue
            k = b * args.n_envs + i
            # Re-key the map RNG per MAP exactly as eval_comparison does, so a map always gets the
            # same formation whatever --n-envs is and this run is comparable to the CSVs.
            env.rng = np.random.default_rng(args.seed + map_idxs[k])
            env.reload_map(env_idx=i, map_idx=map_idxs[k], start_override=starts[k])
            off_all.append(env.last_start_offset_px.clone())
        if budgets is not None:
            k0 = b * args.n_envs
            # Set AFTER the reloads: reload_map does a full reset.
            env.travel_budget_px = torch.tensor(
                [max(1.0, float(x)) for x in budgets[k0:k0 + args.n_envs]],
                dtype=torch.float32, device=args.device)
        ran = run(model, env, steps, args.quartiles, acc,
                  autocast=(args.ir2_dir is None))
        print(f"    batch {b}: {ran} steps, cumulative agent-steps {int(acc.total)}")
    if off_all:
        o = torch.stack(off_all)
        print(f"    start parity: offset from IR2 positions after lattice snap — "
              f"mean {float(o.mean()):.2f} px, worst agent {float(o.max()):.2f} px")

    tot = acc.total
    print(f"\n-- overall  ({int(tot)} agent-steps)")
    for name, v in zip(BUCKETS, acc.bucket):
        print(f"   {name:<11s} {float(v) / float(tot):6.3f}")
    idle = float(acc.bucket[REDUNDANT] + acc.bucket[TRANSIT]) / float(tot)
    print(f"   {'idle_frac':<11s} {idle:6.3f}   (= redundant + transit; compare to the run's "
          f"metric/idle_frac)")
    print(f"   stall_rate  {float(acc.stalled) / float(tot):6.3f}   "
          f"idle&stalled {float(acc.idle_stalled) / float(tot):6.3f}   "
          f"idle&revisit {float(acc.idle_revisit) / float(tot):6.3f}")

    ns = float(acc.steps) or 1.0
    dd, dg = float(acc.duty_direct) / ns, float(acc.duty_group) / ns
    print(f"\n-- connectivity   direct duty {dd:6.3f}   group duty {dg:6.3f}   "
          f"delta {dg - dd:+6.3f}   whole-team-one-flock {float(acc.connected) / ns:6.3f}")
    print(f"   the delta is how often the relay has anything to relay: 0 means the team is never "
          f"in a chain,\n   so the fix is correctness rather than performance.")

    print(f"\n-- by budget quartile (share WITHIN the quartile)")
    print(f"   {'q':<8s} {'productive':>10s} {'redundant':>10s} {'transit':>10s}  {'n':>9s}")
    for qi in range(args.quartiles):
        row = acc.by_q[qi]
        n = row.sum()
        lo, hi = qi * 100 // args.quartiles, (qi + 1) * 100 // args.quartiles
        print(f"   {f'{lo}-{hi}%':<8s} " + " ".join(f"{v:>10s}" for v in pct(row, n))
              + f"  {int(n):>9d}")

    print(f"\n-- by contact (share WITHIN the group)")
    for c, name in ((1, "in-comm"), (0, "out-of-comm")):
        row = acc.by_contact[c]
        n = row.sum()
        print(f"   {name:<12s} " + " ".join(f"{v:>10s}" for v in pct(row, n)) + f"  {int(n):>9d}")
    n_out = acc.by_contact[0].sum()
    print(f"   redundant-while-out-of-comm = {float(acc.by_contact[0][REDUNDANT]) / float(tot):.3f} "
          f"of ALL agent-steps  <- the share the multi-hop relay can address")

    if args.n_agents > 1 and float(acc.rank_n) > 0:
        rn = float(acc.rank_n)
        print(f"\n-- by CONTRIBUTION RANK within each episode  ({int(rn)} episodes closed)")
        print(f"   episode headline (GATE vs the comparison CSV for the same maps): "
              f"max_dist {float(acc.ep_dist) / rn:7.1f}   steps {float(acc.total) / rn / args.n_agents:6.2f}"
              f"   budget-bound {int(acc.ep_bound)}/{int(rn)}")
        # Cross-check: the rank bins must tile the same agent-steps acc.total counted. A gap means
        # episodes were harvested with a stale per-episode tally, i.e. the rank shares are wrong.
        _gap = float(acc.total) - float(acc.ep_steps)
        if abs(_gap) > 0.5:
            print(f"   !! rank bins hold {float(acc.ep_steps):.0f} agent-steps but the run counted "
                  f"{float(acc.total):.0f}  (gap {_gap:+.0f}, {_gap / max(1.0, float(acc.total)):+.1%})")
        print(f"   contrib_imbalance = {float(acc.imb_sum) / max(1.0, float(acc.ep_n)):6.3f}"
              f"   (Sigma_i |share_i - 1/M|; even split = 0)")
        print(f"   {'rank':<8s} {'share':>7s} {'productive':>10s} {'redundant':>10s} "
              f"{'transit':>10s}  {'steps/ep':>9s} {'in-comm':>9s} {'own_cov':>8s} {'syncs/ep':>9s}")
        for r in range(args.n_agents):
            row = acc.rank_bucket[r]
            print(f"   {('top' if r == 0 else 'low' if r == args.n_agents - 1 else f'#{r}'):<8s} "
                  f"{float(acc.rank_share[r]) / rn:>7.3f} "
                  + " ".join(f"{v:>10s}" for v in pct(row, row.sum()))
                  + f"  {float(acc.rank_steps[r]) / rn:>9.1f}"
                  + f" {float(acc.rank_contact[r]) / max(1.0, float(acc.rank_steps[r])):>9.3f}"
                  + f" {float(acc.rank_owncov[r]) / rn:>8.4f}"
                  + f" {float(acc.rank_sync[r]) / rn:>9.2f}")
        print(f"   THE QUESTION: for the LOW rank, is the idle REDUNDANT (it scanned real ground a")
        print(f"   teammate already held -> information/symmetry problem, comm_idle_pen) or TRANSIT")
        print(f"   (it walked over ground it had already mapped itself -> assignment problem,")
        print(f"   frontier-level diversity)? The two fixes are mutually exclusive.")

    print(f"\n-- by agent slot (share WITHIN the agent; flat = the idle is shared, not carried)")
    for a in range(args.n_agents):
        row = acc.by_agent[a]
        print(f"   a{a:<11d} " + " ".join(f"{v:>10s}" for v in pct(row, row.sum())))

    print(f"\n-- TRANSIT steps: BF path length to the agent's OWN nearest frontier (PIXELS)")
    n_tr = float(acc.transit_n)
    if acc.transit_d and n_tr > 0:
        d = torch.cat(acc.transit_d)
        qs = torch.tensor([0.10, 0.50, 0.90, 0.99])
        p = torch.quantile(d, qs)
        nofr = float(acc.transit_nofront) / n_tr
        print(f"   reachable frontier on {1.0 - nofr:5.3f} of transit steps")
        print(f"   p10 {float(p[0]):7.1f}   p50 {float(p[1]):7.1f}   "
              f"p90 {float(p[2]):7.1f}   p99 {float(p[3]):7.1f}   mean {float(d.mean()):7.1f}")
        print(f"   NO reachable frontier on {nofr:5.3f} of transit steps, of which:")
        nn = float(acc.transit_nofront) or 1.0
        print(f"      no frontier AT ALL in own map     {float(acc.stranded_none) / nn:5.3f}"
              f"   <- locally finished")
        print(f"      frontiers exist but DISCONNECTED  {float(acc.stranded_discon) / nn:5.3f}"
              f"   <- fused-in region across an unexplored gap")
        print(f"      mean own_cov on those steps       {float(acc.stranded_owncov) / nn:5.3f}"
              f"   <- if far below 1.0 the agent is stranded, not done")
    else:
        print("   (no transit steps)")

    print(f"\n-- is TRANSIT going anywhere? progress = -d(dist to own nearest frontier)/displacement")
    pn = float(acc.prog_n)
    if pn > 0:
        print(f"   toward  (>+0.5)  {float(acc.prog_toward) / pn:6.3f}   "
              f"<- the step went down the path to the frontier")
        print(f"   sideways(|.|<=.5){float(acc.prog_flat) / pn:6.3f}   "
              f"<- moved, closed no distance")
        print(f"   away    (<-0.5)  {float(acc.prog_away) / pn:6.3f}")
        print(f"   mean progress    {float(acc.prog_sum) / pn:6.3f}   ({int(pn)} scored transit steps)")
        print(f"   NOTE the nearest frontier can change identity between steps, which shows up as "
              f"noise in both tails; the MEAN is the number to read.")
        wp = float(env_cfg.n_hops * nr)
        print(f"\n   mean progress by distance to that frontier (ego window half-width = {wp:.0f} px)")
        names = (f"inside window (<{wp:.0f})", f"1-2x window ({wp:.0f}-{2*wp:.0f})",
                 f"beyond 2x   (>{2*wp:.0f})")
        for bi, nm in enumerate(names):
            n_b = float(acc.prog_by_d_n[bi])
            val = f"{float(acc.prog_by_d_sum[bi]) / n_b:6.3f}" if n_b > 0 else "     -"
            print(f"      {nm:<26s} {val}   ({int(n_b)} steps)")
    else:
        print("   (nothing scorable)")

    tn = float(acc.turn_n)
    if tn > 0:
        print(f"\n-- does TRANSIT hold a heading? cos(turn) between consecutive transit steps")
        print(f"   mean cos      {float(acc.turn_sum) / tn:6.3f}   "
              f"(+1 = committed walk, 0 = random turn, <0 = reversing)")
        print(f"   reversals     {float(acc.turn_rev) / tn:6.3f}   (cos < 0)")
        print(f"   backtracks    {float(acc.turn_back) / tn:6.3f}   (cos < -0.7)   "
              f"({int(tn)} scored pairs)")

    if acc.run_steps:
        rl = torch.cat(acc.run_steps)
        rs = torch.cat(acc.run_straight)
        q = torch.tensor([0.5, 0.9, 0.99])
        pl = torch.quantile(rl, q)
        print(f"\n-- TRANSIT RUNS (one uninterrupted commute between two pieces of real work)")
        print(f"   runs {int(rl.numel())}   steps/run  mean {float(rl.mean()):6.2f}  "
              f"p50 {float(pl[0]):5.0f}  p90 {float(pl[1]):5.0f}  p99 {float(pl[2]):5.0f}")
        print(f"   straightness |net|/path   mean {float(rs.mean()):6.3f}   "
              f"(1 = a straight route, 0 = ended where it started)")
        long_ = rl >= 10
        if bool(long_.any()):
            print(f"   runs >=10 steps: {float(long_.float().mean()):5.3f} of runs, "
                  f"{float(rl[long_].sum() / rl.sum()):5.3f} of all transit steps, "
                  f"straightness {float(rs[long_].mean()):6.3f}")

        d0 = torch.cat(acc.run_d0)
        pth = torch.cat(acc.run_path)
        dend = torch.cat(acc.run_dend)
        ok = torch.isfinite(d0) & long_
        if bool(ok.any()):
            a, b, c = d0[ok], pth[ok], dend[ok]
            ratio = (b / a.clamp(min=1.0))
            qq = torch.tensor([0.25, 0.5, 0.75])
            print(f"\n   LONG RUNS (>=10 steps), {int(ok.sum())} of them — WAS THE WALK FORCED?")
            print(f"      d0  dist to own nearest frontier AT RUN START (px)   "
                  f"p25 {float(torch.quantile(a, qq[0])):7.1f}  "
                  f"p50 {float(torch.quantile(a, qq[1])):7.1f}  "
                  f"p75 {float(torch.quantile(a, qq[2])):7.1f}")
            print(f"      path walked over the run (px)                        "
                  f"p25 {float(torch.quantile(b, qq[0])):7.1f}  "
                  f"p50 {float(torch.quantile(b, qq[1])):7.1f}  "
                  f"p75 {float(torch.quantile(b, qq[2])):7.1f}")
            print(f"      path / d0   (~1 = went straight to the nearest, >>1 = went elsewhere)  "
                  f"p50 {float(torch.quantile(ratio, qq[1])):6.2f}")
            print(f"      d_end dist to nearest frontier when the run ENDED    "
                  f"p50 {float(torch.quantile(c, qq[1])):7.1f}")
            # The verdict split. A long run that began with work within one sensor radius was NOT
            # forced: the agent had something to do and walked away from it.
            for thr in (80.0, 160.0):
                near = float((a < thr).float().mean())
                print(f"      long runs that STARTED with a frontier < {thr:5.0f} px away: "
                      f"{near:5.3f}  <- had work in reach")

            un, ub, db = (torch.cat(acc.run_u_near)[ok], torch.cat(acc.run_u_best)[ok],
                          torch.cat(acc.run_d_best)[ok])
            fin = torch.isfinite(db)
            print(f"\n      IS THE LONG WALK RATIONAL? utility of the nearest vs the best frontier")
            print(f"         u(nearest)  p50 {float(torch.quantile(un, qq[1])):6.3f}"
                  f"     u(best)  p50 {float(torch.quantile(ub, qq[1])):6.3f}")
            print(f"         d(best) px  p50 {float(torch.quantile(db[fin], qq[1])):7.1f}"
                  f"   vs path walked p50 {float(torch.quantile(b, qq[1])):7.1f}")
            eq = float((un >= 0.9 * ub).float().mean())
            print(f"         nearest frontier was >=90% as good as the best: {eq:5.3f}")
            print(f"         -> high share + long walk = the detour bought nothing;"
                  f" low share = utility genuinely sits far and the lever is vf_gamma")
    print()


if __name__ == "__main__":
    main()
