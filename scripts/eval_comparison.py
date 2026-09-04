"""MARLauder side of the MARLauder-vs-IR2 comparison — emits IR2's own CSV, map for map.

The protocol is frozen in eval/comparison/README.md: the metrics are IR2's NATIVE ones, measured
on the SAME 100 maps per split (eval/comparison/map_indices_{split}.json, dataset parity verified
bit-for-bit by parity_check.py), for M=2 and M=4, never averaged across splits.

Everything here exists to make MARLauder answer IR2's question rather than its own. Three places
where the two systems disagree, and what this script does about each:

  1. WHEN THE EPISODE ENDS. MARLauder stops when the TEAM UNION map is 99% explored; IR2 stops
     when EVERY robot's PRIVATE belief is 99% explored (env.check_done, which loops over robots).
     Their `success` column is literally that termination flag (test_multi_robot_worker.py:122).
     Under the union rule the exchange is optional — the union is complete whether or not the map
     ever reached the other robot — so scoring MARLauder under it would answer a strictly easier
     question. Forced here via done_mode="own".
  2. WHAT `explored` MEANS. IR2's evaluate_team_exploration_rate (env.py:624) averages
     evaluate_exploration_rate(agent_id) over agents, and that reads all_robot_belief[a][a] — each
     robot's OWN belief. It is a per-robot mean, NOT the union, despite "team" in the name. We
     report the same mean, and carry the union along as the extra `explored_union` column so the
     two are never confused again.
  3. WHAT A STEP IS. An IR2 step is a waypoint decision plus the A* traverse to it; a MARLauder
     step is one lattice hop of at most NR·√2 px. `steps` is recorded for completeness but is NOT
     comparable between the systems — `max_dist` (metres of robot travel) is the headline, and it
     is comparable.

Already identical, so nothing to correct: sensor range (80 px both), the radio model (both run the
log-distance path-loss model with P_T=-20, thresh=-70, γ=2/4, d₀=35, PL₀=31, X_g,K ~ U[0,13]
resampled per episode — IR2's PROXIMITY_COMMS_RANGE is dead code under
USE_SIGNAL_STRENGTH_NOT_PROXIMITY=True), and the coverage denominator (ground-truth free pixels).

    python scripts/eval_comparison.py --ckpt runs/v10_difficult_.../ckpt_best.pt
    python scripts/eval_comparison.py --ckpt <ckpt> --splits complex --agents 2 --batch 25

Writes eval/comparison/results/marlauder_{split}_M{M}[_{tag}].csv, then run analyze.py.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
from eval.ckpt_loader import load_model_from_ckpt

_COMPARISON_DIR = _REPO / "eval" / "comparison"

# IR2's native per-split episode caps (their test_parameter.py MAX_EPS_STEPS). Not a MARLauder
# choice: the cap decides how many episodes end in truncation rather than success, so it has to be
# theirs for the success rates to mean the same thing.
IR2_CAPS = {"hybrid": 196, "corridor": 196, "complex": 384}

CSV_FIELDS = ["eps", "num_robots", "max_dist", "steps", "explored", "success", "connectivity",
              "explored_union",
              # --- APPENDIX: behaviour diagnostics. No IR2 counterpart, never a substitute for the
              # columns above. They exist to answer a question `success` cannot: WHETHER the team
              # reached per-agent 99% the intended way (split the map, then deliberately meet to
              # exchange) or degenerately (never separate, so both maps are identical for free and
              # no rendezvous is ever needed). The degenerate solution scores a perfect `success`
              # while demonstrating none of the coordination the thesis claims.
              "pair_dist_mean",      # mean inter-agent distance / canvas diagonal, time-averaged
              "pair_dist_max",       # the furthest they ever got apart (same normalisation)
              "comm_duty",           # fraction of steps the pair was in contact. →1.0 = glued
              "sensing_overlap",     # fraction of steps the LiDAR disks overlapped (MARVEL metric)
              "n_syncs",             # paid map-exchange events (MARLauder-only: sync_min_gap etc.)
              "pair_dist_mean_px",   # same as pair_dist_mean but in PIXELS — the cross-system one
              "pair_dist_max_px",
              "own_gap_final",       # union coverage − weakest robot's own coverage, at episode end
              "contrib_imbalance",   # |share_i − 1/M| spread of union-new cells found per agent
              # ATTRIBUTION PARITY (only non-zero with --attr-parity): the same spread recomputed
              # under IR2's accounting. `_seq` = single-claimant at our cadence, `_ir2` = single
              # claimant at IR2's per-map stride. See EnvCfg.attr_ir2_parity.
              "contrib_imbalance_seq",
              "contrib_imbalance_ir2",
              # --- comparison v2 §6.2: how far our agents ended up from IR2's actual start
              # positions after snapping to the lattice. 0 only if IR2 happened to start on one of
              # our nodes. Reported per episode so the claim "same starting positions" carries its
              # own error bar instead of being an assertion.
              "start_offset_mean_px", "start_offset_max_px"]


def _all_connected(comm_mask: torch.Tensor) -> torch.Tensor:
    """[N, M, M] pairwise contact → [N] bool: are all M agents in ONE connected component?

    IR2 sets connectivity_rate = (len(agents_comms_broken) == 0) (env.py:386), and an agent is
    "broken" whenever it sits outside the single largest flock — so their flag is true exactly
    when one component holds everybody. Contact is transitive there (multi-hop relaying), which is
    why this closes the mask over M hops instead of just testing pairs.
    """
    N, M, _ = comm_mask.shape
    if M == 1:
        return torch.ones(N, dtype=torch.bool, device=comm_mask.device)
    reach = comm_mask | torch.eye(M, dtype=torch.bool, device=comm_mask.device).unsqueeze(0)
    for _ in range(M):                       # M squarings ≫ enough to close a component of size M
        reach = (reach.float() @ reach.float()) > 0
    return reach[:, 0, :].all(dim=-1)


@torch.no_grad()
def _run_chunk(model, env: Explorer, map_idxs: list[int], cap: int, device: str,
               noise_seed: int, map_seed: int,
               budgets: list[float] | None = None,
               starts: list[list] | None = None,
               strides: list[float] | None = None) -> list[dict]:
    """One batched rollout: len(map_idxs) maps as len(map_idxs) parallel envs, one episode each.

    Envs are FROZEN at their own `done` (the env auto-resets them in place, so every recorded
    quantity has to stop advancing there or it would describe the next episode).
    """
    K = len(map_idxs)
    assert env.N == K, f"env n_envs ({env.N}) must equal the chunk size ({K})"
    M = env.M
    # Reseed BEFORE the reloads, not after: reload_map runs a full env reset, and that reset is
    # what draws the episode's shadowing noise (X_g, K). Reseeding afterwards would pin a stream
    # nothing in a one-episode-per-env rollout ever draws from again, and the channel each map
    # actually met would silently be whatever the constructor's stream happened to hold.
    env.reseed_channel_noise(noise_seed)     # pin the shadowing stream: same channel for every ckpt
    start_off: list = []
    for i, midx in enumerate(map_idxs):
        # Re-key the map RNG per MAP, not per slot. reload_map pulls one draw from env.rng to seed
        # _spread_starts_graph, which scatters the M agents around the map's single anchor start —
        # so without this the starting formation depends on where the map fell in the chunk, and
        # --batch would silently change the results. Keyed by map index, a map always gets the same
        # formation: the same checkpoint re-run agrees with itself, and the per-map pairing against
        # IR2 stays honest. (Measured drift when unpinned: max_dist 2021 vs 2016 on hybrid_M2.)
        env.rng = np.random.default_rng(map_seed + int(midx))
        # comparison v2 §6.2: pin the agents to IR2's own start positions when supplied, so the two
        # systems answer the same question from the same place. reload_map snaps each to the
        # nearest free lattice node and records the residual in env.last_start_offset_px.
        env.reload_map(env_idx=i, map_idx=int(midx),
                       start_override=(starts[i] if starts is not None else None))
        if starts is not None:
            start_off.append(env.last_start_offset_px.clone())

    # comparison v2 §7.1: PER-MAP travel budget = the distance IR2 spent on that same map. Set
    # after the reloads because reload_map does a full reset. This drives BOTH truncation and the
    # actor's travel_frac observation (Explorer.budget_px), which is what keeps the policy
    # on-distribution: v20 trains with rdv_urgency_mode="budget", so the rendezvous pull ramps on
    # the observed fraction of budget spent. A budget the agent cannot see is a budget it cannot
    # plan against.
    if budgets is not None:
        env.travel_budget_px = torch.tensor([max(1.0, float(b)) for b in budgets],
                                            dtype=torch.float32, device=device)
    if strides is not None:
        env.attr_stride_px = torch.tensor([max(1.0, float(x)) for x in strides],
                                          dtype=torch.float32, device=device)

    h_act, h_crit = model.init_hidden(K, device)
    obs = env.obs

    active = torch.ones(K, dtype=torch.bool, device=device)
    travel = torch.zeros(K, M, device=device)          # cumulative px per agent
    steps = torch.zeros(K, device=device)
    explored_own = torch.zeros(K, device=device)       # IR2 `explored`: mean over agents of own map
    explored_union = torch.zeros(K, device=device)
    success = torch.zeros(K, device=device)
    connected = torch.zeros(K, device=device)
    # Appendix accumulators (see CSV_FIELDS). Computed per-EPISODE from the raw tensors rather
    # than read from info["metrics"], whose values are already reduced to batch scalars.
    canvas_diag = float((env.H ** 2 + env.W ** 2) ** 0.5)
    triu = torch.triu(torch.ones(M, M, device=device), diagonal=1).bool()
    offdiag = ~torch.eye(M, dtype=torch.bool, device=device)
    pd_sum = torch.zeros(K, device=device)
    pd_max = torch.zeros(K, device=device)
    # Raw-pixel twins of the two above. IR2 reads the unpadded PNG while our canvas is zero-padded
    # to a fixed per-split size, so "fraction of the canvas diagonal" denotes a different physical
    # length on each side and the normalised columns are NOT cross-system comparable. Pixels are.
    pdpx_sum = torch.zeros(K, device=device)
    pdpx_max = torch.zeros(K, device=device)
    duty_sum = torch.zeros(K, device=device)
    overlap_sum = torch.zeros(K, device=device)
    n_syncs = torch.zeros(K, device=device)
    own_gap_final = torch.zeros(K, device=device)
    contrib_imb = torch.zeros(K, device=device)
    contrib_imb_seq = torch.zeros(K, device=device)
    contrib_imb_ir2 = torch.zeros(K, device=device)

    was_training = model.training
    model.eval()
    for _t in range(cap):
        out = model.act(obs, h_act, h_crit, deterministic=True)
        obs, _r, done, info = env.step(out["action"])
        h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
        a = active.float()

        # Distance travelled, LATCHED from the env rather than re-accumulated here. The env's own
        # travel_px is the exact quantity its truncation test compares against the budget, so
        # reading it makes "distance reported" and "distance charged" the same number by
        # construction. The previous version summed (env.pos - pos_prev) AFTER step() returned —
        # but step() auto-resets a finished env at its tail (explorer.py: _reset_envs), so on the
        # step an episode ended it added the teleport from the last position to the NEW episode's
        # spawn: 200-800 px of phantom travel on the headline metric, in the direction that makes
        # MARLauder look worse. info["travel_px"] is cloned pre-reset for exactly this reason.
        travel = torch.where(active.unsqueeze(-1), info["travel_px"].to(device).float(), travel)
        steps += a

        own = info["own_cov"].to(device).float()                            # [K, M] pre-auto-reset
        explored_own = torch.where(active, own.mean(dim=-1), explored_own)
        explored_union = torch.where(active, info["explored_rate"].to(device).float(), explored_union)
        connected = torch.where(active, _all_connected(info["comm_mask"].to(device)).float(), connected)
        # `success` = IR2's termination flag. done_mode="own" makes info["terminated"] fire on
        # "every robot holds 99% of the map", so this is their check_done, not ours.
        # --- appendix: separation / contact / division-of-labour, accumulated while active ---
        if M > 1:
            # info["pos"], NOT env.pos: on the step an episode ends, env.pos already holds the
            # next episode's spawn (see the travel latch above), which would put a phantom
            # separation into the last sample of every episode.
            ipos = info["pos"].to(device).float()
            dmat = torch.cdist(ipos, ipos)
            pdist_px = dmat[:, triu].mean(-1)                                          # [K] px
            pdist = pdist_px / canvas_diag
            pd_sum += pdist * a
            pd_max = torch.maximum(pd_max, pdist * a)
            pdpx_sum += pdist_px * a
            pdpx_max = torch.maximum(pdpx_max, pdist_px * a)
            cm = info["comm_mask"].to(device)
            duty_sum += cm[:, offdiag].float().mean(-1) * a
            overlap_sum += ((dmat[:, triu]
                             < 2.0 * env.cfg.sensor_range_px).float().mean(-1)) * a
            if "sync_paid" in info:
                n_syncs += info["sync_paid"].to(device).float().sum(-1) * a
        newly = active & done.to(device)
        success = torch.where(newly, info["terminated"].to(device).float(), success)
        # Episode-end snapshots, latched on the step this env finished (info is pre-auto-reset).
        own_gap_final = torch.where(
            newly, (info["explored_rate"].to(device).float() - own.min(dim=-1).values).clamp(min=0.0),
            own_gap_final)
        if M > 1:
            nov = info["novel_cells_ep"].to(device).float()                            # [K, M]
            share = nov / nov.sum(-1, keepdim=True).clamp(min=1.0)
            contrib_imb = torch.where(newly, (share - 1.0 / M).abs().sum(-1), contrib_imb)
            for _key, _dst in (("novel_cells_seq_ep", "seq"), ("novel_cells_ir2_ep", "ir2")):
                _n = info[_key].to(device).float()
                _sh = _n / _n.sum(-1, keepdim=True).clamp(min=1.0)
                _v = torch.where(newly, (_sh - 1.0 / M).abs().sum(-1),
                                 contrib_imb_seq if _dst == "seq" else contrib_imb_ir2)
                if _dst == "seq":
                    contrib_imb_seq = _v
                else:
                    contrib_imb_ir2 = _v
        active = active & ~done.to(device)
        if not bool(active.any().item()):
            break
    if was_training:
        model.train()

    max_dist = travel.max(dim=-1).values
    denom = steps.clamp(min=1.0)                       # time-average over the steps each env ran
    soff = (torch.stack(start_off) if start_off else torch.zeros(K, M, device=device))
    return [{
        "start_offset_mean_px": float(soff[i].mean().item()),
        "start_offset_max_px": float(soff[i].max().item()),
        "num_robots": M,
        "max_dist": float(max_dist[i].item()),
        "steps": int(steps[i].item()),
        "explored": float(explored_own[i].item()),
        "success": int(success[i].item()),
        "connectivity": int(connected[i].item()),
        "explored_union": float(explored_union[i].item()),
        "pair_dist_mean": float((pd_sum / denom)[i].item()),
        "pair_dist_max": float(pd_max[i].item()),
        "pair_dist_mean_px": float((pdpx_sum / denom)[i].item()),
        "pair_dist_max_px": float(pdpx_max[i].item()),
        "comm_duty": float((duty_sum / denom)[i].item()),
        "sensing_overlap": float((overlap_sum / denom)[i].item()),
        "n_syncs": float(n_syncs[i].item()),
        "own_gap_final": float(own_gap_final[i].item()),
        "contrib_imbalance": float(contrib_imb[i].item()),
        "contrib_imbalance_seq": float(contrib_imb_seq[i].item()),
        "contrib_imbalance_ir2": float(contrib_imb_ir2[i].item()),
    } for i in range(K)]


def _load_ir2_strides(ir2_dir: Path, split_name: str, M: int, n: int) -> list[float]:
    """Per-map ATTRIBUTION STRIDE in px = IR2's own max_dist / steps on that map.

    IR2 credits discoveries once per graph edge traversed, and this is how long one of those edges
    actually was. It is an UPPER bound on their mean stride — max_dist is the FASTEST robot's
    cumulative travel while `steps` counts the episode, so the slower robots moved less per step.
    That bias is deliberate: a coarser stride makes the "it is only an artefact of counting"
    hypothesis EASIER to confirm, so a null result under it is the conservative one.
    (The instrumented run did not publish per_robot_dist, or the mean would be used directly.)
    """
    with (ir2_dir / f"ir2_{split_name}_M{M}.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["eps"]))
    if len(rows) != n:
        sys.exit(f"[eval_comparison] stride source has {len(rows)} rows, expected {n}")
    return [max(1.0, float(r["max_dist"]) / max(1.0, float(r["steps"]))) for r in rows]


def _load_ir2_reference(ir2_dir: Path, split_name: str, M: int, n: int):
    """(per-map travel budget, per-map start positions) from an instrumented IR2 run.

    The budget is IR2's OWN `max_dist` on that map — the max over robots of cumulative travel,
    which is the same reduction our truncation uses (`travel_px.amax(dim=1)`), so the two systems
    are held to the same quantity rather than to two things that share a name.

    Row i of the IR2 CSV is map i of map_indices_{split}.json (their driver sorts by `eps` and is
    driven with Env(map_index=episode) over the same 100-file list), which is what makes the
    downstream Wilcoxon paired.
    """
    csv_f = ir2_dir / f"ir2_{split_name}_M{M}.csv"
    js_f = ir2_dir / f"starts_{split_name}_M{M}.json"
    if not csv_f.is_file():
        sys.exit(f"[eval_comparison] PROTOCOL v2 needs {csv_f} — run the IR2 side first")
    if not js_f.is_file():
        sys.exit(f"[eval_comparison] PROTOCOL v2 needs {js_f} — the IR2 run must be the "
                 f"INSTRUMENTED one (comparison/compat), which exports start positions")
    with csv_f.open() as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: int(r["eps"]))
    if len(rows) != n:
        sys.exit(f"[eval_comparison] {csv_f.name} has {len(rows)} rows, expected {n} — the two "
                 f"sides would not be paired map-for-map")
    budgets = [float(r["max_dist"]) for r in rows]
    js = json.loads(js_f.read_text())
    starts = []
    for i in range(n):
        e = js[str(i)]
        if len(e["starts"]) != M:
            sys.exit(f"[eval_comparison] {js_f.name} episode {i} has {len(e['starts'])} start "
                     f"positions, expected M={M}")
        starts.append(e["starts"])
    return budgets, starts


def _run_cell(ckpt: Path, split_name: str, M: int, entries: list[dict], cap: int,
              batch: int, device: str, noise_seed: int, map_seed: int,
              max_travel_px: float = 0.0, comm_relay: bool | None = None,
              budgets: list[float] | None = None,
              starts: list[list] | None = None,
              strides: list[float] | None = None,
              env_overrides: dict | None = None) -> list[dict]:
    """One (split, M) cell of the comparison grid → 100 episode rows."""
    pack_idxs = [int(e["pack_idx"]) for e in entries]
    split = load_split(f"test/{split_name}", device=device)
    peek = torch.load(ckpt, map_location="cpu", weights_only=False)
    penv = dict((peek.get("cfg", {}) or {}).get("env", {}) or {})
    # comm_relay must be PINNED when two checkpoints from different pipelines are compared:
    # from_ckpt_dict falls back to the dataclass default (False) for a pre-relay ckpt while a
    # post-relay one restores True, so the two would be scored on DIFFERENT radio physics — and
    # `connectivity` is a published CSV column, which relay changes directly. None = whatever the
    # checkpoint says, so every CSV produced before this flag existed still reproduces bit-exact.
    if comm_relay is not None:
        penv["comm_relay"] = bool(comm_relay)
    # ABLATION knobs, written into the checkpoint's own env dict so from_ckpt_dict restores
    # everything else untouched. These change what the ACTOR observes, so a value other than the
    # one the checkpoint trained with puts the policy off-distribution: the result measures how
    # sensitive the learned behaviour is to that input, NOT what a policy retrained at that value
    # would do. Label such runs as ablations and never as a checkpoint's score.
    for _k, _v in (env_overrides or {}).items():
        penv[_k] = _v

    rows: list[dict] = []
    for start in range(0, len(pack_idxs), batch):
        chunk = pack_idxs[start:start + batch]
        env_cfg = EnvCfg.from_ckpt_dict(
            penv, n_envs=len(chunk), n_agents=M, max_episode_steps=cap,
            max_travel_px=max_travel_px,     # 0.0 → step cap only (the frozen native protocol)
            # MUST be forced to 0. from_ckpt_dict keeps every valid EnvCfg field found in the
            # checkpoint, so v20's TRAINING budget (max_travel_frac=0.03) silently came back and
            # truncated the evaluation — a per-map ceiling of 3840 px on corridor where IR2 spends
            # 7204, i.e. our own training handicap carried into their test. It also takes
            # precedence over max_travel_px, so setting the flat budget alone did nothing.
            # See PROTOCOL_V2_DISTANZA.md §2.
            max_travel_frac=0.0,
            done_mode="own",                 # <- the IR2 stopping rule; see the module docstring
            map_seed=map_seed,               # constructor draw; _run_chunk re-keys it per map
        )
        env = Explorer(split, env_cfg, seed=0)
        # n_agents=M overrides the ckpt's agent count on purpose: M=4 is a ZERO-SHOT transfer of an
        # M=2 policy (the encoder is per-agent and the critic pools count-invariantly, so the
        # weights are literally the same ones). A finetuned M=4 policy would be a separate,
        # separately-labelled run — never a substitute for this number.
        model, _ = load_model_from_ckpt(ckpt, device, n_agents=M, verbose=(start == 0))
        sl = slice(start, start + len(chunk))
        rows.extend(_run_chunk(model, env, chunk, cap, device, noise_seed, map_seed,
                               budgets=(budgets[sl] if budgets is not None else None),
                               starts=(starts[sl] if starts is not None else None),
                               strides=(strides[sl] if strides is not None else None)))
        print(f"  [{split_name} M={M}] {min(start + batch, len(pack_idxs))}/{len(pack_idxs)} maps",
              flush=True)
        del model, env
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for i, r in enumerate(rows):
        r["eps"] = i                          # episode index == position in map_indices, so the
    return rows                               # paired Wilcoxon lines up map-for-map with IR2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="checkpoint to evaluate")
    ap.add_argument("--splits", nargs="+", default=["hybrid", "corridor", "complex"],
                    choices=["hybrid", "corridor", "complex"])
    ap.add_argument("--agents", nargs="+", type=int, default=[2, 4],
                    help="team sizes; 4 is a zero-shot transfer of the M=2 weights")
    ap.add_argument("--batch", type=int, default=25,
                    help="maps evaluated in one batched rollout (VRAM knob; the 100 maps of a cell "
                         "are split into ceil(100/batch) chunks and the result is identical)")
    ap.add_argument("--noise-seed", type=int, default=777,
                    help="seed for the radio-shadowing stream, so two checkpoints meet the same channel")
    ap.add_argument("--map-seed", type=int, default=0,
                    help="seed for the per-map agent start formation (keyed by map index, so it is "
                         "independent of --batch). Change it only to measure spawn sensitivity")
    ap.add_argument("--max-travel-px", type=float, default=0.0,
                    help="TRAVEL BUDGET per episode in px (0 = off → IR2's native step caps, the "
                         "frozen protocol). An IR2 step is a waypoint teleport of arbitrary length "
                         "while ours is one lattice hop (<=22.63px), so equal step caps give us only "
                         "~50%% of IR2's measured travel on corridor and ~43%% on complex. Distance "
                         "is the unit that means the same thing on both sides. IR2's measured means: "
                         "hybrid 3422, corridor 7204, complex 16966 px (M=2).")
    ap.add_argument("--cap", type=int, default=0,
                    help="override the per-episode STEP cap (0 = IR2 native, or auto when "
                         "--max-travel-px is set). With a travel budget the step cap is only a "
                         "safety net for a stalling policy, which burns no distance.")
    ap.add_argument("--comm-relay", dest="comm_relay", action="store_true", default=None,
                    help="PIN multi-hop relay ON for this run instead of taking it from the "
                         "checkpoint. Required when comparing a pre-relay checkpoint against a "
                         "post-relay one, or the two meet different radio physics and the "
                         "`connectivity` column is not comparable. IR2 itself relays (env.py:424), "
                         "so ON is the faithful setting. Default (unset) = whatever the ckpt says.")
    ap.add_argument("--no-comm-relay", dest="comm_relay", action="store_false", default=None)
    ap.add_argument("--ir2-dir", type=Path, default=None,
                    help="PROTOCOL v2: directory holding the instrumented IR2 run "
                         "(ir2_{split}_M{M}.csv + starts_{split}_M{M}.json). Switches the episode "
                         "budget from IR2's STEP cap to IR2's own per-map DISTANCE, and pins our "
                         "agents to IR2's start positions. An IR2 step is a graph-edge traverse of "
                         "up to 160 px while ours is a lattice hop of at most 22.63, so equal step "
                         "caps hand us ~45%% of their metres on complex — the cap is written in the "
                         "one unit the protocol itself calls incomparable. See "
                         "ir2_comparison_export/PROTOCOL_V2_DISTANZA.md.")
    ap.add_argument("--vf-gamma", type=float, default=None,
                    help="ABLATION: per-hop discount of the VALUE-FIELD actor input (EnvCfg."
                         "vf_gamma, trained at 0.97). It sets how far down the BF tree utility "
                         "still pulls: 0.97^20hops=0.54 but 0.80^20=0.012, so lowering it makes "
                         "the agent myopic. Off-distribution for the checkpoint — see the note in "
                         "_run_cell.")
    ap.add_argument("--radar-gamma", type=float, default=None,
                    help="ABLATION: same idea for the COARSE long-range radar channel "
                         "(EnvCfg.radar_gamma; v20 trained at 0.97, dataclass default 0.92). The "
                         "value field and the radar are the two inputs that advertise distant "
                         "utility; changing only one leaves the other still calling.")
    ap.add_argument("--env-set", action="append", default=[], metavar="KEY=VALUE",
                    help="ABLATION: set any EnvCfg field, e.g. --env-set "
                         "force_full_occupancy_sharing=True. Repeatable. Same warning as "
                         "--vf-gamma: anything the actor observes puts the policy "
                         "off-distribution, so the result is a mechanism probe, not a score.")
    ap.add_argument("--attr-parity", action="store_true",
                    help="Also report contrib_imbalance recomputed under IR2's ACCOUNTING: "
                         "single-claimant (their loop folds each robot into the merged belief "
                         "before the next is measured) and coarse-grained (they credit once per "
                         "graph edge, we credit once per lattice hop). Answers whether our worse "
                         "imbalance is behaviour or bookkeeping. Requires --ir2-dir for the "
                         "per-map stride. Measurement only — nothing feeds a reward or the "
                         "termination test, and `contrib_imbalance` itself is untouched.")
    ap.add_argument("--tag", default="", help="suffix for the output CSVs, e.g. --tag v10")
    ap.add_argument("--out-dir", type=Path, default=_COMPARISON_DIR / "results")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    _env_ovr = {}
    if args.vf_gamma is not None:
        _env_ovr["vf_gamma"] = float(args.vf_gamma)
    if args.radar_gamma is not None:
        _env_ovr["radar_gamma"] = float(args.radar_gamma)
    for _kv in args.env_set:
        if "=" not in _kv:
            sys.exit(f"[eval_comparison] --env-set wants KEY=VALUE, got {_kv!r}")
        _k, _v = _kv.split("=", 1)
        try:
            _env_ovr[_k.strip()] = ast.literal_eval(_v.strip())
        except (ValueError, SyntaxError):
            _env_ovr[_k.strip()] = _v.strip()          # bare string value
    if _env_ovr:
        print(f"[eval_comparison] ABLATION env overrides: {_env_ovr} — the policy is "
              f"OFF-DISTRIBUTION at these values; label the output as an ablation", flush=True)

    if not args.ckpt.is_file():
        print(f"[eval_comparison] checkpoint not found: {args.ckpt}")
        sys.exit(1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    print(f"[eval_comparison] ckpt={args.ckpt} splits={args.splits} agents={args.agents} "
          f"device={args.device}", flush=True)
    print(f"{'cell':<18}{'max_dist':>12}{'steps':>8}{'explored':>10}{'success':>9}"
          f"{'conn':>7}{'union':>8}", flush=True)

    for split_name in args.splits:
        idx_file = _COMPARISON_DIR / f"map_indices_{split_name}.json"
        entries = json.loads(idx_file.read_text())["entries"]
        # Step cap: explicit override > auto (travel-budget mode) > IR2 native. In travel-budget
        # mode the step cap must not be the binding constraint, so it is sized at 2x the hops a
        # perfectly efficient robot would need (min axial hop = nr px) — a policy that actually
        # moves always exhausts the DISTANCE first, and a stalling one still terminates.
        if args.cap > 0:
            cap = args.cap
        elif args.max_travel_px > 0:
            cap = int(math.ceil(args.max_travel_px / 16.0)) * 2
        else:
            cap = IR2_CAPS[split_name]
        if args.max_travel_px > 0:
            print(f"[{split_name}] travel budget {args.max_travel_px:.0f}px "
                  f"(step cap {cap} = safety net; IR2 native was {IR2_CAPS[split_name]})", flush=True)
        for M in args.agents:
            budgets = starts = None
            strides = None
            if args.ir2_dir is not None:
                budgets, starts = _load_ir2_reference(args.ir2_dir, split_name, M, len(entries))
                if args.attr_parity:
                    strides = _load_ir2_strides(args.ir2_dir, split_name, M, len(entries))
                    _env_ovr["attr_ir2_parity"] = True
                    print(f"[{split_name} M={M}] ATTRIBUTION PARITY on — IR2 stride "
                          f"mean {sum(strides)/len(strides):.1f} px/step "
                          f"[{min(strides):.1f}, {max(strides):.1f}] vs our hop <=22.63", flush=True)
                # The step cap is demoted to a safety net: a policy that stalls burns no distance
                # and would never truncate. 2x the hops a perfectly efficient robot would need at
                # the minimum axial hop (nr px) — anything that actually moves exhausts the
                # DISTANCE first, so this never binds for a working policy.
                cap = int(math.ceil(max(budgets) / 16.0)) * 2
                print(f"[{split_name} M={M}] PROTOCOL v2 — per-map budget from IR2: "
                      f"mean {sum(budgets)/len(budgets):.0f} px, range "
                      f"[{min(budgets):.0f}, {max(budgets):.0f}]; step cap {cap} = safety net "
                      f"(IR2 native was {IR2_CAPS[split_name]})", flush=True)
            rows = _run_cell(args.ckpt, split_name, M, entries, cap, args.batch,
                             args.device, args.noise_seed, args.map_seed, args.max_travel_px,
                             comm_relay=args.comm_relay, budgets=budgets, starts=starts,
                             strides=strides, env_overrides=_env_ovr)
            if args.ir2_dir is not None:
                _o = [r["start_offset_mean_px"] for r in rows]
                _x = [r["start_offset_max_px"] for r in rows]
                print(f"  start parity: offset from IR2 positions after lattice snap — "
                      f"mean {sum(_o)/len(_o):.2f} px, worst agent {max(_x):.2f} px", flush=True)
            out = args.out_dir / f"marlauder_{split_name}_M{M}{suffix}.csv"
            with out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows({k: r[k] for k in CSV_FIELDS} for r in rows)
            n = len(rows)
            print(f"{split_name + '_M' + str(M):<18}"
                  f"{sum(r['max_dist'] for r in rows) / n:>12.0f}"
                  f"{sum(r['steps'] for r in rows) / n:>8.1f}"
                  f"{sum(r['explored'] for r in rows) / n:>10.3f}"
                  f"{sum(r['success'] for r in rows) / n:>9.2f}"
                  f"{sum(r['connectivity'] for r in rows) / n:>7.2f}"
                  f"{sum(r['explored_union'] for r in rows) / n:>8.3f}   → {out.name}", flush=True)

    print("\n[eval_comparison] done. Aggregate against IR2 with:\n"
          f"  python eval/comparison/analyze.py{' --tag ' + args.tag if args.tag else ''}", flush=True)


if __name__ == "__main__":
    main()
