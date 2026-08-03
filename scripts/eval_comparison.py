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
              "n_syncs",             # paid map-exchange events (rising-edge contacts)
              "own_gap_final",       # union coverage − weakest robot's own coverage, at episode end
              "contrib_imbalance"]   # |share_i − 1/M| spread of union-new cells found per agent


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
               noise_seed: int, map_seed: int) -> list[dict]:
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
    for i, midx in enumerate(map_idxs):
        # Re-key the map RNG per MAP, not per slot. reload_map pulls one draw from env.rng to seed
        # _spread_starts_graph, which scatters the M agents around the map's single anchor start —
        # so without this the starting formation depends on where the map fell in the chunk, and
        # --batch would silently change the results. Keyed by map index, a map always gets the same
        # formation: the same checkpoint re-run agrees with itself, and the per-map pairing against
        # IR2 stays honest. (Measured drift when unpinned: max_dist 2021 vs 2016 on hybrid_M2.)
        env.rng = np.random.default_rng(map_seed + int(midx))
        env.reload_map(env_idx=i, map_idx=int(midx))

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
    duty_sum = torch.zeros(K, device=device)
    overlap_sum = torch.zeros(K, device=device)
    n_syncs = torch.zeros(K, device=device)
    own_gap_final = torch.zeros(K, device=device)
    contrib_imb = torch.zeros(K, device=device)

    was_training = model.training
    model.eval()
    for _t in range(cap):
        pos_prev = env.pos.clone()                                          # [K, M, 2] px
        out = model.act(obs, h_act, h_crit, deterministic=True)
        obs, _r, done, info = env.step(out["action"])
        h_act, h_crit = out["hidden_actor"], out["hidden_critic"]
        a = active.float()

        # Distance actually covered this step. env.pos is the post-move node centre and a lattice
        # move is a straight segment, so the norm IS the path length — the same quantity IR2
        # accumulates from its A* traverse. Collisions shorten the segment, and this follows.
        travel += (env.pos - pos_prev).norm(dim=-1) * a.unsqueeze(-1)
        steps += a

        own = info["own_cov"].to(device).float()                            # [K, M] pre-auto-reset
        explored_own = torch.where(active, own.mean(dim=-1), explored_own)
        explored_union = torch.where(active, info["explored_rate"].to(device).float(), explored_union)
        connected = torch.where(active, _all_connected(info["comm_mask"].to(device)).float(), connected)
        # `success` = IR2's termination flag. done_mode="own" makes info["terminated"] fire on
        # "every robot holds 99% of the map", so this is their check_done, not ours.
        # --- appendix: separation / contact / division-of-labour, accumulated while active ---
        if M > 1:
            pdist = torch.cdist(env.pos, env.pos)[:, triu].mean(-1) / canvas_diag      # [K]
            pd_sum += pdist * a
            pd_max = torch.maximum(pd_max, pdist * a)
            cm = info["comm_mask"].to(device)
            duty_sum += cm[:, offdiag].float().mean(-1) * a
            overlap_sum += ((torch.cdist(env.pos, env.pos)[:, triu]
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
        active = active & ~done.to(device)
        if not bool(active.any().item()):
            break
    if was_training:
        model.train()

    max_dist = travel.max(dim=-1).values
    denom = steps.clamp(min=1.0)                       # time-average over the steps each env ran
    return [{
        "num_robots": M,
        "max_dist": float(max_dist[i].item()),
        "steps": int(steps[i].item()),
        "explored": float(explored_own[i].item()),
        "success": int(success[i].item()),
        "connectivity": int(connected[i].item()),
        "explored_union": float(explored_union[i].item()),
        "pair_dist_mean": float((pd_sum / denom)[i].item()),
        "pair_dist_max": float(pd_max[i].item()),
        "comm_duty": float((duty_sum / denom)[i].item()),
        "sensing_overlap": float((overlap_sum / denom)[i].item()),
        "n_syncs": float(n_syncs[i].item()),
        "own_gap_final": float(own_gap_final[i].item()),
        "contrib_imbalance": float(contrib_imb[i].item()),
    } for i in range(K)]


def _run_cell(ckpt: Path, split_name: str, M: int, entries: list[dict], cap: int,
              batch: int, device: str, noise_seed: int, map_seed: int,
              max_travel_px: float = 0.0) -> list[dict]:
    """One (split, M) cell of the comparison grid → 100 episode rows."""
    pack_idxs = [int(e["pack_idx"]) for e in entries]
    split = load_split(f"test/{split_name}", device=device)
    peek = torch.load(ckpt, map_location="cpu", weights_only=False)
    penv = (peek.get("cfg", {}) or {}).get("env", {}) or {}

    rows: list[dict] = []
    for start in range(0, len(pack_idxs), batch):
        chunk = pack_idxs[start:start + batch]
        env_cfg = EnvCfg.from_ckpt_dict(
            penv, n_envs=len(chunk), n_agents=M, max_episode_steps=cap,
            max_travel_px=max_travel_px,     # 0.0 → step cap only (the frozen native protocol)
            done_mode="own",                 # <- the IR2 stopping rule; see the module docstring
            map_seed=map_seed,               # constructor draw; _run_chunk re-keys it per map
        )
        env = Explorer(split, env_cfg, seed=0)
        # n_agents=M overrides the ckpt's agent count on purpose: M=4 is a ZERO-SHOT transfer of an
        # M=2 policy (the encoder is per-agent and the critic pools count-invariantly, so the
        # weights are literally the same ones). A finetuned M=4 policy would be a separate,
        # separately-labelled run — never a substitute for this number.
        model, _ = load_model_from_ckpt(ckpt, device, n_agents=M, verbose=(start == 0))
        rows.extend(_run_chunk(model, env, chunk, cap, device, noise_seed, map_seed))
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
    ap.add_argument("--tag", default="", help="suffix for the output CSVs, e.g. --tag v10")
    ap.add_argument("--out-dir", type=Path, default=_COMPARISON_DIR / "results")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

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
            rows = _run_cell(args.ckpt, split_name, M, entries, cap, args.batch,
                             args.device, args.noise_seed, args.map_seed, args.max_travel_px)
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
