"""Generate an inspector trace that SHOWS the teammate belief propagating, for the web viewer.

No well-trained 2-agent policy currently loads (architecture drift), and the smoke checkpoint keeps
the agents together → the belief stays collapsed. So we still load a checkpoint (for real logits /
value / GAT attention in the trace) but OVERRIDE the actions with a separation heuristic: agent 1
holds, agent 0 walks away exploring. Agent 0 then loses comm with the stationary agent 1, and
agent 0's belief of agent 1 diffuses / carves / piles at the frontier — visible in the inspector's
"teammate_pot" field, stepping through time.

    python scripts/trace_belief_web.py --ckpt runs/<run>/ckpt_best.pt \\
        --split train/easy --map-idx 50 --steps 160 --comm-range 40 --out runs/<run>

Then open the inspector from the web dashboard (the run shows "traces"); select the
`teammate_pot` field to watch agent 0's belief of the stationary teammate propagate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from eval.ckpt_loader import load_model_from_ckpt
from eval.trace import capture_trace


def build_action_fn(env):
    node_xy = env.graph.node_xy
    eidx_static = env.graph.edge_idx_static
    K = env.graph.edge_len.shape[0]
    last0 = [-1]

    def action_fn(env, obs):
        am = obs["action_mask"][0].bool()               # [M, K]
        a1_xy = env.pos[0, 1]
        curr0 = int(env.curr_idx_global[0, 0])
        nbrs = eidx_static[curr0]
        nbr_xy = node_xy[nbrs.clamp(min=0)]
        dist_away = (nbr_xy - a1_xy).pow(2).sum(-1)      # walk agent0 away from agent1
        valid0 = am[0].clone()
        if last0[0] >= 0:
            valid0[K - 1 - last0[0]] = False             # no immediate backtrack
        score0 = torch.where(valid0, dist_away, torch.full_like(dist_away, -1e9))
        a0 = int(score0.argmax())
        last0[0] = a0
        invalid1 = (~am[1]).nonzero()
        a1 = int(invalid1[0]) if invalid1.numel() else 0  # masked slot → agent1 holds
        return torch.tensor([[a0, a1]], device=am.device)

    return action_fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--map-idx", type=int, default=50)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--comm-range", type=float, default=40.0)
    ap.add_argument("--comm-model", choices=["los", "signal_strength", "ckpt"], default="los",
                    help="'ckpt' keeps the checkpoint's own comm model/range (as trained)")
    ap.add_argument("--belief-mode", choices=["uniform", "pathfront"], default="pathfront",
                    help="which teammate-belief model to record (bel field)")
    ap.add_argument("--frontier-min-unknown", type=int, default=None,
                    help="pathfront: min UNKNOWN 8-neighbours for a node to count as an opening "
                         "(env default 4). Lower it to stop dropping openings the observer has "
                         "only partly revealed.")
    ap.add_argument("--frontier-min-util", type=float, default=None,
                    help="pathfront: min PRE-diffusion utility seed (util_raw) for a node to count "
                         "as an opening (env default 0.02)")
    ap.add_argument("--use-policy", action="store_true",
                    help="drive with the TRAINED policy (agents explore & separate on their own) "
                         "instead of the scripted agent1-holds/agent0-walks-away override")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True, help="run dir to write the trace into")
    args = ap.parse_args()

    from env.maps import load_split as _load_split  # canonical loader
    model, env_peek = load_model_from_ckpt(args.ckpt, args.device, n_agents=2)
    split = _load_split(args.split, device=args.device)

    trace_env = dict(env_peek or {})
    trace_env["use_teammate_belief"] = True
    trace_env["belief_mode"] = args.belief_mode
    # What counts as an OPENING. These two gates overlap and can disagree: the unknown-neighbour
    # COUNT drops a node the moment the observer reveals a few of its neighbours, even while it
    # still has most of its ribbon, whereas the SEED (util_raw) measures what is actually left to
    # reveal. Exposed here so the two can be compared on the same episode instead of argued about.
    if args.frontier_min_unknown is not None:
        trace_env["pf_frontier_min_unknown"] = int(args.frontier_min_unknown)
    if args.frontier_min_util is not None:
        trace_env["pf_frontier_min_util"] = float(args.frontier_min_util)
    # PRINT WHAT IS ACTUALLY IN FORCE. A checkpoint stores its own env cfg and `from_ckpt_dict`
    # restores it, which is right for an eval but means a value CHANGED IN THE CODE SINCE THE
    # CHECKPOINT WAS SAVED DOES NOT APPLY unless overridden here. That is not hypothetical: the
    # v10 checkpoint pins pf_frontier_min_unknown=4, so traces regenerated from it after the
    # default moved to 1 came out byte-identical and looked like the fix had not been applied.
    from env.explorer import EnvCfg as _EC
    _eff = _EC.from_ckpt_dict(trace_env, n_envs=1, n_agents=2, max_episode_steps=args.steps + 1)
    print(f"[trace] frontier semantics IN FORCE: min_unknown={_eff.pf_frontier_min_unknown} "
          f"min_util={_eff.pf_frontier_min_util:g}  "
          f"(ckpt pinned min_unknown="
          f"{(env_peek or {}).get('pf_frontier_min_unknown', 'absent')}, "
          f"min_util={(env_peek or {}).get('pf_frontier_min_util', 'absent')})")
    if args.comm_model != "ckpt":                 # 'ckpt' → leave the trained comm model/range alone
        trace_env["comm_model"] = args.comm_model
        trace_env["comm_range_px"] = float(args.comm_range)

    if args.use_policy:
        action_fn = None                 # let the TRAINED policy drive → agents explore & separate on their own
    else:
        # scripted separation override (agent1 holds, agent0 walks away). Needs an env handle for graph
        # statics; capture_trace builds its own env, so pre-build an identical probe for the closure.
        from env.explorer import EnvCfg, Explorer
        probe_cfg = EnvCfg.from_ckpt_dict(trace_env, n_envs=1, n_agents=2, max_episode_steps=args.steps + 1)
        probe = Explorer(split, probe_cfg, seed=int(args.map_idx))
        action_fn = build_action_fn(probe)

    # The SPLIT and the COMM RANGE belong in the tag. Without them a hybrid run and a complex run
    # of the same checkpoint and map index write to the same folder and silently replace each
    # other, and two comm ranges — the whole point of comparing — cannot coexist.
    sp = args.split.replace("/", "-")
    cr = "ckpt" if args.comm_model == "ckpt" else f"{args.comm_model}{args.comm_range:g}"
    tag = (f"belief_{args.belief_mode}_{'policy' if args.use_policy else 'scripted'}"
           f"_{args.ckpt.stem}_{sp}_m{args.map_idx}_{cr}")
    capture_trace(model, split, trace_env, 2, int(args.map_idx), int(args.steps),
                  args.out, tag, args.device, action_fn=action_fn)
    print(f"[trace] wrote {args.out}/traces/{tag}  → open the inspector and pick the '{tag}' episode")


if __name__ == "__main__":
    main()
