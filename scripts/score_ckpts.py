"""Score saved checkpoints with ONE harness — the same eval suite training uses.

Why this exists: `eval/score` in metrics.jsonl is written by whatever env cfg / split / agent
count that run happened to use, so two runs' numbers are not comparable, and a single tick is
not reproducible (bf16 argmax flips + success_rate is a binomial on 32 maps: +-12.6 points at
2 sigma). This script re-scores an explicit list of checkpoints under identical settings and
reports mean +- spread over repeats, so a difference can be read against the noise floor.

    python scripts/score_ckpts.py --ckpt runs/A/ckpt_best.pt runs/A/ckpt_stop.pt \\
        --splits train/difficult,test/complex --repeats 3

`--stochastic` samples from the policy instead of taking the argmax. Training rollouts sample
(driver.py:291) while eval takes the argmax (driver.py:463), so a policy whose GREEDY readout
degenerates into a limit cycle scores 0 here while every train/* metric stays flat. Running both
modes on the same weights separates "policy degraded" from "argmax degenerated".

Loading goes through eval/ckpt_loader.py (auto-detects encoder depth from the state dict). Do
NOT hand-roll checkpoint loading: a layer-count mismatch does not raise, it silently drops the
GAT weights via load_state_dict(strict=False) and the score becomes fiction.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split
from eval.ckpt_loader import load_model_from_ckpt
from train.driver import TrainCfg, _eval_map_idxs, _run_eval_suite

# eval/score = coverage_auc - 0.5*contrib_imbalance_norm - 0.25*sensing_overlap - 0.25*idle_rate_max
# (driver.py:533-538). The first four keys are exactly those four terms, in that order, so a score
# difference can be ATTRIBUTED instead of guessed: two checkpoints can trade a better AUC for a
# worse imbalance and land at the same score, which is a completely different result from a tie.
# The equity terms carry weight 0.5+0.25 of the composite and appear NOWHERE in the IR2 comparison
# CSV (eval_comparison.py:63-75 = steps/explored/success/connectivity/max_dist), so for the paper
# endpoint the outcome block below is what decides — read both, never the composite alone.
REPORT = ["eval/score",
          "eval/coverage_auc", "eval/contrib_imbalance_norm", "eval/sensing_overlap",
          "eval/idle_rate_max",
          # --- outcomes: what the IR2 comparison actually measures ---
          "eval/success_rate", "eval/steps_to_90", "eval/own_coverage_final", "eval/score_own",
          "eval/sync_gap", "eval/n_syncs", "eval/max_comm_gap", "eval/comm_duty", "eval/fairness"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--splits", default="test/complex", help="comma-separated")
    ap.add_argument("--n-maps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--steps", type=int, default=768, help="max_episode_steps for the eval env")
    ap.add_argument("--n-agents", type=int, default=None, help="default: from ckpt")
    ap.add_argument("--stochastic", action="store_true",
                    help="sample actions instead of argmax (see module docstring)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    # The eval env is rebuilt from the CHECKPOINT's saved EnvCfg. from_ckpt_dict falls back to
    # dataclass defaults for absent keys, so a checkpoint predating a field silently pins the OLD
    # semantics; override explicitly when a code change is meant to apply.
    ap.add_argument("--comm-relay", dest="comm_relay", action="store_true", default=None)
    ap.add_argument("--no-comm-relay", dest="comm_relay", action="store_false", default=None)
    ap.add_argument("--max-travel-frac", type=float, default=None)
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    mode = "SAMPLED" if args.stochastic else "ARGMAX"
    print(f"[score_ckpts] mode={mode} maps={args.n_maps} repeats={args.repeats} splits={splits}")

    results: dict[tuple[str, str], dict[str, list[float]]] = {}
    for ck in args.ckpt:
        model, env_peek = load_model_from_ckpt(ck, args.device, n_agents=args.n_agents,
                                               verbose=False)
        M = int(getattr(model, "M", args.n_agents or 2))
        peek = dict(env_peek or {})
        if args.comm_relay is not None:
            peek["comm_relay"] = bool(args.comm_relay)
        if args.max_travel_frac is not None:
            peek["max_travel_frac"] = float(args.max_travel_frac)
        for sp in splits:
            split = load_split(sp, device=args.device)
            env_cfg = EnvCfg.from_ckpt_dict(peek, n_envs=1, n_agents=M,
                                            max_episode_steps=args.steps)
            env = Explorer(split, env_cfg, seed=args.seed)
            tcfg = TrainCfg(n_agents=M, device=args.device, seed=args.seed)
            idxs = _eval_map_idxs(env, args.n_maps)
            acc: dict[str, list[float]] = {}
            for rep in range(args.repeats):
                # Each repeat is a genuinely independent draw: the suite re-pins the channel and
                # map RNG from cfg.seed, so the seed must move for the repeat to mean anything.
                tcfg.seed = args.seed + 1000 * rep
                out = _run_eval_suite(model, env, tcfg, map_idxs=idxs,
                                      deterministic=not args.stochastic)
                for k in REPORT:
                    acc.setdefault(k, []).append(float(out.get(k, float("nan"))))
            results[(str(ck), sp)] = acc
            del env
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    for sp in splits:
        print(f"\n===== {sp}   ({mode}, {args.n_maps} maps, {args.repeats} repeats)")
        for ck in args.ckpt:
            acc = results[(str(ck), sp)]
            name = f"{ck.parent.name}/{ck.stem}"
            cells = []
            for k in REPORT:
                v = acc[k]
                m = statistics.mean(v)
                s = (max(v) - min(v)) if len(v) > 1 else 0.0
                cells.append(f"{k.split('/')[-1]}={m:.4f}" + (f"(+-{s:.4f})" if len(v) > 1 else ""))
            print(f"  {name:<40} " + "  ".join(cells))


if __name__ == "__main__":
    main()
