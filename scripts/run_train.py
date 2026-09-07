"""MAPPO training driver.

    python scripts/run_train.py --n-envs 128 --total-steps 5_000_000 --out runs/run_001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from env.explorer import EnvCfg
from paths import RUNS_ROOT
from scripts.train_args import build_parser
from jsonio import jsonable
from train.driver import TrainCfg, train
from train.mappo import MAPPOCfg


class _Tee:
    """Duplicates writes to the real stream + a log file, so the web dashboard's live console
    works no matter how training was launched (web launcher already redirects stdout to a file
    itself; a bare CLI run previously had no train.log at all → 'Show log' button never appeared)."""

    def __init__(self, stream, logfile) -> None:
        self._stream = stream
        self._logfile = logfile

    def write(self, data) -> int:
        self._stream.write(data)
        self._logfile.write(data)
        self._logfile.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._logfile.flush()


def _unique_dir(base: Path) -> Path:
    """base if free, else base_2, base_3, ... — never returns a dir that already exists."""
    if not base.exists():
        return base
    n = 2
    while (cand := base.parent / f"{base.name}_{n}").exists():
        n += 1
    return cand


def main() -> None:
    ap = build_parser()
    args = ap.parse_args()
    out_was_explicit = args.out is not None

    # Auto-create a fresh run dir when --out is omitted: runs/<run-name|run>_<timestamp>,
    # collision-safe (suffix _2, _3, ... if that exact name is somehow already taken — e.g.
    # two launches within the same second). Never silently reuses/overwrites another run.
    if args.out is None:
        import time as _time
        _stamp = _time.strftime("%Y%m%d_%H%M%S")
        _base = (args.wandb_run_name or "run").strip().replace("/", "_") or "run"
        args.out = _unique_dir(RUNS_ROOT / f"{_base}_{_stamp}")
        print(f"[out] auto run dir → {args.out}")
    else:
        args.out = Path(args.out)

    # An EXPLICIT --out that already holds a PRIOR TRAINING (a checkpoint or a status.json) is
    # someone reusing a name on purpose (or by mistake) — never overwrite silently. Checked via
    # actual training artifacts, not "any file in the dir": the web launcher always drops
    # params.json + opens train.log into a fresh dir BEFORE spawning this process, so a plain
    # "non-empty" check would misfire on every single web-launched run.
    _has_prior_training = args.out.exists() and (
        any(args.out.glob("*.pt")) or (args.out / "status.json").exists()
    )
    if out_was_explicit and _has_prior_training:
        if args.force:
            print(f"[out] --force: overwriting existing run dir {args.out}")
        elif sys.stdin.isatty():
            resp = input(f"[out] '{args.out}' already holds a previous training. Overwrite? [y/N] ").strip().lower()
            if resp not in ("y", "yes"):
                print("[out] aborted.")
                sys.exit(1)
        else:
            print(f"[out] ERROR: '{args.out}' already holds a previous training. "
                  f"Pass --force to overwrite, or pick a different --out. Refusing to run non-interactively.")
            sys.exit(1)

    # IR2-style MANUAL curriculum: --stage picks a single split + its episode length and
    # disables any auto-advance. One stage per run; relaunch with the next --stage to advance.
    if args.stage is not None:
        if args.curriculum or args.curriculum_gated:
            ap.error("--stage (manual curriculum) is mutually exclusive with "
                     "--curriculum / --curriculum-gated (automatic). Pick one.")
        _STAGE_STEPS = {"easy": 196, "difficult": 384}      # IR2 values
        args.split = f"train/{args.stage}"
        args.max_episode_steps = _STAGE_STEPS[args.stage]
        print(f"[stage] MANUAL curriculum: stage='{args.stage}' "
              f"split='{args.split}' max_episode_steps={args.max_episode_steps}")

    # Mirror stdout/stderr into <out>/train.log so the web dashboard's live log console works
    # even for a bare CLI launch (previously train.log only existed when the web launcher had
    # redirected the subprocess's stdout to it → 'Show log' never appeared for CLI runs). Only
    # do this when stdout is a tty: the web launcher already redirects stdout to a real train.log
    # file itself, so tee-ing again here would duplicate every line.
    if sys.stdout.isatty():
        Path(args.out).mkdir(parents=True, exist_ok=True)
        _logf = open(Path(args.out) / "train.log", "a", buffering=1)
        _logf.write(f"\n$ {' '.join(sys.argv)}\n\n")
        sys.stdout = _Tee(sys.stdout, _logf)
        sys.stderr = _Tee(sys.stderr, _logf)

    # Same gap for params.json: the web launcher writes one (form params the dashboard's
    # "Params" button reads), but a bare CLI launch never did → the button never appeared for
    # those runs. Write the fully-resolved argparse values here too, unless the web launcher
    # already dropped its own params.json moments ago (don't clobber its richer {params, cmd}).
    _params_path = Path(args.out) / "params.json"
    if not _params_path.exists():
        Path(args.out).mkdir(parents=True, exist_ok=True)
        # jsonable(): --revisit-streak-cap defaults to +inf, and json.dumps emits the bare token
        # `Infinity` for it — readable by Python's json.load, rejected by JSON.parse, so every
        # params.json written so far is invalid JSON to the dashboard and to any external tool.
        _params_path.write_text(json.dumps(
            jsonable({"params": vars(args), "cmd": " ".join(sys.argv)}), indent=2, default=str))

    cfg = TrainCfg(
        split=args.split,
        out_dir=args.out,
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        n_agents=args.n_agents,
        rollout_len=args.rollout_len,
        n_hops=args.n_hops,
        lr_actor=args.lr,
        use_gru=args.gru and not args.no_gru,   # default OFF; --gru opts in, --no-gru forces off
        gat_actor=not (args.no_gat_actor or args.no_gat),
        gat_critic=not args.no_gat,
        device=args.device,
        seed=args.seed,
        compile=args.compile,
        init_ckpt=args.init_ckpt,
        eval_on_ckpt=args.eval_on_ckpt,
        eval_n_maps=args.eval_n_maps,
        eval_map_idx=args.eval_map_idx,
        eval_split=(
            args.eval_split if args.eval_split is not None
            else ("test/complex" if args.curriculum else args.split)
        ),
        eval_every=args.eval_every,
        eval_steps=(args.max_episode_steps if args.eval_steps < 0 else args.eval_steps),
        trace_steps=args.trace_steps,
        curriculum=args.curriculum,
        curriculum_gated=args.curriculum_gated,
        curriculum_stage_splits=tuple(
            s.strip() for s in args.curriculum_stage_splits.split(",") if s.strip()
        ),
        curriculum_stage_steps=tuple(
            int(s) for s in args.curriculum_stage_steps.split(",") if s.strip()
        ),
        curriculum_gate_score=args.curriculum_gate_score,
        curriculum_min_stage_iters=args.curriculum_min_stage_iters,
        eval_suite_splits=tuple(
            s.strip() for s in args.eval_suite_splits.split(",") if s.strip()
        ),
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=tuple(args.wandb_tags),
        score_w_imbalance=args.score_w_imbalance,
        score_w_overlap=args.score_w_overlap,
        score_w_idle=args.score_w_idle,
        env=EnvCfg(
            n_envs=args.n_envs,
            n_agents=args.n_agents,
            nr=16,                              # lattice spacing — 16px → N_max≈1200 nodes
            max_episode_steps=args.max_episode_steps,
            max_travel_px=args.max_travel_px,
            max_travel_frac=args.max_travel_frac,
            done_mode=args.done_mode,
            sensor_range_px=args.sensor_range,
            comm_range_px=args.comm_range,
            comm_model=args.comm_model,
            comm_relay=args.comm_relay,
            ss_thresh=args.ss_thresh,
            n_hops=args.n_hops,
            force_full_comm=args.force_full_comm,
            force_full_pos_sharing=args.force_full_pos_sharing,
            force_full_occupancy_sharing=args.force_full_occupancy_sharing,
            novel_scan_weight=args.novel_scan_weight,
            rdv_dense_weight=args.rdv_weight,
            rdv_offer_frac=args.rdv_offer_frac,
            rdv_clamp_pos=args.rdv_clamp_pos,
            rdv_urgency_T=args.rdv_urgency_T,
            comm_idle_pen=args.comm_idle_pen,
            # The env only pays for the overlap tensor when the loss will actually consume it.
            div_overlap=(args.div_weight > 0.0),
            rdv_urgency_mode=args.rdv_urgency_mode,
            rdv_urgency_start=args.rdv_urgency_start,
            completion_bonus=args.completion_bonus,
            step_penalty_coef=args.step_penalty,
            sync_give_weight=args.sync_weight,
            sync_weight_m_scale=args.sync_weight_m_scale,
            sync_recv_ratio=args.sync_recv_ratio,
            sync_min_gap=args.sync_min_gap,
            teammate_obs=not args.no_teammate_obs,
            vf_gamma=args.vf_gamma,
            revisit_penalty_coef=args.revisit_pen,
            revisit_window=args.revisit_window,
            stall_penalty_coef=args.stall_pen,
            stall_streak_beta=args.stall_streak_beta,
            stall_streak_cap=args.stall_streak_cap,
            revisit_streak_beta=args.revisit_streak_beta,
            revisit_streak_decay=args.revisit_streak_decay,
            revisit_streak_cap=args.revisit_streak_cap,
            radar_gamma=args.radar_gamma,
            radar_util_norm=args.radar_util_norm,
            belief_mode=args.belief_mode,
            pf_frontier_min_unknown=args.pf_frontier_min_unknown,
            pf_frontier_min_util=args.pf_frontier_min_util,
            radar_team_source=args.radar_team_source,
            map_seed=args.map_seed,
        ),
        ppo=MAPPOCfg(
            ent_coef=args.ent_coef,
            n_minibatches=args.minibatches,
            clip_eps=args.clip_eps,
            k_epochs=args.k_epochs,
            max_grad_norm=args.max_grad_norm,
            lam=args.gae_lambda,
            gamma=args.gamma,
            vf_coef=args.vf_coef,
            tbptt_steps=args.tbptt_steps,
            diag_grad=args.diag_grad,
            div_weight=args.div_weight,
        ),
    )
    train(cfg, log_every=1,
          ckpt_pct=() if args.no_milestone_ckpt else (20, 40, 60, 80, 100))


if __name__ == "__main__":
    main()
