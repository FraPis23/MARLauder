#!/usr/bin/env python3
"""Paired A/B of the multi-hop relay on ONE checkpoint: does it explore more, sooner, for less travel?

    python scripts/ab_relay.py --ckpt runs/v19_m4_20260811_122528/ckpt_stop.pt \\
        --split test/complex --n-agents 4

Answers the three questions the eval suite does NOT: it scores coverage and time but never reports
DISTANCE, which is IR2's headline metric (max over robots of path length) and half of what "better
exploration" means here.

Design notes, because a sloppy A/B on this env is worse than none:

  * PAIRED. Both arms run the SAME map indices, and every statistic is reported as a per-map
    difference. Map luck on test/complex dwarfs the effect being measured, so comparing two
    independent means would mostly measure which maps each arm happened to draw.
  * RADIO NOISE PINNED. reseed_channel_noise() is called with the same seed at the start of each
    arm. The per-episode shadowing draw otherwise depends on how many actions the policy sampled
    earlier, and it moves comm duty and sync counts by ~30% between two evaluations of the SAME
    checkpoint.
  * FROZEN AT FIRST DONE. step() auto-resets a finished env, so every counter is masked by `alive`
    and stops at the episode's own end. Otherwise a fast arm gets a second episode folded in.
  * N>1 envs. A batch-dim bug in the env is invisible at P=1.

Caveat to keep in mind when reading the output: the checkpoint was trained WITHOUT the relay, so
this measures ZERO-SHOT transfer. It is a lower bound on what the feature is worth — a policy
trained with it could position a robot as a bridge, which this one has no reason to do.

MEASURED NOISE FLOOR — RUN `--control` BEFORE BELIEVING ANY NUMBER BELOW IT.
`--control` runs BOTH arms with the relay off, so every delta is zero by construction. On
v19 ckpt_stop / M=4 / test/complex / 32 maps / 768 steps / fp32 it printed:

    success_rate    -11.1%  (1/32 "better")     steps_to_90     -3.8%
    own_cov_min      -1.4%                      travel_px_max   -0.9%
    offer_frac       +7.5%                      comm_duty       +2.3%
    one_flock_frac   +3.2%                      transit_frac    -0.3%

That is the resolution of this bench, and `--fp32` does NOT remove it — three identical bf16
A/B invocations had already disagreed in SIGN (success -25.0% vs -3.8%; travel +2.3% vs -2.1%),
and the OFF arm's own success_rate moved 0.875 -> 0.8125 between runs of the SAME config. A
single lattice-hop difference early in an episode sends a chaotic system somewhere else entirely.

So: an effect under ~10 points of success_rate, ~4% of steps_to_90 or ~1% of travel CANNOT be
claimed from one run. Either it clears the floor by a wide margin (the relay's connectivity gain
does: one_flock_frac +132..149% against a +3.2% floor) or it needs many more maps and seeds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from env.explorer import EnvCfg, Explorer  # noqa: E402
from env.maps import load_split  # noqa: E402
from eval.ckpt_loader import load_model_from_ckpt  # noqa: E402

PRODUCTIVE, REDUNDANT, TRANSIT = 0, 1, 2


@torch.no_grad()
def run_arm(model, env: Explorer, map_idxs: list[int], steps: int, noise_seed: int,
            amp: bool = True) -> dict:
    """Deterministic episodes on the given maps. Returns per-map arrays, all frozen at first done."""
    N, M = env.N, env.M
    dev = env.dev
    out = {k: [] for k in ("explored", "own_min", "own_mean", "success", "s90", "s90_own",
                           "travel_max", "travel_mean", "steps", "overlap", "duty", "connected",
                           "syncs", "productive", "redundant", "transit",
                           "g", "staleness", "contact", "offer_frac")}
    for b in range(0, len(map_idxs), N):
        chunk = map_idxs[b:b + N]
        for i, midx in enumerate(chunk):
            env.reload_map(env_idx=i, map_idx=int(midx))
        # Pin the shadowing stream per BATCH, identically in both arms.
        env.reseed_channel_noise(noise_seed + b)
        h_act, h_crit = model.init_hidden(N, str(dev))
        obs = env.obs
        alive = torch.ones(N, dtype=torch.bool, device=dev)
        z = lambda: torch.zeros(N, dtype=torch.float64, device=dev)  # noqa: E731
        expl, own_min, own_mean = z(), z(), z()
        tmax, tmean, nsteps = z(), z(), z()
        # AGENT_SCALAR_DIM order (models/actor_critic.py): [g, staleness, travel_frac, contact,
        # offer_frac]. g is the rendezvous surplus gate: it is what tells the actor "I owe this
        # teammate map, go meet him", and it is driven by _own_expl_at_comm, which the relay
        # resets on GROUP contact. If the relay quietly zeroes g without actually delivering the
        # map (chains are rare), the policy stops seeking the physical rendezvous that does.
        g_s, stale_s, cont_s, off_s = z(), z(), z(), z()
        succ, s90, s90o = z(), torch.full((N,), float(steps), device=dev), \
            torch.full((N,), float(steps), device=dev)
        ov, duty, conn, syn = z(), z(), z(), z()
        buckets = torch.zeros((N, 3), dtype=torch.float64, device=dev)
        for t in range(steps):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=amp and str(dev).startswith("cuda")):
                o = model.act(obs, h_act, h_crit, deterministic=True)
            h_act, h_crit = o["hidden_actor"], o["hidden_critic"]
            obs, _r, done, info = env.step(o["action"])
            a = alive.to(torch.float64)
            am = alive.view(N, 1).expand(N, M)

            # Latched "value at this episode's last live step".
            er = info["explored_rate"].double()
            oc = info["own_cov"].double()
            expl = torch.where(alive, er, expl)
            own_min = torch.where(alive, oc.amin(1), own_min)
            own_mean = torch.where(alive, oc.mean(1), own_mean)
            tv = info["travel_px"].double()
            tmax = torch.where(alive, tv.amax(1), tmax)
            tmean = torch.where(alive, tv.mean(1), tmean)
            nsteps += a
            s90 = torch.where(alive & (er >= 0.9) & (s90 >= steps),
                              torch.full_like(s90, float(t + 1)), s90)
            s90o = torch.where(alive & (oc.amin(1) >= 0.9) & (s90o >= steps),
                               torch.full_like(s90o, float(t + 1)), s90o)
            succ = torch.where(alive & done, info["terminated"].double(), succ)

            # Per-step means need the per-ENV value; info["metrics"] is already reduced over N, so
            # take the env-resolved quantities from the raw masks instead.
            eye = torch.eye(M, dtype=torch.bool, device=dev).view(1, M, M)
            pd = torch.cdist(env.pos, env.pos)
            triu = torch.triu(torch.ones(M, M, device=dev), diagonal=1).bool()
            ov += (pd[:, triu] < 2.0 * env.cfg.sensor_range_px).double().mean(-1) * a
            duty += (info["comm_mask"] & ~eye).double().flatten(1).mean(-1) * a
            conn += info["comm_group"][:, 0, :].all(-1).double() * a
            syn += (info["sync_paid"].amax(1) > 0).double() * a

            sc = obs["agent_scalars"].double()                          # [N, M, 5]
            g_s += sc[..., 0].mean(-1) * a
            stale_s += sc[..., 1].mean(-1) * a
            cont_s += sc[..., 3].mean(-1) * a
            off_s += sc[..., 4].mean(-1) * a

            bk = info["idle_bucket"].long()
            oh = torch.zeros((N, M, 3), dtype=torch.float64, device=dev)
            oh.scatter_(2, bk.unsqueeze(-1), 1.0)
            buckets += (oh * am.unsqueeze(-1).double()).sum(1)

            alive = alive & ~done
            if not bool(alive.any()):
                break
        n = nsteps.clamp(min=1)
        for k, v in (("explored", expl), ("own_min", own_min), ("own_mean", own_mean),
                     ("success", succ), ("s90", s90.double()), ("s90_own", s90o.double()),
                     ("travel_max", tmax), ("travel_mean", tmean), ("steps", nsteps),
                     ("overlap", ov / n), ("duty", duty / n), ("connected", conn / n),
                     ("syncs", syn), ("g", g_s / n), ("staleness", stale_s / n),
                     ("contact", cont_s / n), ("offer_frac", off_s / n)):
            out[k].append(v[:len(chunk)].cpu())
        tot = buckets.sum(-1).clamp(min=1)
        for k, i in (("productive", PRODUCTIVE), ("redundant", REDUNDANT), ("transit", TRANSIT)):
            out[k].append((buckets[:, i] / tot)[:len(chunk)].cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def build(ckpt_env: dict, split, args, relay: bool) -> Explorer:
    cfg = EnvCfg.from_ckpt_dict(dict(ckpt_env or {}), n_envs=args.n_envs, n_agents=args.n_agents,
                                map_seed=args.seed, comm_relay=relay)
    return Explorer(split, cfg, seed=args.seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--split", default="test/complex")
    ap.add_argument("--n-agents", type=int, required=True)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-maps", type=int, default=32)
    ap.add_argument("--steps", type=int, default=0, help="0 = the checkpoint's max_episode_steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="NULL EXPERIMENT: run BOTH arms with the relay off. Every number must "
                         "come out 0. Whatever it prints instead is this harness's noise floor, "
                         "and no smaller effect from the real A/B can be believed.")
    ap.add_argument("--fp32", action="store_true",
                    help="Disable bf16 autocast. The policy is already deterministic and the maps "
                         "and radio noise are pinned, so under fp32 two identical configs should "
                         "trace bit-identical episodes; under bf16 a 1-ulp logit difference flips "
                         "an argmax and the trajectories diverge completely from there.")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    assert args.n_envs > 1

    model, env_peek = load_model_from_ckpt(args.ckpt, args.device, n_agents=args.n_agents)
    model.eval()
    split = load_split(args.split, device=args.device)
    n_split = int(getattr(split, "n", 0)) or args.n_maps
    k = min(args.n_maps, n_split)
    map_idxs = [int(round(i * (n_split - 1) / max(1, k - 1))) for i in range(k)]

    arms = (("OFF", False), ("ON", False if args.control else True))
    envs = {name: build(env_peek, split, args, relay) for name, relay in arms}
    steps = args.steps or int(envs["OFF"].cfg.max_episode_steps)
    mode = "NULL CONTROL (both arms relay OFF)" if args.control else "A/B multi-hop relay"
    print(f"\n=== {mode}   ckpt={args.ckpt.parent.name}/{args.ckpt.name}")
    print(f"    M={args.n_agents}  split={args.split}  maps={k}  steps<={steps}  "
          f"N={args.n_envs} envs  PAIRED, radio noise pinned, "
          f"{'fp32' if args.fp32 else 'bf16 autocast'}")
    if args.control:
        print(f"    Every delta below MUST be 0. Anything else is the noise floor.\n")
    else:
        print(f"    NOTE zero-shot: the checkpoint trained WITHOUT the relay.\n")

    res = {}
    for name, env in envs.items():
        res[name] = run_arm(model, env, map_idxs, steps, noise_seed=args.seed + 777,
                            amp=not args.fp32)
        print(f"    arm {name} done")

    # Higher is better for the first block, lower for the second.
    HIGHER = [("explored", "explored_final"), ("own_min", "own_cov_min"),
              ("own_mean", "own_cov_mean"), ("success", "success_rate"),
              ("connected", "one_flock_frac"), ("duty", "comm_duty"), ("syncs", "syncs/ep"),
              ("productive", "productive_frac")]
    LOWER = [("s90", "steps_to_90"), ("s90_own", "steps_to_90_own"),
             ("travel_max", "travel_px_max"), ("travel_mean", "travel_px_mean"),
             ("steps", "episode_steps"), ("overlap", "sensing_overlap"),
             ("transit", "transit_frac"), ("redundant", "redundant_frac")]

    print(f"\n{'metric':<20s} {'OFF':>10s} {'ON':>10s} {'delta':>10s} {'delta %':>9s} "
          f"{'maps better':>12s}")
    print("-" * 76)

    def row(key: str, label: str, higher: bool) -> None:
        a, b = res["OFF"][key].double(), res["ON"][key].double()
        d = (b - a)
        ma, mb = float(a.mean()), float(b.mean())
        rel = (mb - ma) / abs(ma) * 100.0 if abs(ma) > 1e-12 else float("nan")
        better = int(((d > 0) if higher else (d < 0)).sum())
        print(f"{label:<20s} {ma:>10.4f} {mb:>10.4f} {float(d.mean()):>+10.4f} "
              f"{rel:>+8.1f}% {better:>7d}/{len(d)}")

    for key, label in HIGHER:
        row(key, label, higher=True)
    print()
    for key, label in LOWER:
        row(key, label, higher=False)

    # Not "better/worse" — these are the actor's rendezvous inputs, printed to explain any change
    # in the outcomes above. A collapse in g/offer_frac with a rise in contact is the signature of
    # the relay satisfying the "we are talking" signal without delivering the map.
    print(f"\n  actor rendezvous scalars (diagnostic, no better/worse direction)")
    for key in ("g", "offer_frac", "staleness", "contact"):
        a, b = res["OFF"][key].double(), res["ON"][key].double()
        ma, mb = float(a.mean()), float(b.mean())
        rel = (mb - ma) / abs(ma) * 100.0 if abs(ma) > 1e-12 else float("nan")
        print(f"{key:<20s} {ma:>10.4f} {mb:>10.4f} {mb - ma:>+10.4f} {rel:>+8.1f}%")

    # Coverage per unit of distance — the efficiency number the whole diagnosis pointed at.
    print()
    for name in ("OFF", "ON"):
        r = res[name]
        cpd = (r["explored"] / r["travel_max"].clamp(min=1)).mean()
        print(f"    coverage_per_travel_px ({name}) = {float(cpd):.3e}")

    # Paired significance, the cheap honest version: a sign test on the per-map differences of the
    # two headline outcomes. With ~32 maps, an effect that flips fewer than ~22 of them is not
    # distinguishable from map noise at this sample size.
    print()
    for key, label in (("explored", "explored_final"), ("travel_max", "travel_px_max")):
        d = (res["ON"][key] - res["OFF"][key]).double()
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        print(f"    sign test {label:<16s} ON better on {pos}, worse on {neg}, ties "
              f"{len(d) - pos - neg}")
    print()


if __name__ == "__main__":
    main()
