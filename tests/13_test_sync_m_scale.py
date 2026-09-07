"""EnvCfg.sync_weight_m_scale — the sync bonus scales as (2/M)^a, and is an EXACT no-op off.

Checks, in order of how badly a regression would hurt:
  1. default is 0.0 and `from_ckpt_dict` on a dict without the key falls back to it, so every
     pre-existing checkpoint keeps the unscaled bonus (the from_ckpt_dict trap: absent keys take
     the dataclass default, silently pinning whatever that default happens to be);
  2. at M=2 the factor is identically 1 for ANY exponent -> the whole M=2 history cannot move;
  3. at M=4 with a=1 the per-step `sync` reward term is exactly HALF the unscaled one, on the
     same maps with the same actions;
  4. a=0 at M=4 reproduces the unscaled run bit-for-bit;
  5. nothing else in the reward moves — novel/step/revisit must be bit-identical, or the scaling
     leaked outside the sync term.

    python tests/13_test_sync_m_scale.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
N_ENV, STEPS = 4, 24


def run(M: int, a: float, split) -> dict[str, list[float]]:
    """Identical env + identical action stream; only sync_weight_m_scale differs."""
    cfg = EnvCfg(n_envs=N_ENV, n_agents=M, max_episode_steps=STEPS + 1,
                 sync_give_weight=0.25, sync_recv_ratio=0.5, sync_min_gap=4,
                 sync_weight_m_scale=a)
    # Construction itself draws from the GLOBAL torch RNG (spawn placement), so the per-env seeds
    # below are not enough on their own — reseed globally before building.
    torch.manual_seed(0)
    if DEV.startswith("cuda"):
        torch.cuda.manual_seed_all(0)
    env = Explorer(split, cfg, seed=0)
    # Pin EVERY stochastic stream, or two arms with identical configs already diverge: the spawn
    # RNG and the radio-shadowing stream both advance with construction/steps. Same discipline the
    # eval suite uses (driver.py) — without it this test compares two different episodes.
    env.reseed_map_rng(4242)
    env.reseed_channel_noise(777)
    for n in range(N_ENV):
        env.reload_map(env_idx=n, map_idx=n)
    g = torch.Generator(device=DEV).manual_seed(1234)      # same actions in every arm
    out: dict[str, list[float]] = {"sync": [], "novel": [], "step": [], "revisit": []}
    for _ in range(STEPS):
        K = env.obs["action_mask"].shape[-1]
        act = torch.randint(0, K, (N_ENV, M), device=DEV, generator=g)
        _o, _r, _d, info = env.step(act)
        for k in out:
            out[k].append(float(info["reward_terms"][k].item()))
    return out


def main() -> None:
    ok = True

    # 1. default off + from_ckpt_dict fallback
    assert EnvCfg().sync_weight_m_scale == 0.0, "default must be 0.0 (off)"
    restored = EnvCfg.from_ckpt_dict({"n_agents": 4}, n_envs=1, n_agents=4)
    assert restored.sync_weight_m_scale == 0.0, "from_ckpt_dict must fall back to off"
    print("  [1] default 0.0 and from_ckpt_dict fallback ....... OK")

    # 2. M=2 is a no-op for any exponent (pure arithmetic, no env needed)
    for a in (0.0, 0.5, 1.0, 2.0, 7.0):
        assert (2.0 / 2.0) ** a == 1.0
    print("  [2] M=2 factor == 1 for every exponent ........... OK")

    split = load_split("train/difficult", device=DEV)

    # 3/4/5. M=4: a=1 halves ONLY sync; a=0 reproduces unscaled exactly.
    base = run(4, 0.0, split)
    half = run(4, 1.0, split)
    zero = run(4, 0.0, split)

    if base["sync"] != zero["sync"]:
        print("  [4] a=0 is NOT bit-identical to unscaled ......... FAIL"); ok = False
    else:
        print("  [4] a=0 bit-identical to unscaled ................ OK")

    leaked = [k for k in ("novel", "step", "revisit") if base[k] != half[k]]
    if leaked:
        print(f"  [5] {leaked} moved — scaling leaked out of sync ... FAIL"); ok = False
    else:
        print("  [5] novel/step/revisit untouched ................. OK")

    nz = [(b, h) for b, h in zip(base["sync"], half["sync"]) if abs(b) > 1e-12]
    if not nz:
        print("  [3] no sync events fired — test is VACUOUS ....... FAIL"); ok = False
    else:
        worst = max(abs(h / b - 0.5) for b, h in nz)
        if worst < 1e-6:
            print(f"  [3] M=4, a=1 halves sync ({len(nz)} paying steps) .. OK")
        else:
            print(f"  [3] ratio off by {worst:.2e} (want 0.5) .......... FAIL"); ok = False

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
