#!/usr/bin/env python3
"""Frontier-diversity loss: the overlap tensor must mean what the loss assumes it means.

    python tests/14_test_frontier_div.py

The loss (train/mappo.py, MAPPOCfg.div_weight) is

    L = mean_over_ordered_pairs  sum_{k,l} pi_i(k) . O[i,j,k,l] . pi_j(l)

so it is only as sound as O = obs["div_overlap"] (env/explorer.py::_branch_overlap), which claims
to be "the discounted frontier mass agent i's exit k and agent j's exit l lead to IN COMMON".
Assertions, in the order in which a wrong O would do damage:

    1. OFF is an exact no-op — the key is absent, so the buffer allocates nothing and the update
       never sees it. This is what lets every pre-existing checkpoint replay unchanged.
    2. shape/symmetry: O[i,j,k,l] == O[j,i,l,k]. Both index orders appear in the pair mean.
    3. the arithmetic is the arithmetic — a synthetic label/mass pair with a hand-computed answer,
       then the SAME pair scaled x100. Each agent is normalised to unit mass BEFORE the product,
       so an agent standing in a utility-rich region cannot dominate the term just by having more
       mass, and the number stays comparable across steps, maps and M.
    3b. on a real map every pair obeys Cauchy-Schwarz, or the term is not an inner product.
    4. IDENTICAL STATE (same node AND same belief) gives cross-overlap EXACTLY equal to
       self-overlap: the top of the scale is real and reachable — the case the loss exists to
       punish. Agents left in their own spawn state must sit strictly below it.
       NOTE co-location alone is NOT identical state: each agent carries its own occupancy, seeded
       at its own spawn, so two agents on the same node still hold different maps and therefore
       different utility fields. That is precisely the asymmetry the loss cannot see and does not
       need to — it prices the shared TARGET, not the shared position.
    5. the gradient does what it is for: with mass shared only between i's exit a and j's exit b,
       descent must drive pi_i(a).pi_j(b) down. Checked on the real loss expression.

N>1 envs throughout: a batch-dim bug is invisible at P=1 (a gather on dim 1 killed v15 at it=0).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

from env.explorer import EnvCfg, Explorer  # noqa: E402
from env.maps import load_split  # noqa: E402

DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
N_ENVS, M = 4, 3
SPLIT = "train/difficult"
MAP_IDX = 50
TOL = 1e-4


def build(div: bool, identical: bool = False) -> Explorer:
    cfg = EnvCfg(n_envs=N_ENVS, n_agents=M, n_hops=6, sensor_range_px=80.0,
                 max_episode_steps=500, div_overlap=div, map_seed=7)
    split = load_split(SPLIT, device=DEV)
    env = Explorer(split, cfg, seed=3)
    for n in range(N_ENVS):
        env.reload_map(env_idx=n, map_idx=MAP_IDX + n)
    if identical:
        # Identical state has TWO halves and both are needed. The position is the tree ROOT; the
        # per-agent occupancy is the tree itself. force_full_occupancy_sharing would not do it
        # here: it acts inside step() at fusion time, so it has no effect on a freshly reset env.
        # Copy both from agent 0 and refresh — now the M trees are literally the same object.
        env.pos[:, 1:] = env.pos[:, :1]
        env.world.occupancy_torch[:, 1:] = env.world.occupancy_torch[:, :1]
        env._refresh_obs()
    return env


def main() -> None:
    fail = 0

    # ---- 1. OFF is an exact no-op -------------------------------------------------------------
    off = build(div=False)
    if "div_overlap" in off.obs:
        print("FAIL 1: div_overlap present with EnvCfg.div_overlap=False"); fail += 1
    else:
        print("ok 1  OFF: key absent -> buffer allocates nothing, update never sees it")
    del off

    env = build(div=True)
    O = env.obs["div_overlap"]
    K = env.K

    # ---- 2. shape + symmetry ------------------------------------------------------------------
    want = (N_ENVS, M, M, K, K)
    if tuple(O.shape) != want:
        print(f"FAIL 2: shape {tuple(O.shape)} != {want}"); fail += 1
    else:
        asym = (O - O.permute(0, 2, 1, 4, 3)).abs().max().item()
        if asym > TOL:
            print(f"FAIL 2: O[i,j,k,l] != O[j,i,l,k], max |delta| = {asym:.3e}"); fail += 1
        else:
            print(f"ok 2  shape {want}, symmetric under (i,k)<->(j,l) to {asym:.1e}")

    # ---- 3. the arithmetic, on a synthetic case with a known answer ---------------------------
    B, V = N_ENVS * M, env.N_max
    lab = torch.full((B, V), -1, dtype=torch.long, device=DEV)
    mas = torch.zeros((B, V), device=DEV)
    # agent 0: all mass on node 5, reached through exit 0.
    # agent 1: half on node 5 through exit 3, half on node 9 through exit 4.
    # agent 2: everything on node 9 through exit 7 -> shares with agent 1, never with agent 0.
    for n in range(N_ENVS):
        i0, i1, i2 = n * M, n * M + 1, n * M + 2
        lab[i0, 5] = 0; mas[i0, 5] = 2.0
        lab[i1, 5] = 3; mas[i1, 5] = 1.0
        lab[i1, 9] = 4; mas[i1, 9] = 1.0
        lab[i2, 9] = 7; mas[i2, 9] = 4.0
    Osyn = env._branch_overlap(lab, mas)
    exp01, exp12 = 1.0 * 0.5, 0.5 * 1.0
    got01 = Osyn[0, 0, 1, 0, 3].item()
    got12 = Osyn[0, 1, 2, 4, 7].item()
    tot02 = Osyn[0, 0, 2].sum().item()
    ok3 = (abs(got01 - exp01) < TOL and abs(got12 - exp12) < TOL and tot02 < TOL
           and abs(Osyn[0, 0, 1].sum().item() - exp01) < TOL)
    Oscaled = env._branch_overlap(lab, mas * 100.0)
    scale_err = (Osyn - Oscaled).abs().max().item()
    if not ok3:
        print(f"FAIL 3: O[0,1,0,3]={got01:.4f} (want {exp01}), O[1,2,4,7]={got12:.4f} "
              f"(want {exp12}), disjoint pair total={tot02:.2e} (want 0)"); fail += 1
    elif scale_err > TOL:
        print(f"FAIL 3: not scale-free — x100 changes O by {scale_err:.3e}"); fail += 1
    else:
        print(f"ok 3  arithmetic exact (shared {got01:.3f} / {got12:.3f}, disjoint pair 0), "
              f"invariant to a x100 mass rescale ({scale_err:.1e})")

    # ---- 3b. Cauchy-Schwarz on a real map -----------------------------------------------------
    tot = O.sum(dim=(-1, -2))                                                       # [N, M, M]
    dg = torch.stack([tot[:, m, m] for m in range(M)], dim=1)                       # [N, M]
    viol = (tot ** 2) - dg.unsqueeze(2) * dg.unsqueeze(1) * (1.0 + 1e-4)
    if bool((viol > 0).any()) or not bool(torch.isfinite(tot).all()):
        print("FAIL 3b: Cauchy-Schwarz violated (or non-finite) on a real map"); fail += 1
    else:
        n_desert = int((dg <= TOL).sum())
        print(f"ok 3b Cauchy-Schwarz holds for all {N_ENVS * M * M} pairs, all finite "
              f"({n_desert} desert agent(s) at exactly 0)")

    # ---- 4. identical state -> maximal overlap ------------------------------------------------
    co = build(div=True, identical=True)
    Oc = co.obs["div_overlap"]
    cross = Oc[:, 0, 1].sum(dim=(-1, -2))
    self0 = Oc[:, 0, 0].sum(dim=(-1, -2))
    live = self0 > TOL
    if not bool(live.any()):
        print("SKIP 4: every agent in a desert on these maps")
    else:
        ratio = cross[live] / self0[live]
        if abs(ratio.min().item() - 1.0) > 1e-3 or abs(ratio.max().item() - 1.0) > 1e-3:
            print(f"FAIL 4: identical-state cross/self = "
                  f"{ratio.min().item():.4f}..{ratio.max().item():.4f} (want 1)"); fail += 1
        else:
            print(f"ok 4  identical state (same node + same belief): cross/self = "
                  f"{ratio.min().item():.5f}..{ratio.max().item():.5f}")
    sep, sep_self = O[:, 0, 1].sum(dim=(-1, -2)), O[:, 0, 0].sum(dim=(-1, -2))
    liv = sep_self > TOL
    if bool(liv.any()):
        r = sep[liv] / sep_self[liv]
        print(f"      same pair spawned apart: {r.min().item():.4f}..{r.max().item():.4f} "
              f"-> the metric discriminates")
        if r.min().item() >= 0.999:
            print("FAIL 4b: separated agents overlap as much as identical ones"); fail += 1

    # ---- 5. the gradient moves probability off the shared branch ------------------------------
    a, b = 2, 5
    Os = torch.zeros((1, 2, 2, K, K), device=DEV)
    Os[0, 0, 1, a, b] = 1.0
    Os[0, 1, 0, b, a] = 1.0
    logits = torch.zeros((1, 2, K), device=DEV, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=5.0)
    pair_off = ~torch.eye(2, dtype=torch.bool, device=DEV)
    p0 = torch.softmax(logits, -1).detach()
    before = (p0[0, 0, a] * p0[0, 1, b]).item()
    for _ in range(50):
        p = torch.softmax(logits, -1)
        ov = torch.einsum("nik,nijkl,njl->nij", p, Os, p)
        loss = ov[:, pair_off].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    p1 = torch.softmax(logits, -1).detach()
    after = (p1[0, 0, a] * p1[0, 1, b]).item()
    if not (after < before * 0.5):
        print(f"FAIL 5: pi_0({a})*pi_1({b}) {before:.4f} -> {after:.4f}, want a clear drop")
        fail += 1
    else:
        others = p1[0, 0].clone(); others[a] = 0.0
        print(f"ok 5  descent moves the shared pair {before:.4f} -> {after:.4f}; mass went to "
              f"the free exits (max other = {others.max().item():.3f})")

    print("\nALL PASS" if fail == 0 else f"\n{fail} FAILURE(S)")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
