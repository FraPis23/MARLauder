"""MAPPO update with chunked encoder + truncated BPTT.

NOTE on TBPTT: the GRUCells are OFF by default (TrainCfg.use_gru=False, --gru to opt in), so on the
shipped config there is no recurrence to back-propagate through and tbptt_steps only controls how
the rollout is chunked for the encoder. The chunking machinery is kept because it is what makes the
--gru arm trainable at all, and because chunk size still drives update-time memory.

v0.2 changes vs v0.1:
- Encoder is called ONCE per TBPTT chunk on the reshaped [T*N*M, N_max, F] batch
  (instead of T separate forward passes). Cuts 4-10× of update wall time.
- Minibatches over the N axis: each PPO epoch splits envs into K groups and
  optimizes each group separately. Better gradient diversity at large N.
- Single optimizer + single GradScaler (no actor/critic split).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.value_normalizer import ValueNormalizer
from train.buffer import Rollout

# AMP dtype: bf16 on Ampere+ (same fp32 range → value-target spikes can't overflow → no
# GradScaler, no NaN-skip collapses), fp16 + GradScaler as the pre-Ampere fallback.
AMP_DTYPE = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)


@dataclass
class MAPPOCfg:
    gamma: float = 0.99
    lam: float = 0.95
    # clip ε 0.15 (paper says ≤0.2; we run tighter because this task is MORE non-stationary
    # than the paper's benchmarks — intra-episode obs shift + dense evolving shaping → smaller
    # policy drift per update is safer). k_epochs stays low (4) for the same reason.
    clip_eps: float = 0.15
    # ---- FRONTIER-DIVERSITY auxiliary loss. Weight on
    #   E_{k~pi_i, l~pi_j} [ shared discounted frontier mass down i's exit k and j's exit l ]
    # averaged over ordered agent pairs, using obs["div_overlap"] built by the env
    # (EnvCfg.div_overlap — turn BOTH on or this is silently inert).
    # It is an ACTOR loss, not a reward: the return, the advantage and the critic target are
    # untouched, which is what v18 got wrong when it moved the reward budget at M=4.
    # It is over FRONTIER mass, not action indices: two agents 300 px apart share no action index,
    # which is why the v17/J.1/J.2 local-logit port was identically zero exactly when the
    # duplication was being decided. The term is also self-extinguishing — when the reachable
    # frontier sets are disjoint the overlap is 0 and the gradient vanishes, so it stops pushing
    # once the team has actually split.
    div_weight: float = 0.0
    ent_coef: float = 0.01
    # Once per iteration, backprop the policy-gradient term and the entropy bonus SEPARATELY and
    # log ||grad|| of each. Comparing the two loss VALUES is not enough to claim one dominates:
    # pg is a clipped surrogate whose value can be near zero while its gradient is not. Costs two
    # extra backward passes on one chunk per iteration; off by default.
    diag_grad: bool = False
    vf_coef: float = 0.5
    k_epochs: int = 4
    tbptt_steps: int = 16
    n_minibatches: int = 1
    # Paper Tab.7 uses 10.0; we keep a tighter leash (2.0) because the dense shaping reward
    # spikes gradients — but 0.5 was throttling learning, so 2.0 is the middle ground.
    max_grad_norm: float = 2.0
    use_amp: bool = True
    # MAPPO paper §3.3 / Alg.1 — clipped value loss (max of unclipped and
    # V_old-clipped squared error). Clip range reuses clip_eps, in the
    # value-normalized space (same space as the MSE target returns_norm).
    clip_vloss: bool = True
    # Huber delta for the value loss (paper Tab.7 = 10.0). 0.0 → plain squared
    # error. Robust to value-target outliers; with value normalization it rarely
    # triggers but caps the gradient on return spikes.
    huber_delta: float = 10.0


def _value_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    v_old: torch.Tensor,
    clip_eps: float,
    clip_vloss: bool,
    huber_delta: float,
) -> torch.Tensor:
    """PPO value loss. Optionally V_old-clipped (max of the two errors) and/or Huber.

    All tensors are in the value-normalized space. `v_old` is the rollout-time
    critic output (re-normalized by the caller).
    """
    def err(x: torch.Tensor) -> torch.Tensor:
        if huber_delta > 0.0:
            return F.huber_loss(x, target, delta=huber_delta, reduction="none")
        return (x - target) ** 2

    loss_unclipped = err(pred)
    if not clip_vloss:
        return loss_unclipped.mean()
    pred_clipped = v_old + (pred - v_old).clamp(-clip_eps, clip_eps)
    loss_clipped = err(pred_clipped)
    return torch.max(loss_unclipped, loss_clipped).mean()


def _slice_chunk_obs(obs: dict, t0: int, t1: int, env_idx: torch.Tensor) -> dict:
    return {k: v[t0:t1, env_idx] for k, v in obs.items()}


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vnorm: ValueNormalizer,
    rollout: Rollout,
    cfg: MAPPOCfg,
    device: str,
) -> dict:
    T, N, M, d = rollout.T, rollout.N, rollout.M, rollout.d_hidden
    K = rollout.K
    advantages, returns = rollout.compute_gae(gamma=cfg.gamma, lam=cfg.lam)
    # advantages [T, N, M], returns [T, N] (team-mean target for shared V).
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # EXPLAINED VARIANCE of the value function, on the data the critic has NOT yet been updated on
    # — so it measures prediction, not fit. ev = 1 - Var(returns - V)/Var(returns):
    #   ~1.0  the critic explains the returns; advantages are informative
    #   ~0.0  no better than predicting the mean return
    #   <0    worse than the mean, i.e. the advantages fed to the policy gradient are mostly noise
    # v_loss alone cannot say this: it is an absolute error whose scale depends on the return
    # scale, so it is not comparable between M=2 and M=4 runs, and a "flat low" v_loss is equally
    # consistent with a good critic and with a critic that has given up and predicts the mean.
    with torch.no_grad():
        _r = returns.flatten()
        _v = rollout.values[:T].flatten()
        _vr = _r.var()
        explained_var = float(1.0 - (_r - _v).var() / _vr.clamp(min=1e-8)) if _vr > 1e-8 else 0.0
    vnorm.update(returns)
    returns_norm = vnorm.normalize(returns)
    use_amp = cfg.use_amp and device.startswith("cuda")
    # GradScaler only exists to keep fp16 gradients out of the denormal range; bf16 shares the
    # fp32 exponent so scaling is pointless — disabled, every scaler call becomes a passthrough.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and AMP_DTYPE is torch.float16)

    n_mb = max(1, cfg.n_minibatches)
    assert N % n_mb == 0, f"n_envs={N} must be divisible by n_minibatches={n_mb}"
    mb_size = N // n_mb

    stats = {"pg_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0,
             "div_loss": 0.0,
             "updates": 0, "nan_skips": 0}
    diag_done = False        # cfg.diag_grad: measure the two gradient norms once per iteration

    for epoch in range(cfg.k_epochs):
        # Shuffle env indices each epoch for minibatching.
        perm = torch.randperm(N, device=device)
        for mb in range(n_mb):
            env_idx = perm[mb * mb_size:(mb + 1) * mb_size]
            mb_actor_h0  = rollout.h_actor_init[env_idx].detach()        # [Nmb, M, d]
            mb_critic_h0 = rollout.h_critic_init[env_idx].detach()       # [Nmb, d]
            h_act = mb_actor_h0
            h_crit = mb_critic_h0

            for c0 in range(0, T, cfg.tbptt_steps):
                c1 = min(T, c0 + cfg.tbptt_steps)
                chunk_len = c1 - c0
                h_act = h_act.detach()
                h_crit = h_crit.detach()
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=use_amp):
                    # Slice chunk obs along T and N axes.
                    chunk_obs = _slice_chunk_obs(rollout.obs, c0, c1, env_idx)
                    if model.use_encoder:
                        # ONE encoder pass for the whole chunk.
                        enc = model.encode_chunk(chunk_obs)
                        curr_emb_chunk = enc["curr_emb"]                   # [T_chunk, Nmb, M, d]
                        nbr_embs_chunk = enc["nbr_embs"]                   # [T_chunk, Nmb, M, K, d]
                    else:
                        # --no-gat: no encoder at all; the critic embedding is the cheap
                        # raw-feature mean⊕max projection (actor side is VF-only anyway).
                        curr_emb_chunk = model.critic_feat_emb(
                            chunk_obs["node_feat"], chunk_obs["node_valid"])  # [T_chunk, Nmb, M, d]
                        nbr_embs_chunk = None

                    chunk_pg = 0.0; chunk_vl = 0.0; chunk_ent = 0.0
                    chunk_clip = 0.0; chunk_kl = 0.0; chunk_div = 0.0
                    div_on = cfg.div_weight > 0.0 and "div_overlap" in chunk_obs and M > 1
                    if div_on:
                        _pair_off = ~torch.eye(M, dtype=torch.bool, device=device)
                    last_h_act = h_act
                    last_h_crit = h_crit
                    for tt in range(chunk_len):
                        t = c0 + tt
                        action_t = rollout.actions[t, env_idx]              # [Nmb, M]
                        old_logp_t = rollout.logp[t, env_idx]
                        adv_t = advantages[t, env_idx]                       # [Nmb, M] per-agent
                        ret_t = returns_norm[t, env_idx]
                        action_mask_t = chunk_obs["action_mask"][tt]        # [Nmb, M, K]

                        if t > c0:
                            nonterm = (~rollout.dones[t - 1, env_idx]).float()
                            last_h_act = last_h_act * nonterm.view(-1, 1, 1)
                            last_h_crit = last_h_crit * nonterm.view(-1, 1)

                        ev = model.evaluate_step_from_enc(
                            curr_emb=curr_emb_chunk[tt],
                            nbr_embs=nbr_embs_chunk[tt] if nbr_embs_chunk is not None else None,
                            action_mask=action_mask_t,
                            action=action_t,
                            hidden_actor=last_h_act,
                            hidden_critic=last_h_crit,
                            prev_action=chunk_obs["prev_action"][tt],
                            agent_scalars=chunk_obs["agent_scalars"][tt],
                            value_field=chunk_obs["value_field"][tt],
                            critic_global=chunk_obs["critic_global"][tt] if "critic_global" in chunk_obs else None,
                        )
                        new_logp = ev["logp"]
                        new_val = ev["value"]
                        new_ent = ev["entropy"]
                        last_h_act = ev["hidden_actor"]
                        last_h_crit = ev["hidden_critic"]

                        ratio = (new_logp - old_logp_t).exp()
                        unclipped = ratio * adv_t
                        clipped = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                        pg = -torch.min(unclipped, clipped).mean()
                        # V_old (rollout-time critic output, stored denormalized)
                        # → renormalize into the target space for clipped value loss.
                        v_old_norm = vnorm.normalize(rollout.values[t, env_idx])
                        vl = _value_loss(
                            new_val, ret_t, v_old_norm,
                            clip_eps=cfg.clip_eps,
                            clip_vloss=cfg.clip_vloss,
                            huber_delta=cfg.huber_delta,
                        )
                        ent = new_ent.mean()
                        if div_on:
                            # ov[n,i,j] = sum_{k,l} pi_i(k) O[n,i,j,k,l] pi_j(l): the mass agents i
                            # and j are both heading for, under their CURRENT policies. O carries no
                            # gradient (it is a property of the state); both policies do.
                            p_t = ev["probs"].to(torch.float32)                 # [Nmb, M, K]
                            O_t = chunk_obs["div_overlap"][tt].to(torch.float32)
                            ov = torch.einsum("nik,nijkl,njl->nij", p_t, O_t, p_t)
                            chunk_div = chunk_div + ov[:, _pair_off].mean()
                        chunk_pg = chunk_pg + pg
                        chunk_vl = chunk_vl + vl
                        chunk_ent = chunk_ent + ent
                        with torch.no_grad():
                            chunk_clip = chunk_clip + ((ratio - 1).abs() > cfg.clip_eps).float().mean()
                            chunk_kl = chunk_kl + (old_logp_t - new_logp).mean()
                    chunk_pg /= chunk_len
                    chunk_vl /= chunk_len
                    chunk_ent /= chunk_len
                    chunk_clip /= chunk_len
                    chunk_kl /= chunk_len

                    actor_loss = chunk_pg - cfg.ent_coef * chunk_ent
                    if div_on:
                        chunk_div = chunk_div / chunk_len
                        actor_loss = actor_loss + cfg.div_weight * chunk_div
                    if cfg.diag_grad and not diag_done:
                        # Separate backward for each actor-loss term. retain_graph keeps the graph
                        # alive for the real optimizer step below, so this only costs compute.
                        params = [p for p in model.parameters() if p.requires_grad]

                        def _gnorm(term: torch.Tensor) -> float:
                            gs = torch.autograd.grad(term, params, retain_graph=True,
                                                     allow_unused=True)
                            sq = sum(float((g.detach().float() ** 2).sum().item())
                                     for g in gs if g is not None)
                            return sq ** 0.5

                        try:
                            g_pg = _gnorm(chunk_pg)
                            g_ent = _gnorm(cfg.ent_coef * chunk_ent)
                            stats["g_pg"] = g_pg
                            stats["g_ent"] = g_ent
                            stats["g_ent_over_pg"] = g_ent / max(g_pg, 1e-12)
                        except RuntimeError as exc:
                            stats["g_pg"] = stats["g_ent"] = stats["g_ent_over_pg"] = float("nan")
                            print(f"[diag_grad] skipped ({exc})")
                        diag_done = True
                    critic_loss = cfg.vf_coef * chunk_vl
                    # Scale loss by chunk_len/T for unbiased multi-chunk gradients,
                    # and divide by n_mb*k_epochs so total magnitude is invariant
                    # to minibatching/epoch choices.
                    loss = (actor_loss + critic_loss) * (chunk_len / T)

                # NaN/inf guard. Late in training the value targets can spike and, under
                # fp16 AMP, overflow → v_loss=NaN → the step poisons every weight and the
                # policy collapses to uniform (entropy=ln|A|) for the rest of the run
                # (observed in 3 sweep trials). Gate the optimizer step on a finite loss
                # AND finite grad norm so a single bad minibatch is dropped, not fatal.
                if torch.isfinite(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    if torch.isfinite(total_norm):
                        scaler.step(optimizer)
                    else:
                        stats["nan_skips"] += 1
                    scaler.update()
                else:
                    stats["nan_skips"] += 1

                stats["pg_loss"] += float(chunk_pg.detach().item())
                if div_on:
                    stats["div_loss"] += float(chunk_div.detach().item())
                stats["v_loss"] += float(chunk_vl.detach().item())
                stats["entropy"] += float(chunk_ent.detach().item())
                stats["kl"] += float(chunk_kl.detach().item())
                stats["clipfrac"] += float(chunk_clip.detach().item())
                stats["updates"] += 1
                # Carry hidden across chunks within this minibatch.
                h_act = last_h_act.detach()
                h_crit = last_h_crit.detach()

    n = max(1, stats["updates"])
    for k in ("pg_loss", "v_loss", "entropy", "kl", "clipfrac", "div_loss"):
        stats[k] /= n
    # Computed once per rollout, before any gradient step — not an average over minibatches.
    stats["explained_var"] = explained_var
    return stats
