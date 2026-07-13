"""Shared checkpoint → model loader for eval/trace scripts.

Auto-detects architecture from the checkpoint itself — n_layers from the actual encoder layer
count in the state dict (authoritative), d_hidden/n_heads/n_agents/use_gru from the saved
training cfg — so an eval or trace ALWAYS reconstructs the exact architecture the checkpoint was
trained with. A layer/head/dim MISMATCH doesn't error: `load_state_dict(strict=False)` silently
drops the shape-mismatched keys, so the model loads fine but is missing most of its trained GAT
depth — every eval score and every attention number in a trace becomes fiction, with no crash to
flag it (this is exactly what happened to run "fra-run": trained at 6-hop/6-layer, evaluated
with eval_ckpt.py's old hardcoded --n-layers=2 default → 1-4% explored; re-run with the correct
depth → 56-83% explored, same checkpoint).

trace_episode.py had this auto-detection; eval_ckpt.py didn't (hardcoded defaults) — the two
scripts had drifted apart. Sharing one function here means there's only one place to get it
right, and no way for the two callers to disagree again.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch

from models.actor_critic import MarlActorCritic


def load_model_from_ckpt(
    ckpt_path: Path,
    device: str,
    n_agents: int | None = None,
    d_hidden: int | None = None,
    n_heads: int | None = None,
    n_layers: int | None = None,
    verbose: bool = True,
) -> tuple[MarlActorCritic, dict]:
    """Load a checkpoint into a freshly-built, ARCHITECTURE-MATCHED model.

    Every arg defaults to None → auto-detected/read from the checkpoint. Pass an explicit value
    only to deliberately override (e.g. probing a checkpoint under a different head count).

    Returns (model, env_peek) — env_peek is the checkpoint's saved EnvCfg dict (includes n_hops,
    so an eval/trace env is built with the SAME ego-window radius training used, not a default).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_peek = ckpt.get("cfg", {})
    cfg = cfg_peek if isinstance(cfg_peek, dict) else {}
    env_peek = cfg.get("env", {}) if isinstance(cfg, dict) else {}

    sd = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ckpt["model"].items()}
    sd = {k.replace("encoder._orig_mod.", "encoder."): v for k, v in sd.items()}

    # n_layers: count distinct encoder GAT layers actually present in the state dict
    # (authoritative — cfg could be stale/missing on very old checkpoints).
    lyr = {int(m.group(1)) for k in sd for m in [re.search(r"encoder\.layers\.(\d+)\.", k)] if m}
    det_layers = (max(lyr) + 1) if lyr else 2
    det_d = sd["encoder.input_proj.weight"].shape[0] if "encoder.input_proj.weight" in sd else 128
    det_agents = int(cfg.get("n_agents", 2))

    n_layers_final = n_layers if n_layers is not None else det_layers
    d_hidden_final = d_hidden if d_hidden is not None else int(cfg.get("d_hidden", det_d))
    n_heads_final  = n_heads  if n_heads  is not None else int(cfg.get("n_heads", 4))
    n_agents_final = n_agents if n_agents is not None else det_agents
    use_gru = bool(cfg.get("use_gru", True))   # honor a GRU-ablation checkpoint
    gat_actor = bool(cfg.get("gat_actor", True))   # honor a VF-only-ablation checkpoint
    gat_critic = bool(cfg.get("gat_critic", True))  # honor a full --no-gat checkpoint

    if verbose:
        print(f"[ckpt] arch: n_layers={n_layers_final} d={d_hidden_final} n_heads={n_heads_final} "
              f"n_agents={n_agents_final} use_gru={use_gru} (detected layers={det_layers}, d={det_d})")

    model = MarlActorCritic(n_agents=n_agents_final, d=d_hidden_final, n_heads=n_heads_final,
                            n_layers=n_layers_final, use_gru=use_gru, gat_actor=gat_actor,
                            gat_critic=gat_critic).to(device)
    msd = model.state_dict()
    dropped = [k for k in sd if k in msd and msd[k].shape != sd[k].shape]
    if dropped and verbose:
        print(f"[ckpt] WARNING: {len(dropped)} state-dict keys shape-mismatched and were "
              f"DROPPED (arch override differs from the checkpoint?): {dropped[:5]}"
              f"{'...' if len(dropped) > 5 else ''}")
    for k in dropped:
        del sd[k]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, env_peek
