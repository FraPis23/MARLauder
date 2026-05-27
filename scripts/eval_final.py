"""Batch eval final.pt on multiple random maps. Reports coverage stats."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import numpy as np
import torch

from env.explorer import EnvCfg, Explorer
from env.maps import load_split, sample_batch
from eval.rollout import EvalCfg, EvalRollout
from models.actor_critic import MarlActorCritic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="Path to final.pt or ckpt_*.pt")
    ap.add_argument("--split", default="train/easy")
    ap.add_argument("--n-maps", type=int, default=5, help="Num maps to eval")
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None, help="Dir for GIFs (default: ckpt dir)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=args.device)
    cfg_dict = ckpt.get("cfg", {})

    # Strip _orig_mod. prefix inserted by torch.compile
    sd = {k.replace("encoder._orig_mod.", "encoder."): v
          for k, v in ckpt["model"].items()}

    # Infer architecture from weights (safer than cfg_dict which may be stale).
    # encoder.input_proj.bias has shape [d] exactly.
    # critic_pre.0.weight has shape [d, n_agents * d].
    d_hidden = int(sd["encoder.input_proj.bias"].shape[0])
    n_agents = int(sd["critic_pre.0.weight"].shape[1]) // d_hidden
    n_heads = cfg_dict.get("n_heads", 4)
    n_layers = cfg_dict.get("n_layers", 2)

    print(f"[load] {args.ckpt}")
    print(f"       n_agents={n_agents}  d={d_hidden}  heads={n_heads}  layers={n_layers}")

    model = MarlActorCritic(n_agents=n_agents, d=d_hidden, n_heads=n_heads, n_layers=n_layers).to(args.device)
    model.load_state_dict(sd)
    model.eval()

    import imageio.v2 as imageio
    out_dir = args.out if args.out is not None else args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.split, device=args.device)
    rng = np.random.default_rng(args.seed)
    map_indices = rng.integers(0, split.n, size=args.n_maps).tolist()

    explored_list: list[float] = []
    for map_i, map_idx in enumerate(map_indices):
        print(f"\n[{map_i+1}/{args.n_maps}] map={map_idx}...", end=" ", flush=True)

        env_cfg = EnvCfg(n_envs=1, n_agents=n_agents, max_episode_steps=args.steps)
        env = Explorer(split, env_cfg, seed=map_idx)

        # Load specific map
        gt_new, starts_new, fc_new = sample_batch(
            split, 1, indices=np.array([map_idx]), seed=0, device=args.device,
        )
        env.world.gt_torch[0] = gt_new[0]
        env.world.occupancy_torch[0] = 0
        env.world.occupancy_logodds_torch[0] = 0.0
        env.starts[0] = starts_new[0]
        env.free_total[0] = fc_new[0]
        env.visited_step[0] = -1
        env.t[0] = 0

        row0, col0 = int(starts_new[0, 0]), int(starts_new[0, 1])
        agent_pos = env._spread_starts_graph(row0, col0)
        env.pos[0] = agent_pos
        for ag in range(n_agents):
            env.last_known_pos[0, :, ag] = agent_pos[ag]

        env.world.set_positions(env.pos)
        env.world.scan()
        env.last_union[0] = (env.world.occupancy_torch[0] == 1).any(dim=0).view(-1).float().sum()
        env._refresh_obs()

        # Run eval
        rollout = EvalRollout(env, model, EvalCfg(max_steps=args.steps, deterministic=True,
                                                  draw_edges=True))
        frames, stats = rollout.run()
        explored = stats["final_explored"]
        explored_list.append(explored)

        gif_path = out_dir / f"eval_map{map_idx:05d}.gif"
        imageio.mimsave(gif_path, frames, duration=80, loop=0)
        print(f"explored={explored*100:5.1f}%  → {gif_path.name}")

    explored_array = np.array(explored_list)
    print(f"\n[summary] {args.n_maps} maps on {args.split}")
    print(f"  mean  {explored_array.mean()*100:5.1f}%")
    print(f"  std   {explored_array.std()*100:5.1f}%")
    print(f"  min   {explored_array.min()*100:5.1f}%")
    print(f"  max   {explored_array.max()*100:5.1f}%")


if __name__ == "__main__":
    main()
