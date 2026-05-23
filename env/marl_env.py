"""Env MARL vettorizzato N_env x M_agenti, tutto-GPU (baseline comms perfette).

- Belief CONDIVISA per mondo (comms perfette). Ogni agente ha il proprio ego-grafo.
- Azione discreta: vicino del nodo centrale dell'ego-lattice (8 direzioni) + STAY.
- Anti-collisione: action-masking + risoluzione conflitti (due agenti non finiscono sullo
  stesso nodo). Vincolo di proprieta' dell'env, non appreso (vedi memoria env-rules).
- Spawn: tutti gli agenti nella stessa "stanza" attorno allo start, su nodi distinti.
- Reward stile IR2: nuova area esplorata (team) - costo passo.

Convenzioni: belief 0=unknown,1=free,2=ostacolo. Coord (x=col,y=row).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from env.world_warp import WarpWorld
from env.frontier import frontier_centers
from env.graph_lattice import EgoLattice, build_anchors, frontier_coarse_warp, _DIRS
from env.comms import PositionProvider, all_connected

FSCALE = 4   # downsample frontier (res 4, come IR2)

STAY = 8


def _guidepost(coords: torch.Tensor, pos: torch.Tensor,
               anchors: torch.Tensor, amask: torch.Tensor) -> torch.Tensor:
    """Cosine similarity di ogni nodo ego verso il nearest valid anchor.

    coords [N,M,KK,2], pos [N,M,2], anchors [N,A,2], amask [N,A] -> [N,M,KK].
    Per nodi che puntano nella direzione di una frontiera globale: valore alto.
    Per nodi che puntano lontano da tutte le frontiere: valore basso/negativo.
    Se nessun anchor valido: 0.
    """
    import torch.nn.functional as F
    N, M, KK, _ = coords.shape
    A = anchors.shape[1]
    # direzione da agente a ogni nodo: [N,M,KK,2]
    node_dir = coords - pos.unsqueeze(2)
    node_dir = F.normalize(node_dir, dim=-1, eps=1e-6)
    # direzione da agente a ogni anchor: [N,M,A,2]
    anc = anchors.unsqueeze(1).expand(N, M, A, 2)
    anc_dir = anc - pos.unsqueeze(2)
    anc_dir = F.normalize(anc_dir, dim=-1, eps=1e-6)
    # cosine similarity: [N,M,KK,A]
    sim = torch.einsum("nmkd,nmad->nmka", node_dir, anc_dir)
    # maschera anchor invalidi
    mask = amask.unsqueeze(1).unsqueeze(2).expand(N, M, KK, A)
    sim = sim.masked_fill(~mask, -2.0)
    # max similarity su tutti gli anchor validi: [N,M,KK]
    guide = sim.amax(dim=-1)
    # zero se nessun anchor valido
    has_anchor = amask.any(dim=-1).unsqueeze(1).unsqueeze(2)   # [N,1,1]
    return guide * has_anchor.float()
N_ACT = 9   # 8 direzioni + stay


@dataclass
class EnvConfig:
    n_envs: int = 64
    n_agents: int = 4
    K: int = 21
    spacing: float = 20.0
    sensor_range: float = 80.0
    n_rays: int = 720
    util_range: float = 70.0
    a_max: int = 64
    spawn_radius: float = 60.0
    max_steps: int = 256
    cov_done: float = 0.99
    reward_scale: float = 500.0
    step_cost: float = 0.01
    collision_penalty: float = 0.05      # penalty su tentativo bloccato da conflitto
    device: str = "cuda:0"


class MarlExploreEnv:
    def __init__(self, gt: torch.Tensor, free_counts: torch.Tensor, starts: torch.Tensor,
                 cfg: EnvConfig):
        self.cfg = cfg
        self.N, self.M = cfg.n_envs, cfg.n_agents
        self.device = cfg.device
        self.h, self.w = gt.shape[1], gt.shape[2]
        self.gt = gt.to(self.device)
        self.free_counts = free_counts.to(self.device).float()
        self.starts = starts.to(self.device)

        self.world = WarpWorld(self.gt, n_agents=self.M, sensor_range=cfg.sensor_range,
                               n_rays=cfg.n_rays, device=self.device)
        env_idx = torch.arange(self.N, device=self.device).repeat_interleave(self.M)  # [N*M]
        self.lat = EgoLattice(self.N * self.M, env_idx=env_idx, K=cfg.K, spacing=cfg.spacing,
                              util_range=cfg.util_range, device=self.device)
        self.kc = (cfg.K // 2) * cfg.K + (cfg.K // 2)   # indice nodo centrale
        # offset xy per le 8 direzioni (dj=x, di=y)
        self.dir_xy = torch.tensor([[d[1], d[0]] for d in _DIRS],
                                   dtype=torch.float32, device=self.device) * cfg.spacing
        self.pos = torch.zeros((self.N, self.M, 2), dtype=torch.float32, device=self.device)
        self.step_count = torch.zeros(self.N, dtype=torch.int32, device=self.device)
        self.dist = torch.zeros((self.N, self.M), dtype=torch.float32, device=self.device)

        # comms: baseline tutti-connessi, posizioni vere dietro il provider (vedi env/comms.py)
        self.provider = PositionProvider()
        self.connectivity = all_connected(self.N, self.M, self.device)

    # ---------- spawn ----------
    def _spawn(self):
        """Stessa stanza attorno allo start, nodi distinti (separati >= spacing)."""
        for e in range(self.N):
            r0, c0 = int(self.starts[e, 0]), int(self.starts[e, 1])
            free = (self.gt[e] == 1)
            if r0 < 0:
                fr = torch.nonzero(free)
                r0, c0 = [int(v) for v in fr[torch.randint(len(fr), (1,), device=self.device)][0]]
            yy, xx = torch.nonzero(free, as_tuple=True)
            d2 = (xx - c0) ** 2 + (yy - r0) ** 2
            near = d2 <= self.cfg.spawn_radius ** 2
            cand = torch.stack([xx[near].float(), yy[near].float()], dim=-1)  # [P,2]
            chosen = [torch.tensor([float(c0), float(r0)], device=self.device)]
            order = torch.randperm(len(cand), device=self.device)
            for j in order:
                p = cand[j]
                if all(torch.norm(p - q) >= self.cfg.spacing * 0.8 for q in chosen):
                    chosen.append(p)
                if len(chosen) >= self.M:
                    break
            while len(chosen) < self.M:           # fallback: ammassati allo start
                chosen.append(chosen[0].clone())
            self.pos[e] = torch.stack(chosen[: self.M])

    # ---------- reset ----------
    def reset(self) -> dict:
        self.world.reset_belief()
        self.step_count.zero_()
        self.dist.zero_()
        self._spawn()
        self.world.set_positions(self.pos)
        self.world.scan()
        self.explored_prev = self._explored()
        return self._observe()

    def _explored(self) -> torch.Tensor:
        return (self.world.belief_torch == 1).sum(dim=(1, 2)).float()   # [N]

    # ---------- osservazione ----------
    def _observe(self) -> dict:
        belief = self.world.belief_torch
        fcoarse = frontier_coarse_warp(belief, scale=FSCALE)    # [N, H/4, W/4] kernel fuso
        g = self.lat.build(self.pos.reshape(self.N * self.M, 2), belief, fcoarse, fscale=FSCALE)
        edges = g["edges"].reshape(self.N, self.M, self.lat.KK, len(_DIRS))
        valid = g["valid"].reshape(self.N, self.M, self.lat.KK)
        util = g["utility"].reshape(self.N, self.M, self.lat.KK)
        coords = g["coords"].reshape(self.N, self.M, self.lat.KK, 2)

        # action-mask: edge valido dal nodo centrale + STAY
        mask = torch.zeros((self.N, self.M, N_ACT), dtype=torch.bool, device=self.device)
        mask[:, :, :8] = edges[:, :, self.kc, :] == 1
        mask[:, :, STAY] = True
        mask = self._mask_collisions(mask)

        # anchor dai centri di frontiera coarse (coord riportate a full-res x FSCALE)
        centers, cvalid, ccount = frontier_centers(fcoarse, stride=4, min_count=2, coord_scale=FSCALE)
        anchors, amask = build_anchors(centers, cvalid, ccount, a_max=self.cfg.a_max)

        # guidepost: cosine similarity di ogni nodo ego verso il nearest valid anchor.
        # Nodi che puntano verso una frontiera globale ricevono guidepost alto
        # anche se utility locale = 0 (agente nel centro dell'area esplorata).
        # Shape [N, M, KK], range [-1, 1], 0 se nessun anchor valido.
        guidepost = _guidepost(coords, self.pos, anchors, amask)

        # posizioni compagni (relative) dietro il provider, gated da connettivita'
        teammate_rel, teammate_known = self.provider(self.pos, self.connectivity)
        return {"coords": coords, "valid": valid, "edges": edges, "utility": util,
                "guidepost": guidepost,
                "action_mask": mask, "anchors": anchors, "anchor_mask": amask,
                "teammate_rel": teammate_rel, "teammate_known": teammate_known,
                "frontier_coarse": fcoarse, "fscale": FSCALE, "belief": belief, "pos": self.pos.clone()}

    def _target_cells(self, actions: torch.Tensor) -> torch.Tensor:
        """actions [N,M] -> celle target (x,y) float [N,M,2]."""
        tgt = self.pos.clone()
        move = actions < 8
        dirs = self.dir_xy[actions.clamp(max=7)]          # [N,M,2]
        tgt = torch.where(move.unsqueeze(-1), self.pos + dirs, self.pos)
        return tgt

    def _mask_collisions(self, mask: torch.Tensor) -> torch.Tensor:
        """Vieta azioni che porterebbero un agente sulla cella corrente di un altro agente."""
        cur = self.pos.round()                            # [N,M,2]
        for d in range(8):
            tgt = (self.pos + self.dir_xy[d]).round()     # [N,M,2]
            # collisione se tgt coincide con la cella corrente di un ALTRO agente
            diff = (tgt.unsqueeze(2) - cur.unsqueeze(1)).abs().sum(-1)   # [N,M,M]
            same = diff < 1.0
            eye = torch.eye(self.M, dtype=torch.bool, device=self.device).unsqueeze(0)
            collide = (same & ~eye).any(dim=2)            # [N,M]
            mask[:, :, d] &= ~collide
        return mask

    def _resolve(self, actions: torch.Tensor):
        """Risolve conflitti: due agenti non finiscono sulla stessa cella (priorita' a indice basso).
        Ritorna (tgt [N,M,2], collided [N,M] bool: True se forzato a restare per conflitto)."""
        tgt = self._target_cells(actions)
        collided = torch.zeros((self.N, self.M), dtype=torch.bool, device=self.device)
        for m in range(1, self.M):
            for prev in range(m):
                clash = (tgt[:, m].round() - tgt[:, prev].round()).abs().sum(-1) < 1.0
                tgt[clash, m] = self.pos[clash, m]        # resta fermo
                collided[clash, m] |= True
        return tgt, collided

    # ---------- step ----------
    def step(self, actions: torch.Tensor):
        """actions [N,M] in [0,8]. Ritorna (obs, reward[N,M], done[N], info)."""
        new_pos, collided = self._resolve(actions)
        self.dist += torch.norm(new_pos - self.pos, dim=-1)
        self.pos = new_pos
        self.world.set_positions(self.pos)
        self.world.scan()

        explored = self._explored()
        d_explored = (explored - self.explored_prev).clamp(min=0)
        self.explored_prev = explored
        coverage = explored / self.free_counts.clamp(min=1)

        team = (d_explored / self.cfg.reward_scale).unsqueeze(1)       # [N,1]
        reward = team.expand(-1, self.M) - self.cfg.step_cost          # [N,M]
        reward = reward - collided.float() * self.cfg.collision_penalty  # penalty conflitto

        self.step_count += 1
        done = (coverage >= self.cfg.cov_done) | (self.step_count >= self.cfg.max_steps)

        obs = self._observe()
        info = {"coverage": coverage, "explored": explored, "dist": self.dist.clone()}
        return obs, reward, done, info
