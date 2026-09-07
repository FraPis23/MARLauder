# Tests

Property tests for the simulator, the model and the reward. They are plain scripts (no pytest
dependency): each one runs a real `Explorer` and asserts, then exits non-zero on failure.

Run them from the repository root, with the root on `PYTHONPATH`:

```bash
PYTHONPATH=. python tests/00_test_toolchain.py
PYTHONPATH=. bash  tests/run_all.sh          # the whole suite, stops on first failure
```

| test | what it pins |
|---|---|
| `00_test_toolchain` | torch and Warp see the same GPU and share tensors zero-copy |
| `01_test_maps` | split loading, start markers, canvas padding |
| `02_test_lidar` | Warp LiDAR writes a sane log-odds occupancy grid |
| `03_test_frontier` | the conv2d frontier detector fires only on FREE cells bordering UNKNOWN |
| `04_test_model_shapes` | `MarlActorCritic` accepts a real obs dict; shapes and gradients check out |
| `05_test_smoke_mappo` | a full collect → GAE → PPO update cycle completes with finite losses |
| `06_test_teammate_belief` | uniform belief: collapse at comm, one-hop growth, Σp = 1 |
| `07_test_belief_integration` | belief on a real rollout: no NaN in feat[4]/[5]/[6], φ and geo_pair finite |
| `08_test_pathfront_belief` | pathfront belief: Σp = 1 in transit and bloom, hypotheses freeze at comm break |
| `09_test_map_merge` | on rendezvous both agents' maps show the union, in the render *and* the graph |
| `10_test_teammate_blocks` | a visible teammate's cell is masked out of the action space, without stranding anyone |
| `11_test_sync_reward` | the sync-event reward pays for a real exchange and for nothing else (tether, flicker, re-gift) |
| `12_test_comm_relay` | multi-hop relay: A–B–C converge in one step; an exact no-op when disabled |
| `13_test_sync_m_scale` | `(2/M)^a` sync scaling is identity at M=2 and exactly half at M=4, a=1 |
| `14_test_frontier_div` | the frontier-overlap tensor means what the diversity loss assumes |

**Map data.** Only `00_test_toolchain` and `06_test_teammate_belief` run on synthetic input
alone. Every other test builds a real `Explorer`, so it needs the preprocessed map packs under
`data/` — regenerate them with `scripts/preprocess_maps.py` (the packs are ~9 GB and are not in
the repository). Without them those tests fail at `load_split` with
`FileNotFoundError: split '<name>' missing under <root>`, which is the expected outcome, not a
regression.
