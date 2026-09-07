# Pipelines

End-to-end recipes. Each is a thin wrapper over `scripts/run_train.py` or
`scripts/eval_comparison.py` whose value is the **exact flag set** and the comments explaining why
each value is what it is. Run them from anywhere; they `cd` to the repository root themselves.

| script | what it does | cost |
|---|---|---|
| `train_full.sh` | The four chained stages that produce the released M=4 policy: easy M=2 from scratch → difficult M=2 → M=4 → M=4 with the sync bonus scaled by `(2/M)^1`. | ~69 h on a 16 GB GPU |
| `eval_ir2_comparison.sh` | Evaluates a checkpoint against the frozen IR2 protocol (100 fixed maps × 3 splits × M∈{2,4}) and prints the paired comparison table. | ~1 h |
| `ablation_sync_scaling.sh` | Null control for the sync-bonus M-scaling: stage 4 with `--sync-weight-m-scale 0.0`. Separates "the reward prevented the collapse" from "the optimizer restart did". | ~18 h |
| `ablation_frontier_diversity.sh` | Stage 4 plus the auxiliary frontier-diversity loss (`--div-weight 1.0`). | ~18 h |

The three that consume a warm start take it as a required environment variable, so nothing depends
on a run directory that only ever existed on the original machine:

```bash
CKPT=runs/<your-run>/ckpt_best.pt      bash pipelines/eval_ir2_comparison.sh
INIT_CKPT=runs/<your-run>/ckpt_best.pt bash pipelines/ablation_sync_scaling.sh
```

Every one of them exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. At M=4 with 6 hops,
32 environments fit in 15.5 GiB by a hair and the first backward pass goes OOM through
fragmentation without it.
