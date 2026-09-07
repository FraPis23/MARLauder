# Diagnostic tools

Offline analyses that produced measurements cited in the source comments and in the paper.
None of them train; all take a saved checkpoint and report. Run from the repository root.

| tool | question it answers |
|---|---|
| `idle_diag.py` | *Why* did an agent find nothing this step? Splits `idle_frac` into its two disjoint regimes — TRANSIT (walking over ground it already mapped: a routing problem) and REDUNDANT (scanning ground a teammate already held: an information problem) — which have opposite fixes. |
| `ab_relay.py` | Paired A/B of the multi-hop comm relay on one checkpoint: does it explore more, sooner, for less travel? Reports per-map differences, because map luck on `test/complex` dwarfs the effect. |
| `probe_reachability.py` | Is the own-99% objective reachable at all for this checkpoint, split and travel budget? Run it *before* spending GPU hours on a `--done-mode own` run. |
| `pf_scenarios.py` | Hand-scripted scenarios for the pathfront teammate belief on small ASCII maps, one HTML page per case. Every number on those pages is a property of `env/teammate_belief_pathfront.py` alone — no policy, no GPU. |
| `replay_failed.py` | Replays the comparison episodes that failed, reproducing the harness exactly (same chunking, seeds, start overrides and per-map budget), and draws what the agents actually did. |

`replay_failed.py` additionally needs **matplotlib**, which is deliberately not in
`requirements.txt` (nothing in the training or evaluation path uses it). Install it with
`pip install matplotlib`; the tool exits with that instruction rather than a traceback if it is
missing. `analyze_run.py --plot` uses it too, and degrades to ASCII sparklines without it.
