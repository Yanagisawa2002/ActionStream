# SmolVLA official-RTC sync gate report

**Verdict: NO-GO.** Formal RTC was not run because neither frozen three-suite zero-delay sync gate passed. This is a gate result, not a measured RTC effect.

## What was real

- v2 validated 33 learned-policy episodes and 33 traces; v3 validated 18 episodes and 18 traces.
- The checkpoint drove the LIBERO robot through the complete 50-action sync FIFO. Recorded successful Object, Spatial, and Goal episodes were visually inspected.
- Development task selection and both canary attempts used named, disjoint task/reset pairs. v2 was preserved after failure; v3 used a new namespace and stronger Goal screen.

## Frozen canary results

| Attempt | Suite | Task | States | Success | Steps | Infer p50/p95 ms | Peak CUDA MiB |
|---|---|---:|---|---:|---|---:|---:|
| v2 | libero_goal | 1 | 37,39 | 1/2 | 95,300 | 195.7/203.3 | 925.6 |
| v2 | libero_object | 0 | 37,39 | 2/2 | 137,140 | 193.0/206.0 | 925.6 |
| v2 | libero_spatial | 0 | 37,39 | 2/2 | 76,82 | 179.7/188.6 | 925.6 |
| v3 | libero_goal | 5 | 45,46 | 2/2 | 133,130 | 200.6/205.4 | 925.6 |
| v3 | libero_object | 0 | 45,46 | 1/2 | 142,300 | 200.2/208.5 | 925.6 |
| v3 | libero_spatial | 0 | 45,46 | 2/2 | 79,81 | 179.2/185.4 | 925.6 |

v2 failed on Goal (1/2). v3 used a separately frozen stronger Goal screen and then failed on Object (1/2). Spatial passed both attempts; Goal v3 passed 2/2. The gate requires every suite to pass 2/2, so partial success cannot unlock the holdout.

## Formal RTC boundary

- Formal holdout records: **0**.
- Official RTC records in the formal holdout: **0**.
- Paired RTC effect / 95% CI: **unavailable by design**.
- The pre-frozen holdout configs remain sealed for outcomes: states 40–44 were never loaded and their network traces were never consumed. No additional task/reset tuning was performed after the v3 failure.

The legacy diagnostic also ruled out chunk-vs-step postprocessing as the gate fix (maximum action difference 0.0); the remaining limitation is reset-dependent open-loop sync robustness, not missing learned motion or a fabricated RTC result.

## Evidence

Raw v2/v3 archive SHA-256: `8176c702c3af827937997cadd8c945d4c6a4f1b6f5464ee82f48cf9b684af340`.
See `development_task_screen.csv`, `canary_table.csv`, `failure_taxonomy.csv`, `sync_gate_by_suite.png`, and `summary.json`. Videos are retained in the raw archive; the failed v3 Object reset has trace/predicate evidence but was not recorded because capture was predeclared for episode 0 only.
