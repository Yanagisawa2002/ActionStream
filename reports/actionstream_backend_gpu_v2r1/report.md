# ActionStream GPU benchmark v2

Status: **PASS**. Method verdict: **LIMITED_OPERATING_ENVELOPE**.

This is a new frozen namespace; GPU benchmark v1 was not rewritten. Every
runtime/task-family cell ran in a fresh process. Warmup happened inside the
actual inference execution context, startup latency is separate from scored
steady-state latency, and driver residency/utilization came from sampled
`nvidia-smi` records.

## Main result

| Profile | Sync | Official Async | Latest-only | Aligned | Guarded |
|---|---:|---:|---:|---:|---:|
| 0 ms | 9/9 | 7/9 | 9/9 | 8/9 | 8/9 |
| 250 ms | 9/9 | 7/9 | 9/9 | 9/9 | 9/9 |
| 950 ms | 9/9 | 7/9 | 9/9 | 5/9 | 5/9 |
| 600 +/- 400 ms jitter | 9/9 | 7/9 | 9/9 | 9/9 | 9/9 |
| 250 -> 2000 ms burst | 9/9 | 4/9 | 8/9 | 7/9 | 2/9 |

The preregistered primary reference is official latest-only. At fixed 950 ms,
aligned minus latest-only success was
`-44.4` percentage points (paired 95%
CI `[-77.8,
-11.1]`) and the paired completion-step
difference was `62.89` (95% CI
`[5.11, 122.44]`).

If this cell regressed, aligned is explicitly a runtime with a limited
operating envelope; no universal-superiority claim is made. Aligned versus
official Async remains a secondary contrast in `paired_effects.csv`.

## Numbered findings

1. Engineering evidence gate: `{'canary_record_count': True, 'holdout_record_count': True, 'three_task_families': True, 'canary_sync_zero_delay': True, 'fresh_runtime_processes': True, 'actual_worker_warmup': True, 'true_request_rate': True, 'nvidia_smi_residency': True, 'standard_telemetry': True}`.
2. Formal evidence contains `225` paired task/runtime/network episodes;
   raw episode rows are in `episode_table.csv`.
3. Process-isolated GPU evidence covers `15` fresh runtime processes;
   process VRAM, p50/p95 GPU utilization, model-load time, and sampler coverage
   are in `gpu_systems.csv`.
4. Standard backend telemetry validated `40070` events
   across `90` JSONL files with
   `0` invalid files/events.
5. The network axis is a set of five frozen categorical operating points, not
   an interpolated continuous latency curve.

## Next experiment

Only repair the frozen SmolVLA sync gate and, if that gate passes, run the
official RTC comparison. v6 selector and Isaac Lab-Arena remain paused.
