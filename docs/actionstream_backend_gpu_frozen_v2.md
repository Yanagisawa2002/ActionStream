# ActionStream backend GPU benchmark v2: frozen protocol

GPU benchmark v2 is a new evidence namespace. It does not edit, replace, or
reinterpret GPU benchmark v1. Candidate code is commit
`b3da160f7fe22cc3182e5e1991c27e5cdac5e12a`; every protocol pins the source and
measurement file hashes independently.

## Frozen engineering and measurement contract

- A clean clone installs the root package, pinned LeRobot fork, and backend
  plugin with `uv sync --locked --all-packages`. The actual
  `lerobot-rollout --help` subprocess must expose `actionstream`.
- Process transport startup and request deadlines are separate. Reset and stop
  can terminate a child during startup or a hung request. The frozen stress
  gate is ten cold-start reset cycles, one hard deadline/recovery, one hung
  stop, bounded reset/stop latency, and zero orphan transport children.
- Each task-family/runtime benchmark cell is a new OS process. A process may
  run all frozen network profiles for that one runtime, but it may not mix
  runtimes.
- Every episode performs two non-scored inference calls through the actual
  worker or synchronous execution context. The first call is startup latency;
  scored calls after warmup are steady state. Warmup state 45 is never scored.
- `nvidia-smi` samples driver process residency, device utilization, and device
  memory every 200 ms. PyTorch allocated peak remains a separate metric.
- Request/s is completed or started scored inference requests divided by
  scored episode wall time. It is not action throughput or a synthetic loop.
- ActionStream episode events use telemetry JSONL schema v1. Official runtimes
  report unsupported queue/recovery fields as unavailable, not fabricated zero.

## Frozen scientific matrix

The three genuinely different families remain Object task 5, Spatial task 7,
and Goal task 2. Reset identity includes suite, task, initial-state index, and
seed. A pre-freeze scan of 1,961 JSON/JSONL files found no prior episode using
states 45--49 for those exact tasks. State 46 is canary; states 47--49 are
sealed holdout. All twelve seeded canary/holdout network traces are unique and
disjoint from every other loadable frozen config. Fixed 0, 250, and 950 ms are
shared deterministic operating points by definition.

Canary runs six profiles, including a backend-only disconnect/recovery probe.
The latter is an engineering test and cannot enter official baseline effects.
Holdout runs five comparable profiles and 225 paired episodes:
3 families x 5 profiles x 5 runtimes x 3 resets. Runtimes are sync, official
weighted Async, official latest-only, ActionStream aligned, and guarded.

The primary contrast is aligned versus official latest-only. Aligned versus
official weighted Async is secondary. If aligned has lower observed paired
success or higher paired completion steps at fixed 950 ms after corrected
measurement, the frozen verdict is `LIMITED_OPERATING_ENVELOPE`; selector
tuning must not continue. v6 selector and Arena remain paused. After this
engineering/GPU closure, the only next experiment is a new SmolVLA sync gate
and official RTC if that gate passes.
