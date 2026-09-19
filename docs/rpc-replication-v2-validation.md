# RPC replication v2 local validation receipt

Date: 2026-09-19. Scope: local engineering, CPU tests, and preregistration only.
No SSH connection, GPU benchmark, policy warmup, remote diagnostic, deployment,
v1 Run 2, or remote shutdown was performed.

## Environment and source

- Branch: `codex/actionstream/rpc-replication-v2-executor`.
- Starting commit: `f049925e382a1030fd1fca8450309585d3a58687`.
- Python 3.12.11; PyTorch `2.11.0+cpu`; LeRobot 0.6.2.
- Existing interpreter: `C:/Users/cgliu/OneDrive/Documents/ActionStream/.venv/Scripts/python.exe`.
- Installed LeRobot archive provenance identifies pinned revision
  `73e1584473028a2d53ecfc856f5290db84507f90`.
- `PYTHONPATH` points to this worktree's `src` and `integrations/lerobot/src`;
  inspected imports confirmed both local packages resolve to this worktree.
- Existing environment reused without installing or changing dependencies.
  This is not a fresh Linux locked-install validation.
- `CUDA_VISIBLE_DEVICES` empty; installed torch is CPU-only.

## Results

| Check | Result |
|---|---|
| Full repository pytest suite, final implementation + v2 protocol | **223 passed**, 0 failed/errors/skipped; 27.062 s |
| Focused RPC transport + new executor tests | **26 passed** |
| Earlier affected runtime/LeRobot/RPC contracts | **84 passed**; final full suite includes subsequent additions |
| Frozen deterministic loopback matrix | **6/6 cases**, 40 requests; all expected counts match |
| Ruff check: `src tests scripts/release scripts/engineering integrations/lerobot` | PASS |
| CI-configured Ruff format check | PASS, 68 files |
| `compileall`: `src tests integrations/lerobot/src` | PASS |
| Strict public release audit, without license bypass | PASS |
| Git whitespace checks | PASS |
| V1 formal manifest + every referenced file | PASS, 123/123 unchanged |
| Diagnostic manifest + every referenced file | PASS, 77/77 unchanged |
| V1 config, aborted external-validity config and `uv.lock` | Original SHA256 values unchanged |

The full suite includes plugin discovery through the real local `lerobot-rollout
--help` subprocess, registry contracts, lifecycle/reset boundaries and TCP engine
integration. It does not execute the upstream LeRobot test suite, native LIBERO
simulator, X-VLA weights, or remote two-host GPU workload.

An initial run exposed a pre-existing fast simulated-controller test issue:
Python 3.12 Windows `monotonic()` can quantize several control timestamps to the
same tick, producing a zero median interval. Only that test now uses the
higher-resolution monotonic `perf_counter` clock; production benchmark behavior
and historical results are unchanged. Initial failure logs are retained.
The first shutdown test also exposed Windows blocked-receive behavior; server
shutdown now performs both socket shutdown and close before joining handlers.
An initial small (20 ms) loopback test invocation failed under host scheduling;
subsequent focused and complete suites and the separate frozen matrix passed.
CPU scheduling checks are not real-time latency guarantees.

The implementation suite was green (217 tests at that point), with static checks,
strict audit and the frozen matrix passing, before the v2 configuration was
created. Additional executor and protocol guards are included in the final 223.

## Reproduction and local evidence

Run from this checkout with the interpreter and `PYTHONPATH` above. Prefix the
interpreter directory to `PATH` for the CLI-discovery test.

```text
python -m pytest -q -o faulthandler_timeout=45 --junitxml=artifacts/rpc_v2_engineering/final_full_tests.xml
python -m pytest tests/test_rpc_transport.py tests/test_rpc_executor.py -q
python -m actionstream.rpc_matrix --config configs/rpc_fault_matrix_v1.json --output <new-output-directory>
ruff check src tests scripts/release scripts/engineering integrations/lerobot
python -m compileall -q src tests integrations/lerobot/src
python scripts/release/audit_public_release.py --output <audit-output.json>
git diff --cached --check
```

Local ignored evidence is under `artifacts/rpc_v2_engineering/`:
`final_full_tests.xml`, `final_full_tests.log`, `executor_tests.xml`,
`targeted_tests.xml`, `fault_matrix_final/summary.json`,
`format_check.log`, `release_audit_final.json`, `preservation.json`,
`preservation_final.log`, and `validation_summary.json`.
These are engineering evidence, not formal episode results.

V2 configuration SHA256:
`661ee2fbcd565ec2a04deda57d8021cc088f05d2a3cb915bc59e379486fbb4f1`.

## Exact changed files

- `.github/workflows/ci.yml`: include executor and protocol guards in CI.
- `src/actionstream/rpc_transport.py`: persistent executor, lifecycle safety,
  timing/identity telemetry, deadline classes and absolute receive budget.
- `src/actionstream/rpc_server.py`: optional durable server telemetry and final counters.
- `src/actionstream/libero_rpc_client.py`: explicit paired timeout flags,
  action-wait consistency and final ERROR receipt telemetry.
- `src/actionstream/lerobot_inference.py`: report selected TCP budget in start telemetry.
- `tests/test_rpc_executor.py`: event-gated CPU lifecycle, timing and integration regressions.
- `tests/test_rpc_replication_v2_protocol.py`: frozen cohort, claim and CLI-budget guards.
- `tests/test_runtime.py`: high-resolution clock for the fast synthetic controller test.
- `configs/rpc_remote_transport_replication_v2.json`: outcome-informed, unexecuted v2.
- `docs/rpc-remote-transport-replication-v2.md`: architecture, semantics and claims.
- `docs/rpc-replication-v2-validation.md`: this validation receipt.

V1 remains VALID_NEGATIVE and permanently counts as v1 Run 1/51. V2 has zero
executed formal runs. Local engineering passes do not imply that the ~6.6 s cold
penalty has disappeared, that 15 s is an SLA, or that CUDA can be preempted.
