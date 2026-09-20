# RPC runtime hardening: maintenance validation

Date: 2026-09-20. Maintenance baseline: merged master
`626ca8fa329aef071f3108a646d7de7f2e53f2c8`, whose tree equals completed-result
commit `4e426a6913694427a1c9bbf442397acfdd368cf3`.
This is a focused software fix, not a new experiment or validation protocol.

## Confirmed old defects and new contract

The old CLI default was `0.0.0.0` even though the service had no authentication.
Idle reset swallowed connection, timeout, EOF and other failures. Active reset
closed the socket and skipped the remote reset entirely. Worker reset exceptions
already produced server error responses, but the client swallowed those failures.
That existing server behavior is retained and now covered by explicit wire tests;
success/failure telemetry and reset capability are added.

The maintained CLI defaults to loopback and rejects non-loopback binds without
`--allow-unauthenticated-remote`, before worker/listener creation. Reset immediately
invalidates old work, queues reset on the same persistent executor, and returns
success only after a matching ACK. Failure raises `RemoteResetError` and blocks
inference until a later explicit reset succeeds. A new client also requires that
ACK, preventing reconstruction from bypassing uncertain remote state. The separate
20 s configurable reset budget includes local I/O wait, connection, remote queue,
callback and ACK; it does not preempt remote compute or promise an SLA.

## CPU validation results

| Check | Result |
|---|---|
| Full pytest suite | **258 passed**, 0 failures/errors/skips; 35.324 s JUnit time |
| Focused RPC, executor, reset, plugin, lifecycle and registry contracts | **89 passed**, 0 failures/errors/skips; 11.939 s |
| Existing deterministic loopback fault matrix | **6/6 PASS**, 40 inference requests; original expected counts unchanged |
| Ruff lint: source/tests/release/engineering/plugin | PASS |
| CI-configured Ruff formatting | PASS, 69 files |
| compileall: source/tests/plugin | PASS |
| Strict public-release audit without license bypass | PASS; includes staged new regression tests/docs |
| Frozen inputs, tracked reports and sealed evidence | PASS; details below |

The full suite contains 35 added parametrized cases in `tests/test_rpc_reset.py`.
They cover IPv4/IPv6 wildcard opt-in, rejection before factory/listener creation,
startup security/capability receipts, required initial ACK, successful idle reset,
callback failure as a wire error, active stateful ordering, reset timeout,
poisoned inference rejection, reconnect/recreation rejection, explicit recovery,
malformed ACK/EOF/response timeout, cancellation around ACK commit, client I/O-lock
budget, and engine fatal-state propagation. Existing tests continue to cover one
executor, serialized policy calls, stale-result rejection and shutdown draining.
Old socket tests now explicitly reset new clients; the active-reset case waits
asynchronously for remote execution instead of assuming immediate reset success.

The pre-merge code review confirmed and fixed two additional findings: maintained
remote-listener examples were missing the acknowledgement flag, and blocking DNS
resolution escaped the advertised timeout. Three new regressions cover bounded
DNS waits with one outstanding lookup per client, DNS failure/recovery, and a
single shared connection budget across all resolved addresses. Late DNS completion
cannot open a socket or clear reset-required state. The code review also checked
generation/ACK races, persistent executor ordering, callback errors, startup
receipts and plugin configuration. No remaining merge-blocking finding was found
within this maintenance scope. This records an agent-led code review, not an
independent human approval or security certification.

The local environment is Python 3.12.11 / PyTorch 2.11.0+cpu. Existing dependencies
were reused, with PYTHONPATH pointing to this worktree's `src` and
`integrations/lerobot/src`, interpreter scripts on PATH for real CLI discovery,
and CUDA_VISIBLE_DEVICES empty. No weights, simulator, GPU experiment, benchmark
extension, dependency update or new protocol ran. Both remote hosts were checked
read-only: matching experiment hostnames, zero GPU device nodes, Python 3.12.3.
No remote checkout was modified, and neither server was shut down in this task.

CI now includes the new reset tests in the existing formatting and CPU quality
job. The PR's live checks provide Linux and pinned/current LeRobot compatibility
results; this receipt records the local pre-commit validation rather than claiming
future CI results. Packaging audit PASS is not authentication, production-security
certification or physical robot safety evidence.

Commands (with the environment above):

```text
python -m pytest -q -o faulthandler_timeout=45 --junitxml=artifacts/rpc_runtime_hardening_20260920/full.xml
python -m pytest tests/test_rpc_reset.py tests/test_rpc_transport.py tests/test_rpc_executor.py tests/test_lerobot_plugin.py tests/test_lerobot_lifecycle.py tests/test_lerobot_lifecycle_boundaries.py tests/test_upstream_registry_contract.py -q --junitxml=artifacts/rpc_runtime_hardening_20260920/focused.xml
python -m actionstream.rpc_matrix --config configs/rpc_fault_matrix_v1.json --output artifacts/rpc_runtime_hardening_20260920/review_loopback_matrix
python -m ruff check src tests scripts/release scripts/engineering integrations/lerobot
python -m ruff format --check <the explicit CI path list>
python -m compileall -q src tests integrations/lerobot/src
python scripts/release/audit_public_release.py --output artifacts/rpc_runtime_hardening_20260920/release_audit.json
git diff --check
git diff --cached --check
```

## Frozen evidence preservation

All tracked `configs/`, `reports/`, `uv.lock`, the v1 matrix, historical v2 design
and engineering receipt, and the completed interview case match the result commit.
Only maintained README/RPC documentation is updated to distinguish new behavior.

| Source | Preserved SHA256 / verification |
|---|---|
| V2 config | `661ee2fbcd565ec2a04deda57d8021cc088f05d2a3cb915bc59e379486fbb4f1` |
| V1 config | `b2fc8f8a0f7f66eb60cddbd79645647b7313435deaafc84831dd396616749019` |
| Aborted external-validity config | `7d663780396e28c0bc4b867a363e087a40a09cd8bc84e1fab9d24122bc614887` |
| Completed experiment manifest | `9c8e376f83ca4dce162b452defd2d9c42ea6bc8e44478133951846c8068fbe2f` |
| Whole v2 evidence manifest | `3c8cc214071455ac0ad2a10101239ec85af5e8e41a1e8aca02751c90d87f7864`; 2,889 entries verified |
| V2 Run1 manifest | `9d4ad51b781679eb0481c3d498074ab1bd3548c5b86c7e9a260a9b6cbc98d4b4`; 140 entries verified |
| V1 formal evidence manifest | `9b0e223c00b92f907a7d0c0803c94f41a07e01a46fe052d0ac28bc4f9609329d`; 123 entries verified |
| V1 diagnostic manifest | `f515e1f6324c34d7c0cd4933df3c437971bfe1c34e8ac399fff504aa9171a0be`; 77 entries verified |

V1 raw evidence was checked in its original worktree. V2 raw evidence stayed in
its sealed directory. Maintenance logs are separate ignored artifacts under
`artifacts/rpc_runtime_hardening_20260920/`; no raw GPU evidence is added to Git.
The formal result remains **PARTIAL SUPPORT / 7 of 8 / full hard-gate NO-GO**.

## Exact maintenance files and deferred cleanup

- `src/actionstream/rpc_transport.py`: ACK barrier, required-reset state, error
  propagation, bounded reset control, generation checks, minimal telemetry and
  callback capability; IPv6 literal binding for explicitly requested listeners.
- `src/actionstream/rpc_server.py`: loopback default, explicit remote opt-in,
  startup receipt and removal of an unused duplicate stop event.
- `src/actionstream/libero_rpc_client.py`: configurable reset timeout CLI plumbing.
- `src/actionstream/rpc_matrix.py`: confirmed initial reset before unchanged cases.
- `integrations/lerobot/src/lerobot_policy_actionstream/__init__.py`: plugin reset budget.
- `tests/test_rpc_reset.py`: new focused CPU/socket cases.
- `tests/test_rpc_transport.py`, `tests/test_rpc_executor.py`: migrate setup to
  acknowledged reset and preserve executor/lifecycle assertions.
- `tests/test_lerobot_plugin.py`: reset-budget propagation and initial required state.
- `.github/workflows/ci.yml`: include the new regression file.
- `README.md`, `docs/remote-rpc-validation.md`, `integrations/lerobot/README.md`:
  maintained-vs-historical boundary and migration links.
- `docs/rpc-runtime-hardening.md`: runtime contract, operator duties, recovery and
  intentionally retained compatibility aliases/timeout/reporting fields.
- `docs/rpc-runtime-hardening-validation.md`: this maintenance receipt.

No broad interface rewrite was performed. Legacy control timeout names,
process/reconnect compatibility aliases, experiment timing fields and the generic
engine's injected-transport label are deliberately retained; removing them would
require a separately reviewed migration. The core engine's sticky fatal-state
policy and existing inference budgets are unchanged.
