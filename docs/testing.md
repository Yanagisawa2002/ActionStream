# Runtime correctness test map

ActionStream currently keeps its tests in a mostly flat `tests/` directory to
avoid a large path-only migration. This document supplies the reviewer-facing
taxonomy instead. A future directory reorganization should not be mixed with
runtime behavior changes.

## Correctness risks and regression tests

| Risk / invariant | Primary regression coverage |
|---|---|
| stale prefix is discarded against the current controller step | `test_aligned_chunk_drops_elapsed_prefix_and_bounded_hold_stops`, `test_control_step_advance_before_merge_uses_current_age` |
| fully stale work cannot replace a usable queue | `test_fully_stale_chunk_is_rejected_while_existing_queue_is_usable` |
| fallback is allowed only after depletion under the configured policy | `test_fully_stale_chunk_uses_latest_only_only_after_depletion` |
| reset rejects an in-flight old-generation chunk | `test_reset_rejects_inflight_chunk_and_resets_provider_on_worker`, `test_reset_after_acceptance_check_must_not_repopulate_queue` |
| delayed pre-reset response cannot enter the new episode | `test_delivery_scheduler_reset_rejects_pending_pre_reset_response` |
| delivery reorder cannot overwrite a newer response | `test_delivery_scheduler_rejects_out_of_order_older_response` |
| reset/failure races do not poison the next generation | `test_old_generation_failure_must_not_fail_reset_episode`, `test_reset_between_failure_accounting_and_fatal_commit` |
| serialized dispatch rejects reset/task/foreign receipts | `test_serialized_dispatcher_rejects_reset_task_and_foreign_receipts` |
| reconnect does not replace the persistent executor | `test_deadline_reconnect_does_not_replace_executor_or_accept_old_result`, `test_worker_exception_does_not_replace_or_stop_executor` |
| queued work invalidated by disconnect never enters compute | `test_obsolete_queued_request_never_enters_compute`, `test_raw_disconnect_before_queued_compute_is_dropped` |
| reset runs on the same serialized executor as inference | `test_reset_is_serialized_on_the_same_executor`, `test_active_reset_orders_stateful_execution_and_gates_inference_until_ack` |
| a new/recreated client cannot infer before acknowledged reset | `test_new_client_requires_confirmed_reset_and_idle_reset_allows_inference`, `test_active_reset_timeout_reconnect_and_recreation_cannot_bypass_reset` |
| reset error/EOF/timeout/malformed ACK fails closed | `test_failed_or_invalid_idle_reset_never_allows_inference`, `test_reset_requires_valid_wire_ack` |
| cancellation after ACK cannot clear reset-required state | `test_cancel_after_ack_before_reset_commit_cannot_clear_required_state` |
| DNS lookup obeys a bounded budget without retry-thread accumulation | `test_dns_timeout_is_bounded_reuses_pending_lookup_and_requires_reset` |
| DNS failure can recover only through a later explicit reset | `test_dns_error_propagates_and_later_explicit_reset_recovers` |
| all resolved addresses share one connection budget | `test_multiple_resolved_addresses_share_one_connect_budget` |
| socket/request deadline is absolute even under streaming response | `test_streaming_response_cannot_extend_absolute_deadline` |
| a process timeout kills the hung child and permits recovery | `test_process_deadline_kills_hung_transport_and_recovers_with_new_observation` |
| repeated process resets do not leak child owners | `test_process_reset_preempts_repeated_hung_calls_without_orphans` |

The referenced tests live primarily in:

- `tests/test_lerobot_inference.py`
- `tests/test_lerobot_lifecycle.py`
- `tests/test_lerobot_lifecycle_boundaries.py`
- `tests/test_rpc_transport.py`
- `tests/test_rpc_executor.py`
- `tests/test_rpc_reset.py`

## Useful review commands

Core lifecycle and queue semantics:

```bash
uv run pytest \
  tests/test_lerobot_inference.py \
  tests/test_lerobot_lifecycle.py \
  tests/test_lerobot_lifecycle_boundaries.py
```

TCP framing, reset, reconnect, and executor ownership:

```bash
uv run pytest \
  tests/test_rpc_transport.py \
  tests/test_rpc_executor.py \
  tests/test_rpc_reset.py
```

The deterministic real-socket fault matrix is separate from pytest:

```bash
uv run actionstream-rpc-matrix \
  --config configs/rpc_fault_matrix_v1.json \
  --output /tmp/actionstream-rpc-matrix
```

## CI layers

The public CI has three complementary jobs.

1. **quality-and-core** checks Ruff, the maintained runtime/plugin/RPC test suite,
   the real-socket fault matrix, public-release audit, and frozen evidence replay.
2. **upstream-lerobot-contract** validates the pinned historical LeRobot integration
   contract and upstream rollout tests.
3. **upstream-current-extension-contract** validates the refreshed extension point
   against the recorded current-main snapshot.

Passing CPU CI establishes these software contracts. It does not establish
physical robot safety, production network security, GPU-driver recovery, or remote
CUDA preemption.
