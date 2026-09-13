# Episode ownership and dispatch contract

This contract applies to `ActionStreamInferenceEngine` and its built-in direct
and process transports. It preserves the LeRobot tensor API, one policy owner,
latest mailbox, age alignment, bounded hold, fallback, and request-budget defaults.
It does not establish learned-policy performance or physical-robot acceptance.

## Identity and state

| Identity | Lifetime and meaning |
|---|---|
| `engine_id` | Unique engine instance; retained across stop/start. |
| `epoch` | Advances at the beginning of every reset and stop. It owns the mailbox, request ordinals, queue, held action, and response heap. |
| `request_ordinal` | Increases within an epoch for each admitted attempt, including retries. |
| `source_step` | Observation step from which the request was made; alignment uses the latest control step at queue commit. |
| `task_revision` | Increases on a changed requested task, including changing back to an earlier task. Captured with the task at request admission; never reused on reset. |

Worker results, failures, deliveries, telemetry events, and action receipts carry
their originating identity. Before mutating engine response metrics or the queue,
the engine checks the epoch, reset gate, shutdown, and fatal state while holding
the same lock used for invalidation. A second guarded check at fatal commitment
prevents a reset between failure accounting and fatal publication from poisoning
the new episode. Delivery order, latest-step trimming, and queue replacement
commit together; no callback runs between the check and mutation.

| Operation/state | Contract |
|---|---|
| Before start | Observations may be staged. Actions are unavailable. |
| `start()` | Starts one owner and, optionally, one delivery thread. Calling it while already started does not create another owner. Call `resume()` to enable admission. |
| `pause()` | Stops new admission; preserves the mailbox, queued tail, holds, and already admitted work. It is not an episode boundary. |
| `reset()` | Invalidates the epoch and cancellation token first, clears queue/last action/mailbox/heap/budget and episode counters, then tears down the transport outside the commit locks. Observations arriving during teardown retain latest-mailbox semantics, but admission and action reads remain gated until teardown finishes. |
| Provider reset | Runs only on the policy owner before inference. The pending flag clears only after a successful reset in the same epoch. An ordinary failure retries the full reset under the existing failure budget; inference never follows a partially failed reset. An obsolete reset result or exception cannot clear the new pending flag or fail the new episode. |
| `stop()` | Invalidates before cancellation and joins, clears queued and held actions, and preserves final episode counters. A successful stop permits a fresh start with fresh counters and connectivity state. |
| Stop join failure | Sets fatal state and retains the existing owner reference. `start()` cannot launch a second owner. After the blocked call ends, a subsequent successful `stop()` is required before restart. Teardown exceptions are reported and re-raised. |
| Fatal | Rejects action reads and response commits and wakes the workers. `reset()` does not clear fatal state. A successful stop/start clears the engine's fatal state; only the caller may clear its shared shutdown event. |

Reset completion means transport teardown is finished, not that the deferred
provider reset or first inference has completed. `ready` indicates an open engine
lifecycle, not a populated action queue. The caller owns environment reset and
observation publication; `notify_observation(dict)` cannot identify an old frame
that the caller publishes after resetting the engine.

## Lock order and external operations

- The lifecycle mutex serializes start/reset/stop. Workers never acquire it.
  It may span transport teardown and bounded joins, so concurrent control calls
  wait for the current lifecycle operation.
- Engine commit order is **condition → queue → base task lock**. A path needing
  fewer locks takes a prefix or skips an unneeded lock; no path acquires the
  condition while already holding the queue or task lock.
- Model inference, provider reset, delay providers, delivery observers, telemetry
  writes, transport IPC/cancellation, and joins run outside condition/queue locks.
  Start/reset/stop from either worker callback raise `RuntimeError`; this avoids
  self-join and a callback waiting on the lifecycle operation joining it.
- The process transport uses **request lock → state lock**. Its state lock can
  span child termination/join. Engine telemetry reads transport properties before
  acquiring engine commit locks, so it cannot carry those locks into that join.

The engine counters and queue depth form one consistent snapshot. Transport
diagnostics are sampled independently and are not an atomic snapshot with the
engine counters. Process restart/cancellation counts cover the transport object's
lifetime. Reset clears transport latency samples; generation guards prevent old
completions from writing them back. `chunks_rejected_reset` deliberately counts
obsolete-work cleanup in the receiving reset episode; it is not a new-episode
inference failure. Obsolete work after stop does not update episode counters.

## Transport admission and cancellation

The original injected `infer(observation, task, *, timeout_s)` API remains valid.
Built-in transports additionally implement
`infer_cancellable(observation, task, *, timeout_s, cancellation_event)`. The
engine supplies the immutable per-epoch event, which is set on invalidation.
The process transport checks it under the lock that publishes the child: reset
either prevents an obsolete request from starting a child or terminates that
child. Admission of the next epoch waits for old transport teardown to finish.

Legacy custom transports still receive the original call and benefit from engine
commit rejection and the teardown admission gate. A custom transport that owns
persistent provider state must implement the cancellation extension (or an
equivalent internally serialized reset contract) to prevent obsolete late entry
from recreating that state. Engine queue isolation cannot enforce arbitrary
external side effects inside an injected callable.

Direct calls cannot be forcibly interrupted; their timeout is advisory, and new
provider work waits for the old owner call to return. Process tests exercise
child timeout, reset, and stop with CPU fixtures. They do not certify all blocking
IPC or platform failures: startup has a separate timeout, sending a request
precedes the response deadline, and tensor deserialization may block. No actual
network, CUDA hang, driver recovery, or simulator acceptance is implied.

## Task changes and actions already fetched

`get_action(obs_frame)` still returns `Tensor | None`. `set_task()` preserves
LeRobot's requested/dispatched-task lag and does not flush the queued tail or
reset the policy. To make a goal change an episode boundary, the caller can
serialize `set_task()` followed by `reset()` with observation publication and
dispatch. Physical environment reset is separate and remains caller-owned.

The opt-in `get_action_with_revision(obs_frame)` returns an `ActionStreamAction`
with a caller-owned tensor clone and frozen identity fields. Use
`is_action_current(packet)` immediately before dispatch. It rejects stopped,
resetting, fatal, foreign-engine, old-epoch, and old-requested-task packets.
It does not enforce one-time use or supersede every earlier request within the
same task/episode; bounded hold intentionally permits repeats.

For a dispatcher on a different thread, **all** goal/reset/stop operations and
the validation/write pair must share a caller-owned boundary, for example:

```python
from threading import Lock

control = Lock()

def change_goal(task):
    with control:
        engine.set_task(task)
        engine.reset()

def stop():
    with control:
        engine.stop()

def dispatch(packet):
    with control:
        if not engine.is_action_current(packet):
            return False
        robot.send_action(packet.action)
        return True
```

A revision check alone is not an atomic robot write. Reset cannot retract a
command already sent, and this API does not validate the physical suitability
of an action. Delivery observers receive copied identity values including
`engine_id`, epoch, request, task revision, and source step even after reset.
Their exceptions affect the engine only if their response still belongs to its
current epoch. Callbacks may arrive after invalidation or on different threads;
consumers must use identity and terminal status rather than assume callback
arrival is a serialized state-transition log. Telemetry emission timestamps can
also follow the corresponding commit.

## CPU verification

Run the [CPU dispatch example](../scripts/engineering/demo_lifecycle_revision.py)
with `uv run python scripts/engineering/demo_lifecycle_revision.py`. It uses a
mock actuator on the lifecycle thread and rejects an already fetched packet
after changing the task and resetting, without loading weights or a simulator.

`tests/test_lerobot_lifecycle.py` retains the eleven regression cases from the
fixed candidate audit. `tests/test_lerobot_lifecycle_boundaries.py` adds teardown
gating, provider retry, transport late entry, stop/restart/fatal ownership,
observer/error ownership, and a serialized consumer that rejects stale receipts.
Event handoffs force the relevant interleavings; timeout polling only bounds
completion waits. The original backend suite retains mailbox, alignment,
out-of-order, hold, retry, and process-orphan checks.

```bash
uv run pytest tests/test_lerobot_lifecycle.py \
  tests/test_lerobot_lifecycle_boundaries.py tests/test_lerobot_inference.py \
  tests/test_lerobot_plugin.py tests/test_clean_install.py
```

Passing these fixtures establishes the exercised CPU contracts. It does not
replace an exact-checkpoint native baseline, Linux acceptance, real policy
rollouts, or target-hardware validation.
