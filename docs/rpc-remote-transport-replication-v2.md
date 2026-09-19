# Remote transport replication v2: engineering and preregistration

Status: **PRE_REGISTERED_NOT_EXECUTED**. No GPU benchmark, warmup, diagnostic,
remote deployment, or v1 Run 2 was performed for this change.

v2 is **outcome-informed by v1 and its postmortem**. v1 Run 1 remains
**VALID_NEGATIVE**, permanently counted as **v1 Run 1/51**. v1 demonstrated that
its original one-size 5 s deadline was incompatible with the observed
connection-first inference path. v2 evaluates a revised remote transport
architecture and deadline semantics; it is not an independent replication of v1.
It is a robustness/systems experiment, with no unseen-task generalization,
unseen-state external validity, physical robot safety, or remote CUDA preemption claim.

## Frozen ancestry and evidence

The implementation starts at `f049925e382a1030fd1fca8450309585d3a58687`.
The v1 configuration, external-validity configuration, lockfile and original
artifact directories are unchanged. Original artifacts stay in the original
`actionstream-rpc-reset-audit-20260918` worktree; they are not copied over or replaced.

| Item | SHA256 |
|---|---|
| v1 configuration | `b2fc8f8a0f7f66eb60cddbd79645647b7313435deaafc84831dd396616749019` |
| v1 formal evidence manifest (123 files) | `9b0e223c00b92f907a7d0c0803c94f41a07e01a46fe052d0ac28bc4f9609329d` |
| Postmortem manifest (77 files) | `f515e1f6324c34d7c0cd4933df3c437971bfe1c34e8ac399fff504aa9171a0be` |
| Postmortem diagnosis JSON | `3ffa024420cc3d847c998d62ac634caada839458a00d49f654d404a5fb8217d4` |

Postmortem diagnostic calls do not count toward either formal 51-run matrix.
Four warm observations do not establish a population p95 or SLA. The original
second-request breakdown (~1.993 s lock wait + ~6.443 s processing) is a proxy
from retained PCAP/timers, not direct lock-acquisition telemetry.

## Persistent execution and lifecycle

`RpcInferenceServer` receives one already-constructed worker and starts one
persistent inference thread for its lifetime. TCP handlers validate frames,
assign connection/request identities, enqueue FIFO jobs and route each result
only to its originating connection. Inference and idle worker reset callbacks
execute on that same thread. Reconnect never constructs a policy or executor.
No concurrent model calls are permitted. CPU result materialization also happens
on the executor; response serialization and injected delivery delay stay in the
connection handler.

Handlers monitor disconnect while waiting. The executor also checks obsolescence
immediately before admission. Reset/cancel closes the client socket and invalidates
its generation; late replies cannot cross a generation even when a connection
attempt finishes concurrently with cancellation. Queued jobs observed obsolete
are skipped. A disconnect can race with compute admission, so already-admitted
work may finish. Neither a timeout nor a disconnect preempts running CUDA.

Active client reset preserves the existing immediate generation-cancellation
contract; it does not issue a concurrent model reset. Idle reset is queued and
serialized with inference. Other clients are not globally reset.

Shutdown stops admission, closes sockets, joins handlers, drains pending jobs as
obsolete and joins the executor after any active callback returns. There is no
bounded shutdown guarantee for a callback that never returns. The serving
process must not claim that a timed join or socket close terminated a kernel.

## Timing definitions

All durations use local monotonic clocks, not cross-host clock subtraction.

| Response field | Definition |
|---|---|
| `server_queue_wait_s` | Immediately before queue insertion to executor admission after the obsolete check. Excludes request decode and pre-inference fault stall. |
| `server_compute_s` | Immediately around the inference callable, including its own preprocess/infer/postprocess. This is callback wall time, not pure kernel time. |
| `server_service_s` | Executor admission to response-ready payload; includes validation, CPU materialization and error handling, excludes queue wait and response delivery. |
| `server_inference_s` | Compatibility interval: queue wait plus worker/validation time. As in v1, it includes scheduling wait and excludes successful output CPU materialization, serialization and delivery. v1 waited for a lock; v2 waits for the executor. |
| `server_lock_wait_s` | Omitted because there is no policy inference lock; no fabricated measured zero. |
| `connection_id` | Unique ID for each accepted TCP connection. |
| `connection_request_ordinal` | 1-based validated inference ordinal on that connection; controls excluded, pre-inference fault disconnects included. |
| `global_request_ordinal` | 1-based validated inference ordinal per fresh server, used for fault matching. Legacy `request_ordinal` remains its alias. |

No CUDA synchronization is introduced into the generic transport. Existing X-VLA
backend synchronization is unchanged. For another asynchronous callback,
`server_compute_s` can exclude GPU completion; `server_service_s` can include it
through output materialization. Neither field is advertised as kernel profiling.
Injected pre-compute errors/disconnects have unavailable compute timing, not zero.
Reset-control timing is excluded from inference distributions.

## Deadline and failure semantics

The legacy CLI defaults remain unchanged. v2 explicitly supplies:

```text
--startup-inference-timeout-s 15.0
--steady-inference-timeout-s 5.0
--action-wait-timeout-s 20.0
--connect-timeout-s 3.0
--control-timeout-s 3.0
```

The first inference request on each newly established TCP connection gets 15 s;
all later inference requests on that connection get 5 s. Control/reset frames do
not consume the first-inference ordinal. A server error does consume it but does
not itself close a healthy connection. After reconnect, including
`periodic_pre_inference_disconnect`, the next request gets 15 s again.

The budget begins at serialized client request admission and includes connection
establishment when necessary, serialization, wire I/O, server queue/service,
injected delivery delay and decoding. Connect is additionally capped at 3 s.
Each receive fragment uses the remaining absolute budget, so a trickling response
cannot continually renew socket timeout. Python encoding/decoding is not
preempted; expiration is checked before results can be accepted.

The 20 s outer action wait covers cumulative retry/backoff time. It can cancel a
later retry before that retry's full startup budget expires; it is not a guarantee
that every retry can run for 15 s. Both budgets exceed the legacy 5 s fallback
only when explicitly configured. No engine-side 5 s watchdog overrides the TCP
startup selection; engine telemetry previews the selected transport budget and
per-request RPC JSONL records the authoritative selected value.

15 s is an experimental safety budget chosen after the observed ~6.6–6.9 s first
request, **not a measured SLA**. There is no inference warmup. If persistent
execution retains or removes the cold penalty, preserve and report the later
experimental observation without assuming the outcome now.

Client counters retain `deadlines`, `transport_errors`, `server_errors`,
`reconnects`, and `cancellations`; new fields separate startup/steady deadlines.
Historical `reconnects` and `process_restarts` still count initial connection
establishment too. `connections_established` names this explicitly;
`reconnections=max(0,connections_established-1)` excludes the first establishment.
`cancelled_invalidated_requests` counts requests ending in deadline/cancellation,
not an invented count of stale packets observed on the wire.

Server counters include `executor_starts`, `inference_calls`, `reset_calls`,
`dropped_obsolete_before_compute` (inference or reset, distinguished in raw events),
and `invalidated_results` (executor completion found obsolete before routing).
The latter is not a total count of every response later lost during delivery.

The v2 server must use `--telemetry-jsonl <unique-run-path>/server_telemetry.jsonl`.
Client runs write `rpc_telemetry.jsonl`, including all attempts, chosen budget,
generation, outcome, UTC timestamps and available response fields. Server logs
retain late completion and queue disposal even when no client accepts a reply.
Final server stdout reports counters. Both COMPLETED and ERROR episode receipts
retain final engine/RPC telemetry and JSONL hashes. These logs contain no
observation/action payloads.

## Preregistration and validation scope

[`rpc_remote_transport_replication_v2.json`](../configs/rpc_remote_transport_replication_v2.json)
retains the exact v1 51-run matrix: Object 5, Spatial 7, Goal 2; no-fault states
25–29; fault states 25–27; identical seeds and all four fault conditions. The v1
[matrix table](rpc-remote-transport-replication-v1-matrix.md) therefore also gives
v2's tuple ordering, but v2 uses a separate run namespace and result ledger.
The same two-host requirement, 20 Hz absolute controller, 300-step cap,
descriptive task success and fresh server process per episode-condition remain.
Historical states are not reclassified as fresh. No v1 count is transferred to v2.

CPU tests cover a timed-out active A continuing remotely while B reconnects,
executor identity/thread reuse, queue disposal after reset/cancel/disconnect/deadline,
shutdown with active/queued work, reset serialization, startup/steady budgets,
periodic fault reconnect, errors, timing decomposition, absolute receive deadlines,
engine integration and ERROR receipts. CI includes these new tests.

Validation receipts and commands are recorded in
[`rpc-replication-v2-validation.md`](rpc-replication-v2-validation.md).
CPU and local LeRobot contracts do not establish remote GPU performance or task
success. GPU outcomes remain unavailable/not run. Commit this protocol and
implementation before any later authorized two-host execution.
