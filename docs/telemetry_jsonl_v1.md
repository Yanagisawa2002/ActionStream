# ActionStream telemetry JSONL v1

`ActionStreamInferenceEngine` can append one UTF-8 JSON object per event by
setting `telemetry_jsonl_path`. Every line is flushed before the call returns,
so a terminated rollout retains all events emitted before termination.

Required envelope fields are `schema_version`, `engine_id`, `event`,
`utc_unix_ns`, `monotonic_ns`, `pid`, and `thread`. Event-specific scalar fields
carry epochs, source/control steps, queue depth and age, latency, discard,
depletion, timeout, cancellation, recovery, and fallback provenance. Unknown
event-specific fields are allowed so schema v1 can add non-breaking telemetry.

The stream intentionally excludes observations, camera frames, action tensors,
task payloads, credentials, and exception messages. Consumers should join on
`engine_id`, order within one process by `monotonic_ns`, and treat wall-clock
timestamps as cross-process display metadata only.

For request and action events, combine `engine_id`, `epoch`, and
`request_ordinal`; `task_revision` identifies the requested task captured at
admission. Reset/stop events carry the new epoch while delayed outcomes retain
the old one. `response_delivery_status` carries the observer status and commit
time/step where available. Event emission and callbacks occur outside commit
locks and may follow reset; emission order is not the queue's commit order.
See the [lifecycle contract](lifecycle_contract.md) for metric lifetimes,
transport snapshot limits, and consumer revision checks.

`deadline_enforced` distinguishes hard process isolation from the direct local
CUDA mode. A direct-mode timeout is detected only after the call returns;
process mode terminates the isolated client at the request deadline and starts
a new child for the next request.

The machine-readable contract is
`schemas/actionstream.telemetry.v1.schema.json`.
