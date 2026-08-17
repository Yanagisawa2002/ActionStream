# M6-G0 SO-101 + ACT portability audit

Formal outcome: **NO-GO**. The smallest credible port is documented for feasibility only; M6-G0 does not implement or authorize physical-robot deployment.

## Smallest credible insertion

The client insertion point is in `RobotClient.receive_actions`, after deserialization and before `_aggregate_action_queues`, with corresponding request metadata captured in `control_loop_observation` ([`RobotClient.receive_actions`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/robot_client.py#L269-L337), [`RobotClient.control_loop_observation`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/robot_client.py#L408-L453)). A custom `aggregate_fn` is insufficient because it sees only two tensors for an already-matched timestep ([`AGGREGATE_FUNCTIONS`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/configs.py#L29-L34), [`RobotClient._aggregate_action_queues`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/robot_client.py#L224-L267)).

The credible opt-in path would:

1. Add `request_id`, `request_generation`, and explicit observation/intended-step metadata to the timed request/action objects.
2. Have `PolicyServer` preserve and echo that provenance when producing the chunk.
3. Have `RobotClient` compute elapsed control steps at arrival, reject old generations, remove the elapsed prefix, and atomically replace the queue.
4. Make startup block, bounded repeat-last-command, and watchdog expiry explicit telemetry and configuration.

Existing timestamps can be retained, but they are not sufficient alone: the current first action reuses the observation timestep and the timed objects carry no request generation ([`PolicyServer._time_action_chunk`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/policy_server.py#L312-L320), [`TimedData, TimedAction, TimedObservation`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/helpers.py#L202-L235)).

## Component impact

| Component | Required change | Conclusion |
|---|---|---|
| SO-101 follower | No scheduler algorithm in the driver; configure target clipping, watchdog/stop behavior, and log the command actually sent. | Safety integration required, no hardware work in M6. |
| ACT policy | None to architecture, weights, or training. | ACT remains an ordinary chunk producer. |
| PolicyServer | Echo request ID/generation and explicit origin/intended-step convention. | Required for robust provenance. |
| RobotClient | Add opt-in arrival-age alignment, generation rejection, atomic replacement, and explicit underrun/hold states. | Primary implementation surface. |
| Timed data/config | Add provenance fields and scheduler/hold/watchdog configuration. | Required and additive. |
| Aggregate callback | No sufficient implementation is possible with only `(old, new)` tensors. | Not the port boundary. |

## Files and estimated surface

A credible upstream implementation would touch four production files and their focused tests:

- `src/lerobot/async_inference/helpers.py`
- `src/lerobot/async_inference/configs.py`
- `src/lerobot/async_inference/policy_server.py`
- `src/lerobot/async_inference/robot_client.py`
- `tests/async_inference/test_helpers.py`
- `tests/async_inference/test_policy_server.py`
- `tests/async_inference/test_robot_client.py`

Estimated implementation size is roughly 150-250 production lines plus tests and benchmark fixtures. This is an engineering estimate, not a validated port measurement. Pickled timed dataclasses already travel inside the existing RPC payload ([`RobotClient.send_observation`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/robot_client.py#L183-L215), [`PolicyServer.SendObservations`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/policy_server.py#L173-L212)); a protobuf schema change is not obviously required, but mixed-version client/server compatibility would need an explicit test.

## Telemetry and safety controls

Required telemetry: request ID/generation, observation timestamp and control step, inference start/end, client receive/insertion step, prefix drop count, queue depth before/after, executed intended/actual index, hold/underrun reason, command age, and command actually sent.

Required pre-hardware controls: startup block until a valid command, bounded hold duration, communication watchdog, queue/generation reset on reconnect or episode reset, monotonic local arrival timing, explicit clock-skew handling for cross-host timestamps, joint/velocity/workspace limits, accessible emergency stop, and SO-101 relative-target clipping. LeRobot exposes `max_relative_target`, and `SOFollower.send_action` clips before writing motor goals ([`SOFollowerConfig.max_relative_target`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/robots/so_follower/config_so_follower.py#L27-L39), [`SOFollower.send_action`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/robots/so_follower/so_follower.py#L205-L230)).

## Compatibility risks and upstreamability

- The official action label convention starts at the observation label; changing it globally could break queue/tests, so the new convention must be opt-in or separately represented.
- Pickle compatibility across mixed client/server versions needs explicit fallback behavior.
- Host wall clocks are used for cross-host timestamp diagnostics; safety decisions should prefer a local monotonic arrival clock or verified clock sync.
- Replacing a queue after a fully stale result can starve execution; a bounded fallback policy must be specified before hardware.
- Extra observation/request generation must remain compatible with the server's queue-of-one replacement and duplicate/similarity filters ([`PolicyServer._obs_sanity_checks, _enqueue_observation`](https://github.com/huggingface/lerobot/blob/62600065cdb349c1e41b0403c511f12ebfa686eb/src/lerobot/async_inference/policy_server.py#L268-L310)).

An opt-in scheduler plus diagnostics could plausibly be reviewed as an upstream PR because the surface is localized and ACT does not change. However, the predeclared M6 result is not `PORT GO`, and official scheduling was not equivalent-or-better in enough families for `TOOLING GO / ALGORITHM OVERLAP`. The formal next step is therefore **termination of this direction**, not a hardware port or PR.
