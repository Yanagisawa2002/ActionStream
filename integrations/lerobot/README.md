# ActionStream backend for `lerobot-rollout`

Maintained TCP runtime: the server CLI defaults to `127.0.0.1`; non-loopback
listeners require `--allow-unauthenticated-remote` and still have no authentication.
The plugin exposes `tcp_reset_timeout_s` (default 20 s) independently of connect,
inference and legacy control budgets. Episode reset invalidates old responses and
waits for a successful remote reset ACK; failure propagates and blocks inference.
Low-level clients must also call reset before first use. See
[`docs/rpc-runtime-hardening.md`](../../docs/rpc-runtime-hardening.md) for the full
contract, recovery lifecycle and historical-v2 distinction.

This small package registers `--inference.type=actionstream` with LeRobot's
rollout inference factory. It depends on the generic third-party inference-engine
registry proposed in
`upstream/lerobot/0001-feat-rollout-allow-third-party-inference-engines.patch`.
A refreshed, smaller patch against LeRobot main `30074f7` is retained as
`upstream/lerobot/0002-current-main-third-party-inference-builders.patch`.

The backend executes action-chunk inference on one worker, aligns returned chunks
to observation age, rejects pre-reset and timed-out responses, bounds queue-empty
holds, and exposes queue/fallback/transport telemetry. It is validated with mock
transport failures and simulator rollouts; it is not a real-robot safety claim.

From a clean ActionStream clone, one command installs the root package, the
pinned LeRobot revision, and this plugin:

```bash
uv sync --locked --all-packages
```

`uv run lerobot-rollout --help` then exercises LeRobot's actual CLI subprocess
and must list `actionstream` as an inference choice.

## Transport modes

`transport_mode=direct` keeps local CUDA policy inference in the worker thread.
Its timeout is advisory because Python cannot safely interrupt a blocked CUDA
call; telemetry therefore records `deadline_enforced=false`.

`transport_mode=process` owns a restartable inference client in a child process.
It can enforce a hard local deadline by terminating that child on timeout, reset,
or stop before accepting more work.

`transport_mode=tcp` crosses a real socket boundary. Configure `tcp_host`,
`tcp_port`, `tcp_connect_timeout_s`, and `tcp_control_timeout_s`. The client uses
versioned length-prefixed frames, reconnects after transport failure, and closes
the connection on request deadlines or lifecycle invalidation. A late response
from an invalidated connection cannot cross the engine's episode generation.

TCP I/O deadlines are not remote CUDA cancellation. If the server has already
entered a kernel, closing the client connection does not kill that kernel. Use a
server-side process-isolation layer when remote compute itself must be preempted.

## RPC server contract

A trusted `module:factory` can be exposed with:

```bash
uv run actionstream-rpc-server \
  --factory my_package.worker:make_worker \
  --host 0.0.0.0 --port 50051
```

The factory returns a callable `(observation, task) -> torch.Tensor`; a callable
`reset()` method is used when present. For the generic `lerobot-rollout` plugin,
the worker must consume the same raw robot-observation contract that the rollout
engine forwards. ActionStream does not auto-convert arbitrary robot observations
into LIBERO-native observations.

The repository also contains
`actionstream.xvla_rpc_worker:make_xvla_remote_worker`, but that worker consumes
native LIBERO observations and belongs to the separate two-host external-validity
runner documented in `docs/remote-rpc-validation.md`. It should not be used as a
generic `lerobot-rollout` server without an explicit observation adapter.

The server supports deterministic response delay/jitter, stalls, disconnects,
dropped responses, and injected server errors. These knobs are for controlled
fault experiments, not production traffic shaping.

Run the CPU/loopback failure contract with:

```bash
uv run actionstream-rpc-matrix \
  --config configs/rpc_fault_matrix_v1.json \
  --output /tmp/actionstream-rpc-matrix
```

Set `telemetry_jsonl_path` on the rollout backend to append schema-v1 lifecycle,
request, queue, discard, depletion, recovery, and fallback events. Frames,
observations, actions, credentials, and model payloads are deliberately excluded.
The field contract is documented in `docs/telemetry_jsonl_v1.md` and validated by
`schemas/actionstream.telemetry.v1.schema.json`.
