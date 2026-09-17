# Real RPC and fault-validation boundary

ActionStream now has a real TCP transport in addition to the historical in-process
delivery scheduler and child-process transport. This document separates what is
already executable in CPU CI from the GPU/simulator experiment that still needs a
separate client host and GPU server.

## What is implemented

`TcpInferenceTransport` uses a persistent TCP connection with a versioned,
length-prefixed JSON protocol. Torch tensors and NumPy arrays are transmitted as
shape/dtype/raw-byte payloads. The transport records connects/reconnects,
cancellations, deadlines, transport/server errors, bytes, round-trip latency and
server inference latency.

The server exposes a trusted `module:factory` worker and supports deterministic:

- response delay and bounded jitter;
- pre-inference disconnects;
- request stalls;
- response drops after inference;
- explicit server errors.

The server serializes model calls so reconnecting clients do not issue concurrent
calls into one policy object. Reset callbacks use the same model lock.

## Cancellation boundary

A TCP request deadline is a hard deadline on the client I/O wait, not a claim that
the remote CUDA kernel has been interrupted. Closing a socket or resetting the
client generation prevents a late response from being accepted by the current
ActionStream episode. If the remote worker already entered CUDA, that work may
continue until it returns.

Use server-side process isolation when the compute itself must be preempted.
ActionStream's existing `ProcessInferenceTransport` demonstrates that stronger
local contract by killing its child process. The TCP transport intentionally does
not claim equivalent remote GPU termination.

## CI fault contract

Run:

```bash
uv run actionstream-rpc-matrix \
  --config configs/rpc_fault_matrix_v1.json \
  --output /tmp/actionstream-rpc-matrix
```

This creates real loopback TCP connections and verifies the frozen failure
taxonomy for healthy traffic, delivery jitter, deadlines, disconnects, dropped
responses and explicit server errors. It is transport/lifecycle evidence only. It
is not X-VLA, LIBERO, remote-host, or GPU evidence.

## Pinned X-VLA worker

On the GPU server, use the same owned store and EGL environment as the existing
native X-VLA evidence:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ACTIONSTREAM_RPC_STORE=/path/to/owned/store
export ACTIONSTREAM_RPC_SUITE=libero_object
export ACTIONSTREAM_RPC_TASK_IDS=5
export ACTIONSTREAM_RPC_DEVICE=cuda

uv run actionstream-rpc-server \
  --factory actionstream.xvla_rpc_worker:make_xvla_remote_worker \
  --host 0.0.0.0 --port 50051 \
  --ready-file /tmp/actionstream-rpc-ready.json
```

Run a separate fresh server process per LIBERO family/condition. The current
worker deliberately reuses `LeRobotBackend`, so policy construction and official
pre/post-processing stay aligned with the prior frozen evidence rather than
creating a second model-loading path.

The LeRobot client selects:

```text
--inference.type=actionstream
--inference.transport_mode=tcp
--inference.tcp_host=<gpu-server>
--inference.tcp_port=50051
--inference.delivery_scheduler_enabled=true
--inference.minimum_request_interval_steps=1
```

Exact rollout arguments remain the same as the pinned ActionStream/LeRobot
benchmark environment.

## Remote GPU external-validity protocol

`configs/rpc_external_validity_v1.json` preregisters the next execution boundary:

- Object task 5, Spatial task 7 and Goal task 2;
- five candidate no-fault resets per family;
- three candidate fault resets per family;
- no-fault, mild jitter, WAN-like jitter, long-delay jitter and periodic
  disconnect conditions;
- exact identity reuse across network conditions;
- fresh runtime processes by family/condition;
- complete queue, RPC, recovery, task and GPU telemetry.

Candidate reset indices 20-24 are not authorized merely because they appear in
the config. Before GPU execution, the orchestrator must verify that none collide
with any consumed ActionStream evidence. A collision makes the run `NO_RUN`; new
identities must be frozen before collecting outcomes.

The hard gates are mechanism gates: exact declared coverage, zero stale actions
crossing reset, zero accepted out-of-order responses, no unexplained errors in the
no-fault condition, exercised injected faults, recovery after faults, and hashed
raw receipts. Task success by family is reported as an outcome rather than being
preselected as a superiority gate.

## Required report structure

The resulting report should contain, for every family and condition:

1. task success and completion steps with failures counted at the episode cap;
2. request rate, queue depletion and queue-age percentiles;
3. RPC round-trip p50/p95/p99 and server inference p50/p95;
4. deadlines, disconnects, reconnects, stale/out-of-order rejection and recovery
   latency;
5. GPU utilization p50/p95 and process VRAM maximum;
6. exact task/reset/seed identities and content hashes of raw receipts.

A remote result must preserve negative outcomes and family regressions. It should
not collapse network recovery, task success and GPU efficiency into one headline
score.
