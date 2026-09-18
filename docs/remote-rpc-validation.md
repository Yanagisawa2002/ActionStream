# Real RPC and fault-validation boundary

ActionStream now has a real TCP transport in addition to the historical in-process
delivery scheduler and child-process transport. This document separates what is
already executable in CPU CI from the GPU/simulator experiment that still needs a
separate LIBERO client host and GPU inference server.

## What is implemented

`TcpInferenceTransport` uses a persistent TCP connection with a versioned,
length-prefixed JSON protocol. Torch tensors and NumPy arrays are transmitted as
shape/dtype/raw-byte payloads. The transport records connects/reconnects,
cancellations, deadlines, transport/server errors, bytes, round-trip latency and
server inference latency.

The server exposes a trusted `module:factory` worker and supports deterministic:

- application-level response delay and bounded jitter;
- pre-inference disconnects;
- request stalls;
- response drops after inference;
- explicit server errors.

The server serializes model calls so reconnecting clients do not issue concurrent
calls into one policy object. Reset callbacks use the same model lock.

These injected delays happen around a real TCP request/response, but they are not
Linux `tc netem` or a claim about packet-level impairment. A two-host no-fault run
still measures the real network path; the deterministic delay/jitter cases isolate
application-delivery sensitivity on top of that path.

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
is not X-VLA, LIBERO, two-host, or GPU evidence.

## Two-host X-VLA / LIBERO path

The GPU server and the environment client intentionally have different jobs.

### GPU server

The server owns X-VLA, CUDA and the official policy/environment processor chain.
Use the same owned store and EGL environment as the existing native X-VLA
evidence. A fresh server process is required for every episode/condition so the
policy RNG seed and fault ordinal cannot leak across paired runs.

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ACTIONSTREAM_RPC_STORE=/path/to/owned/store
export ACTIONSTREAM_RPC_SUITE=libero_object
export ACTIONSTREAM_RPC_TASK_IDS=5
export ACTIONSTREAM_RPC_SEED=2026091725
export ACTIONSTREAM_RPC_DEVICE=cuda

uv run actionstream-rpc-server \
  --factory actionstream.xvla_rpc_worker:make_xvla_remote_worker \
  --host 0.0.0.0 --port 50051 \
  --ready-file /tmp/actionstream-rpc-ready.json
```

`XVLARemoteWorker` consumes **native LIBERO observations** and returns the final
30 x 7 environment-action chunk after the same official pre/post-processing used
by `LeRobotBackend`.

### LIBERO client

The environment host does not load X-VLA weights. It owns the LIBERO simulator,
ActionStream scheduling/queue/lifecycle state, and the TCP client:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

uv run actionstream-libero-rpc-client \
  --suite libero_object \
  --task-id 5 \
  --initial-state-index 25 \
  --seed 2026091725 \
  --host <gpu-server> --port 50051 \
  --output /tmp/object5-state25-no-fault
```

This explicit client is the X-VLA/LIBERO external-validity path. It is separate
from the generic `lerobot-rollout` TCP plugin surface. The generic plugin forwards
`lerobot-rollout` robot observations and therefore requires a server worker whose
input contract matches that robot. Do not point the generic plugin at
`XVLARemoteWorker` and assume the observation schemas are interchangeable.

## Remote GPU external-validity protocol

`configs/rpc_external_validity_v1.json` preregisters the next execution boundary:

- Object task 5, Spatial task 7 and Goal task 2;
- five candidate no-fault episodes per family, with explicit reset indices and
  policy seeds;
- three candidate fault episodes per family;
- no-fault, three application-delivery delay/jitter conditions, and periodic
  pre-inference disconnects;
- exact task/reset/seed reuse across transport conditions;
- a fresh GPU server process for every episode/condition;
- complete queue, RPC, recovery, task and GPU telemetry.

### Identity preflight result

The required raw-evidence collision audit has now completed, and the experiment is
formally **NO_RUN_INSUFFICIENT_FRESH_IDENTITIES**. The frozen protocol itself was
not modified and no experiment was launched.

| Task family | CLEAN | CONSUMED | UNKNOWN |
|---|---:|---:|---:|
| Object 5 | 0 | 15 | 0 |
| Spatial 7 | 0 | 13 | 2 (35, 39) |
| Goal 2 | 0 | 13 | 2 (35, 39) |

All 45 audited states exist. The audit verified 2,235 deduplicated evidence
records, including 709 warmup initialization records; an executed warmup reset is
treated as consumed. Object states 35–49 are all consumed, so even resolving the
four UNKNOWN states cannot satisfy the protocol requirement of five fresh
identities per family.

No identities were selected, the new-seed mapping is empty, and no protocol
commit was created during the audit. The audit-time HEAD was
`f6fb04ea940fc809564b06552599c1872f6b4e03`. The recorded `audit.json`
SHA-256 is
`07a6af8b07afdce26b5b3d251f13685cbd1bf292c799e4918c50843dd625ba8a`.

The full repository-facing summary is
[`reports/rpc_external_validity_preflight_20260919/README.md`](../reports/rpc_external_validity_preflight_20260919/README.md).

The original hard gates remain the mechanism gates for any future separately
frozen experiment. They are not evaluated here because the identity precondition
failed before execution. Task success, RPC recovery and GPU metrics therefore
remain unmeasured for the planned multi-task remote-host experiment.

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
