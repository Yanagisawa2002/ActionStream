# Real RPC and fault-validation boundary

For the **current maintained runtime**, see [listener and confirmed-reset
semantics](rpc-runtime-hardening.md). That post-experiment maintenance defaults
the CLI to loopback and requires explicit non-loopback exposure acknowledgement;
reset now waits for successful executor completion and a matching ACK, with
inference blocked until confirmation. The experiment narrative and v1 CLI
examples below describe the historical implementation, not a recipe for rerunning
the frozen protocol against changed code. Use the recorded implementation commit
to inspect historical semantics. No frozen result is revised by maintenance.

ActionStream now has a real TCP transport in addition to the historical in-process
delivery scheduler and child-process transport. This document separates what is
validated in CPU CI from the completed two-host GPU/simulator experiment.

Current evidence: **v1 VALID_NEGATIVE, stopped after Run 1; postmortem completed;
v2 COMPLETED with PARTIAL SUPPORT, 7/8 hard gates and full gate NO-GO**.
The [completed v2 report](../reports/rpc_remote_transport_replication_v2/report.md)
is the current result ledger. The [technical case study](case-studies/actionstream-remote-inference-case.md)
connects the failure, profiling, revised design and remaining limitations.
External validity is NOT ESTABLISHED; physical robot safety is NOT EVALUATED.

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

V2 serializes inference and idle-reset callbacks on one persistent executor,
separate from TCP connection threads. Reconnecting does not construct a worker or
executor. V1 used a shared model lock; its timing included waiting for that lock.
See [v2 design and timing semantics](rpc-remote-transport-replication-v2.md).

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

The CLI examples below document the historical v1 interface and 5 s budget;
they are not a v2 execution recipe or authorization to run another experiment.
V2 explicitly froze startup/steady inference at 15/5 s, outer action wait at
20 s and connect/control at 3/3 s, with separate per-run server/client telemetry.
Both the 51-run v2 result and v1's stopped result are complete and preserved.

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

This explicit client is the X-VLA/LIBERO transport replication path. It is separate
from the generic `lerobot-rollout` TCP plugin surface. The generic plugin forwards
`lerobot-rollout` robot observations and therefore requires a server worker whose
input contract matches that robot. Do not point the generic plugin at
`XVLARemoteWorker` and assume the observation schemas are interchangeable.

## Aborted external-validity v1: preserved provenance

`configs/rpc_external_validity_v1.json` is unchanged, including its historical
preregistration status. Its disposition is **ABORTED_BEFORE_EXECUTION_NO_RUN**;
it must not be executed. Pre-run audits found historical initialization (including
warmups) and insufficient evidence coverage to certify other identities CLEAN.
Changing a seed does not restore reset freshness; UNKNOWN has not been reclassified.

- States 10-14: H1-R2; 15-19: H2; 20: H2 CUDA canary.
- States 25-29: rejected after Object task 5 historical execution was confirmed.
- States 30-34: 13 confirmed collisions, two unresolved identities.
- States 35-49: insufficient fresh identities, including all 15 consumed on Object 5.
- Exhaustive task audit: `NO_RUN_NO_FRESH_TASK_IDENTITIES`.

The exhaustive audit is retained at
`artifacts/actionstream_rpc_task_selection_audit_20260919/` (local evidence, not
packaged as a tracked experiment result). Its hashes are:

- `audit.json`: `a6ffb6dc3b79602431333a09f7b0c6e291ef0d740e95871e46dd22fe7af9b2e2`
- `evidence_manifest.sha256`: `9e17bfd4dbe81d53caa1e14c8ccd24d7e9f8a2b36006d09b77034dad14f6962b`

The new protocol also records the preceding audit paths/hashes. None of these
negative conclusions is superseded or rewritten. The immutable v1 config SHA256 is
`7d663780396e28c0bc4b867a363e087a40a09cd8bc84e1fab9d24122bc614887`.

## Archived v1 transport replication protocol

`configs/rpc_remote_transport_replication_v1.json` retains its original
**PRE_REGISTERED_NOT_EXECUTED** bytes, but its final disposition is
**VALID_NEGATIVE / stopped after formal Run 1 of 51**. The remaining v1 runs
were not executed. The text below preserves the original protocol specification;
its future-tense instructions are historical, not the current execution state.
V2 reused the cohort in a separate outcome-informed protocol/result namespace,
with revised execution ownership and budgets. It completed 51 valid runs.
Neither protocol restores external-validity freshness.

**Supported question:** does the real two-host TCP implementation preserve its
lifecycle, queue, deadline/reconnect and fault-recovery behavior while driving
actual X-VLA/LIBERO under the declared conditions?

The fixed cohort is Object (`libero_object`, task 5), Spatial (`libero_spatial`,
task 7), and Goal (`libero_goal`, task 2). Each uses states 25-29 with policy seeds
2026091725-2026091729 respectively. Historical execution is explicitly permitted
and disclosed for this narrower question. There is no fresh/unseen-state claim,
model training, finetuning, parameter update, or outcome-conditioned selection.
Historical task success is not a selection criterion.

The [complete ordered matrix](rpc-remote-transport-replication-v1-matrix.md)
contains 15 no-fault runs (3 families x 5 identities) followed by 36 fault runs
(3 families x states 25,26,27 x 4 conditions). Faults remain application delivery
50 +/- 20 ms, 250 +/- 100 ms, 950 +/- 250 ms, and pre-inference disconnect every
7 requests. The JSON freezes their original fault seeds and all 51 row identities.
Ordering is stage, family (Object/Spatial/Goal), identity, then condition.

Run 1 is Object 5/state 25/seed 2026091725/no fault. It is the formal smoke/canary
and counts as **run 1 of 51**, never an extra run. A valid task failure counts;
a valid negative outcome must not be discarded or repeated. Remaining runs start
only after run 1 is structurally valid. Demonstrated installation, environment,
schema or network setup failures can be fixed and retried with the same frozen
parameters, preserving every invalid attempt, diagnosis and raw hashes. Declared
fault effects or mechanism-gate failures after a valid setup are experimental
outcomes, not excuses for infrastructure retries. Task success cannot determine
attempt validity. Conditions cannot change based on interim results.

Two distinct physical hosts and the exact same protocol commit are mandatory.
A owns X-VLA/CUDA inference; B owns LIBERO and the ActionStream client. Before
execution record hostname, machine-id if accessible, boot ID, GPU UUID, full
nvidia-smi output, driver/CUDA runtime, OS/kernel, Python/uv versions, git HEAD,
diff and status on both hosts; verify B -> A TCP port 50051. Before any rented-host
installation/download run `source /etc/network_turbo`.

Keep 20 Hz, the 300-control-step cap, no client delivery scheduler, minimum request
interval 1, and the existing 5 s inference deadline. Existing connection/control
and action-wait/retry defaults are explicitly frozen in `client_runtime_parameters`.
Every episode-condition uses a fresh server process with the episode seed; no
host reboot is required. Capture GPU telemetry on A throughout each episode,
then stop that owned server, hash raw outputs, validate receipt/telemetry,
stale/out-of-order/deadline/error counters and the observation/action/task contract.
Commit the protocol before any GPU execution. Preparation stops at that commit
for the current phase; commands above are instructions, not execution evidence.

## Claims and analysis boundary

Primary analysis is paired transport-condition comparison within this experiment,
using the same suite/task/reset/seed in each compared condition. Only states 25-27
have all four fault-condition pairs; states 28-29 contribute no-fault coverage only.
Task success is descriptive. Do not compare against historical H1/H2 task-success
numbers as if they formed a clean causal baseline.

This experiment does **not** establish unseen-state external validity, unseen-task
generalization, physical robot safety, packet-level Linux netem behavior, remote
CUDA kernel preemption, ActionStream task-success superiority, or independence
from all prior development exposure.

Unchanged hard transport gates are exact episode coverage, zero stale actions
crossing reset, zero accepted out-of-order responses, zero no-fault transport
errors/deadlines, exercised declared faults, later-request recovery and hashed raw
receipts/telemetry. Report these separately from task success.

## Required report structure

The resulting report should contain, for every family and condition:

1. task success and completion steps with failures counted at the episode cap;
2. request rate, queue depletion and queue-age percentiles;
3. RPC round-trip p50/p95/p99 and server inference p50/p95;
4. deadlines, disconnects, reconnects, stale/out-of-order rejection and recovery
   latency;
5. GPU utilization p50/p95 and process VRAM maximum;
6. exact coverage and task/reset/seed identities, both host identities, exact code
   commit/config hashes, and a SHA256 manifest of all raw receipts and telemetry;
7. all attempts, including separately documented invalid infrastructure attempts.

Failure completion steps equal 300. Missing metrics are unavailable with a reason,
never zero-filled. The [completed v2 report](../reports/rpc_remote_transport_replication_v2/report.md)
and its curated metrics now provide these results; v1 remains stopped after Run 1.

A remote result must preserve negative outcomes and family regressions. It should
not collapse network recovery, task success and GPU efficiency into one headline
score.

## Completed v2 negatives and metric semantics

The original recovery criterion remains **97/100, FAIL**, with terminal-edge
failures in Runs 23, 39 and 47. Recomputed post-hoc counts are 100 disconnects,
99 actual reconnects, 97 uncensored response opportunities, 97 demonstrated
recoveries and three terminal cases. Literal later-RPC presence is 99: two
admitted attempts were cancelled at stop. The opportunity-conditioned 97/97 is
descriptive and does not change the gate or full NO-GO verdict.

All 51 servers reported one executor start. Cold compute remained about 7 s;
successful reconnect-first compute p50/p95 was 90/113 ms. No-fault deadlines
and transport errors were zero. This supports removal of the observed v1
amplification loop within this deployment, not elimination of cold compute.

Run1 raw empty-queue polling was about 95%, while post-first-action depletion
was zero. The raw measure includes startup waiting and is retained alongside
the post-first measure. Under 950 ± 250 ms delivery, the latter reached
87.58–88.30%; high task success does not establish healthy queue supply.
V2's four task failures were Spatial; historical Object failures remain separate.
Compatibility reconnect-like counters can include initial establishment;
explicit connections_established=150 is 51 initial + 99 actual reconnections.

The evidence is historical-cohort LIBERO simulation over SSH-forwarded TCP,
with application-level faults and approximately 1 Hz GPU sampling. It does not
establish physical robot safety, packet-level netem behavior, CUDA preemption,
external validity or production readiness.
