# ActionStream

ActionStream is a LeRobot-compatible asynchronous inference backend for chunked
robot policies under delayed delivery. It separates policy inference from response
delivery, rejects stale work across lifecycle boundaries, bounds queue depletion,
and emits telemetry that makes timing failures inspectable.

The repository has two deliberately separate surfaces:

1. **ActionStream Core** — inference scheduling, transport, lifecycle, queue and
   telemetry infrastructure.
2. **Frozen case studies** — controlled experiments that test the runtime and
   adjacent completion/confirmation mechanisms without promoting failed or narrow
   results into general claims.

[![CI](https://github.com/Yanagisawa2002/ActionStream/actions/workflows/ci.yml/badge.svg)](https://github.com/Yanagisawa2002/ActionStream/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

## Current evidence boundary

| Surface | Current status | What it establishes |
|---|---|---|
| Core lifecycle / stale rejection | validated CPU contracts + upstream compatibility CI | reset/stop generations, queue ownership and failure paths are explicit and tested |
| H1-R2 delayed-delivery pipeline | **GO** at frozen 950 ms simulated delivery | the serialized delivery wait was a real request-supply / queue-depletion bottleneck |
| H2 five-step compute budget | **NO-GO** | 61.6% fewer calls did not clear the frozen success gate |
| RGB completion + continuous confirmation | **GO only within one predeclared LIBERO Object task protocol** | 19/19 physically completed fresh trajectories stopped correctly; one VLA noncompletion was not falsely stopped |
| TCP RPC transport | implemented; deterministic real-socket fault contract in CI | real socket framing, deadlines, reconnects, reset invalidation and fault taxonomy |
| Remote-host X-VLA/LIBERO RPC | **not yet executed** | no remote-GPU, real-network-jitter or multi-task claim yet |
| Physical robot safety | **not evaluated** | no hardware safety, E-stop or real-robot claim |

The current remote-RPC preregistration is the separate **transport replication /
robustness** protocol in
[`configs/rpc_remote_transport_replication_v1.json`](configs/rpc_remote_transport_replication_v1.json):
51 two-host episode-condition runs, with historically exposed identities disclosed.
It is **not executed** and does not establish unseen-state or unseen-task validity.
External-validity v1 in
[`configs/rpc_external_validity_v1.json`](configs/rpc_external_validity_v1.json)
is preserved unchanged as an **aborted, never-executed preregistration (NO_RUN)**:
pre-run audits found identity collisions and incomplete historical coverage.
UNKNOWN remains UNKNOWN. No remote-RPC outcome was observed before this change.
See [`docs/remote-rpc-validation.md`](docs/remote-rpc-validation.md) for the audit
hashes, claim boundaries and [exact 51-run matrix](docs/rpc-remote-transport-replication-v1-matrix.md).
Historical language/visual failures remain preserved rather than rewritten; see
[`docs/embodied-development-status.md`](docs/embodied-development-status.md).

## Review surface

- Core inference engine: [`src/actionstream/lerobot_inference.py`](src/actionstream/lerobot_inference.py)
- Local process deadlines: [`src/actionstream/inference_transport.py`](src/actionstream/inference_transport.py)
- Real TCP transport + fault server: [`src/actionstream/rpc_transport.py`](src/actionstream/rpc_transport.py)
- RPC server CLI: [`src/actionstream/rpc_server.py`](src/actionstream/rpc_server.py)
- Deterministic fault matrix: [`src/actionstream/rpc_matrix.py`](src/actionstream/rpc_matrix.py)
- Pinned X-VLA RPC worker: [`src/actionstream/xvla_rpc_worker.py`](src/actionstream/xvla_rpc_worker.py)
- Episode ownership / lock contract: [`docs/lifecycle_contract.md`](docs/lifecycle_contract.md)
- LeRobot plugin: [`integrations/lerobot/`](integrations/lerobot/)
- Core failure-path tests: [`tests/test_lerobot_inference.py`](tests/test_lerobot_inference.py)
- RPC failure tests: [`tests/test_rpc_transport.py`](tests/test_rpc_transport.py)
- H1-R2 report: [`reports/actionstream_transport_h1_r2/report.md`](reports/actionstream_transport_h1_r2/report.md)
- H2 report: [`reports/actionstream_transport_h2_budget_holdout/report.md`](reports/actionstream_transport_h2_budget_holdout/report.md)
- Frozen completion acceptance: [`reports/completion_acceptance_v3_20260915/README.md`](reports/completion_acceptance_v3_20260915/README.md)
- Historical upstream LeRobot proposal: [huggingface/lerobot#4466](https://github.com/huggingface/lerobot/pull/4466)

## Core runtime contract

```text
latest observation mailbox
          │
          ▼
asynchronous policy worker ── timeout / disconnect telemetry
          │
          ▼
transport: direct | child process | TCP RPC
          │
          ▼
cancellable delivery scheduler
          │
          ▼
reset + stale-response rejection
          │
          ▼
age-aligned action queue ── queue age / discard / depletion telemetry
          │
          ▼
bounded hold → latest-only fallback → rollout controller
```

`ActionStreamInferenceEngine` implements the LeRobot lifecycle used by
`lerobot-rollout`:

- `start`, `stop`, `reset`, pause/resume and task revision ownership;
- latest-observation mailbox rather than unbounded inference fan-out;
- age-aligned action chunk replacement and expired-prefix discard;
- reset-generation and out-of-order response rejection;
- bounded hold during depletion and optional latest-only fallback;
- JSONL telemetry without frames, actions, credentials or model payloads.

### Transport semantics

**Direct** keeps the policy in the LeRobot process. A Python thread cannot safely
interrupt an already-blocked CUDA call, so its inference deadline is advisory.

**Process** owns the inference client in a restartable child process. Deadline,
reset or stop can terminate the child and therefore enforce a stronger local
preemption boundary.

**TCP** crosses a real socket boundary with a persistent connection, versioned
length-prefixed frames and tensor/array serialization. It records reconnects,
client deadlines, transport/server errors, bytes, round-trip latency and server
inference latency. Closing the socket invalidates late responses, but does **not**
claim to terminate a CUDA kernel already running on the remote machine. See the
remote-RPC validation document for that boundary.

## Frozen GPU evidence

The current H1/H2 reports use X-VLA on an RTX 5090, three different LIBERO task
families, fresh runtime processes, disjoint reset identities and a fixed 950 ms
delivery delay.

### H1-R2 — delivery pipeline mechanism GO

H1-R2 changed one variable: whether the 950 ms response-delivery wait blocked the
single inference worker.

| Metric | Serialized aligned | Pipelined aligned |
|---|---:|---:|
| Success | 10/15 | 13/15 |
| Mean steps, failures = 300 | 201.33 | 137.93 |
| Median request rate | 0.902/s | 10.110/s |
| Queue depletion | 46.6% | 0.0% |
| Inference calls | 159 | 1,261 |
| Steady process VRAM max | 4,844 MiB | 4,844 MiB |

Request supply increased 11.43x and all 1,407 depletion pulls disappeared. Mean
paired completion steps improved by 63.40 with 95% CI `[-118.20, -5.47]`.
Success increased by three episodes, but its paired CI crossed zero and Object
regressed from 5/5 to 3/5. The supported claim is the pipeline bottleneck, not
universal policy superiority.

### H2 — fixed compute budget NO-GO

H2 kept the pipelined runtime and changed the minimum request interval from one
to five controller steps.

| Metric | Unbounded pipeline | Five-step budget |
|---|---:|---:|
| Success | 14/15 | 13/15 |
| Mean steps, failures = 300 | 125.67 | 139.80 |
| Inference calls | 1,109 | 426 |
| Median request rate | 9.901/s | 3.462/s |
| Queue depletion | 0.0% | 0.0% |

The budget saved 61.6% of calls while keeping the queue supplied, but lost one
paired success and failed the preregistered success gate. It remains a measured
compute/success frontier, not a promoted scheduler default.

## Completion / confirmation case study

The September 15 frozen acceptance evaluates a separate RGB completion verifier
plus eleven-control continuous confirmation on one finite task: LIBERO Object 5,
tomato sauce into the basket.

On 20 previously unused procedural trajectories, 19 tasks physically completed
and all 19 stopped correctly. The remaining VLA trajectory never physically
completed and never received a success declaration. The run recorded zero
premature first stops, zero missed completed episodes, median confirmation delay
0.55 s, maximum delay 1.50 s, and 19/19 stable two-second post-stop continuations.

This GO belongs to the **adapted verifier + continuous confirmation** combination.
It is not permission to stop on one raw model output, and it does not establish
other tasks, objects, natural occlusions, real robots or rare-error rates.

## Real RPC and fault matrix

Run the deterministic CPU/loopback contract:

```bash
uv run actionstream-rpc-matrix \
  --config configs/rpc_fault_matrix_v1.json \
  --output /tmp/actionstream-rpc-matrix
```

It uses actual TCP sockets and frozen injected failure cases for healthy traffic,
delivery jitter, deadlines, pre-inference disconnects, response drops and server
errors. This is transport evidence only; it is intentionally not presented as
remote-host GPU evidence.

A trusted inference worker can be served with:

```bash
uv run actionstream-rpc-server \
  --factory my_package.worker:make_worker \
  --host 0.0.0.0 --port 50051
```

The pinned X-VLA development worker is available as
`actionstream.xvla_rpc_worker:make_xvla_remote_worker`. Run a fresh server process
per LIBERO family/condition so model state and fault injection remain attributable.

## Install and verify

Requirements are Python 3.12, Git and `uv` 0.11.32. Model weights, LIBERO assets
and simulator dependencies retain their upstream terms and are not redistributed.

```bash
git clone https://github.com/Yanagisawa2002/ActionStream.git
cd ActionStream
uv sync --locked --all-packages
uv run lerobot-rollout --help
```

The real CLI help must expose `actionstream` as an inference-engine choice.

Run the main review checks:

```bash
uv run ruff check src tests scripts/release scripts/engineering integrations/lerobot
uv run pytest tests/test_lerobot_inference.py \
  tests/test_lerobot_lifecycle.py \
  tests/test_lerobot_lifecycle_boundaries.py \
  tests/test_lerobot_plugin.py \
  tests/test_rpc_transport.py
uv run actionstream-rpc-matrix \
  --config configs/rpc_fault_matrix_v1.json \
  --output /tmp/actionstream-rpc-matrix
uv run python scripts/release/audit_public_release.py
```

## LeRobot integration boundary

ActionStream packages a third-party `lerobot-rollout` backend. The repository
currently pins a small LeRobot registry patch because upstream's factory surface
has historically dispatched only built-in inference config types. Upstreaming
that generic extension point is tracked separately from ActionStream's runtime
claims; an unmerged proposal must not be described as upstream support.

The CI contract checks the pinned upstream checkout, applies the patch, discovers
the ActionStream plugin through the real CLI path, and runs both downstream
lifecycle tests and LeRobot's rollout tests.

## Evidence and release discipline

Tracked evidence contains aggregate tables, paired effects, compact figures,
representative media, frozen protocols and content hashes. Raw traces, full
episode videos, model weights, simulator assets and machine-local archives remain
outside ordinary Git when licensing or size makes that necessary.

Negative results stay visible. H2 remains NO-GO; earlier language/visual failures
remain historical records even after the later single-task completion acceptance.
New GPU experiments must use untouched identities and freeze protocol changes
before outcomes are collected.

## Limitations

- H1/H2 cover one X-VLA checkpoint in LIBERO simulation.
- H1's 950 ms condition was simulated in-process delivery, not actual network
  jitter; the new TCP transport exists specifically to close that evidence gap.
- The preregistered remote-host multi-task RPC run has not yet been executed.
- TCP client deadlines do not terminate already-running remote CUDA kernels.
- The completion acceptance covers one finite simulated Object task and one
  trained checkpoint.
- No physical robot, hardware safety controller or E-stop claim exists.
- Short deterministic transport tests are not a production soak, CUDA-OOM test or
  GPU-driver recovery result.

## License

ActionStream source is Apache-2.0. LeRobot, X-VLA, LIBERO, simulator assets and
all policy weights remain under their own terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
