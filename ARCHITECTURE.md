# ActionStream architecture

ActionStream is a fault-aware asynchronous inference runtime for chunked robot
policies. Its core design goal is not simply to run policy inference off-thread;
it is to make ownership and invalidation explicit when observations, inference,
delivery, reset, and action dispatch no longer happen synchronously.

## System boundary

```text
control plane
reset / stop / task revision / generation
                   |
                   v
data plane
observation mailbox
        |
        v
policy worker -> transport -> delayed delivery
        |                         |
        +-------------------------+
                  |
                  v
generation + ordering check
                  |
                  v
age-aligned action queue
                  |
                  v
rollout controller
```

The control plane decides which work is still allowed to affect the current
episode. The data plane moves observations and action chunks. A result is useful
only if both planes still agree on its ownership when the result reaches the queue
commit point.

## Module ownership

| Module | Primary ownership |
|---|---|
| `inference_engine.py` | lifecycle, generation, worker ownership, task revision, failure state |
| `delivery_scheduler.py` | delayed response heap and delivery observer |
| `action_queue.py` | chunk validation, age alignment, stale-prefix discard, hold/fallback, queue commit |
| `inference_types.py` | configuration, telemetry snapshots, action provenance, queue/delivery records |
| `inference_transport.py` | direct and child-process transport abstractions |
| `rpc_client.py` | TCP connection, request admission, cancellation, reset-required state |
| `rpc_protocol.py` | versioned frame encoding and tensor/array serialization |
| `rpc_server_runtime.py` | socket listener, connection handlers, deterministic fault injection |
| `rpc_executor.py` | single persistent remote worker/executor ownership |
| `rpc_telemetry.py` | TCP client telemetry snapshot |
| `lerobot_inference.py` / `rpc_transport.py` | compatibility import surfaces |

## Control plane

Every episode is identified by an engine generation. Reset, stop, and relevant
lifecycle transitions invalidate work from older generations before that work can
be committed back into the current action queue.

Task changes carry a separate task revision. Action receipts include the engine
identity, generation, task revision, request ordinal, source step, and task so a
serialized dispatcher can reject work that is no longer current.

A reset does not mean "clear a Python list." It changes ownership:

1. the current generation is invalidated;
2. queued and delayed work from that generation becomes ineligible;
3. provider/transport reset is performed according to its transport boundary;
4. only work admitted under the new generation can repopulate the queue.

## Data plane

The engine keeps one latest-observation mailbox rather than an unbounded inference
fan-out. If observations arrive while inference is busy, newer observations
supersede older pending observations.

A returned action chunk carries the source controller step. At queue commit:

- the engine re-reads the current controller step;
- expired prefix actions are discarded;
- an entirely stale chunk is rejected while a usable queue exists;
- after queue depletion, the configured bounded-hold/latest-only policy may be
  used;
- scheduled responses older than an already-delivered response are rejected.

The critical queue commit still acquires the engine condition and queue lock
together. Splitting `delivery_scheduler.py` and `action_queue.py` does not create
another generation authority, background worker, or commit lock.

## RPC ownership

The TCP runtime separates four state domains.

**Connection state** lives in `rpc_client.py`: socket creation, DNS resolution,
reconnect, absolute request budgets, bytes, and round-trip telemetry.

**Request/reset state** also lives in the client: generation invalidation,
cancellation, reset-required poisoning, ACK validation, and fail-closed recovery.

**Socket/server state** lives in `rpc_server_runtime.py`: listener ownership,
connection threads, framing dispatch, and deterministic fault injection.

**Executor ownership** lives in `rpc_executor.py`: one queue and one persistent
executor thread are the only path that invokes the remote worker or its reset
callback. Reconnecting a socket does not create another policy owner.

## Fault boundaries

| Boundary | What ActionStream can invalidate | What it cannot claim |
|---|---|---|
| Direct Python call | acceptance of a late result | interruption of a blocked Python/CUDA call |
| Child process | process/request ownership; child can be terminated | hardware-level safety |
| TCP client | socket/request generation and late response acceptance | interruption of a remote CUDA kernel already running |
| RPC reset | future inference until a reset ACK confirms state | authentication, encryption, or physical E-stop behavior |

Three distinctions are central to the project:

> Cancelling a client request is not remote compute preemption.

> Reconnecting a socket is not resetting policy state.

> Task success is not evidence of healthy request supply.

The last point matters because the frozen remote study observed high task success
while post-first-action polling depletion remained very high. Application outcome
and runtime health must therefore be measured separately.

## Maintained invariants

1. A reset or stop can make old work ineligible before that work reaches queue
   commit.
2. The latest-observation mailbox bounds inference fan-out.
3. Delivery ordering and queue alignment are checked at the existing queue commit
   boundary.
4. Remote inference and remote reset are serialized by one persistent executor.
5. A new TCP client starts reset-required; socket recreation alone cannot certify
   remote policy state.
6. Timeout/cancellation telemetry is separated from task-level success.
7. Historical experiment outcomes are not rewritten when maintained runtime
   semantics improve.

## Review path

For runtime correctness, read the modules in this order:

```text
inference_types.py
    -> inference_engine.py
    -> delivery_scheduler.py
    -> action_queue.py
    -> inference_transport.py

rpc_protocol.py
    -> rpc_client.py
    -> rpc_server_runtime.py
    -> rpc_executor.py
```

Then use [docs/testing.md](docs/testing.md) to jump from each invariant to its
regression tests.
