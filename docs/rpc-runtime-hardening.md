# Maintained RPC runtime: listener exposure and confirmed reset

This maintenance change fixes reusable-runtime defects found after the completed
v2 experiment. It does not rerun or change that experiment. The implementation at
`e33eb98c94013da14109ab2ac710414fea63ebaf` and results at
`4e426a6913694427a1c9bbf442397acfdd368cf3` remain historical evidence.
V2 remains **PARTIAL SUPPORT, 7/8 hard gates, full gate set NO-GO**.
Frozen configs, reports and raw manifests retain their original bytes.

## Listener boundary

The ordinary command now binds to loopback:

```bash
actionstream-rpc-server --factory my_worker:build_worker
```

The CLI and `RpcInferenceServer` both default to `127.0.0.1`. The CLI refuses any
non-loopback host, including `0.0.0.0` and `::`, unless the operator supplies
`--allow-unauthenticated-remote`. Validation happens before loading the worker or
creating a listener. `localhost` and numeric loopback addresses need no opt-in.
Other hostnames conservatively require opt-in; they are not trusted as loopback
based on a transient DNS lookup.

```bash
actionstream-rpc-server --factory my_worker:build_worker \
  --host 0.0.0.0 --allow-unauthenticated-remote
```

The acknowledgement flag **does not add authentication or encryption**. The
startup JSON/ready file states `authentication: "none"` and
`listener_exposure: "loopback"` or `"unauthenticated_remote"`. Protect remote
access with SSH, TLS termination, VPN/firewall controls or another operator-owned
boundary. This patch implements none of those mechanisms and makes no production
security certification or physical robot safety claim. The Python server API
still accepts an explicitly supplied remote host; the exposure flag is a CLI
safeguard, not an access-control protocol.

## Reset is an acknowledged state barrier

Every new `TcpInferenceTransport` starts with `reset_required=True`. Call
`reset()` successfully before its first inference, even for a stateless worker.
A new socket or a replacement client is not proof that a persistent worker has
clean state. There is no bypass flag.

For an episode reset:

1. Mark reset required, advance the client generation and close the old socket
   immediately. Any old in-flight or locally queued response/request is invalidated.
2. Send a reset control request on the new connection. It enters the existing
   single persistent executor after any already-admitted policy call.
3. Wait for the worker reset callback to finish and a matching version/request-ID
   ACK to arrive. Only then clear reset required and return successfully.
4. Reject `infer()` while reset is pending or uncertain. On any failed reset,
   raise `RemoteResetError` (an `InferenceTransportError` subclass), preserving
   the original exception as its cause. No inference can silently continue.

The resulting order is old inference completion, worker reset, ACK, new episode
inference. Closing a socket does **not** preempt CUDA; admitted remote work may
continue after a client timeout or cancellation. A queued obsolete job may be
dropped, and an already-running reset may finish after the client gives up. Neither
case is a confirmed client reset without a valid ACK. A concurrent cancel or
another reset invalidates an older ACK; it cannot reopen inference accidentally.

Worker reset exceptions produce an `error` response, never ACK. No callback means
the factory/operator explicitly supplies a **stateless no-op reset** contract.
Arbitrary stateful workers must expose a working `reset()` callback. The startup
receipt identifies `reset_capability: "callback"` or `"stateless_noop"`.

This is a cooperative single-owner episode contract. Multiple independent
controllers sharing one mutable policy must coordinate ownership externally;
there is no cross-client episode lease or isolation mechanism in this patch.

## Independent reset budget and recovery

`TcpInferenceTransport(..., reset_timeout_s=20.0)` uses an absolute budget from
the start of reset. It includes waiting for local request I/O to unwind,
hostname resolution, connection establishment, executor queue wait behind old work, callback execution,
response delivery and ACK validation. The existing connect timeout remains an
additional connect-phase cap. Encoding and callbacks are not forcibly preempted.

DNS resolution and all candidate socket addresses share that connect-phase cap
and the remaining reset budget. The OS resolver itself is not cancellable: one
daemon lookup per client may finish later, but cannot open or publish a socket.
Retries reuse an outstanding lookup rather than accumulate blocked resolver
threads. A late DNS result never clears reset-required state; explicit reset and
its ACK are still necessary.

Twenty seconds is a conservative configurable waiting allowance for one admitted
call plus reset/transport overhead. It avoids inheriting the ordinary 3 s control
budget; it is not a measured latency guarantee. A worker can block longer or
forever, so callers must handle explicit reset failure. Set a budget appropriate
to the worker and shutdown policy. No experimental inference timeout changed.

Public configuration paths are:

| Surface | Reset option |
|---|---|
| Python transport | `reset_timeout_s=20.0` |
| LeRobot plugin config | `tcp_reset_timeout_s=20.0` |
| Native LIBERO client CLI | `--reset-timeout-s 20.0` |

On failure, fix the underlying worker/network problem and call `reset()` again.
Only a positively acknowledged explicit reset clears the transport's required
state. Reconnecting or reconstructing the transport is insufficient. At engine
level, a reset failure also preserves the existing sticky fatal-state contract:
recover the remote reset and use the existing stop/start lifecycle before
resuming the engine. This patch does not weaken the engine's fatal-state policy.

```python
from actionstream.rpc_transport import TcpInferenceTransport

transport = TcpInferenceTransport("127.0.0.1", 50051, reset_timeout_s=20.0)
try:
    transport.reset()  # Must complete before first/new-episode inference.
    actions = transport.infer(observation, task, timeout_s=5.0)
finally:
    transport.close()
```

Existing direct/process transport behavior is unchanged. Low-level TCP users who
previously called `infer()` immediately after construction must add the explicit
reset. The native episode runner and normal LeRobot episode reset path already
call reset; the CPU fault-matrix runner now does so before its unchanged cases.
Successful `reset()` retains the boolean result indicating whether an inference
was cancelled; it returns only after ACK, and may block for the reset budget.

## Minimal telemetry and deferred interface cleanup

RPC client telemetry adds `reset_requests`, `reset_successes`, `reset_failures`,
`reset_timeouts` and `reset_required`. Client success means an accepted ACK.
Server `reset_successes`/`reset_failures` and `reset_succeeded`/`reset_failed`
events describe callback/no-op outcomes; a successful callback does not prove
the ACK reached the client. The existing `reset_calls` counts callback invocations.
Reset controls remain outside inference ordinals, deadlines and latency samples.

The CLI's unused duplicate stop event was removed; server shutdown already owns
the stop event that admission and execution observe.

The API review deliberately retains these compatibility surfaces:

- `control_timeout_s` / `tcp_control_timeout_s` and their CLI spelling remain
  available for legacy controls/config readers. Confirmed reset always uses its
  own budget; neither inference nor outer action-wait budgets are repurposed.
- `process_restarts`, `reconnects` and `server_inference_s` remain compatibility
  fields. Explicit connection counters and queue/compute/service timing are the
  clearer RPC measurements, but removing aliases would break consumers and frozen
  analysis. They are not added to the generic engine interface.
- Startup/steady inference timing and fault/ordinal telemetry stay in the RPC
  module. Extracting an experiment-only reporting API or collapsing public timeout
  names is deferred because it would expand the correctness patch.
- The generic engine's legacy `transport_mode="direct"` label for injected TCP
  remains unchanged; use RPC telemetry for transport identity. A separate enum/API
  migration is deferred.

See [maintenance validation](rpc-runtime-hardening-validation.md) for focused
socket regressions, full CPU checks and preservation verification. These checks
validate maintained software; they do not revise the [frozen v2 result](../reports/rpc_remote_transport_replication_v2/report.md).
