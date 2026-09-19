# Remote inference: resolving a retry/cold-start failure without hiding NO-GO

**Completed result: 51 valid two-host runs, 47 task successes, 7/8 hard gates.
PARTIAL SUPPORT; the full preregistered gate set remains NO-GO.**
The [full report](../../reports/rpc_remote_transport_replication_v2/report.md)
and [frozen result status](../../reports/rpc_remote_transport_replication_v2/final_status.json)
retain the source hashes and negative outcomes.

## Problem

ActionStream supplies chunked X-VLA actions asynchronously to a LIBERO controller
over real two-host RPC. The GPU server and environment client have separate
ownership. Closing a client socket cancels acceptance of its result; it cannot
preempt CUDA already running remotely. Retry behavior must respect that boundary.

## V1 failure

V1 used a single 5 s inference deadline. The connection-first path took roughly
7 s, so the client timed out while remote work continued. Reconnection and retry
entered the serialized server path, interacting with old work and amplifying
waiting. Object task 5/state 25 remained **VALID_NEGATIVE**, permanently counted
as v1 Run 1/51. Execution stopped; the negative run was not replaced.

## Profiling and root cause

Direct-worker measurements separated model initialization (14.762 s), first
inference (6.605 s) and four warm observations (p50 75.60 ms). RPC measurements
showed 6.927 s initially and warm p50 94.29 ms. SSH echo baselines and the warm
RPC-minus-server residual (14.61 ms, including serialization and network handling)
were millisecond-scale, not the seconds-scale bottleneck.

GPU telemetry, packet capture and a reconstructed lock timeline explained why a
timeout did not free remote execution. V1's server timer included infer-lock wait;
the second-request split of about 1.993 s waiting plus 6.443 s processing is a
proxy from retained timers/PCAP, not direct lock-acquisition telemetry.

A new connection to the same warmed worker reproduced a 6.645 s first-use path.
The evidence localized a connection-associated cold path interacting with
serialized retries. It did not isolate the exact CUDA/cuDNN/cuBLAS initialization
components. Four warm calls are not enough for an SLA or population p95.

## Design

V2 separates connection handling from execution ownership. One persistent executor
serializes inference and idle resets; reconnect does not recreate the worker or
executor. Obsolete queued work is dropped, while results from already-admitted
work can be invalidated after cancellation. No CUDA preemption is claimed.

The first inference on each connection gets a 15 s budget; subsequent requests
get 5 s, with a 20 s outer action wait. Explicit queue/compute/service timing and
connection/request identities distinguish backlog from compute. V2 is
**outcome-informed by v1**, not an independent replication. Architecture and
budgets changed together, so their independent causal effects are not identified.

## Validation

The frozen matrix completed 51 valid two-host runs with zero infrastructure-invalid
formal attempts. There were 47 task successes and no no-fault startup/steady
deadlines or transport errors. Cold compute remained about 7 s; no-fault warm
steady RTT was about 104 ms p50. Each fresh server process reported
executor_starts=1.

There were 100 injected disconnects, 99 actual reconnects and 97 demonstrated
later recoveries. Successful reconnect-first compute p50/p95 was 90/113 ms;
recovery including retry/backoff was 166/189 ms. The successful reconnects did
not repay the roughly 7 s cold cost. Within this deployment, the persistent
executor and deadline split removed the timeout/reconnect/cold-start amplification
observed in v1. Cold compute itself was budgeted, not eliminated.

## Tradeoffs and negative results

- **The full gate set remains NO-GO.** The original recovery gate is 97/100,
  FAIL. Runs 23, 39 and 47 ended without a later successful response after their
  last fault, and were neither rerun nor relabeled.
- A separate post-hoc replay found 97/97 uncensored response opportunities
  recovered. Two terminal events had an admitted RPC cancelled at stop; one had
  no later RPC admission. Literal later-request presence is therefore 99.
  Termination-dependent censoring limits this descriptive statistic; it does
  not revise the frozen gate or estimate an unbiased recovery probability.
- At 950 ± 250 ms application delivery, post-first-action polling depletion was
  **87.58–88.30%**. Task success cannot substitute for healthy action supply.
  Raw depletion is also retained: Run1 was about 95% raw but 0% after first action,
  because startup waiting dominated its empty polls.
- All four v2 task failures were Spatial (Runs 8, 9, 36 and 39). Object and Goal
  each achieved 17/17 in v2. Historical Object negatives remain in v1 and H1/H2;
  they are not clean causal baselines or additional v2 failures.
- This was LIBERO simulation, one checkpoint and a historically exposed cohort,
  over SSH-forwarded TCP with application-level faults. GPU telemetry was about
  1 Hz and could miss bursts; observed VRAM outliers remain in the report.
  External validity is **NOT ESTABLISHED** and physical robot safety is
  **NOT EVALUATED**. There is no production-readiness claim.

The post-hoc opportunity-conditioned analysis helps explain the preregistered
failure but does not revise the frozen gate or its NO-GO verdict.
