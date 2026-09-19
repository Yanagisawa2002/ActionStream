# Remote transport replication v2: completed, partial support, full gate NO-GO

**Formal result: 51/51 valid runs, 0 infrastructure-invalid formal attempts, 47/51 task successes. Seven of eight preregistered gates PASS; the full gate set is NO-GO.** The recovery gate remains FAIL for Runs 23, 39 and 47. None was rerun or relabeled.

The strongest supported mechanism is narrow: the v2 persistent executor and startup/steady deadline split removed the timeout-reconnect-cold-start amplification observed in v1 under this tested two-host X-VLA/LIBERO deployment. Cold-start compute itself remained about 7 s. This outcome-informed revision is **not an independent replication of v1**.

The separate [final_status.json](final_status.json) is the completed-result status. The [preregistration config](../../configs/rpc_remote_transport_replication_v2.json) retains its original bytes and historical PRE_REGISTERED_NOT_EXECUTED status; that frozen field is not the current experiment ledger.

## Frozen provenance

- Implementation/protocol commit: `e33eb98c94013da14109ab2ac710414fea63ebaf`.
- Protocol config SHA256: `661ee2fbcd565ec2a04deda57d8021cc088f05d2a3cb915bc59e379486fbb4f1`.
- Authoritative completed experiment manifest SHA256: `9c8e376f83ca4dce162b452defd2d9c42ea6bc8e44478133951846c8068fbe2f`.
- Two distinct RTX 5090 host identities; A owns X-VLA/CUDA, B owns LIBERO/client. TCP traversed an SSH forward from B loopback to A loopback, not direct routed TCP.
- One new model/server process per episode, fixed seed/identity/condition order, no inference warmup, 20 Hz controller, 300-step cap. States 25-29 are historically exposed; only states 25-27 have all fault-condition pairs.
- Source paths and hashes are in final_status.json. Original manifests, run classifications, raw receipts and protocol files remain unchanged. No GPU process, remote connection, experiment or diagnostic was launched during this consolidation.

## Preregistered hard gates

| Gate | Verdict | Evidence |
|---|---|---|
| declared_episode_coverage_exact | **PASS** | 51 exact committed identities in execution order; Run1 reused, never rerun |
| stale_action_crosses_reset | **PASS** | 0 |
| accepted_out_of_order_responses | **PASS** | 0 |
| no_fault_transport_deadlines | **PASS** | 0 |
| no_fault_transport_errors | **PASS** | 0 |
| every_declared_fault_exercised | **PASS** | 36/36 fault episodes exercised their declared condition |
| post_fault_later_request_recovered | **FAIL** | [23, 39, 47] |
| raw_receipts_and_hashes_complete | **PASS** | All 51 local run manifests and both-host raw manifests rehashed; receipt telemetry hashes matched |

The formal all-fault result is **97/100**, not a complete recovery pass. Terminal stopping is not an exemption from the preregistered criterion. Task success cannot override this failed gate.

## V1 → diagnosis → V2

V1 ran real two-host X-VLA/LIBERO with a single 5 s inference deadline. Approximately 7 s first inference exceeded that budget; closing the client socket did not preempt remote CUDA. Reconnect/retry entered the serialized infer-lock path while older work could still be running. The preserved PCAP/timer ordering supports timeout, retry overlap and lock/backlog amplification. Object 5/state 25 remained VALID_NEGATIVE, counted permanently as v1 Run 1/51, and v1 stopped. It is not part of v2's 51 runs.

The completed postmortem separated initialization, worker execution, RPC, SSH echo baseline, GPU sampling, packet capture and lock-timeline reconstruction. These were non-scored diagnostic calls, not additional formal episodes.

| Diagnostic measurement | Observed value |
|---|---:|
| Worker initialization | 14.762 s |
| First direct inference | 6.605 s |
| Warm direct inference p50 | 75.60 ms |
| First RPC RTT | 6.927 s |
| Warm RPC RTT p50 | 94.29 ms |
| Warm RTT minus server interval p50 | 14.61 ms |
| New connection, same warmed worker, first call | 6.645 s |

The residual includes serialization, decoding, SSH/network and response handling; it is not pure SSH overhead. The millisecond-scale baseline/residual did not explain seconds of first-use work. Existing v1 server timing began before the infer lock; the approximate second-request decomposition (1.993 s wait + 6.443 s service) is a PCAP/timer proxy, not direct lock-acquisition telemetry. Four warm direct/RPC observations do not establish a population p95 or SLA. Internal CUDA/cuDNN/cuBLAS initialization was not isolated.

V2 moved GPU execution ownership out of connection handlers into one persistent executor. Validated inference and idle-reset jobs enter the same serialized queue; handlers retain connection framing and result routing. Reconnect does not construct another worker/executor. Obsolete queued work can be dropped; admitted work can finish and be invalidated. The explicit budgets are 15 s for the first inference on each connection, 5 s thereafter, and 20 s for outer action wait. Queue, compute and service timing plus connection/request identities make wait and execution distinguishable. Both architecture and budgets changed together, so their independent causal contributions are not separated.

## Task results and negative outcomes

| Family | No fault | 50 ± 20 ms | 250 ± 100 ms | 950 ± 250 ms | Disconnect every 7 |
|---|---:|---:|---:|---:|---:|
| Object | 5/5 | 3/3 | 3/3 | 3/3 | 3/3 |
| Spatial | 3/5 | 2/3 | 3/3 | 3/3 | 2/3 |
| Goal | 5/5 | 3/3 | 3/3 | 3/3 | 3/3 |

V2 task failures are Spatial Run 8 (no fault/state 27), Run 9 (no fault/state 28), Run 36 (50 ms/state 27), and Run 39 (disconnect/state 27), all at the 300-step cap. Object and Goal each had 17/17 task successes in v2. Historical Object negatives are reported separately below.

Historical Object negatives remain visible and separate: v1 Object Run 1 is VALID_NEGATIVE; [H1-R2](../actionstream_transport_h1_r2/report.md) regressed Object from 5/5 to 3/5 (states 13,14), and [H2](../actionstream_transport_h2_budget_holdout/report.md) retained Object state-16 failures under both runtimes. These are not additional v2 outcomes or clean causal task-success baselines.

The exact 51 identities and actual/failure-capped steps are in [coverage.csv](coverage.csv). [paired_comparisons.json](paired_comparisons.json) preserves all 36 same-suite/task/state/seed comparisons; failures count as 300. Success is descriptive, not a universal improvement claim or a collapsed performance score.

## Cold, steady and reconnect latency

| Condition | Steady RTT p50 / p95 / p99 ms | Steady compute p50 / p95 ms |
|---|---|---|
| tcp_no_injected_fault | 103.84 / 111.08 / 144.95 | 77.44 / 83.48 |
| application_delivery_jitter_50ms | 161.97 / 203.72 / 236.36 | 81.32 / 121.13 |
| application_delivery_jitter_250ms | 423.10 / 495.96 / 515.87 | 109.78 / 139.41 |
| application_delivery_jitter_950ms | 994.64 / 1341.64 / 1348.57 | 102.52 / 132.72 |
| periodic_pre_inference_disconnect | 104.28 / 119.00 / 145.76 | 77.42 / 91.75 |

Cold process-first compute p50/p95: **7.028/7.285 s**; RTT: **7.197/8.068 s** (51 first requests). Successful reconnect-first compute: **90/113 ms**, RTT: **115/138 ms** (97 requests). Successful recovery latency, including retry/backoff: **166/189 ms**. All startup and steady deadlines and all server errors were zero; the 100 transport errors correspond to injected disconnects.

All 51 fresh processes reported executor_starts=1; actual reconnects occurred within nine unchanged server PIDs. The frozen single-thread queue path and counters support executor continuity and serialized policy calls. Native OS executor thread IDs were not exported. Successful reconnects did not repay the roughly 7 s cold path; the original cold path was budgeted, not eliminated.

Warm distributions include successful steady-budget requests only. Process-first and reconnect-first startup requests are separate; failed/cancelled attempts remain in raw telemetry and counters. Quantiles use linear interpolation; these small, asynchronous historical-cohort observations are descriptive, not an SLA.

## POST-HOC DESCRIPTIVE ANALYSIS: recovery opportunity

**This is separate from the preregistered verdict.** [recompute_posthoc.py](recompute_posthoc.py) verifies the authoritative manifest, all 51 run manifests and their files, both-host raw manifests and RPC/engine receipt hashes, then recomputes events from raw logs. It does not import policy, simulator or networking code. See [posthoc_recovery.json](posthoc_recovery.json) and the 100-event [posthoc_events.csv](posthoc_events.csv).

| Recomputed quantity | Count |
|---|---:|
| Injected disconnects | 100 |
| Connections established, all 51 runs | 150 |
| Actual reconnects | 99 |
| Events followed by an admitted TCP inference attempt | 99 |
| Usable observed-response opportunities | 97 |
| Successful recoveries among those opportunities | 97 |
| Terminal-edge events without an uncensored response opportunity | 3 |

Definitions are operational and expose the two different denominators:

1. An injected disconnect is every seventh validated single-client inference ending in transport_error. All available server global ordinals must match that attempted order. Faults precede server enqueue, so their identities are joined from surrounding ordinals and client errors, not invented server enqueue records.
2. Literal later-request presence means a later admitted RPC attempt in rpc_telemetry.jsonl. This count is 99. An engine provider-reset cancellation before RPC admission does not count as an RPC request.
3. A usable observed-response opportunity means at least one later **non-cancelled** RPC attempt completed before engine_stop_requested. An observed error/deadline would remain in this denominator. The rule does not require success for eligibility.
4. Terminal-edge means either no later RPC admission, or the only later attempt spans episode stop and is cancelled with reset_or_stop=true. Request presence and terminal-edge censoring therefore overlap for two events; the literal request-presence category is not falsely reported as 97.

| Run / fault ordinal | Later admitted RPC | Evidence at stop | Formal result |
|---|---|---|---|
| 23 / 77 | Yes; startup budget 15 s | Next RPC cancelled after stop; server result invalidated | No later success; recovery FAIL |
| 39 / 140 | No | Provider-reset cancellation before RPC admission at the 300-step boundary | No later success; recovery FAIL |
| 47 / 56 | Yes; startup budget 15 s | Next RPC cancelled after stop; server result invalidated | No later success; recovery FAIL |

**Descriptively, 97/97 usable observed recovery opportunities recovered. Formally, 97/100 all-fault events recovered: the original gate is FAIL and the complete gate set remains NO-GO.**

> The post-hoc opportunity-conditioned analysis helps explain the preregistered failure but does not revise the frozen gate or its NO-GO verdict.

Termination-dependent censoring may be informative. The hypothetical outcome had a cancelled episode continued is unknown; 97/97 is not an unbiased recovery probability, an unconditional 100% recovery claim, or a replacement gate. Opportunity ordering uses client/engine timestamps on host B, not cross-host subtraction.

## Queue supply: preserve both metrics

Raw depletion counts empty polls divided by empty polls + dequeued actions + bounded holds. Post-first-action depletion applies the same ratio starting at the first accepted valid chunk. These are polling-event ratios, not fractions of physical elapsed control time. Run 1 had about 95.03% raw depletion, almost entirely before first action availability, and 0% afterward. Unified analysis uses first chunk acceptance; its sealed original report used first dequeue, without changing either original file.

| Family | Condition | Raw polling depletion | Post-first-action depletion | Queue age p50 / p95 steps |
|---|---|---:|---:|---|
| Object | tcp_no_injected_fault | 95.24% | 0.00% | 4.0 / 4.0 |
| Object | application_delivery_jitter_50ms | 95.58% | 0.00% | 5.0 / 7.0 |
| Object | application_delivery_jitter_250ms | 96.24% | 0.00% | 13.0 / 18.0 |
| Object | application_delivery_jitter_950ms | 97.45% | 88.30% | 21.0 / 29.0 |
| Object | periodic_pre_inference_disconnect | 95.37% | 0.00% | 4.0 / 6.0 |
| Spatial | tcp_no_injected_fault | 93.15% | 0.00% | 4.0 / 4.0 |
| Spatial | application_delivery_jitter_50ms | 94.23% | 0.00% | 5.0 / 7.0 |
| Spatial | application_delivery_jitter_250ms | 96.26% | 0.00% | 12.0 / 17.0 |
| Spatial | application_delivery_jitter_950ms | 97.39% | 87.58% | 21.0 / 29.0 |
| Spatial | periodic_pre_inference_disconnect | 93.50% | 0.00% | 4.0 / 6.0 |
| Goal | tcp_no_injected_fault | 96.16% | 0.00% | 4.0 / 5.0 |
| Goal | application_delivery_jitter_50ms | 96.68% | 0.00% | 5.0 / 8.0 |
| Goal | application_delivery_jitter_250ms | 97.34% | 0.00% | 13.0 / 18.0 |
| Goal | application_delivery_jitter_950ms | 97.97% | 88.22% | 20.0 / 29.0 |
| Goal | periodic_pre_inference_disconnect | 96.39% | 0.00% | 4.0 / 7.0 |

**The 950 ± 250 ms condition produced 87.58–88.30% post-first-action polling depletion. High task success does not establish healthy action supply.** Ordinary age-alignment prefix discard is separate from an action crossing reset. Cross-reset actions and accepted out-of-order responses remained zero.

Historical reconnect-like compatibility counters include initial establishment. Explicit v2 connections_established=150 equals 51 initial connections + 99 actual reconnections. Actual reconnections are not the 100 injected fault count: the final Run39 fault had no new TCP request. Two of the 99 post-reconnect inference attempts were cancelled, leaving 97 demonstrated recoveries.

## Measurement limitations and review data

The curated [metrics.json](metrics.json) retains family-by-condition request rates, completion steps, queue age, queue waits, latency distributions, deadlines/errors, connection counters, recovery, stale handling and GPU measurements. It is a projection of the hash-verified frozen aggregate, not a new experiment. Tracked CSV/JSON summaries use LF line endings for portable replay; source hashes identify the unchanged original raw bytes.

- Historical cohort, one X-VLA checkpoint, LIBERO simulation; five no-fault identities and three paired fault identities per family. External validity is NOT ESTABLISHED and physical robot safety is NOT EVALUATED.
- Outcome-informed architecture and budget changes are coupled; no isolated causal attribution, independent v1 replication, generalization, task-superiority or production-readiness claim.
- Application delivery jitter over SSH-forwarded TCP is not packet-level netem. Closing a socket does not preempt CUDA.
- Native executor thread ID is unavailable. Approximately 1 Hz GPU sampling can miss bursts; device/process queries are sequential and GPU episode windows compare unsynchronized host UTC clocks.
- Retained sampled process-VRAM peaks include 6,246 MiB (Run24) and 12,890 MiB (Run34); allocator causes are untraced. No steady-memory guarantee follows from these sampled maxima.
- Run1 capture had 13 kernel packet drops in the request direction; all 2,582 accepted response action payloads across v2 were captured and validated finite float32 [30,7]. Runs2–51 used response-direction capture only.
- No extra mid-episode reset schedule was injected beyond the fixed lifecycle. Zero reset-crossing is limited to that tested lifecycle.

## Reproduction and validation

Materialize the original ignored evidence bundle at its recorded root, then run locally:

```bash
python reports/rpc_remote_transport_replication_v2/recompute_posthoc.py \
  --evidence-root artifacts/rpc_remote_transport_replication_v2 \
  --output /tmp/actionstream-v2-posthoc-replay
```

This only hashes/reads existing files and writes separate descriptive outputs. Raw evidence is intentionally excluded from normal Git; a public checkout without that bundle cannot independently replay the original measurements. Tracked CSV/JSON summaries retain source identities and hashes. See [validation.md](validation.md) for CPU tests, static/format checks, strict release audit and protocol/evidence preservation. No new experiment is authorized or proposed by this report.
