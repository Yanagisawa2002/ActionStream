# ActionStream engineering and GPU v2r1 closeout

## Verdict

- Engineering/backend contract: **PASS**, with a residual production-hardening
  gap described below.
- X-VLA GPU benchmark v2r1 measurement: **PASS**.
- X-VLA aligned method claim: **LIMITED_OPERATING_ENVELOPE**, not a general
  improvement.
- SmolVLA sync gate: **NO-GO (4/6)**; official RTC formal holdout was not run.
- v6 selector and Isaac Lab-Arena remained paused.

The old GPU v1 and invalid v2 namespaces were not edited or relabelled. v2r1
used previously unused reset identities and seeded network traces.

## Engineering closure

| Contract | Evidence | Result |
|---|---|---:|
| Clean clone install | `uv sync --locked --all-packages` at `583c2fa` | PASS |
| Real CLI subprocess | `lerobot-rollout --help` exposes `sync,rtc,actionstream` | PASS |
| Targeted clean-clone tests | real subprocess, backend, and archive tests | 15 passed |
| Remote targeted tests | inference/transport suite | 14 passed |
| Existing CI quality/core selection | lint, format, and targeted tests | 79 passed |
| CI-equivalent plus portable-archive tests | same checks plus new audit tests | 81 passed |
| Process request deadline | configured 0.200 s; observed request cancellation 0.233 s | PASS with 33 ms overshoot |
| Startup separated from request | process startup 4.193 s; request 0.233 s | PASS |
| Hung reset stress | 10 process cancellations/restarts | max 0.407 ms |
| Hung stop stress | process cancellation, no child left behind | 0.408 ms |
| Orphan process check | remaining transport children | 0 |
| Standard telemetry | schema-v1 JSONL; X-VLA formal validation | 40,070 events / 90 files / 0 invalid |
| Relocated archive replay | basename-anchored, containment-checked path rebase | PASS |

This proves deterministic cancellation and lifecycle behavior against the
process stress fixture. It is not yet a production soak: ten cycles do not
replace thousands of cycles, abrupt GPU-driver failure, OOM, or a real remote
service killed while CUDA is executing.

The transferred archive was also revalidated without rewriting frozen rows.
The portable auditor resolved 126 telemetry files, 315 traces, and 165 videos
through containment-checked local roots; both manifests matched all 760 files,
and 55,842 telemetry events had zero schema/hash errors. This closes the
remote-absolute-path replay gap discovered during local reanalysis.

## X-VLA frozen GPU result

Hardware: one NVIDIA GeForce RTX 5090 with 32,607 MiB reported memory. Each
task-family/runtime cell used a fresh process. Formal holdout size was 225
paired episodes across Object, Spatial, and Goal task families.

| Network profile | Sync | Official Async | Latest-only | Aligned | Guarded |
|---|---:|---:|---:|---:|---:|
| 0 ms | 9/9 | 7/9 | 9/9 | 8/9 | 8/9 |
| 250 ms | 9/9 | 7/9 | 9/9 | 9/9 | 9/9 |
| 950 ms | 9/9 | 7/9 | 9/9 | 5/9 | 5/9 |
| 600 +/- 400 ms jitter | 9/9 | 7/9 | 9/9 | 9/9 | 9/9 |
| 250 -> 2000 ms burst | 9/9 | 4/9 | 8/9 | 7/9 | 2/9 |

The preregistered 950 ms aligned-minus-latest contrast is negative: success
`-44.4 pp`, paired 95% CI `[-77.8, -11.1]`; completion steps `+62.89`, 95% CI
`[+5.11, +122.44]`. At zero delay aligned also lost one success and used 50.89
more steps on average. At 250 ms it retained 9/9 success but was 4.67 steps
slower, 95% CI `[+2.89, +6.33]`.

The only clean primary positive cell was jitter: both methods were 9/9, while
aligned used 12.44 fewer steps, 95% CI `[-22.22, -2.11]`. This is why the method
is described as having a limited operating envelope.

### GPU/system measurements

| Runtime | Fresh processes | Process VRAM max across families | Mean model load | GPU util p95 range |
|---|---:|---:|---:|---:|
| Sync | 3 | 4,767-4,844 MiB | 15.51 s | 31-39% |
| Official Async | 3 | 4,767-4,844 MiB | 15.48 s | 58-63.2% |
| Latest-only | 3 | 4,767-4,844 MiB | 15.61 s | 63% |
| Aligned | 3 | 4,767-4,844 MiB | 15.49 s | 62.95-66% |
| Guarded | 3 | 4,767-4,846 MiB | 15.40 s | 61-64% |

At 950 ms, mean true request rates were 0.428, 7.304, 7.512, 0.875, and
0.876 request/s for Sync, Official Async, Latest-only, Aligned, and Guarded.
Their mean steady-state inference p50/p95 latencies were respectively
94.2/102.9, 88.1/140.5, 87.5/133.3, 157.2/246.8, and 159.2/247.6 ms.

The mechanism-level failure is visible in telemetry: at 950 ms, aligned and
guarded reached about 29.2 queue-age steps and accumulated 1,002/1,039
depletion events. Guarded had zero fallback activations in that cell, so it
failed identically to aligned. Under burst, Guarded activated fallback 21 times
but fell to 2/9, showing that fallback existence is not fallback quality.

The image `fixed0950_content_pair.png` shows the first captured paired reset per
family, with latest-only on the left and aligned on the right. Goal and Object
match the numerical failures; Spatial succeeds for both.

## SmolVLA gate and RTC boundary

The development screen followed the frozen “numerically smallest 2/2 task”
rule and selected Object task 3, Spatial task 0, and Goal task 5. Canary used
new states 29 and 30.

| Family / task | State 29 | State 30 | Gate |
|---|---:|---:|---:|
| Object / 3 | success, 131 steps | success, 120 steps | 2/2 |
| Spatial / 0 | fail, 300 steps | success, 80 steps | 1/2 |
| Goal / 5 | success, 143 steps | fail, 300 steps | 1/2 |

The pinned upstream checkpoint reports RTC capability, and the sync canary has
valid actual-worker warmup, request-rate, and `nvidia-smi` evidence. Across the
three fresh processes, model load was 6.53-6.95 s, steady process VRAM was
2,047-2,124 MiB, GPU-utilization p95 was 16.05-17%, inference p50/p95 ranged
178.9-235.3 / 188.8-255.7 ms, and true request rate ranged 0.37-0.46 request/s.

Those system metrics do not rescue the policy gate. Because only 4/6 sync
episodes succeeded, the formal RTC comparison is `NOT_RUN_BY_FROZEN_SYNC_GATE`.
Changing tasks after reading this canary would invalidate the protocol.

## Remaining blockers

1. Aligned is scientifically negative at fixed 950 ms and already regresses at
   zero delay; the current method cannot support a broad superiority claim.
2. Queue depletion is still the decisive runtime defect. Guarded fallback is
   either inactive when needed (950 ms) or harmful when activated (burst).
3. SmolVLA is not stable enough on the frozen three-family sync gate, so there
   is no formal RTC effect or RTC GPU throughput result.
4. Engineering lifecycle tests are fixture-level and short; production
   confidence still needs long soak, OS/network fault injection, CUDA OOM/kill,
   and recovery under a real inference service.
5. The evidence is learned-policy LIBERO simulation. It is valid GPU systems
   evidence, but it is not real-robot safety evidence.
6. A full unfiltered repository sweep is not green: it reaches 3 failures in
   old M6/M8 tests. All depend on ignored historical output fixtures that are
   absent from the checkout; the M8 assertion additionally decodes localized
   Windows stderr as UTF-8 and masks that missing-file cause. The actual CI
   quality/core selection is green (79 passed), and the local extended
   selection including the portable-archive tests is green (81 passed), but
   this broader test debt should remain visible.

## Evidence map

- `report.md`, `main_table.csv`, `paired_effects.csv`, `gpu_systems.csv`, and
  `latency_success_operating_points.png`: generated X-VLA report.
- `engineering/transport_stress/summary.json` and
  `engineering/transport_stress/transport_events.jsonl`: deadline/reset/stop
  stress evidence.
- `clean_clone_receipt.json`: clean install and true CLI subprocess receipt.
- `portable_archive_validation.json`: relocated path, manifest, artifact hash,
  and telemetry replay validation.
- `../actionstream_backend_gpu_smolvla_rtc_v4/`: immutable SmolVLA NO-GO
  report, raw canary rows, receipts, and inspected content frames.
- The 58 MiB raw archive stays under ignored `artifacts/`; its SHA-256 and video
  inspection receipt are in `local_evidence_receipt.json`.
