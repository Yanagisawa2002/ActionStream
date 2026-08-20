# ActionStream

ActionStream is a delay-aligned inference backend and frozen GPU benchmark for
chunked robot policies. It connects LeRobot rollouts to an asynchronous GPU
worker, rejects stale responses, bounds queue depletion, and records the
telemetry needed to explain failures under latency, jitter, and outages.

[![CI](https://github.com/Yanagisawa2002/ActionStream/actions/workflows/ci.yml/badge.svg)](https://github.com/Yanagisawa2002/ActionStream/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

[![ActionStream v1.1.0 paired GPU demo](release/v1.1.0/media/actionstream_v1_1_demo_poster.png)](release/v1.1.0/media/actionstream_v1_1_demo.mp4)

The 74-second demo shows the strongest paired result and the failure
boundaries: an official Async failure recovered by aligned execution under
frozen jitter, a fixed-950 ms regression against latest-only, a guarded-runtime
NO-GO, and native learned-policy Isaac development runs.

## What is implemented

- `ActionStreamInferenceEngine` with `start`, `stop`, `reset`, `get_action`,
  and `notify_observation` lifecycle boundaries.
- One asynchronous inference worker and a thread-safe, delay-aligned action
  queue.
- Stale-response rejection, bounded hold on depletion, inference timeout,
  disconnect/recovery handling, and latest-only fallback.
- Per-episode latency, queue age, discard, depletion, fallback, GPU throughput,
  and CUDA-memory telemetry.
- A frozen paired protocol with disjoint resets and network traces, bootstrap
  confidence intervals, replay provenance, failure taxonomy, and videos.
- A third-party LeRobot inference-engine extension proposed upstream in
  [huggingface/lerobot#4466](https://github.com/huggingface/lerobot/pull/4466).

## Runtime path

```text
observation snapshot
        │
        ▼
asynchronous GPU worker ── timeout / disconnect telemetry
        │
        ▼
stale-response rejection
        │
        ▼
delay-aligned action queue ── queue age / discard / depletion telemetry
        │
        ▼
bounded hold → latest-only fallback → robot or simulator
```

The backend is policy-agnostic at its interface. The formal v1.1 benchmark
uses a frozen X-VLA checkpoint through LeRobot and three genuinely different
LIBERO task families: Goal, Object, and Spatial.

## Formal GPU result

The X-VLA matrix contains 375 holdout condition-episodes: 15 paired episodes
for every runtime/network cell. Canary plus holdout evidence totals 435
episodes, 435 traces, 135 videos, and 75,043 verified final 7-D actions.

| Frozen profile | Latest-only | Aligned | Guarded | Official Async | Sync-hold |
|---|---:|---:|---:|---:|---:|
| 0 ms | 11/15 | 12/15 | 12/15 | 10/15 | 14/15 |
| 250 ms | 13/15 | 13/15 | 13/15 | 9/15 | 14/15 |
| 950 ms | 14/15 | 10/15 | 8/15 | 11/15 | 14/15 |
| 600 ± 400 ms jitter | 14/15 | 14/15 | 12/15 | 8/15 | 14/15 |
| 250→2000 ms burst/outage | 12/15 | 13/15 | 8/15 | 9/15 | 14/15 |

### Primary contrast: official latest-only

Latest-only remains the preregistered primary reference. Aligned matched it at
250 ms and jitter and gained one paired success under burst/outage, but the
burst confidence interval reaches zero. At fixed 950 ms, aligned regressed by
26.7 percentage points (95% paired bootstrap CI `[-53.3, -6.7]`) and required
41.53 more steps (`[3.40, 84.33]`). Guarded is a formal NO-GO.

### Secondary contrast: official Async

This is explicitly a secondary, post-holdout analysis of unchanged frozen
cells. Under jitter, aligned improved success from 8/15 to 14/15: +40.0
percentage points (`[13.3, 66.7]`) and -62.33 mean steps
(`[-105.20, -21.53]`). At 250 ms it improved 9/15 to 13/15 and reduced mean
steps by 45.60 (`[-92.60, -1.20]`). At 950 ms it did not improve Async.

This contrast does not replace latest-only as the primary reference and does
not support a universal-superiority claim.

## GPU systems evidence

- Hardware: one RTX 5090.
- Peak allocated CUDA memory during scored episodes: 12,258.95 MiB.
- The report includes GPU-synchronized inference p50/p95, delivery p50/p95,
  actions/s, queue age, discard, depletion, recovery, and fallback.
- Unsupported upstream queue-age fields are `N/A`, not zero.
- CUDA allocated memory is not total `nvidia-smi` process residency.
- Sampled network profiles are operating points, not a continuous curve.

See the [formal report](reports/actionstream_backend_gpu_v1/formal_xvla/report.md),
[paired effects](reports/actionstream_backend_gpu_v1/formal_xvla/paired_effects.csv),
[secondary Async contrast](reports/actionstream_backend_gpu_v1/formal_xvla/secondary_paired_effects.csv),
[failure taxonomy](reports/actionstream_backend_gpu_v1/formal_xvla/failure_taxonomy.csv),
and [latency-success figure](reports/actionstream_backend_gpu_v1/formal_xvla/latency_success_operating_points.png).

## Native Isaac status

X-VLA CUDA inference drove native Franka motion with the official render/state
bridge in three development tasks. Object0 and Spatial2 passed their task and
40 N collision gates. Goal2 reached both success predicates but hit 96.46 N,
so it is a safety NO-GO. All three use development reset 0; no native paired
holdout or real-robot safety claim is made.

Isaac Lab-Arena was not executed. The checked-in adapter is structural only.

## Install

Requirements are Python 3.12, Git, and `uv` 0.11.32. The lock file pins the
complete dependency graph, including the LeRobot revision.

```bash
git clone https://github.com/Yanagisawa2002/ActionStream.git
cd ActionStream
uv sync --locked --all-packages
```

That single sync installs the root package, the pinned LeRobot fork revision,
and the `lerobot_policy_actionstream` entry-point package. Verify the real CLI
discovery path with `uv run lerobot-rollout --help`; `actionstream` must appear
as an inference-engine choice.

The benchmark also needs separately licensed policy weights, LIBERO assets,
and a CUDA environment. They are not redistributed by this repository.

## Run the core checks

```bash
uv run ruff check src tests scripts/release integrations/lerobot
uv run pytest
uv run python scripts/release/audit_public_release.py
```

Regenerate the formal report from a retained raw evidence directory:

```bash
uv run python scripts/analysis/generate_actionstream_backend_gpu_v1_report.py \
  --input-root /path/to/actionstream_backend_gpu_v1 \
  --output-dir /tmp/actionstream-report
```

Generate the release demo after extracting the content-addressed raw archive:

```bash
uv run python scripts/release/generate_v1_1_demo.py \
  --evidence-root /path/to/extracted/evidence
```

## Repository map

```text
src/actionstream/             inference backend and benchmark logic
integrations/lerobot/         third-party LeRobot plugin
tests/                        runtime, report, and integration contracts
docs/                         frozen protocols and system boundaries
reports/                      curated aggregate evidence
release/v1.1.0/               demo, hashes, and release notes
upstream/lerobot/             reviewable upstream patch boundary
```

Raw traces, full video sets, model weights, simulator assets, and archives stay
outside ordinary Git. Curated reports and release media retain hashes back to
the preserved evidence.

## Scope and limitations

- Results cover simulation, not a physical robot.
- X-VLA is the only model with a completed formal v1.1 GPU matrix.
- SmolVLA v2 and v3 each failed a separately frozen three-suite sync gate
  (Goal 1/2, then Object 1/2). Formal RTC was therefore not run and its paired
  effect is unavailable; see the [NO-GO gate report](reports/actionstream_backend_gpu_v1/smolvla_rtc_gate_v2_v3/report.md).
- Sync-hold pauses the simulator during inference and is an upper-bound
  behavioral reference, not a real-time deployment mode.
- Disconnect/recovery is a backend canary, not a paired baseline effect.
- The benchmark demonstrates engineering observability and bounded failure
  behavior; it does not prove deployment safety.

## Release and provenance

- [v1.1.0 release notes](release/v1.1.0/RELEASE_NOTES.md)
- [v1.1.0 asset manifest](release/v1.1.0/asset_manifest.json)
- [public-release boundary](docs/public_release.md)
- [third-party notices](THIRD_PARTY_NOTICES.md)
- [Apache-2.0 license](LICENSE)

The full development history is preserved. v1.1.0 curates the review surface
instead of deleting negative results: aligned loses at 950 ms, guarded is a
NO-GO, native Goal2 fails the collision gate, and SmolVLA RTC remains sealed
behind its failed sync gate.

## Citation

If this repository helps your work, cite the release URL and include the exact
Git commit plus the relevant frozen protocol/report hash.
