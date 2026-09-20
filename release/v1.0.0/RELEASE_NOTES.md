# ActionStream v1.0.0

ActionStream v1.0.0 is the first validated release of the minimal closed-loop
LIBERO benchmark for delay-aware asynchronous action-chunk execution with a
frozen X-VLA policy.

## Headline result

At the calibrated 950 ms pressure point:

| Mode | Success | Mean steps | Mean wall time | Mean holds |
|---|---:|---:|---:|---:|
| `sync_hold` | 30/30 | 211.3 | 13.67 s | 84.87 |
| `async_naive` | 19/30 | 550.2 | 30.30 s | 1.50 |
| `async_aligned` | 29/30 | 150.3 | 10.37 s | 16.37 |

Against `async_naive`, alignment improved paired success by 33.3 percentage
points (95% episode-bootstrap CI 13.3 to 53.3), reduced environment steps by
399.9 (CI 315.1 to 482.3 fewer), and reduced wall time by 19.93 seconds
(CI 16.03 to 23.67 seconds less).

The original underrun claim is not supported as stated. Alignment produced
14.87 more hold steps than naive async (CI 9.53 to 24.40 more), while producing
68.50 fewer holds than `sync_hold` (CI 57.10 to 75.83 fewer). No fully stale
chunk occurred.

## What is included

- Validated M0 preflight, M1 official baseline, M2 parity, and M3 executor
  evidence.
- Episode-level paired M3 statistics with 10,000 fixed-seed bootstrap
  resamples.
- The real-time `sync_hold` baseline at 0 and 200 ms.
- Four-point queue-pressure calibration and the preregistered 950 ms
  selection.
- The full paired 950 ms evaluation of `sync_hold`, `async_naive`, and
  `async_aligned`.
- Three reproducible PDF/PNG figures, a portable machine-readable summary, and
  a 38-second H.264 demonstration.

## Release assets

- `media/actionstream_v1_demo.mp4`: captioned overview, 1280x720 at 30 fps.
- `media/actionstream_v1_demo_poster.png`: video poster and README preview.
- `figures/m4_pressure_headline.{pdf,png}`: success and observed episode
  efficiency under pressure.
- `figures/m3_latency_effects.{pdf,png}`: paired M3 effects at 0 and 200 ms.
- `figures/queue_pressure_calibration.{pdf,png}`: delivery age, headroom, and
  pressure-point selection.
- `evidence_summary.json`: portable summary with no capture-host path
  dependency.
- `asset_manifest.json`: release-media input/output hashes and encoding
  metadata.

The LIBERO footage in the video comes from the official synchronous baseline
and is shown only as task context. The M4 comparison is represented by frozen
charts and real queue telemetry; the video does not present baseline footage as
an M4 mode recording.

## Integrity and reproducibility

The validated runtime and historical `outputs/m3` and `outputs/m4` artifacts are
pinned at commit
[`d84cb64e9e48b681e083828b24df5a38709c78c8`](https://github.com/Yanagisawa2002/ActionStream/tree/d84cb64e9e48b681e083828b24df5a38709c78c8).
Generated outputs are intentionally not retained on the current default branch;
the release integrity manifest records their Git object IDs.

Regenerate and verify only the release layer:

```bash
python -m pip install -r requirements-release.txt
python scripts/release/generate_figures.py
python scripts/release/generate_release_summary.py
python scripts/release/generate_demo_video.py
python scripts/release/verify_release.py
```

These commands read frozen JSON/JSONL/NPZ evidence. They do not import the
benchmark runtime, execute LIBERO, invoke X-VLA, or write under `outputs/`.

## Boundaries

- M3 success was ceilinged at 180/180, so it did not establish equal
  robustness.
- M4 supports success preservation and observed episode efficiency against
  naive async under the tested pressure protocol.
- M4 does not demonstrate success superiority over `sync_hold`.
- Underrun evidence is mixed by comparator.
- Fully stale chunk rejection remains structurally tested but empirically
  unobserved.
- Results cover one frozen policy, one LIBERO suite, three tasks, and the
  recorded simulator/hardware environment.
