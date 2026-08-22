# ActionStream

ActionStream is a LeRobot-compatible asynchronous inference backend for
chunked robot policies under delayed delivery. It separates GPU inference from
response delivery, rejects stale work, bounds queue depletion, and emits the
telemetry needed to explain timing failures.

[![CI](https://github.com/Yanagisawa2002/ActionStream/actions/workflows/ci.yml/badge.svg)](https://github.com/Yanagisawa2002/ActionStream/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

[![ActionStream v1.1 paired GPU demo](release/v1.1.0/media/actionstream_v1_1_demo_poster.png)](release/v1.1.0/media/actionstream_v1_1_demo.mp4)

The 61.9-second demo shows one frozen mechanism-level GO and the next frozen
NO-GO. It does not hide the compute cost, the task-family regression, or the
missing real-robot evidence.

## Review surface

- [Inference backend](src/actionstream/lerobot_inference.py)
- [Cancellable process transport](src/actionstream/inference_transport.py)
- [LeRobot plugin](integrations/lerobot/)
- [Backend and failure-path tests](tests/test_lerobot_inference.py)
- [H1-R2 delivery-pipeline report](reports/actionstream_transport_h1_r2/report.md)
- [H2 compute-budget report](reports/actionstream_transport_h2_budget_holdout/report.md)
- [LeRobot upstream PR #4466](https://github.com/huggingface/lerobot/pull/4466)

The v1.1 review branch intentionally excludes the earlier selector, Isaac,
Arena, ROS, and raw experiment archives from its diff. Existing v1.0 history on
`master` is unchanged.

## Runtime contract

```text
latest observation mailbox
          │
          ▼
asynchronous policy worker ── timeout / disconnect telemetry
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

`ActionStreamInferenceEngine` provides the LeRobot lifecycle expected by
`lerobot-rollout`:

- `start`, `stop`, and `reset` with bounded worker/process joins;
- `notify_observation` through a latest-observation mailbox;
- `get_action` from a thread-safe aligned queue;
- reset-generation and out-of-order response rejection;
- bounded hold during depletion and optional latest-only fallback;
- direct local inference or a restartable child-process transport;
- schema-v1 JSONL telemetry without frames, actions, credentials, or payloads.

The direct CUDA path has an advisory timeout because Python cannot safely
interrupt a blocked CUDA call. The process transport owns the inference client
in a child process and can enforce request deadlines by terminating it.

### Telemetry surface

Each JSONL event has a schema version, monotonic timestamp, run ID, episode ID,
and lifecycle generation. Depending on the event, the backend records:

- request source step, start/completion/delivery timestamps, and deadline mode;
- inference and transport latency without serializing model payloads;
- queue depth, queue age, aligned-prefix discard, and stale rejection;
- depletion, bounded hold, fallback, timeout, restart, and recovery counters;
- reset/stop cancellation and child-process termination outcomes.

This keeps traces useful for postmortems without logging observations, actions,
credentials, or licensed model data.

## Frozen GPU evidence

Both current reports use X-VLA on an RTX 5090, three genuinely different
LIBERO task families, fresh runtime processes, disjoint reset identities, and a
fixed 950-ms delivery delay.

### H1-R2: delivery-pipeline mechanism GO

H1-R2 changed one variable: whether the 950-ms response-delivery wait blocked
the single inference worker.

| Metric | Serialized aligned | Pipelined aligned |
|---|---:|---:|
| Success | 10/15 | 13/15 |
| Mean steps, failures = 300 | 201.33 | 137.93 |
| Median request rate | 0.902/s | 10.110/s |
| Queue depletion | 46.6% | 0.0% |
| Inference calls | 159 | 1,261 |
| Steady process VRAM max | 4,844 MiB | 4,844 MiB |

Request supply increased 11.43x and all 1,407 depletion pulls disappeared.
Mean paired completion steps improved by 63.40 with 95% CI
`[-118.20, -5.47]`. Success improved by three episodes, but its paired CI
crosses zero; Object also regressed from 5/5 to 3/5. This supports the pipeline
bottleneck, not universal policy superiority.

### H2: fixed compute budget NO-GO

H2 kept the pipelined runtime and changed one variable: the minimum request
interval from one to five controller steps.

| Metric | Unbounded pipeline | Five-step budget |
|---|---:|---:|
| Success | 14/15 | 13/15 |
| Mean steps, failures = 300 | 125.67 | 139.80 |
| Inference calls | 1,109 | 426 |
| Median request rate | 9.901/s | 3.462/s |
| Queue depletion | 0.0% | 0.0% |

The budget saved 61.6% of calls and kept the queue supplied, but lost one
paired success and failed the preregistered success gate. H2 therefore records
a compute/success frontier rather than a new default scheduler.

## Install and verify

Requirements are Python 3.12, Git, and `uv` 0.11.32. Policy weights, LIBERO
assets, and simulator dependencies retain their upstream terms and are not
redistributed.

```bash
git clone https://github.com/Yanagisawa2002/ActionStream.git
cd ActionStream
uv sync --locked --all-packages
uv run lerobot-rollout --help
```

The real CLI help must expose `actionstream` as an inference-engine choice.

Run the review-branch checks:

```bash
uv run ruff check src tests scripts/release scripts/engineering integrations/lerobot
uv run pytest tests/test_lerobot_inference.py tests/test_lerobot_plugin.py \
  tests/test_clean_install.py
uv run python scripts/release/audit_public_release.py
```

Exercise hard process deadlines and reset/stop preemption with the deterministic
fixture:

```bash
uv run python scripts/engineering/stress_inference_transport.py \
  --output-dir /tmp/actionstream-transport-stress --cycles 10 \
  --deadline-seconds 0.2
```

## Evidence and reproduction boundary

Tracked evidence contains aggregate tables, paired effects, compact figures,
the representative demo, and content hashes. Raw traces, full episode videos,
model weights, simulator assets, and transfer archives remain outside ordinary
Git.

The demo's H1 footage is a same-task, same-reset Spatial pair: serialized fails
at 300 steps and pipelined succeeds at 123. The H2 footage is explicitly labeled
as a representative pair where both variants succeed; the decisive state-19
budget failure has scored trace evidence but was not captured by the frozen
episode-0 video contract.

See [release notes](release/v1.1.0/RELEASE_NOTES.md), the
[demo manifest](release/v1.1.0/media/asset_manifest.json), and the
[release boundary](docs/public_release.md).

## Limitations

- The result covers one X-VLA checkpoint in LIBERO simulation.
- H1-R2 tests fixed 950-ms in-process delivery, not real remote RPC jitter.
- H2 is a strict NO-GO; no adaptive scheduler is released as superior.
- SmolVLA failed its frozen sync gate, so formal RTC remains unavailable.
- Native learned-policy Isaac and Isaac Lab-Arena holdouts are not included.
- No physical robot, safety controller, E-stop, or hardware-safety claim exists.
- The short transport stress fixture is not a production soak, CUDA-OOM test,
  or GPU-driver recovery result.

## License

ActionStream source is Apache-2.0. LeRobot, X-VLA, LIBERO, simulator assets,
and all policy weights remain under their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
