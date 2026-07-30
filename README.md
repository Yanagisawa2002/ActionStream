# ActionStream

ActionStream is a minimal closed-loop LIBERO benchmark for a frozen
[`lerobot/xvla-libero`](https://huggingface.co/lerobot/xvla-libero) policy. It
tests whether compensating for observation age when an asynchronous action
chunk arrives is safer and more efficient than replacing the queue with the
entire stale chunk.

The completed 0/200 ms matrix supports an **efficiency benefit over naive
asynchronous replacement**, but not the full hypothesis yet. All six
conditions reached 30/30 success and none exhausted its action queue, so this
run cannot establish a success-rate or underrun advantage. Synchronous
execution also remained more step-efficient than aligned async.

## Hypothesis

For a frozen X-VLA policy, dropping the stale prefix of a newly generated
action chunk according to the age of its source observation will reduce queue
underruns and preserve task success better than synchronous execution and
naive asynchronous chunk replacement.

## Frozen protocol

- Suite: `libero_object`; task IDs `0,1,2`.
- Ten episodes per task and condition, with initial-state indices
  `0,2,...,18` and paired policy/environment seeds `142,...,151`.
- Absolute control, episode length 800, batch size 1.
- X-VLA checkpoint revision
  `12e8783e996944f5c97e490d37d4c145484ed70a`.
- Checkpoint defaults: `chunk_size=30`, `n_action_steps=30`; the official
  baseline has no `n_action_steps` override.
- LIBERO reports a 20 Hz controller frequency. M3 uses monotonic-clock pacing
  and records coalesced whole missed ticks and dispatch lateness.
- Async modes request a new chunk every 10 executed control steps. Sync is the
  parity runner and consumes each 30-step chunk before replanning.
- One persistent local inference worker; at most one inference call is in
  flight. Pending observations are coalesced to the latest request.
- Injected delay is scheduled after inference and does not occupy the worker.
  The full matrix uses 0 and 200 ms; 100 and 300 ms are configurable.
- The policy, processors, environments, action queue, worker counters, and all
  Python/NumPy/Torch RNGs are reset at episode boundaries. No training,
  fine-tuning, demonstrations, RPC, or upstream LeRobot patch is used.

The three modes are:

1. `sync`: infer a 30-action chunk on the control thread, wait for optional
   delivery delay, then execute the complete chunk.
2. `async_naive`: infer from the latest requested observation in the
   background and replace the pending queue with the complete returned chunk.
3. `async_aligned`: compute
   `age_steps = current_control_step - observation_control_step`, drop that
   many actions, and replace the queue with the remainder. A chunk whose age
   is at least 30 steps is discarded as fully stale.

Before the first asynchronous result, the environment waits. If a queue later
underruns, it repeats the last finite, nonzero, final 7D absolute command; an
all-zero absolute hold is rejected.

## Processor and action contract

The custom runner reuses the public LeRobot 0.6.0 environment and checkpoint
processor factories in evaluator order:

```text
preprocess_observation
-> task-description insertion
-> X-VLA LIBERO environment preprocessor
-> checkpoint policy preprocessor
-> predict_action_chunk()
-> checkpoint policy postprocessor, once per time step
-> X-VLA LIBERO environment postprocessor, once per time step
```

The model emits finite `[1,30,20]` `float32` actions. Official postprocessing
converts each time step to a finite `[1,7]` LIBERO absolute action, producing a
final `[1,30,7]` chunk. Only these final 7D commands enter the external queue.
Image rotation/normalization, `domain_id`, state conversion, 6D rotation
conversion, and 20D-to-7D conversion are therefore not reimplemented here.

## Environment and commands

The measured environment was Ubuntu 24.04.4 under WSL2, Python 3.12.3,
LeRobot 0.6.0, PyTorch 2.11.0+cu130, CUDA runtime 13.0, and an NVIDIA RTX 4090
with 24,563.5 MiB VRAM. ActionStream uses an isolated canonical hf-libero
configuration at `$HOME/.cache/actionstream/libero-config` so an unrelated
global LIBERO checkout cannot affect the tasks or initial states.

Run from the repository root in Linux. On this machine, enter the intended WSL
distribution explicitly because `docker-desktop` is the Windows default:

```bash
wsl.exe -d Ubuntu-24.04
cd /mnt/c/Users/cgliu/OneDrive/Documents/ActionStream
```

Bootstrap or verify the pinned environment. Preserve the checked-in evidence
by selecting a fresh output root, then execute the gates in order:

```bash
bash scripts/bootstrap_wsl.sh
export ACTIONSTREAM_OUTPUT_ROOT="$PWD/reproduced_outputs"

bash scripts/run_preflight.sh
bash scripts/run_official_smoke.sh
bash scripts/run_official_baseline.sh
bash scripts/run_custom_sync_smoke.sh
bash scripts/run_custom_sync_parity.sh
bash scripts/run_m3_smoke.sh
bash scripts/run_m3_matrix.sh

$HOME/.venvs/actionstream/bin/python -m actionstream.results \
  "$ACTIONSTREAM_OUTPUT_ROOT"/m3/*/episodes.jsonl \
  --output-dir "$ACTIONSTREAM_OUTPUT_ROOT/summary"

$HOME/.venvs/actionstream/bin/python -m pytest
$HOME/.venvs/actionstream/bin/python -m compileall -q src tests
$HOME/.local/bin/uv pip check \
  --python $HOME/.venvs/actionstream/bin/python
```

The gate scripts create the official and custom-parity manifests after their
evaluations pass. The benchmark refuses to overwrite episode JSONL, and
`ACTIONSTREAM_OUTPUT_ROOT` makes a repeat run independent of the checked-in
evidence. To run the two additional configured delay levels:

```bash
ACTIONSTREAM_OUTPUT_ROOT="$PWD/delay_sweep_outputs" \
ACTIONSTREAM_DELAYS_MS="0 100 200 300" \
  bash scripts/run_m3_matrix.sh
```

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| M0 preflight | Passed | Canonical environment reset/step, official model/processors loaded, raw and final action contracts verified |
| M1 official baseline | Passed | 30/30 overall; task 0: 10/10, task 1: 10/10, task 2: 10/10 |
| M2 custom sync parity | Passed | 30/30 versus official 30/30; absolute difference 0, allowed difference 2 |
| M3 local executor | Passed | Six fixed conditions, 180/180 successful episodes, 180 traces and 29,073 final 7D actions validated |

M0 measured a 1.231 s first CUDA inference and 3,524.0 MiB peak allocated
VRAM after model load. Across M3, steady-state median peak allocation was
3,522.0-3,522.6 MiB. Each fresh condition process had a first-episode CUDA
warm-up peak of 12,252.0 MiB; later episodes returned to about 3.5 GiB.
The evidence does not isolate which CUDA subsystem requested that transient
workspace. The official 30-episode baseline took 563.61 s total
(18.79 s/episode).

## M3 aggregate results

Values are means across the same 30 paired episodes. `infer` is pure model
latency; `delivery` includes inference, scheduling, and injected delay.

| mode | delay | success | steps mean +/- sd | wall mean | infer p50 / p95 | delivery p50 / p95 | holds | stale prefix mean / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sync` | 0 ms | 30/30 | 126.9 +/- 11.9 | 8.83 s | 0.112 / 0.332 s | 0.151 / 0.379 s | 0 | 0 / 0 |
| `async_naive` | 0 ms | 30/30 | 189.7 +/- 19.8 | 11.95 s | 0.113 / 0.175 s | 0.128 / 0.192 s | 0 | 0 / 0 |
| `async_aligned` | 0 ms | 30/30 | 136.6 +/- 13.0 | 9.23 s | 0.115 / 0.236 s | 0.131 / 0.255 s | 0 | 2.57 / 3 |
| `sync` | 200 ms | 30/30 | 126.9 +/- 11.9 | 10.43 s | 0.210 / 0.321 s | 0.490 / 0.624 s | 0 | 0 / 0 |
| `async_naive` | 200 ms | 30/30 | 246.3 +/- 26.6 | 15.22 s | 0.239 / 0.260 s | 0.452 / 0.477 s | 0 | 0 / 0 |
| `async_aligned` | 200 ms | 30/30 | 142.7 +/- 54.9 | 10.08 s | 0.244 / 0.400 s | 0.457 / 0.615 s | 0 | 8.11 / 10 |

Paired aligned-versus-naive results:

- At 0 ms, alignment reduced environment steps by 53.1
  (95% paired bootstrap CI 50.0 to 56.3), or 28.0%, and wall time by 2.71 s
  (22.7%). It won all 30 step-count pairs.
- At 200 ms, alignment reduced environment steps by 103.5
  (95% CI 79.9 to 118.7), or 42.0%, and wall time by 5.14 s (33.8%). It won
  29 of 30 step-count pairs.
- Against sync, alignment used 9.7 more steps at 0 ms and 15.8 more at 200 ms
  on average. Thus this result supports stale-prefix alignment over naive
  async replacement, not superiority over synchronous execution.

Two aligned-200 ms task-1 trajectories account for much of its variance:
initial-state 8/seed 146 took 204 steps and initial-state 16/seed 150 took 417
steps. Both succeeded. Their inference and delivery medians were normal,
with zero holds, fully stale chunks, request replacements, or cross-episode
leaks. Trace-only inspection suggests repeated gripper-sign changes clustered
near queue replacements, especially in the 417-step trajectory; without video
this is a diagnostic clue, not a scene-level causal claim. The events and
action traces are retained rather than filtered.

## Evidence

- [M0 preflight](outputs/preflight/preflight.json)
- [M1 official manifest](outputs/xvla_baseline/manifest.json)
- [M2 parity manifest](outputs/custom_sync_parity/manifest.json)
- [Final M3 aggregate report](outputs/summary/summary.md)
- [Final M3 machine-readable summary](outputs/summary/summary.json)
- [Final M3 validation manifest](outputs/summary/manifest.json)
- Episode rows: `outputs/m3/<mode>_delay<ms>/episodes.jsonl`
- Per-step action/timing traces:
  `outputs/m3/<mode>_delay<ms>/traces/*.npz`

Only `outputs/m3` and `outputs/summary` are final M3 evidence. Directories with
`pre_fixes`, `pre_rng`, `pre_math`, or `pre_persistent_worker` in their names
are intentionally preserved development evidence and must not be pooled with
the final matrix.

## Limitations and next experiment

The primary unresolved limitation is task/delay saturation: every condition
succeeded and the 30-action queue absorbed the tested inference and delivery
latencies without an underrun or fully stale chunk. The benchmark therefore
exercised alignment but not its intended safety boundary. The 20 Hz controller
was also paced in a WSL-hosted simulator rather than a hard real-time system;
missed ticks and lateness are measurements, not real-time guarantees.

The next justified step is a preregistered stress sweep that keeps the frozen
policy, tasks, states, and seeds fixed while running the already-supported
100/300 ms delays and a smaller queue/replan headroom chosen to produce some
underruns without changing model actions. That directly tests the still-open
queue-underrun and success-preservation claims before adding more tasks or
models.
