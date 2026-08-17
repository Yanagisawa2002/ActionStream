# Isaac Lab-Arena integration status

## Outcome

The scalable benchmark contract and the ActionStream-side adapter are prepared,
but no Arena learned-policy result is claimed. The current GPU host fails the
official execution preflight before simulator startup: it is an AutoDL container
without Docker, `/isaac-sim`, `isaacsim`, `isaaclab`, or an Arena checkout. Arena
0.2.1's own contributor instructions require its Docker source environment.
Installing a second container runtime and a new Isaac Sim stack inside this
container was intentionally not attempted.

The machine-readable observation is in
`reports/isaaclab_arena_v1/readiness_20260818.json`.

## What is ready

- Arena `release/0.2.1` is pinned at
  `8b4a3a47fc53de23e8205089d71109a2e2348acd`.
- The frozen matrix contains three distinct task families: pick-and-place,
  button pressing, and a sequential put-and-close-door task.
- Each cell requests eight vectorized DROID environments. The official DROID
  embodiment supplies two external RGB cameras and one wrist RGB camera.
- Development and holdout reset seeds and network traces are disjoint. One seed
  is executed per Arena process because Arena 0.2.1 applies its seed at process
  scope.
- X-VLA and SmolVLA checkpoints and all transitive model revisions are pinned.
- The bridge detaches simulator observations before worker access, rejects a
  missing visual/state contract, replays frozen network delay, records runtime
  telemetry, and converts absolute targets into bounded current-state-relative
  DROID IK actions.
- `ActionStreamArenaPolicy` integrates the aligned and guarded backends with
  Arena's `PolicyBase` and explicit cleanup path.

## What is not ready

- No Arena environment has been instantiated on the current server.
- No Arena GPU-vector throughput, task success, failure taxonomy, or video has
  been measured.
- Sync, official LeRobot latest-only, and official RTC are frozen cells but not
  yet vector-safe Arena adapters. The plugin rejects them instead of presenting
  a local imitation as an official baseline.
- The DROID-to-LIBERO observation/action bridge has structural and unit-test
  validation only. A learned GPU smoke must verify non-zero policy-driven motion,
  camera content, coordinate direction, and gripper polarity before any formal
  holdout.
- Arena is currently alpha; the exact pin is part of every receipt because its
  API and task assets may change.

## Resume boundary

Use a host that can run Arena's official Docker workflow with Isaac Sim 6.0 and
mount the ActionStream checkout and existing model cache. Run
`scripts/arena/preflight.py`, then one development cell through the generated
one-cell job file. Do not open the holdout until all observation/action/checkpoint,
GPU-inference, video-content, and provenance gates pass. A zero-action smoke or
successful import is only simulator evidence, never learned-policy success.
