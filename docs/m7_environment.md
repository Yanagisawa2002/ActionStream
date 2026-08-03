# M7 ROS 2 / Isaac Sim environment

Audit timestamp: 2026-08-03 (Asia/Singapore). Repository start state:
`master` at `e786a25c5eeadffaed9bf35d6785fe4468ba7462`. The pre-existing
accepted results under `outputs/m3/`, `outputs/m4/`, `outputs/m5_g0/`, and
`outputs/m6_g0/` were not modified or regenerated. New live-simulator evidence
is isolated under `outputs/m7_g0/isaac_smoke/`.

## Readiness result

The native Isaac runtime is **END_TO_END_INTEGRATION_COMPLETE_TASK_UNSUCCESSFUL**.
The machine passes the installed Isaac Sim 6.0.1 Compatibility Checker;
ActionStream messages, executor, and adapter build in NVIDIA's Windows Pixi
Jazzy environment; and a live headless episode completed against the official
Franka scene. The bounded run `m7-isaac-smoke-0004` exited 0 after strict
executor/adapter progression from Observation 0 through step 180, then
automatically terminated with `success=false` and `reason=step_limit`. This was
the expected result for a safe-hold-only smoke without a policy. A second
development smoke, `m7-isaac-profile-a-dev-2026080300-aligned-0002`, then ran
the complete path: Isaac Observation -> in-Kit scripted Python policy -> frozen
Profile A 950 ms injector -> native C++ `aligned_async` executor -> Isaac
actuation -> atomic recorder. Bootstrap completed, all 180 control steps ran,
and the independent replay audit passed 1,070 events with zero invariant
violations. The task itself failed: `success=false`, `reason=step_limit`.

This is not a paired Isaac benchmark result. Only one development seed and one
strategy ran; no policy-driven lift success, rosbag, policy-driven video,
three-strategy Profile A/Profile B comparison, Isaac holdout, or G0 decision is
claimed. The separate validated video is a safe-hold-only recording-path check.
The remaining environment limitation is narrower: a standalone pre-Kit
`import rclpy` and
therefore Python-based `ros2` CLI commands fail with a Windows DLL
procedure-resolution error. The supported adapter path succeeds because it
starts `SimulationApp`, enables and updates `isaacsim.ros2.bridge`, and only
then imports `rclpy` and the generated ActionStream messages. It does not
preload private DLLs and never invokes the deterministic test plant.

The preserved development evidence is under `outputs/m7_g0/isaac_smoke/`:
`aligned_async.events.jsonl`, `aligned_async.summary.json`,
`aligned_async.audit.json`, the exact frozen trace, executor parameters, and
`launch_audit.json`. The launch audit records both the failed orchestration
attempt and the single corrected attempt, including commands and cleanup.

## Current machine and installed target

| Item | Audited value | Status |
|---|---|---|
| Host | Windows 11 Home 25H2, build 26200, x86-64 | Compatibility Checker: supported |
| CPU / RAM | Intel Core Ultra 7 265K, 20 cores / 33.68 GB | supported |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9 | supported |
| Driver | 591.86 | supported; pinned checker's minimum is 537.58 |
| CUDA toolkit | 13.0, build 13.0.88 | installed; Isaac uses its packaged runtime |
| Default shell Python | 3.13.5, PyTorch 2.6.0+cu124 | not the M7 runtime |
| M7 environment | `C:\IsaacSim-ros_workspaces\jazzy_ws` | official workspace, clean commit `dd3eeede7912755996a18f4884285d9f50843f79` |
| Environment manager | Pixi 0.75.0 | installed at `C:\Users\cgliu\AppData\Local\pixi\bin\pixi.exe` |
| M7 Python | 3.12.13 | pinned environment |
| Isaac packages | `isaacsim`, `isaacsim-app`, `isaacsim-core`, `isaacsim-robot`, `isaacsim-ros2` all 6.0.1.0 | installed |
| ROS 2 | Jazzy; `rclpy` 7.1.9, `rosgraph-msgs` 2.0.3 | post-bridge import passed; standalone pre-Kit import remains blocked |
| RMW | `rmw_zenoh_cpp` | frozen by the official Pixi manifest |
| ActionStream native build | messages, C++ executor, Isaac adapter | passed; executor 23/23 GTests passed |
| Live Isaac smoke | `m7-isaac-smoke-0004` | exit 0; terminal step 180, safe-hold timeout |
| Policy-driven Isaac development smoke | Profile A, seed 2026080300, `aligned_async` | exit 0; replay passed; task failed at step 180 |

The compatibility result comes from the installed
`Isaac Sim Compatibility Checker App 6.0.1`, not from a guessed requirements
table. Its rerun reported `System checking result: PASSED`, Windows supported,
RTX 4090 supported, 25.76 GB VRAM, and driver 591.86 above its 537.58 floor.
The unrelated GPU container occupying memory during the audit was not stopped
or modified.

The official environment is the Windows `jazzy_ws` from NVIDIA's
[IsaacSim ROS workspaces](https://github.com/isaac-sim/IsaacSim-ros_workspaces/tree/dd3eeede7912755996a18f4884285d9f50843f79/jazzy_ws).
Its manifest pins Python 3.12, ROS 2 Jazzy, Zenoh, and
`isaacsim[all,extscache,ros2]==6.0.1.0`. NVIDIA documents Python 3.12 as the
required Python version for the pip distribution in the
[Python environment installation guide](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html).

## Simulator and API contract

The adapter uses only the Isaac 6 experimental/current path:

- `SimulationApp`, then `isaacsim.core.experimental.*` scene and prim APIs;
- `SimulationManager.setup_simulation(dt=1/60, device="cpu")` and
  `RenderingManager.set_dt(1/60)` for coherent physics/timeline rates;
- the shipped
  `isaacsim.robot.experimental.manipulators.examples.franka.Franka` class;
- `Franka.set_end_effector_pose(..., ik_method="damped-least-squares")`, whose
  shipped implementation uses the articulation Jacobian and official
  differential IK routine;
- `Franka.set_gripper_position` for the two Panda finger targets;
- `isaacsim.ros2.bridge` enabled before native `rclpy` custom-message traffic.

After scene creation, the adapter follows the installed 6.0.1 experimental
examples' lifecycle: update Kit to finish USD reference loading, configure the
simulation, play and update until tensor views are valid, then pause the
timeline and advance only through explicit `SimulationManager.step()` calls.
This prevents implicit timeline frames from being counted as control physics
steps. `SimulationApp.close(exit_code=...)` preserves loud nonzero failures on
Kit's Windows fast-shutdown path.

This is the same Franka path shown in NVIDIA's current
[Adding a Manipulator Robot tutorial](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/core_api_tutorials/tutorial_core_adding_manipulator.html).
It avoids the deprecated `isaacsim.core.api`, `isaacsim.core.utils`, and old
manipulator-example namespaces. The adapter validates the required Franka
methods after `SimulationApp` starts and fails if the installed extension has
drifted. NVIDIA's
[Simulation Manager documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.simulation_manager/docs/index.html)
defines `setup_simulation` as the coherent timestep/device setup API.

The scene is deliberately small: official Franka Panda, ground plane, one
dynamic 5 cm cube, and one visual lift target. Physics runs at 60 Hz and a ROS
control step consists of exactly three physics steps, so observations advance
at a wall-clock-capped 20 Hz. `/clock` is derived from completed physics steps;
steady-clock and wall-clock nanoseconds are recorded separately. Consumers must
set `use_sim_time=true` when they use ROS clocks. See NVIDIA's
[ROS 2 clock guide](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_clock.html).

## Frozen task and message mapping

Task ID is `scripted_reach_lift_v1`. The public command is seven-dimensional:

```text
[absolute EE x, y, z, absolute axis-angle x, y, z, gripper]
```

The policy emits exactly `+1` (open) or `-1` (closed). The low-level mapping
sign-quantizes any finite nonzero value: positive maps both Panda fingers to
0.04 m and negative maps both to 0.00 m; zero is rejected. Holds ignore their
payload and retain the last accepted EE pose and gripper state.

The simulator's joint/contact state remains private. `Observation.robot_state`
is exactly seven values: EE xyz, EE axis-angle, and signed gripper state.
`Observation.task_state` is exactly twelve values:

```text
object xyz, lift-target xyz, grasped, initial object z,
success streak, success, terminated, episode step
```

The object spawn is a SHA-256 mapping of the episode ID. Success requires at
least 0.12 m lift while the hand remains within 0.18 m for 20 consecutive
control steps. Timeout is 180 steps. These 20-step/180-step gates match the
deterministic reference plant; the Isaac result still requires its own live
calibration and may not be inferred from reference-plant results.

The logical step boundary is:

```text
publish Observation O
accept exactly one RobotCommand with actual_target_step = O + 1
apply/hold it for three 60 Hz physics ticks
publish Observation O + 1
```

Episode mismatch, duplicate actual target, malformed command, and any target
other than `O+1` are rejected. `source_target_step` is retained but is not an
actuator rejection rule: `naive_async` may intentionally execute an
arrival-ordered source action at a different actual step. `EpisodeControl`
uses an explicit `REQUEST`/`STATUS` discriminator on the shared topic so an ACK
cannot be interpreted as a second START/RESET request. REQUEST messages require
`active=false` and `terminated=false`; START/RESET also require
`success=false`, while TERMINATE may carry the outcome in `success`. STATUS
messages carry the acknowledged state flags.

## Capability classifications

`python -m action_stream_isaac.capability_probe --json` returns zero only when
standalone ROS imports work. On this Windows environment it intentionally
returns 2 and `NO_GO_ENVIRONMENT_BLOCKED` because the pre-Kit `rclpy` DLL load
fails. The adapter uses the explicit preflight form below, then performs the
deferred ROS/message imports after enabling the bridge:

```powershell
python -m action_stream_isaac.capability_probe --json --defer-ros-imports-to-kit
```

That command reports `READY_FOR_ISAAC_ROS2_PREFLIGHT` and
`ros_imports_deferred_to_kit=true`; it is not by itself proof that ROS imported.
The adapter's post-bridge import and exact schema validation are the second
gate.

- `NO_GO_ISAAC_UNAVAILABLE`: the exact 6.0.1.0 Isaac app/robot wheels are absent.
- `NO_GO_UNSUPPORTED_HOST`: the host is neither native Windows 11 x86-64 nor
  native Ubuntu 22.04/24.04 x86-64.
- `NO_GO_ENVIRONMENT_BLOCKED`: Isaac exists, but Python 3.12, ROS Jazzy,
  generated messages, ROS CLI, GPU/driver, or EULA readiness is incomplete.
- `READY_FOR_ISAAC_ROS2_PREFLIGHT`: non-ROS pre-Kit gates passed and ROS imports
  are explicitly deferred to the in-Kit bridge validation.
- `READY_FOR_ISAAC_ROS2`: all gates, including standalone ROS imports, passed.

The executable repeats the post-`SimulationApp` Franka API validation. It never
redirects to `action_stream_benchmark.plant`.

## Exact Windows build and launch procedure

Open a PowerShell terminal and enter the supported environment:

```powershell
& 'C:\Users\cgliu\AppData\Local\pixi\bin\pixi.exe' shell --manifest-path 'C:\IsaacSim-ros_workspaces\jazzy_ws\pixi.toml'
```

Inside that Pixi shell, build and source ActionStream into the official native
workspace. Do not reuse ActionStream's Docker-generated `ros2_ws/install` on
Windows:

```powershell
Set-Location 'C:\IsaacSim-ros_workspaces\jazzy_ws'
colcon build --base-paths 'C:\Users\cgliu\OneDrive\Documents\ActionStream\ros2_ws\src' --packages-select action_stream_msgs action_stream_executor action_stream_policy action_stream_benchmark action_stream_isaac --merge-install --cmake-args -DBUILD_TESTING=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
. .\install\setup.ps1
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:ROS_DISTRO = 'jazzy'
$env:RMW_IMPLEMENTATION = 'rmw_zenoh_cpp'
python -m action_stream_isaac.capability_probe --json --defer-ros-imports-to-kit
```

Do not continue unless the last command reports
`READY_FOR_ISAAC_ROS2_PREFLIGHT`. Start the Zenoh router in a second Pixi shell
using its native executable; `ros2 run` currently hits the pre-Kit `rclpy` DLL
blocker:

```powershell
Set-Location 'C:\IsaacSim-ros_workspaces\jazzy_ws'
& '.\.pixi\envs\default\Library\lib\rmw_zenoh_cpp\rmw_zenohd.exe'
```

Launch the adapter headlessly from the built/sourced shell. Direct module
execution is required on the currently audited Windows environment because
`ros2 launch` imports `rclpy` before Kit:

```powershell
python -m action_stream_isaac.isaac_adapter --headless true
```

An orchestrator normally sends the lifecycle request. For a bounded native
adapter/executor smoke, start the executor in a third Pixi shell and then run a
safe-hold episode:

```powershell
& '.\install\lib\action_stream_executor\action_stream_executor_node.exe' --ros-args -r __ns:=/action_stream -p 'safe_hold_command:=[0.45,0.0,0.35,3.141592653589793,0.0,0.0,1.0]'
```

```powershell
python -m action_stream_isaac.isaac_adapter --headless true --auto-start-episode m7-isaac-smoke-0001 --exit-after-auto-episode true --command-timeout-seconds 10
```

The adapter deliberately aborts after five seconds if that active episode does
not receive its next `RobotCommand` (ten seconds in the explicit smoke command);
it will not fabricate an unrecorded hold. The packaged
`franka_reach_lift.launch.py` exposes the same headless, auto-start, and bounded
exit arguments for environments where the ROS 2 CLI imports cleanly.

For the validated in-Kit development path, start the executor with the
persisted parameter file so Windows argument handling cannot reinterpret the
seven-value hold command:

```powershell
& 'C:\Users\cgliu\AppData\Local\pixi\bin\pixi.exe' run --manifest-path 'C:\IsaacSim-ros_workspaces\jazzy_ws\pixi.toml' 'C:\IsaacSim-ros_workspaces\jazzy_ws\install\lib\action_stream_executor\action_stream_executor_node.exe' --ros-args --params-file 'C:\Users\cgliu\OneDrive\Documents\ActionStream\outputs\m7_g0\isaac_smoke\executor.params.yaml'
```

Then run a new bounded episode. `--event-log-path` must not already exist; the
adapter refuses to overwrite evidence:

```powershell
& 'C:\Users\cgliu\AppData\Local\pixi\bin\pixi.exe' run --manifest-path 'C:\IsaacSim-ros_workspaces\jazzy_ws\pixi.toml' python -m action_stream_isaac.isaac_adapter --headless true --auto-start-episode m7-isaac-profile-a-dev-new --exit-after-auto-episode true --command-timeout-seconds 10 --inkit-policy-run true --fault-trace-file 'C:\Users\cgliu\OneDrive\Documents\ActionStream\outputs\m7_g0\isaac_smoke\profile_a_seed_2026080300.trace.json' --event-log-path 'C:\path\to\new\aligned_async.events.jsonl' --strategy aligned_async --profile-id profile_a --seed 2026080300 --request-interval-steps 10 --startup-discovery-seconds 1 --bootstrap-timeout-seconds 20 --terminal-drain-seconds 1
```

The first live policy attempt is retained as a failure audit: its inline ROS
parameter list did not parse, so the adapter's explicit 20-second bootstrap
watchdog exited 1. The sole corrected attempt used `executor.params.yaml`,
exited 0, and produced the persisted event log, summary, replay audit, trace,
and `launch_audit.json`. No further retry or task tuning was performed.

## Exact rosbag recording command

Once the standalone ROS 2 CLI DLL blocker is cleared, run this in another
built/sourced Pixi shell before sending START:

```powershell
ros2 bag record --storage mcap --output 'm7_isaac_smoke_0001' /clock /action_stream/episode_control /action_stream/observation /action_stream/inference_request /action_stream/raw_action_chunk /action_stream/action_chunk /action_stream/robot_command /action_stream/executor_diagnostics /action_stream/runtime_event /action_stream/events
```

For a bounded visual safe-hold MP4, run:

```powershell
.\scripts\m7_record_demo.ps1
```

Add `-Headless` for offscreen capture or `-VideoOutput C:\path\demo.mp4` to
select a new output. The adapter uses the installed NVIDIA
`omni.kit.viewport.utility`, `omni.kit.renderer.capture`, and
`omni.videoencoding` APIs. It passively schedules exactly one LDR frame after
each completed 20 Hz control tick, so it never takes ownership of the paused
simulation timeline; it then encodes H.264 at 1280x720/20 fps, removes only its
own intermediate PNGs, and verifies the MP4. The launcher refuses overwrite or
stale frame reuse and terminates its owned adapter/executor/router process tree.
The validation audit records 180 decoded frames, 9.0 seconds, media hashes, and
visual framing checks in
`outputs/m7_g0/audit/demo_capture_validation.json`. This is a safe-hold visual
demo, not a policy result or benchmark episode. On the audited Windows
environment, `ros2 bag record` is still blocked before recording by the pre-Kit
`rclpy` DLL error; no bag is claimed.

## Blocker-clearing acceptance sequence

Observed in the supported Pixi environment:

1. Native custom-message and executor build passed; executor GTests passed
   23/23.
2. The adapter package built and its pre-Kit capability gate passed.
3. Headless scene startup loaded the official Franka experimental extension and
   asset; its physics tensor, IK, gripper, object, and target APIs initialized.
4. Post-bridge `rclpy` and generated-message imports passed.
5. START produced Observation 0; the C++ executor supplied holds and the adapter
   advanced strictly to Observation 180.
6. Automatic timeout emitted terminal step 180 with `success=false`, and the
   bounded process exited 0.
7. With no executor/router, the watchdog failed loudly with exit 1 rather than
   substituting the deterministic test plant.
8. The in-Kit policy, deterministic Profile A injector, atomic recorder, native
   executor, and terminal lifecycle handshake completed one aligned development
   episode with 18 requests and 18 responses.
9. Independent replay recomputed the metrics from 1,070 events, matched the
   summary, found no semantic invariant violations, and passed. The manipulation
   task did not pass: it reached the frozen 180-step limit without a successful
   lift.

Still required before a full M7 evaluation claim:

1. clear the standalone `rclpy`/`ros2` CLI DLL blocker or move recording and
   Python ROS nodes to another officially supported deployment;
2. record `/clock` and all custom-message topics in MCAP;
3. diagnose the failed reach/lift on development-only inputs and demonstrate a
   successful live lift without changing frozen holdout evidence;
4. verify deterministic RESET live;
5. run all three strategies under paired Profile A/Profile B development and
   frozen Isaac holdout conditions, then replay every episode independently.

The live safe-hold and policy-driven smokes validate Isaac integration, but
neither is a paired benchmark result and neither inherits any deterministic
reference-plant score. The policy smoke's task failure is retained as measured
development evidence rather than hidden or reclassified as success.
