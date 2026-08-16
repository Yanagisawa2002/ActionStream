"""Bounded learned X-VLA -> native Isaac development smoke.

The smoke proves four things before any paired protocol can be frozen: the
exact learned checkpoint performs GPU inference, its finite action chunk passes
through an explicit adapter, the resulting commands physically move the native
Isaac Franka, and the native viewport forms a playable video.  It is not a
task-success benchmark.  The optional LIBERO task scene can use either native
cameras or official LIBERO cameras driven by the current Isaac robot/object
state; the generic smoke retains its duplicate camera2.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np

from actionstream.isaac_learned import (
    LIBERO_HAND_TO_EEF_ROTATION_MAT,
    LIBERO_HAND_TO_EEF_TRANSLATION_XYZ,
    LIBERO_TO_ISAAC_TRANSLATION_XYZ,
    adapter_contract_payload,
    adapter_contract_sha256,
    libero_state_from_isaac,
    map_xvla_chunk_to_isaac,
)

# Isaac's headless viewport forces DLSS even when the launcher requests a
# spatial anti-aliasing mode. Render several static frames before capture so the
# temporal accumulator converges instead of exposing multi-frame robot ghosts
# to the learned policy.
POLICY_OBSERVATION_SETTLE_UPDATES = 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str:
    override = os.environ.get("ACTIONSTREAM_SOURCE_COMMIT")
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _git_source_state(source_files: Sequence[Path]) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = sorted(line for line in status.stdout.splitlines() if line.strip())
    hashes = {
        str(path.resolve()): _sha256_file(path.resolve())
        for path in source_files
        if path.resolve().is_file()
    }
    return {
        "git_commit": _git_commit(),
        "git_dirty": bool(lines) if status.returncode == 0 else None,
        "git_status_porcelain": lines,
        "source_file_sha256": hashes,
        "source_manifest_sha256": _sha256_json(hashes),
    }


class _ViewportRecorder:
    """Synchronous LDR viewport frames plus NVIDIA-bundled MP4 encoding."""

    _EXTENSIONS = (
        "omni.kit.renderer.capture",
        "omni.kit.viewport.utility",
        "omni.videoencoding",
    )

    def __init__(
        self,
        simulation_app: Any,
        *,
        output_directory: Path,
        width: int,
        height: int,
        fps: int,
        timeout_seconds: float,
        camera_position_xyz: Sequence[float] = (1.35, 1.20, 1.05),
        camera_target_xyz: Sequence[float] = (0.38, 0.0, 0.24),
        camera_vertical_fov_degrees: float | None = None,
    ) -> None:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in self._EXTENSIONS:
            manager.set_extension_enabled_immediate(extension, True)
            simulation_app.update()
            if not manager.is_extension_enabled(extension):
                raise RuntimeError(f"Isaac capture extension {extension!r} is unavailable")

        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf, UsdGeom, UsdLux

        viewport = get_active_viewport()
        if viewport is None or viewport.stage is None:
            raise RuntimeError("Isaac has no active viewport/stage")
        light = UsdLux.DomeLight.Define(viewport.stage, "/World/LearnedSmokeDomeLight")
        light.CreateIntensityAttr(1000.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        self.frame_directory = output_directory / "frames"
        self.frame_directory.mkdir(parents=True, exist_ok=False)
        self.output_path = output_directory / "learned_isaac_smoke.mp4"
        self._capture_viewport_to_file = capture_viewport_to_file
        self._viewport = viewport
        self._original_resolution = tuple(viewport.resolution)
        self._restored = False
        self._fps = int(fps)
        self._timeout_seconds = float(timeout_seconds)
        self._frame_paths: list[Path] = []
        self.encoder: str | None = None
        self.encoder_fallback_reason: str | None = None
        viewport.resolution = (int(width), int(height))
        simulation_app.update()
        camera = ViewportCameraState(viewport=viewport)
        camera.set_position_world(Gf.Vec3d(*camera_position_xyz), rotate=False)
        camera.set_target_world(Gf.Vec3d(*camera_target_xyz), rotate=True)
        if camera_vertical_fov_degrees is not None:
            fov = float(camera_vertical_fov_degrees)
            if not 5.0 <= fov <= 150.0:
                raise ValueError("camera vertical field of view must lie in [5,150]")
            camera_prim = viewport.stage.GetPrimAtPath(viewport.camera_path)
            usd_camera = UsdGeom.Camera(camera_prim)
            vertical_aperture = float(usd_camera.GetVerticalApertureAttr().Get())
            focal_length = 0.5 * vertical_aperture / math.tan(math.radians(fov) / 2.0)
            usd_camera.GetFocalLengthAttr().Set(focal_length)
        simulation_app.update()
        simulation_app.update()

    def capture(self, simulation_app: Any) -> Path:
        for _ in range(POLICY_OBSERVATION_SETTLE_UPDATES):
            simulation_app.update()
        path = self.frame_directory / f"frame_{len(self._frame_paths):06d}.png"
        helper = self._capture_viewport_to_file(
            self._viewport, file_path=str(path), is_hdr=False
        )
        task = asyncio.ensure_future(helper.wait_for_result(completion_frames=0))
        deadline = time.monotonic() + self._timeout_seconds
        while not task.done() or not path.is_file() or path.stat().st_size <= 0:
            if task.cancelled():
                raise RuntimeError("Isaac viewport frame capture was cancelled")
            if task.done() and task.exception() is not None:
                raise RuntimeError("Isaac viewport frame capture failed") from task.exception()
            if not simulation_app.is_running():
                raise RuntimeError("Isaac stopped during viewport frame capture")
            if time.monotonic() >= deadline:
                raise RuntimeError("Isaac viewport frame capture timed out")
            simulation_app.update()
        self._frame_paths.append(path)
        return path

    def _restore(self) -> None:
        if not self._restored:
            self._viewport.resolution = self._original_resolution
            self._restored = True

    def finish(self, simulation_app: Any) -> Path:
        if not self._frame_paths:
            raise RuntimeError("cannot encode a zero-frame smoke video")
        self._restore()
        import omni.kit.renderer_capture
        from video_encoding import encode_image_file_sequence

        omni.kit.renderer_capture.acquire_renderer_capture_interface().wait_async_capture()
        frame_pattern = str(self.frame_directory / "frame_%06d.png")
        native_partial = self.output_path.with_name(
            f".{self.output_path.stem}.nvenc.partial.mp4"
        )
        try:
            encoded = encode_image_file_sequence(
                frame_pattern, 0, self._fps, str(native_partial), False
            )
            if (
                not encoded
                or not native_partial.is_file()
                or native_partial.stat().st_size <= 0
            ):
                raise RuntimeError("Isaac video encoder produced no non-empty MP4")
            native_partial.replace(self.output_path)
            self.encoder = "isaac_omni_videoencoding"
        except RuntimeError as exc:
            self.encoder_fallback_reason = f"{type(exc).__name__}: {exc}"
            if native_partial.exists():
                native_partial.unlink()
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError(
                    "Isaac video encoding failed and ffmpeg is unavailable"
                ) from exc
            ffmpeg_partial = self.output_path.with_name(
                f".{self.output_path.stem}.ffmpeg.partial.mp4"
            )
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(self._fps),
                    "-i",
                    frame_pattern,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-y",
                    str(ffmpeg_partial),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if (
                completed.returncode != 0
                or not ffmpeg_partial.is_file()
                or ffmpeg_partial.stat().st_size <= 0
            ):
                raise RuntimeError(
                    "both Isaac and CPU ffmpeg video encoding failed: "
                    f"ffmpeg_exit={completed.returncode}, stderr={completed.stderr[-500:]}"
                ) from exc
            ffmpeg_partial.replace(self.output_path)
            self.encoder = "ffmpeg_libx264_cpu_fallback"
        return self.output_path

    def close(self) -> None:
        self._restore()


class _PolicyWorker:
    def __init__(
        self,
        *,
        python: Path,
        protocol: Path,
        output_directory: Path,
        timeout_seconds: float,
        seed: int,
        suite: str,
        task_id: int,
        initial_state_index: int,
        render_output_directory: Path | None,
    ) -> None:
        self._timeout_seconds = float(timeout_seconds)
        self._stderr_stream = (output_directory / "policy_worker.stderr.log").open(
            "w", encoding="utf-8"
        )
        environment = os.environ.copy()
        # SimulationApp sets interpreter hints for Kit's embedded Python.  They
        # must not leak into the separately provisioned LeRobot virtualenv.
        environment.pop("PYTHONHOME", None)
        environment["VIRTUAL_ENV"] = str(python.parent.parent)
        environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
        source_path = str(Path(__file__).resolve().parents[1])
        venv_site_packages = python.parent.parent / "lib" / "python3.12" / "site-packages"
        if not venv_site_packages.is_dir():
            raise RuntimeError(
                f"policy virtualenv site-packages is missing: {venv_site_packages}"
            )
        # Do not inherit Kit/ROS PYTHONPATH entries.  Explicitly bind the worker
        # to ActionStream source plus the already-provisioned LeRobot venv.
        environment["PYTHONPATH"] = os.pathsep.join(
            (source_path, str(venv_site_packages))
        )
        command = [
            str(python),
            "-m",
            "actionstream.learned_policy_worker",
            "--protocol",
            str(protocol),
            "--model",
            "xvla",
            "--suite",
            suite,
            "--task-id",
            str(task_id),
            "--seed",
            str(seed),
            "--initial-state-index",
            str(initial_state_index),
        ]
        if render_output_directory is not None:
            command.extend(
                ["--render-output-directory", str(render_output_directory)]
            )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_stream,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.ready = self._read()
        if self.ready.get("event") != "ready":
            raise RuntimeError(f"learned policy worker did not become ready: {self.ready}")

    def _read(self) -> dict[str, Any]:
        if self._process.stdout is None:
            raise RuntimeError("policy worker stdout is closed")
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(timeout=self._timeout_seconds)
        finally:
            selector.close()
        if not events:
            raise TimeoutError(
                f"policy worker produced no response within {self._timeout_seconds:.1f}s"
            )
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"policy worker exited with code {self._process.poll()}")
        response = json.loads(line)
        if response.get("event") == "fatal":
            raise RuntimeError(f"policy worker fatal response: {response.get('error')}")
        return response

    def _request(
        self, request: Mapping[str, Any], *, expected_event: str
    ) -> dict[str, Any]:
        if self._process.stdin is None:
            raise RuntimeError("policy worker stdin is closed")
        self._process.stdin.write(
            json.dumps(dict(request), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        self._process.stdin.flush()
        response = self._read()
        if response.get("event") != expected_event:
            raise RuntimeError(f"unexpected policy worker response: {response}")
        if int(response.get("request_id", -1)) != int(request["request_id"]):
            raise RuntimeError("policy worker request/response ID mismatch")
        return response

    def infer(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._request(request, expected_event="inference")

    def evaluate(self, request: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(request)
        payload["render_only"] = True
        return self._request(payload, expected_event="render")

    def close(self) -> None:
        process = self._process
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write('{"command":"close"}\n')
                process.stdin.flush()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10.0)
        finally:
            self._stderr_stream.close()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene-protocol", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081601)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument(
        "--instruction", default="pick up the blue cube and place it on the green marker"
    )
    parser.add_argument("--control-steps", type=int, default=30)
    parser.add_argument("--request-interval-steps", type=int, default=10)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--capture-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--libero-object-task0-scene",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply the development-only canonical LIBERO Object task-0 scene",
    )
    parser.add_argument(
        "--libero-task-scene",
        choices=("object0", "spatial2", "goal5"),
        default=None,
        help="select one audited learned-policy native Isaac task scene",
    )
    parser.add_argument("--libero-assets-root", type=Path, default=None)
    parser.add_argument("--asset-cache-directory", type=Path, default=None)
    parser.add_argument(
        "--official-render-bridge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="infer from official LIBERO cameras after dynamic Isaac robot/object writes",
    )
    return parser


def _selected_task_scene_key(args: Any) -> str | None:
    if args.libero_object_task0_scene and args.libero_task_scene is not None:
        raise ValueError(
            "use either --libero-object-task0-scene or --libero-task-scene, not both"
        )
    if args.libero_object_task0_scene:
        return "object0"
    return args.libero_task_scene


def _validate_args(args: Any) -> tuple[Path, Path, Path, Path]:
    protocol = args.protocol.resolve()
    scene_protocol = args.scene_protocol.resolve()
    # Preserve the virtualenv launcher path.  Path.resolve() dereferences its
    # symlink to the base interpreter and loses the pyvenv/site-packages root.
    policy_python = Path(os.path.abspath(args.policy_python.expanduser()))
    output = args.output_directory.resolve()
    for path, name in (
        (protocol, "protocol"),
        (scene_protocol, "scene protocol"),
        (policy_python, "policy Python"),
    ):
        if not path.is_file():
            raise ValueError(f"{name} does not exist: {path}")
    if output.exists():
        raise ValueError(f"refusing to reuse smoke output directory: {output}")
    if not 1 <= args.control_steps <= 300:
        raise ValueError("control steps must lie in [1,300]")
    if not 1 <= args.request_interval_steps <= args.control_steps:
        raise ValueError("request interval must lie in [1,control steps]")
    if not str(args.instruction).strip():
        raise ValueError("instruction must be non-empty")
    if not 64 <= args.video_width <= 2048 or not 64 <= args.video_height <= 2048:
        raise ValueError("video dimensions must lie in [64,2048]")
    task_scene_key = _selected_task_scene_key(args)
    if task_scene_key == "object0" and args.libero_assets_root is None:
        raise ValueError("LIBERO object0 task scene requires --libero-assets-root")
    if task_scene_key != "object0" and (
        args.libero_assets_root is not None or args.asset_cache_directory is not None
    ):
        raise ValueError("LIBERO asset arguments are only valid for the object0 task scene")
    if args.official_render_bridge and task_scene_key is None:
        raise ValueError(
            "--official-render-bridge requires an audited --libero-task-scene"
        )
    if task_scene_key in {"spatial2", "goal5"} and not args.official_render_bridge:
        raise ValueError("proxy task scenes require --official-render-bridge")
    if task_scene_key is not None:
        if task_scene_key == "object0":
            expected = ("libero_object", 0, 0)
        else:
            from actionstream.isaac_libero_proxy_tasks import proxy_task_spec

            spec = proxy_task_spec(task_scene_key)
            expected = (spec.suite, spec.task_id, spec.initial_state_index)
        actual = (args.suite, args.task_id, args.initial_state_index)
        if actual != expected:
            raise ValueError(
                f"{task_scene_key} requires suite/task/initial-state {expected}, got {actual}"
            )
    if args.initial_state_index < 0:
        raise ValueError("initial state index must be non-negative")
    return protocol, scene_protocol, policy_python, output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        protocol, scene_protocol_path, policy_python, output = _validate_args(args)
    except ValueError as exc:
        print(f"invalid learned Isaac smoke configuration: {exc}", file=sys.stderr)
        return 2
    task_scene_key = _selected_task_scene_key(args)
    output.mkdir(parents=True)
    events_path = output / "events.jsonl"
    policy_actions_path = output / "policy_actions.jsonl"
    simulation_app: Any | None = None
    scene: Any | None = None
    worker: _PolicyWorker | None = None
    recorder: _ViewportRecorder | None = None
    learned_task_scene: Any | None = None
    wrist_camera: Any | None = None
    started_ns = time.time_ns()
    try:
        from action_stream_isaac.capability_probe import probe
        from action_stream_isaac.dynamic_isaac_adapter import (
            CONTROL_DT_SECONDS,
            PHYSICS_STEPS_PER_CONTROL,
            DynamicIsaacScene,
            runtime_configs_from_protocol,
        )

        capability = probe(require_ros_imports=False)
        if not capability.ready:
            raise RuntimeError(f"native Isaac capability probe failed: {capability.to_dict()}")
        scene_protocol = json.loads(scene_protocol_path.read_text(encoding="utf-8"))
        task_config, policy_config = runtime_configs_from_protocol(scene_protocol)
        proxy_scene_spec: Any | None = None
        if task_scene_key in {"spatial2", "goal5"}:
            from actionstream.isaac_libero_proxy_tasks import proxy_task_spec

            proxy_scene_spec = proxy_task_spec(task_scene_key)

        # Start LeRobot before Kit.  SimulationApp mutates process-wide Python
        # interpreter hints for its embedded runtime; a worker forked afterward
        # can otherwise resolve Kit's site-packages instead of its own venv.
        worker = _PolicyWorker(
            python=policy_python,
            protocol=protocol,
            output_directory=output,
            timeout_seconds=args.worker_timeout_seconds,
            seed=args.seed,
            suite=args.suite,
            task_id=args.task_id,
            initial_state_index=args.initial_state_index,
            render_output_directory=(
                output / "official_policy_frames"
                if args.official_render_bridge
                else None
            ),
        )

        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": bool(args.headless)})
        scene = DynamicIsaacScene(
            simulation_app,
            task_config=task_config,
            policy_config=policy_config,
            object_collision_scale_xyz=(
                proxy_scene_spec.collision_scale_xyz
                if proxy_scene_spec is not None
                else None
            ),
        )
        from action_stream_isaac.dynamic_task import scenario_for_seed, scenario_payload

        scenario = scenario_for_seed(args.seed, task_config)
        reset_measurement = scene.reset(scenario)
        effective_instruction = args.instruction
        camera_position = (1.35, 1.20, 1.05)
        camera_target = (0.38, 0.0, 0.24)
        camera_fovy: float | None = None
        coordinate_translation_xyz = LIBERO_TO_ISAAC_TRANSLATION_XYZ
        hand_to_eef_translation_xyz = (0.0, 0.0, 0.0)
        hand_to_eef_rotation_mat = np.eye(3, dtype=np.float64)
        camera_2_source = "duplicate_external_view_development_smoke_only"
        learned_scene_provenance: dict[str, Any] | None = None
        if task_scene_key == "object0":
            from actionstream.isaac_libero_task import (
                LIBERO_WRIST_CAMERA_SETTLE_UPDATES,
                LiberoWristCamera,
                LiberoObjectTask0Scene,
            )

            learned_task_scene = LiberoObjectTask0Scene(
                simulation_app,
                scene,
                assets_root=args.libero_assets_root,
                cache_directory=(
                    args.asset_cache_directory
                    if args.asset_cache_directory is not None
                    else output / "converted_assets"
                ),
            )
            reset_measurement = scene.measure()
            camera_position = learned_task_scene.camera_position_xyz
            camera_target = learned_task_scene.camera_target_xyz
            camera_fovy = learned_task_scene.camera_vertical_fov_degrees
            coordinate_translation_xyz = learned_task_scene.coordinate_translation_xyz
            hand_to_eef_translation_xyz = LIBERO_HAND_TO_EEF_TRANSLATION_XYZ
            hand_to_eef_rotation_mat = np.asarray(
                LIBERO_HAND_TO_EEF_ROTATION_MAT, dtype=np.float64
            )
            camera_2_source = (
                "official_libero_renderer_dynamic_state_bridge"
                if args.official_render_bridge
                else "audited_libero_camera_to_eef_rigid_transform_in_native_isaac"
            )
            learned_scene_provenance = learned_task_scene.provenance
            if effective_instruction == (
                "pick up the blue cube and place it on the green marker"
            ):
                effective_instruction = learned_task_scene.instruction
        elif task_scene_key is not None:
            from actionstream.isaac_libero_proxy_tasks import (
                LiberoProxyTaskScene,
            )

            if proxy_scene_spec is None:
                raise RuntimeError("proxy task scene has no frozen task specification")
            learned_task_scene = LiberoProxyTaskScene(
                simulation_app,
                scene,
                spec=proxy_scene_spec,
            )
            reset_measurement = scene.measure()
            camera_position = learned_task_scene.camera_position_xyz
            camera_target = learned_task_scene.camera_target_xyz
            camera_fovy = learned_task_scene.camera_vertical_fov_degrees
            coordinate_translation_xyz = learned_task_scene.coordinate_translation_xyz
            hand_to_eef_translation_xyz = LIBERO_HAND_TO_EEF_TRANSLATION_XYZ
            hand_to_eef_rotation_mat = np.asarray(
                LIBERO_HAND_TO_EEF_ROTATION_MAT, dtype=np.float64
            )
            camera_2_source = "official_libero_renderer_dynamic_state_bridge"
            learned_scene_provenance = learned_task_scene.provenance
            if effective_instruction == (
                "pick up the blue cube and place it on the green marker"
            ):
                effective_instruction = learned_task_scene.instruction
        recorder = _ViewportRecorder(
            simulation_app,
            output_directory=output,
            width=args.video_width,
            height=args.video_height,
            fps=round(1.0 / CONTROL_DT_SECONDS),
            timeout_seconds=args.capture_timeout_seconds,
            camera_position_xyz=camera_position,
            camera_target_xyz=camera_target,
            camera_vertical_fov_degrees=camera_fovy,
        )
        policy_frame_directory: Path | None = None
        policy_wrist_frame_directory: Path | None = None
        if learned_task_scene is not None and not args.official_render_bridge:
            policy_frame_directory = output / "policy_frames"
            policy_frame_directory.mkdir(exist_ok=False)
            policy_wrist_frame_directory = output / "policy_wrist_frames"
            policy_wrist_frame_directory.mkdir(exist_ok=False)
            wrist_camera = LiberoWristCamera(
                simulation_app,
                output_directory=output / "wrist_frames",
                width=args.video_width,
                height=args.video_height,
                timeout_seconds=args.capture_timeout_seconds,
            )
            if learned_scene_provenance is None:
                raise RuntimeError("learned task scene has no provenance")
            learned_scene_provenance["policy_observation_transform"] = {
                "agentview": "vertical_flip_to_match_hf_libero_raw_mujoco_gl_row_order",
                "eye_in_hand": "vertical_flip_to_match_hf_libero_raw_mujoco_gl_row_order",
            }
            learned_scene_provenance["policy_observation_render_contract"] = {
                "anti_aliasing": "Isaac headless viewport default DLSS",
                "agentview_static_settle_updates": POLICY_OBSERVATION_SETTLE_UPDATES,
                "eye_in_hand_static_settle_updates": (
                    LIBERO_WRIST_CAMERA_SETTLE_UPDATES
                ),
            }
        elif learned_task_scene is not None:
            if learned_scene_provenance is None:
                raise RuntimeError("learned task scene has no provenance")
            learned_scene_provenance["policy_observation_transform"] = {
                "agentview": "official LeRobot/LIBERO formatted observation",
                "eye_in_hand": "official LeRobot/LIBERO formatted observation",
            }
            learned_scene_provenance["policy_observation_render_contract"] = {
                "renderer": "official pinned LIBERO MuJoCo EGL",
                "state_source": "current native Isaac measurement at every policy request",
                "physics_steps_after_state_write": 0,
            }
        initial_eef = np.asarray(reset_measurement.end_effector_xyz, dtype=np.float64)
        maximum_eef_displacement = 0.0
        maximum_command_delta = 0.0
        mapped_commands: list[list[float]] = []
        requests: list[dict[str, Any]] = []
        active_chunk: np.ndarray | None = None
        active_chunk_offset = 0
        total_workspace_clips = 0
        total_translation_limits = 0
        task_success_control_step: int | None = None

        with (
            events_path.open("w", encoding="utf-8") as events,
            policy_actions_path.open("w", encoding="utf-8") as policy_actions,
        ):
            for control_step in range(args.control_steps):
                observation_frame = recorder.capture(simulation_app)
                policy_observation_frame = observation_frame
                policy_wrist_observation_frame = observation_frame
                if policy_frame_directory is not None:
                    from PIL import Image

                    if wrist_camera is None or policy_wrist_frame_directory is None:
                        raise RuntimeError("learned task scene has no wrist camera")
                    policy_observation_frame = (
                        policy_frame_directory / f"frame_{control_step:06d}.png"
                    )
                    with Image.open(observation_frame) as image:
                        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
                            policy_observation_frame
                        )
                    wrist_measurement = scene.measure()
                    wrist_observation_frame = wrist_camera.capture(
                        simulation_app,
                        end_effector_xyz=wrist_measurement.end_effector_xyz,
                        end_effector_wxyz=wrist_measurement.end_effector_wxyz,
                    )
                    policy_wrist_observation_frame = (
                        policy_wrist_frame_directory / f"frame_{control_step:06d}.png"
                    )
                    with Image.open(wrist_observation_frame) as image:
                        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(
                            policy_wrist_observation_frame
                        )
                if control_step % args.request_interval_steps == 0:
                    measurement = scene.measure()
                    joint_positions, joint_velocities = scene.arm_joint_state()
                    robot_state = libero_state_from_isaac(
                        end_effector_xyz=measurement.end_effector_xyz,
                        end_effector_wxyz=measurement.end_effector_wxyz,
                        gripper_aperture_m=measurement.gripper_aperture_m,
                        joint_positions=joint_positions,
                        joint_velocities=joint_velocities,
                        coordinate_translation_xyz=coordinate_translation_xyz,
                        hand_to_eef_translation_xyz=hand_to_eef_translation_xyz,
                        hand_to_eef_rotation_mat=hand_to_eef_rotation_mat,
                    )
                    request = {
                        "request_id": len(requests),
                        "instruction": effective_instruction,
                        "robot_state": robot_state,
                    }
                    if args.official_render_bridge:
                        if learned_task_scene is None:
                            raise RuntimeError(
                                "official render bridge has no learned task scene"
                            )
                        request["official_render_bridge"] = {
                            "pose_mode": "eef_ik",
                            "object_states": learned_task_scene.official_object_states(
                                measurement
                            ),
                        }
                    else:
                        request.update(
                            {
                                "image": str(policy_observation_frame),
                                "image2": str(policy_wrist_observation_frame),
                            }
                        )
                    response = worker.infer(request)
                    mapped = map_xvla_chunk_to_isaac(
                        response["actions"],
                        initial_target_xyz=scene.current_command[:3],
                        workspace_xyz=(
                            policy_config.workspace_x_m,
                            policy_config.workspace_y_m,
                            policy_config.workspace_z_m,
                        ),
                        maximum_translation_per_step_m=(
                            policy_config.maximum_translation_per_step_m
                        ),
                        coordinate_translation_xyz=coordinate_translation_xyz,
                        hand_to_eef_translation_xyz=hand_to_eef_translation_xyz,
                        hand_to_eef_rotation_mat=hand_to_eef_rotation_mat,
                    )
                    active_chunk = mapped.commands
                    active_chunk_offset = 0
                    total_workspace_clips += mapped.clipped_workspace_rows
                    total_translation_limits += mapped.limited_translation_rows
                    request_record = {
                        key: value for key, value in response.items() if key != "actions"
                    }
                    request_record.update(
                        {
                            "control_step": control_step,
                            "mapped_actions_sha256": hashlib.sha256(
                                mapped.commands.astype(np.float64).tobytes(order="C")
                            ).hexdigest(),
                            "mapped_workspace_clips": mapped.clipped_workspace_rows,
                            "mapped_translation_limits": mapped.limited_translation_rows,
                            "robot_state": robot_state,
                            "robot_state_sha256": _sha256_json(robot_state),
                            "policy_image_path": response["image_path"],
                            "policy_image2_path": response["image2_path"],
                            "policy_image_sha256": response["image_sha256"],
                            "policy_image2_sha256": response["image2_sha256"],
                        }
                    )
                    requests.append(request_record)
                    policy_actions.write(
                        json.dumps(
                            {
                                "event": "learned_action_chunk",
                                "control_step": control_step,
                                "request_id": int(response["request_id"]),
                                "policy_actions": response["actions"],
                                "mapped_commands": mapped.commands.tolist(),
                                "policy_actions_sha256": response["actions_sha256"],
                                "mapped_actions_sha256": request_record[
                                    "mapped_actions_sha256"
                                ],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    policy_actions.flush()
                    bridge_at_request = response.get("official_render_bridge")
                    if (
                        isinstance(bridge_at_request, Mapping)
                        and bridge_at_request.get("official_task_success") is True
                    ):
                        task_success_control_step = control_step
                        break
                if active_chunk is None or active_chunk_offset >= len(active_chunk):
                    raise RuntimeError("learned policy produced no executable action")

                before_target = np.asarray(scene.current_command[:3], dtype=np.float64)
                command = active_chunk[active_chunk_offset]
                active_chunk_offset += 1
                scene.set_command(command)
                for _ in range(PHYSICS_STEPS_PER_CONTROL):
                    scene.step()
                measurement = scene.measure()
                if learned_task_scene is not None:
                    learned_task_scene.sync(measurement)
                maximum_command_delta = max(
                    maximum_command_delta,
                    float(np.linalg.norm(command[:3] - before_target)),
                )
                maximum_eef_displacement = max(
                    maximum_eef_displacement,
                    float(np.linalg.norm(np.asarray(measurement.end_effector_xyz) - initial_eef)),
                )
                mapped_commands.append(command.tolist())
                event = {
                    "event": "control_step",
                    "control_step": control_step,
                    "sim_time_seconds": scene.sim_time_seconds(),
                    "command": command.tolist(),
                    "measured_end_effector_xyz": list(measurement.end_effector_xyz),
                    "measured_object_xyz": list(measurement.object_xyz),
                    "gripper_aperture_m": measurement.gripper_aperture_m,
                    "collision": measurement.collision,
                    "joint_or_workspace_limit": measurement.joint_or_workspace_limit,
                }
                events.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
                events.flush()

        final_measurement = scene.measure()
        final_bridge_evaluation: dict[str, Any] | None = None
        if args.official_render_bridge:
            if learned_task_scene is None:
                raise RuntimeError("official final evaluation has no learned task scene")
            final_joint_positions, final_joint_velocities = scene.arm_joint_state()
            final_robot_state = libero_state_from_isaac(
                end_effector_xyz=final_measurement.end_effector_xyz,
                end_effector_wxyz=final_measurement.end_effector_wxyz,
                gripper_aperture_m=final_measurement.gripper_aperture_m,
                joint_positions=final_joint_positions,
                joint_velocities=final_joint_velocities,
                coordinate_translation_xyz=coordinate_translation_xyz,
                hand_to_eef_translation_xyz=hand_to_eef_translation_xyz,
                hand_to_eef_rotation_mat=hand_to_eef_rotation_mat,
            )
            final_bridge_evaluation = worker.evaluate(
                {
                    "request_id": len(requests),
                    "instruction": effective_instruction,
                    "robot_state": final_robot_state,
                    "official_render_bridge": {
                        "pose_mode": "eef_ik",
                        "object_states": learned_task_scene.official_object_states(
                            final_measurement
                        ),
                    },
                }
            )
        recorder.capture(simulation_app)
        video = recorder.finish(simulation_app)
        worker_ready = dict(worker.ready)
        peak_memory = max(float(item["peak_cuda_memory_mib"]) for item in requests)
        actions_array = np.asarray(mapped_commands, dtype=np.float64)
        unique_actions = int(len(np.unique(np.round(actions_array, decimals=6), axis=0)))
        native_task_success = (
            learned_task_scene.task_success(final_measurement)
            if learned_task_scene is not None
            else None
        )
        official_task_success = None
        if final_bridge_evaluation is not None:
            final_bridge = final_bridge_evaluation.get("official_render_bridge")
            if isinstance(final_bridge, Mapping):
                value = final_bridge.get("official_task_success")
                if isinstance(value, bool):
                    official_task_success = value
        development_task_success = (
            official_task_success
            if official_task_success is not None
            else native_task_success
        )
        checks = {
            "learned_checkpoint_loaded": worker_ready.get("event") == "ready",
            "gpu_inference": bool(worker_ready.get("cuda_available")) and peak_memory > 0.0,
            "finite_policy_actions": bool(np.isfinite(actions_array).all()),
            "policy_action_varies": unique_actions >= 2,
            "policy_command_applied": maximum_command_delta > 1e-5,
            "native_franka_moved": maximum_eef_displacement > 0.002,
            "video_recorded": video.is_file() and video.stat().st_size > 0,
            "no_disqualifying_collision": not any(
                json.loads(line)["collision"]
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ),
        }
        if args.official_render_bridge:
            bridge_records = [
                item.get("official_render_bridge") for item in requests
            ]
            checks.update(
                {
                    "official_render_bridge_every_request": bool(bridge_records)
                    and all(isinstance(item, Mapping) for item in bridge_records),
                    "dynamic_object_state_every_request": bool(bridge_records)
                    and all(
                        isinstance(item, Mapping)
                        and int(item.get("dynamic_object_state_count", 0)) >= 1
                        for item in bridge_records
                    ),
                    "official_state_writeback_exact": bool(bridge_records)
                    and all(
                        isinstance(item, Mapping)
                        and bool(item.get("writeback_exact_at_1e-12"))
                        for item in bridge_records
                    ),
                    "official_eef_ik_within_frozen_tolerance": bool(bridge_records)
                    and all(
                        isinstance(item, Mapping)
                        and isinstance(item.get("ik"), Mapping)
                        and bool(item["ik"].get("converged"))
                        and float(item["ik"].get("position_error_l2_m", float("inf")))
                        <= 1e-4
                        for item in bridge_records
                    ),
                    "official_final_task_predicate_available": (
                        official_task_success is not None
                    ),
                }
            )
        source_root = Path(__file__).resolve().parents[2]
        source_state = _git_source_state(
            (
                Path(__file__),
                source_root / "src/actionstream/isaac_learned.py",
                source_root / "src/actionstream/isaac_libero_task.py",
                source_root / "src/actionstream/isaac_libero_proxy_tasks.py",
                source_root / "src/actionstream/learned_policy_worker.py",
                source_root / "src/actionstream/official_render_bridge.py",
                source_root
                / "ros2_ws/src/action_stream_isaac/action_stream_isaac/dynamic_isaac_adapter.py",
            )
        )
        known_limitations = [
            "development smoke only; no manipulation task success is claimed",
            "coordinate calibration uses one development reset reference",
            "canonical distractors are visual-only development geometry",
            "basket collision uses box proxies rather than converted MuJoCo collision meshes",
            "formal paired runtimes and network profiles have not yet run",
        ]
        if args.official_render_bridge:
            known_limitations.insert(
                2,
                "official images are rendered from synchronized state but physics remains native Isaac",
            )
        else:
            known_limitations.insert(
                2,
                "policy frames apply an explicit vertical row-order transform for LIBERO parity",
            )
        if learned_task_scene is None:
            known_limitations.insert(
                1, "camera2 duplicates the external viewport and is not a wrist camera"
            )
        summary = {
            "schema_version": 1,
            "evidence_class": "development_learned_policy_native_isaac_smoke",
            "headline_or_holdout_eligible": False,
            "task_success_claimed": False,
            "development_task_success_observed": development_task_success,
            "native_proxy_task_success_observed": native_task_success,
            "official_libero_task_success_observed": official_task_success,
            "smoke_pass": all(checks.values()),
            "checks": checks,
            "instruction": effective_instruction,
            "seed": args.seed,
            "suite": args.suite,
            "task_id": args.task_id,
            "initial_state_index": args.initial_state_index,
            "task_scene_key": task_scene_key,
            "official_render_bridge": bool(args.official_render_bridge),
            "control_steps": args.control_steps,
            "executed_control_steps": len(mapped_commands),
            "task_success_control_step": task_success_control_step,
            "request_interval_steps": args.request_interval_steps,
            "worker_ready": worker_ready,
            "request_records": requests,
            "final_official_render_evaluation": final_bridge_evaluation,
            "metrics": {
                "request_count": len(requests),
                "unique_mapped_actions": unique_actions,
                "maximum_command_delta_m": maximum_command_delta,
                "maximum_measured_eef_displacement_m": maximum_eef_displacement,
                "peak_cuda_memory_mib": peak_memory,
                "workspace_clipped_chunk_rows": total_workspace_clips,
                "translation_limited_chunk_rows": total_translation_limits,
                "final_object_xyz": list(final_measurement.object_xyz),
            },
            "video_encoder": recorder.encoder,
            "video_encoder_fallback_reason": recorder.encoder_fallback_reason,
            "adapter_contract": adapter_contract_payload(
                coordinate_translation_xyz=coordinate_translation_xyz,
                camera_2_source=camera_2_source,
                hand_to_eef_translation_xyz=hand_to_eef_translation_xyz,
                hand_to_eef_rotation_mat=hand_to_eef_rotation_mat,
            ),
            "adapter_contract_sha256": adapter_contract_sha256(
                coordinate_translation_xyz=coordinate_translation_xyz,
                camera_2_source=camera_2_source,
                hand_to_eef_translation_xyz=hand_to_eef_translation_xyz,
                hand_to_eef_rotation_mat=hand_to_eef_rotation_mat,
            ),
            "learned_task_scene": learned_scene_provenance,
            "provenance": {
                "actionstream_source": source_state,
                "protocol_path": str(protocol),
                "protocol_sha256": _sha256_file(protocol),
                "scene_protocol_path": str(scene_protocol_path),
                "scene_protocol_sha256": _sha256_file(scene_protocol_path),
                "scenario_sha256": _sha256_json(scenario_payload(scenario)),
                "video_path": str(video),
                "video_sha256": _sha256_file(video),
                "events_path": str(events_path),
                "events_sha256": _sha256_file(events_path),
                "policy_actions_path": str(policy_actions_path),
                "policy_actions_sha256": _sha256_file(policy_actions_path),
                "started_wall_time_ns": started_ns,
                "finished_wall_time_ns": time.time_ns(),
            },
            "known_limitations": known_limitations,
        }
        _write_json(output / "summary.json", summary)
        print(json.dumps({"smoke_pass": summary["smoke_pass"], "summary": str(output / 'summary.json')}))
        return 0 if summary["smoke_pass"] else 4
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "evidence_class": "development_learned_policy_native_isaac_smoke_failure",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "started_wall_time_ns": started_ns,
            "failed_wall_time_ns": time.time_ns(),
        }
        _write_json(output / "failure.json", failure)
        print(f"FATAL: learned Isaac smoke failed: {failure['error']}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if worker is not None:
            try:
                worker.close()
            except Exception:
                traceback.print_exc()
        if recorder is not None:
            try:
                recorder.close()
            except Exception:
                traceback.print_exc()
        if wrist_camera is not None:
            try:
                wrist_camera.close()
            except Exception:
                traceback.print_exc()
        if scene is not None:
            try:
                scene.close()
            except Exception:
                traceback.print_exc()
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
