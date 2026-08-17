"""Record reproducible M4 paired behavior videos without changing the runner.

The capture path wraps the frozen M4 backend and stores the already-returned
agent-view observation after every environment step.  It does not add an
extra simulator render call inside the 20 Hz control loop.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from actionstream.benchmark import _make_async_worker, _run_async_episode
from actionstream.lerobot_backend import MODEL_ID, MODEL_REVISION, LeRobotBackend


FROZEN_M4_SOURCE_COMMIT = "d84cb64e9e48b681e083828b24df5a38709c78c8"
CAPTURE_MODES = ("async_naive", "async_aligned")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    override = os.environ.get("ACTIONSTREAM_SOURCE_COMMIT")
    if override:
        return override
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _runtime_identity() -> dict[str, Any]:
    package_versions: dict[str, str | None] = {}
    for distribution in ("lerobot", "transformers", "mujoco", "robosuite", "av"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "packages": package_versions,
    }


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _extract_agentview_frame(observation: Mapping[str, Any]) -> np.ndarray:
    pixels = observation.get("pixels")
    if not isinstance(pixels, Mapping) or not pixels:
        raise ValueError("Observation does not contain a nonempty pixels mapping")

    image = pixels.get("image")
    if image is None:
        image = next(iter(pixels.values()))
    frame = np.asarray(image)
    while frame.ndim > 3 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Unexpected agent-view image shape: {frame.shape}")
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.size and float(frame.max()) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Match LeRobot LiberoEnv.render(), which flips the raw camera on both axes
    # for visualization.  Copy so the vector environment cannot mutate it.
    return frame[::-1, ::-1].copy()


class _FrameCaptureBackend:
    def __init__(self, delegate: LeRobotBackend) -> None:
        self.delegate = delegate
        self.frames: list[np.ndarray] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def reset_episode(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
        observation, info, instruction = self.delegate.reset_episode(**kwargs)
        self.frames = [_extract_agentview_frame(observation)]
        return observation, info, instruction

    def step(self, task_id: int, action: np.ndarray) -> Any:
        output = self.delegate.step(task_id, action)
        self.frames.append(_extract_agentview_frame(output.observation))
        return output


def _load_trace_telemetry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as trace:
        return {
            "queue_depth_before_action": trace["queue_depth_before_action"].copy(),
            "queue_depth_after_action": trace["queue_depth_after_action"].copy(),
            "queue_hold_mask": trace["queue_hold_mask"].copy(),
        }


def _annotated_frame(
    raw_frame: np.ndarray,
    *,
    mode: str,
    step_index: int,
    record: Mapping[str, Any],
    queue_before: int | None,
    queue_after: int | None,
    hold: bool | None,
) -> np.ndarray:
    label = "BEFORE" if mode == "async_naive" else "AFTER"
    accent = (238, 156, 0) if mode == "async_naive" else (0, 166, 118)
    image = Image.fromarray(raw_frame, mode="RGB").resize((600, 600), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (640, 720), (8, 15, 28))
    canvas.paste(image, (20, 78))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    body_font = _font(18)
    small_font = _font(16)

    draw.rectangle((0, 0, 640, 6), fill=accent)
    draw.text((20, 12), f"{label}: {mode}", font=title_font, fill=accent)
    draw.text(
        (20, 47),
        (
            f"task {record['task_id']} | seed {record['seed']} | "
            f"state {record['initial_state_index']} | delay {record['injected_delay_ms']} ms"
        ),
        font=small_font,
        fill=(225, 231, 239),
    )

    outcome = "SUCCESS" if bool(record["success"]) else "FAIL / TIMEOUT"
    outcome_color = (63, 209, 128) if bool(record["success"]) else (255, 107, 107)
    footer = f"step {step_index:03d}/{record['environment_steps']:03d}"
    if queue_before is not None and queue_after is not None:
        footer += f" | queue {queue_before}->{queue_after}"
    if hold is True:
        footer += " | HOLD"
    elif hold is False and step_index > 0:
        footer += " | ACTION"
    draw.rectangle((0, 678, 640, 720), fill=(8, 15, 28))
    draw.text((20, 686), footer, font=body_font, fill=(232, 236, 243))
    outcome_width = draw.textbbox((0, 0), outcome, font=body_font)[2]
    draw.text((620 - outcome_width, 686), outcome, font=body_font, fill=outcome_color)
    return np.asarray(canvas, dtype=np.uint8)


def _encode_video(
    output: Path,
    frames: list[np.ndarray],
    *,
    fps: int,
    mode: str,
    record: Mapping[str, Any],
    telemetry: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 640
    stream.height = 720
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    before = telemetry["queue_depth_before_action"]
    after = telemetry["queue_depth_after_action"]
    holds = telemetry["queue_hold_mask"]
    try:
        for frame_index, raw_frame in enumerate(frames):
            trace_index = frame_index - 1
            queue_before = int(before[trace_index]) if 0 <= trace_index < len(before) else None
            queue_after = int(after[trace_index]) if 0 <= trace_index < len(after) else None
            hold = bool(holds[trace_index]) if 0 <= trace_index < len(holds) else None
            annotated = _annotated_frame(
                raw_frame,
                mode=mode,
                step_index=frame_index,
                record=record,
                queue_before=queue_before,
                queue_after=queue_after,
                hold=hold,
            )
            video_frame = av.VideoFrame.from_ndarray(annotated, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    return {
        "codec": "h264/libx264",
        "width": 640,
        "height": 720,
        "fps": fps,
        "frame_count": len(frames),
        "duration_seconds": len(frames) / fps,
    }


def _decode_video(path: Path) -> tuple[list[np.ndarray], float]:
    frames: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate is not None else 20.0
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"Video contains no decodable frames: {path}")
    return frames, rate


def _validate_pair(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    left_params = left["parameters"]
    right_params = right["parameters"]
    for field in (
        "task_id",
        "episode_index",
        "initial_state_index",
        "base_seed",
        "episode_seed",
        "injected_delay_ms",
        "episode_length",
        "replan_interval_steps",
        "model_id",
        "model_revision",
    ):
        if left_params[field] != right_params[field]:
            raise ValueError(f"Pair mismatch for {field}: {left_params[field]} != {right_params[field]}")
    if left_params["mode"] != "async_naive" or right_params["mode"] != "async_aligned":
        raise ValueError("Pair must be ordered async_naive then async_aligned")
    if left.get("runtime_identity") != right.get("runtime_identity"):
        raise ValueError("Pair runtime identity differs between before and after captures")


def compose_pair(
    *,
    before_receipt_path: Path,
    after_receipt_path: Path,
    output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    if output.exists() or receipt_output.exists():
        raise FileExistsError("Refusing to overwrite an existing paired capture artifact")
    before_receipt = json.loads(before_receipt_path.read_text(encoding="utf-8"))
    after_receipt = json.loads(after_receipt_path.read_text(encoding="utf-8"))
    _validate_pair(before_receipt, after_receipt)

    before_video = before_receipt_path.parent / before_receipt["artifacts"]["video"]["path"]
    after_video = after_receipt_path.parent / after_receipt["artifacts"]["video"]["path"]
    before_frames, before_rate = _decode_video(before_video)
    after_frames, after_rate = _decode_video(after_video)
    if round(before_rate, 6) != round(after_rate, 6):
        raise ValueError(f"Pair frame-rate mismatch: {before_rate} != {after_rate}")
    fps = int(round(before_rate))

    output.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 1280
    stream.height = 720
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    total_frames = max(len(before_frames), len(after_frames))
    try:
        for index in range(total_frames):
            before = before_frames[min(index, len(before_frames) - 1)]
            after = after_frames[min(index, len(after_frames) - 1)]
            combined = np.concatenate((before, after), axis=1)
            video_frame = av.VideoFrame.from_ndarray(combined, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    receipt = {
        "schema_version": 1,
        "artifact_type": "m4_paired_behavior_capture",
        "evidence_boundary": (
            "Fresh same-condition visualization rerun. It is not original M4 episode footage and "
            "does not replace the frozen M4 headline statistics."
        ),
        "parameters": before_receipt["parameters"] | {"modes": ["async_naive", "async_aligned"]},
        "before": {
            "receipt": str(before_receipt_path),
            "success": before_receipt["result"]["success"],
            "environment_steps": before_receipt["result"]["environment_steps"],
        },
        "after": {
            "receipt": str(after_receipt_path),
            "success": after_receipt["result"]["success"],
            "environment_steps": after_receipt["result"]["environment_steps"],
        },
        "video": {
            "path": output.name,
            "sha256": _sha256(output),
            "codec": "h264/libx264",
            "width": 1280,
            "height": 720,
            "fps": fps,
            "frame_count": total_frames,
            "duration_seconds": total_frames / fps,
            "shorter_side_last_frame_held": len(before_frames) != len(after_frames),
        },
    }
    receipt_output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def capture_episode(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.mode}.metrics.jsonl"
    video_path = args.output_dir / f"{args.mode}.mp4"
    receipt_path = args.output_dir / f"{args.mode}.receipt.json"
    if any(path.exists() for path in (metrics_path, video_path, receipt_path)):
        raise FileExistsError(f"Refusing to overwrite an existing capture in {args.output_dir}")

    delegate = LeRobotBackend(
        task_ids=[args.task_id],
        seed=args.base_seed,
        suite=args.suite,
        episode_length=args.episode_length,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    backend = _FrameCaptureBackend(delegate)
    worker = _make_async_worker(backend, injected_delay_ms=args.injected_delay_ms)
    try:
        record = _run_async_episode(
            backend,
            run_id=args.run_id,
            git_commit=_git_commit(),
            task_id=args.task_id,
            episode_index=args.episode_index,
            initial_state_index=args.initial_state_index,
            seed=args.episode_seed,
            injected_delay_ms=args.injected_delay_ms,
            realtime=True,
            replan_interval_steps=args.replan_interval_steps,
            mode=args.mode,
            inference_worker=worker,
            output=metrics_path,
        )
    finally:
        try:
            worker.close()
        finally:
            delegate.close()

    metrics_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    trace_path = Path(record["action_trace_path"])
    telemetry = _load_trace_telemetry(trace_path)
    if len(backend.frames) != int(record["environment_steps"]) + 1:
        raise RuntimeError(
            "Frame/step mismatch: "
            f"{len(backend.frames)} frames for {record['environment_steps']} environment steps"
        )
    video_info = _encode_video(
        video_path,
        backend.frames,
        fps=int(round(float(record["controller_frequency_hz"]))),
        mode=args.mode,
        record=record,
        telemetry=telemetry,
    )

    expected = {
        "success": args.expected_success,
        "environment_steps": args.expected_steps,
    }
    historical_match = (
        (args.expected_success is None or bool(record["success"]) is args.expected_success)
        and (args.expected_steps is None or int(record["environment_steps"]) == args.expected_steps)
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "m4_behavior_capture_rerun",
        "source_commit": _git_commit(),
        "frozen_m4_source_commit": FROZEN_M4_SOURCE_COMMIT,
        "capture_overlay_sha256": _sha256(Path(__file__)),
        "runtime_identity": _runtime_identity(),
        "evidence_boundary": (
            "Fresh same-condition visualization rerun. It is not original M4 episode footage and "
            "does not replace the frozen M4 headline statistics."
        ),
        "parameters": {
            "mode": args.mode,
            "task_id": args.task_id,
            "episode_index": args.episode_index,
            "initial_state_index": args.initial_state_index,
            "base_seed": args.base_seed,
            "episode_seed": args.episode_seed,
            "injected_delay_ms": args.injected_delay_ms,
            "episode_length": args.episode_length,
            "replan_interval_steps": args.replan_interval_steps,
            "suite": args.suite,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
        },
        "historical_reference": expected,
        "historical_outcome_match": historical_match,
        "result": record,
        "frame_sha256": {
            "first": hashlib.sha256(backend.frames[0].tobytes()).hexdigest(),
            "middle": hashlib.sha256(backend.frames[len(backend.frames) // 2].tobytes()).hexdigest(),
            "last": hashlib.sha256(backend.frames[-1].tobytes()).hexdigest(),
        },
        "artifacts": {
            "metrics": {"path": metrics_path.name, "sha256": _sha256(metrics_path)},
            "trace": {"path": str(trace_path.relative_to(args.output_dir)), "sha256": _sha256(trace_path)},
            "video": {"path": video_path.name, "sha256": _sha256(video_path), **video_info},
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def _parse_optional_bool(value: str) -> bool | None:
    if value == "none":
        return None
    return value == "true"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--mode", choices=CAPTURE_MODES, required=True)
    capture.add_argument("--task-id", type=int, default=0)
    capture.add_argument("--episode-index", type=int, default=2)
    capture.add_argument("--initial-state-index", type=int, default=4)
    capture.add_argument("--base-seed", type=int, default=142)
    capture.add_argument("--episode-seed", type=int, default=144)
    capture.add_argument("--injected-delay-ms", type=int, default=950)
    capture.add_argument("--episode-length", type=int, default=800)
    capture.add_argument("--replan-interval-steps", type=int, default=10)
    capture.add_argument("--suite", default="libero_object")
    capture.add_argument("--model-id", default=MODEL_ID)
    capture.add_argument("--model-revision", default=MODEL_REVISION)
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument(
        "--expected-success",
        type=_parse_optional_bool,
        choices=(True, False, None),
        default=None,
    )
    capture.add_argument("--expected-steps", type=int, default=None)

    compose = subparsers.add_parser("compose")
    compose.add_argument("--before-receipt", type=Path, required=True)
    compose.add_argument("--after-receipt", type=Path, required=True)
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument("--receipt-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "capture":
        capture_episode(args)
    else:
        compose_pair(
            before_receipt_path=args.before_receipt,
            after_receipt_path=args.after_receipt,
            output=args.output,
            receipt_output=args.receipt_output,
        )


if __name__ == "__main__":
    main()
