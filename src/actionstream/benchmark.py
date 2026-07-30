"""Custom synchronous and asynchronous ActionStream benchmark runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from actionstream.lerobot_backend import (
    MODEL_ID,
    MODEL_REVISION,
    LeRobotBackend,
    immutable_observation_snapshot,
    thaw_observation_snapshot,
)
from actionstream.runtime import (
    ActionQueue,
    InferencePayload,
    InferenceRequest,
    InferenceResult,
    LatestRequestWorker,
)


RUNTIME_MODES = ("sync", "async_naive", "async_aligned")


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return parsed


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _coalesce_missed_control_ticks(
    scheduled_timestamp: float,
    current_timestamp: float,
    period_seconds: float,
) -> tuple[float, int]:
    """Skip wall-clock ticks that are already a full period in the past."""
    if current_timestamp <= scheduled_timestamp:
        return scheduled_timestamp, 0
    missed_ticks = int((current_timestamp - scheduled_timestamp) // period_seconds)
    return scheduled_timestamp + missed_ticks * period_seconds, missed_ticks


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def _trace_path(output: Path, mode: str, delay_ms: int, task_id: int, episode_index: int) -> Path:
    return (
        output.parent
        / "traces"
        / f"{mode}_delay{delay_ms}_task{task_id}_episode{episode_index}.npz"
    )


def _write_trace(
    path: Path,
    *,
    actions: list[np.ndarray],
    dispatch_timestamps: list[float],
    inference_events: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        actions=np.asarray(actions, dtype=np.float32),
        dispatch_timestamps=np.asarray(dispatch_timestamps, dtype=np.float64),
        inference_events_json=np.asarray(json.dumps(inference_events)),
    )


def _run_sync_episode(
    backend: LeRobotBackend,
    *,
    run_id: str,
    git_commit: str,
    task_id: int,
    episode_index: int,
    initial_state_index: int,
    seed: int,
    injected_delay_ms: int,
    realtime: bool,
    replan_interval_steps: int,
    mode: str,
    output: Path,
) -> dict[str, Any]:
    if mode != "sync":
        raise ValueError(f"Synchronous runner received mode={mode!r}")
    observation, _, instruction = backend.reset_episode(
        task_id=task_id,
        seed=seed,
        initial_state_index=initial_state_index,
    )
    controller_hz = backend.controller_frequency_hz(task_id)
    period = 1.0 / controller_hz
    torch.cuda.reset_peak_memory_stats()

    episode_started = time.monotonic()
    control_epoch: float | None = None
    scheduled: float | None = None
    pending_actions: deque[np.ndarray] = deque()
    model_latencies: list[float] = []
    delivery_latencies: list[float] = []
    inference_events: list[dict[str, Any]] = []
    actions_executed: list[np.ndarray] = []
    dispatch_timestamps: list[float] = []
    discontinuities: list[float] = []
    previous_action: np.ndarray | None = None
    time_to_first_action: float | None = None
    deadline_misses = 0
    dispatch_lateness: list[float] = []
    success = False
    step_count = 0

    while step_count < backend.episode_length:
        if not pending_actions:
            request_timestamp = time.monotonic()
            output_chunk = backend.infer_action_chunk(observation, instruction)
            inference_end_timestamp = time.monotonic()
            if injected_delay_ms:
                time.sleep(injected_delay_ms / 1000.0)
            delivery_timestamp = time.monotonic()

            model_latencies.append(output_chunk.model_latency_seconds)
            delivery_latencies.append(delivery_timestamp - request_timestamp)
            inference_events.append(
                {
                    "observation_control_step": step_count,
                    "request_timestamp": request_timestamp,
                    "start_timestamp": request_timestamp,
                    "end_timestamp": inference_end_timestamp,
                    "delivery_timestamp": delivery_timestamp,
                    "model_inference_latency_seconds": output_chunk.model_latency_seconds,
                    "raw_shape": list(output_chunk.raw_shape),
                    "raw_dtype": output_chunk.raw_dtype,
                }
            )
            pending_actions.extend(output_chunk.actions)
            if control_epoch is None:
                control_epoch = delivery_timestamp
                scheduled = control_epoch

        if control_epoch is None or scheduled is None:
            raise RuntimeError("Synchronous runner reached control without a first chunk")
        if realtime:
            scheduled, missed_ticks = _coalesce_missed_control_ticks(
                scheduled,
                time.monotonic(),
                period,
            )
            deadline_misses += missed_ticks
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        action = np.asarray(pending_actions.popleft(), dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid queued environment action: {action}")

        dispatched = time.monotonic()
        if realtime:
            dispatch_lateness.append(max(0.0, dispatched - scheduled))
        if time_to_first_action is None:
            time_to_first_action = dispatched - episode_started
        step_output = backend.step(task_id, action)

        actions_executed.append(action.copy())
        dispatch_timestamps.append(dispatched)
        if previous_action is not None:
            discontinuities.append(float(np.linalg.norm(action - previous_action)))
        previous_action = action

        step_count += 1
        observation = step_output.observation
        success = success or step_output.success
        if success or step_output.terminated or step_output.truncated:
            break
        scheduled += period

    episode_finished = time.monotonic()
    trace_path = _trace_path(output, "sync", injected_delay_ms, task_id, episode_index)
    _write_trace(
        trace_path,
        actions=actions_executed,
        dispatch_timestamps=dispatch_timestamps,
        inference_events=inference_events,
    )

    return {
        "run_id": run_id,
        "git_commit": git_commit,
        "lerobot_version": "0.6.0",
        "model_id": backend.model_id,
        "model_revision_sha": backend.model_revision,
        "suite": backend.suite,
        "task_id": task_id,
        "episode_index": episode_index,
        "initial_state_index": initial_state_index,
        "seed": seed,
        "runtime_mode": "sync",
        "injected_delay_ms": injected_delay_ms,
        "success": success,
        "environment_steps": step_count,
        "wall_clock_episode_seconds": episode_finished - episode_started,
        "time_to_first_action_seconds": time_to_first_action,
        "inference_calls": len(model_latencies),
        "inference_latency_p50_seconds": _percentile(model_latencies, 50),
        "inference_latency_p95_seconds": _percentile(model_latencies, 95),
        "observation_to_delivery_p50_seconds": _percentile(delivery_latencies, 50),
        "observation_to_delivery_p95_seconds": _percentile(delivery_latencies, 95),
        "queue_underrun_hold_steps": 0,
        "stale_chunks_discarded": 0,
        "stale_prefix_mean_steps": 0.0,
        "stale_prefix_max_steps": 0,
        "control_deadline_misses": deadline_misses if realtime else None,
        "control_dispatch_lateness_p50_seconds": _percentile(dispatch_lateness, 50),
        "control_dispatch_lateness_p95_seconds": _percentile(dispatch_lateness, 95),
        "action_discontinuity_mean_l2": (
            float(np.mean(np.asarray(discontinuities, dtype=np.float64)))
            if discontinuities
            else None
        ),
        "action_discontinuity_max_l2": max(discontinuities, default=None),
        "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
        "controller_frequency_hz": controller_hz,
        "realtime_control": realtime,
        "replan_interval_steps": 30,
        "policy_rng_seed": seed,
        "policy_rng_reset_per_episode": True,
        "action_trace_path": str(trace_path),
    }


def _run_async_episode(
    backend: LeRobotBackend,
    *,
    run_id: str,
    git_commit: str,
    task_id: int,
    episode_index: int,
    initial_state_index: int,
    seed: int,
    injected_delay_ms: int,
    realtime: bool,
    replan_interval_steps: int,
    mode: str,
    output: Path,
) -> dict[str, Any]:
    if mode not in {"async_naive", "async_aligned"}:
        raise ValueError(f"Invalid asynchronous mode: {mode}")
    if not realtime:
        raise ValueError("Asynchronous modes require --realtime")
    if replan_interval_steps <= 0:
        raise ValueError("replan_interval_steps must be positive")

    observation, _, instruction = backend.reset_episode(
        task_id=task_id,
        seed=seed,
        initial_state_index=initial_state_index,
    )
    controller_hz = backend.controller_frequency_hz(task_id)
    period = 1.0 / controller_hz
    torch.cuda.reset_peak_memory_stats()

    episode_id = f"{run_id}-task{task_id}-episode{episode_index}"
    queue = ActionQueue()
    queue.reset_episode(episode_id)

    def infer(request: InferenceRequest) -> InferencePayload:
        worker_observation = thaw_observation_snapshot(request.observation)
        output_chunk = backend.infer_action_chunk(
            worker_observation,
            request.task_instruction,
        )
        return InferencePayload(
            actions=output_chunk.actions,
            model_inference_latency_seconds=output_chunk.model_latency_seconds,
            metadata={
                "raw_shape": list(output_chunk.raw_shape),
                "raw_dtype": output_chunk.raw_dtype,
            },
        )

    worker = LatestRequestWorker(
        infer,
        delivery_delay_seconds=injected_delay_ms / 1000.0,
    )
    worker.reset_episode(episode_id)

    episode_started = time.monotonic()
    model_latencies: list[float] = []
    delivery_latencies: list[float] = []
    inference_events: list[dict[str, Any]] = []
    actions_executed: list[np.ndarray] = []
    dispatch_timestamps: list[float] = []
    discontinuities: list[float] = []
    previous_action: np.ndarray | None = None
    time_to_first_action: float | None = None
    deadline_misses = 0
    dispatch_lateness: list[float] = []
    success = False
    step_count = 0
    episode_finished: float | None = None

    def submit_request(control_step: int, current_observation: dict[str, Any]) -> None:
        worker.submit(
            InferenceRequest(
                observation=immutable_observation_snapshot(current_observation),
                task_instruction=instruction,
                episode_id=episode_id,
                observation_control_step=control_step,
                request_timestamp=time.monotonic(),
            )
        )

    def record_result(
        result: InferenceResult,
        *,
        merge: bool,
        delivered: bool = True,
    ) -> None:
        model_latencies.append(result.model_inference_latency_seconds)
        if delivered:
            delivery_latencies.append(result.delivery_timestamp - result.request_timestamp)
        event: dict[str, Any] = {
            "observation_control_step": result.observation_control_step,
            "request_timestamp": result.request_timestamp,
            "start_timestamp": result.start_timestamp,
            "end_timestamp": result.end_timestamp,
            "delivery_timestamp": result.delivery_timestamp,
            "model_inference_latency_seconds": result.model_inference_latency_seconds,
            "raw_shape": list(result.metadata.get("raw_shape", [])),
            "raw_dtype": result.metadata.get("raw_dtype"),
            "merged": merge,
            "published_before_episode_end": delivered,
        }
        if merge:
            outcome = queue.replace(
                result,
                current_control_step=step_count,
                mode=mode,
            )
            event["merge"] = {
                "accepted": outcome.accepted,
                "reason": outcome.reason,
                "age_steps": outcome.age_steps,
                "dropped_prefix_steps": outcome.dropped_prefix_steps,
                "fully_stale": outcome.fully_stale,
                "queue_length": outcome.queue_length,
            }
        inference_events.append(event)

    submit_request(0, observation)
    try:
        if not worker.wait_for_result(timeout=120):
            raise TimeoutError("Timed out waiting for the first asynchronous action chunk")
        for result in worker.drain_results():
            record_result(result, merge=True)
        if not queue.has_safe_action:
            raise RuntimeError("Initial inference completed without a safe final action")

        control_epoch = time.monotonic()
        scheduled = control_epoch
        while step_count < backend.episode_length:
            scheduled, missed_ticks = _coalesce_missed_control_ticks(
                scheduled,
                time.monotonic(),
                period,
            )
            deadline_misses += missed_ticks
            remaining = scheduled - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

            for result in worker.drain_results():
                record_result(result, merge=True)

            action, _ = queue.next_action()
            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid queued environment action: {action}")

            dispatched = time.monotonic()
            dispatch_lateness.append(max(0.0, dispatched - scheduled))
            if time_to_first_action is None:
                time_to_first_action = dispatched - episode_started
            step_output = backend.step(task_id, action)

            actions_executed.append(action.copy())
            dispatch_timestamps.append(dispatched)
            if previous_action is not None:
                discontinuities.append(float(np.linalg.norm(action - previous_action)))
            previous_action = action

            step_count += 1
            observation = step_output.observation
            success = success or step_output.success
            if success or step_output.terminated or step_output.truncated:
                break
            if step_count % replan_interval_steps == 0:
                submit_request(step_count, observation)
            scheduled += period
        episode_finished = time.monotonic()
    finally:
        worker.close()

    # Count completed requests that arrived after the terminal environment step,
    # but never merge them into a completed episode's queue.
    for result in worker.drain_all_results():
        record_result(result, merge=False, delivered=False)

    if episode_finished is None:
        raise RuntimeError("Asynchronous episode ended without a completion timestamp")
    trace_path = _trace_path(output, mode, injected_delay_ms, task_id, episode_index)
    _write_trace(
        trace_path,
        actions=actions_executed,
        dispatch_timestamps=dispatch_timestamps,
        inference_events=inference_events,
    )
    stale_prefixes = queue.stale_prefix_lengths

    return {
        "run_id": run_id,
        "git_commit": git_commit,
        "lerobot_version": "0.6.0",
        "model_id": backend.model_id,
        "model_revision_sha": backend.model_revision,
        "suite": backend.suite,
        "task_id": task_id,
        "episode_index": episode_index,
        "initial_state_index": initial_state_index,
        "seed": seed,
        "runtime_mode": mode,
        "injected_delay_ms": injected_delay_ms,
        "success": success,
        "environment_steps": step_count,
        "wall_clock_episode_seconds": episode_finished - episode_started,
        "time_to_first_action_seconds": time_to_first_action,
        "inference_calls": worker.calls_started,
        "inference_latency_p50_seconds": _percentile(model_latencies, 50),
        "inference_latency_p95_seconds": _percentile(model_latencies, 95),
        "observation_to_delivery_p50_seconds": _percentile(delivery_latencies, 50),
        "observation_to_delivery_p95_seconds": _percentile(delivery_latencies, 95),
        "queue_underrun_hold_steps": queue.hold_steps,
        "stale_chunks_discarded": queue.stale_chunks_discarded,
        "stale_prefix_mean_steps": (
            float(np.mean(np.asarray(stale_prefixes, dtype=np.float64)))
            if stale_prefixes
            else 0.0
        ),
        "stale_prefix_max_steps": max(stale_prefixes, default=0),
        "control_deadline_misses": deadline_misses,
        "control_dispatch_lateness_p50_seconds": _percentile(dispatch_lateness, 50),
        "control_dispatch_lateness_p95_seconds": _percentile(dispatch_lateness, 95),
        "action_discontinuity_mean_l2": (
            float(np.mean(np.asarray(discontinuities, dtype=np.float64)))
            if discontinuities
            else None
        ),
        "action_discontinuity_max_l2": max(discontinuities, default=None),
        "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
        "controller_frequency_hz": controller_hz,
        "realtime_control": True,
        "replan_interval_steps": replan_interval_steps,
        "policy_rng_seed": seed,
        "policy_rng_reset_per_episode": True,
        "queue_replacements": queue.replacements,
        "pending_requests_replaced": worker.pending_requests_replaced,
        "old_episode_chunks_rejected": queue.old_episode_chunks_rejected,
        "old_episode_results_discarded": worker.old_episode_results_discarded,
        "action_trace_path": str(trace_path),
    }


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.output.exists() and not args.append:
        raise FileExistsError(f"Refusing to overwrite existing metrics: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    initial_state_indices = args.initial_state_indices
    if len(initial_state_indices) != args.episodes_per_task:
        raise ValueError(
            "--initial-state-indices must contain exactly "
            f"{args.episodes_per_task} entries, got {len(initial_state_indices)}"
        )
    if args.mode != "sync" and not args.realtime:
        raise ValueError("Asynchronous modes require --realtime")

    run_id = args.run_id or f"{args.mode}-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    git_commit = _git_commit()
    backend = LeRobotBackend(
        task_ids=args.task_ids,
        seed=args.seed,
        suite=args.suite,
        episode_length=args.episode_length,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )

    records: list[dict[str, Any]] = []
    try:
        with args.output.open("a", encoding="utf-8", buffering=1) as stream:
            for task_id in args.task_ids:
                for episode_index, initial_state_index in enumerate(initial_state_indices):
                    runner = _run_sync_episode if args.mode == "sync" else _run_async_episode
                    record = runner(
                        backend,
                        run_id=run_id,
                        git_commit=git_commit,
                        task_id=task_id,
                        episode_index=episode_index,
                        initial_state_index=initial_state_index,
                        seed=args.seed + episode_index,
                        injected_delay_ms=args.injected_delay_ms,
                        realtime=args.realtime,
                        replan_interval_steps=args.replan_interval_steps,
                        mode=args.mode,
                        output=args.output,
                    )
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    records.append(record)
                    print(
                        f"{args.mode} delay={args.injected_delay_ms}ms "
                        f"task={task_id} episode={episode_index} "
                        f"success={record['success']} steps={record['environment_steps']}"
                    )
    finally:
        backend.close()
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=RUNTIME_MODES, required=True)
    parser.add_argument("--task-ids", type=_parse_int_csv, default=[0, 1, 2])
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument(
        "--initial-state-indices",
        type=_parse_int_csv,
        default=list(range(0, 20, 2)),
        help="Comma-separated fixed LIBERO init-state indices, reused for every task.",
    )
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--episode-length", type=int, default=800)
    parser.add_argument("--injected-delay-ms", type=int, choices=[0, 100, 200, 300], default=0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--replan-interval-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
