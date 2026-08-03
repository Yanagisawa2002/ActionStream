"""Single-authority ROS recorder for native M8 dynamic episodes.

The legacy M7 recorder synthesizes lifecycle rows from ``EpisodeControl``.
M8 instead treats the dynamic Isaac bridge's enriched ``RuntimeEvent`` rows as
the sole lifecycle authority.  This module adapts all ROS topics into the pure
``M8EventRecorder`` while keeping the raw JSONL atomic and import-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from action_stream_benchmark.m8_recorder import M8EventRecorder
from action_stream_benchmark.m8_replay import FAIRNESS_FIELDS, HOLDOUT_FAIRNESS_FIELDS


_PROTECTED_DETAIL_FIELDS = {
    "schema_version",
    "milestone",
    "evidence_class",
    "event_index",
    "event_type",
    "episode_id",
    "seed",
    "profile_id",
    "strategy",
    "split",
}


def _sim_time_ns(message: Any) -> int:
    return int(message.sim_stamp.sec) * 1_000_000_000 + int(message.sim_stamp.nanosec)


def _json_detail(value: object) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip().startswith("{"):
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _queue_detail(value: object) -> dict[str, int]:
    """Decode the stable C++ ``expired=N,duplicates=N`` detail."""

    result = {
        "expired_actions_removed": 0,
        "duplicate_target_actions_removed": 0,
    }
    if not isinstance(value, str):
        return result
    for item in value.split(","):
        key, separator, raw = item.partition("=")
        if not separator or not raw.isdigit():
            continue
        if key == "expired":
            result["expired_actions_removed"] = int(raw)
        elif key == "duplicates":
            result["duplicate_target_actions_removed"] = int(raw)
    return result


@dataclass(frozen=True, slots=True)
class DynamicRecorderConfig:
    output_path: Path
    episode_id: str
    seed: int
    profile_id: str
    strategy: str
    split: str
    expected_scenario_sha256: str
    expected_fault_trace_sha256: str

    def validate(self) -> None:
        if not self.episode_id:
            raise ValueError("dynamic recorder episode_id must be non-empty")
        if self.seed < 0:
            raise ValueError("dynamic recorder seed must be non-negative")
        if self.split not in {"baseline_gate", "development", "frozen_holdout"}:
            raise ValueError(f"unknown dynamic recorder split: {self.split}")
        for name, value in (
            ("expected_scenario_sha256", self.expected_scenario_sha256),
            ("expected_fault_trace_sha256", self.expected_fault_trace_sha256),
        ):
            if len(value) != 64:
                raise ValueError(f"{name} must be a SHA-256 hex digest")


class DynamicM8RecorderCore:
    """Pure lifecycle adapter around :class:`M8EventRecorder`.

    Cross-topic delivery can put a reset observation ahead of the start event,
    so ordinary rows are buffered until the authoritative enriched start is
    received.  The end row is staged until ``close`` so terminal observations
    and executor drain events cannot be lost behind an early atomic commit.
    """

    def __init__(self, config: DynamicRecorderConfig) -> None:
        config.validate()
        self.config = config
        self._recorder: M8EventRecorder | None = None
        self._buffered: list[tuple[str, dict[str, Any]]] = []
        self._pending_end: dict[str, Any] | None = None
        self._closed = False
        self._fairness: dict[str, Any] | None = None

    @property
    def fairness(self) -> Mapping[str, Any] | None:
        return self._fairness

    @staticmethod
    def _clean(payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in payload.items()
            if str(key) not in _PROTECTED_DETAIL_FIELDS
        }

    def _start(self, payload: Mapping[str, Any]) -> None:
        if self._recorder is not None:
            raise RuntimeError("duplicate authoritative M8 episode_start")
        required = (
            HOLDOUT_FAIRNESS_FIELDS
            if self.config.split == "frozen_holdout"
            else FAIRNESS_FIELDS
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"M8 episode_start fairness fields missing: {missing}")
        fairness = {name: payload[name] for name in required}
        if fairness["scenario_sha256"] != self.config.expected_scenario_sha256:
            raise ValueError("episode_start scenario hash differs from matrix")
        if fairness["fault_trace_sha256"] != self.config.expected_fault_trace_sha256:
            raise ValueError("episode_start fault trace hash differs from matrix")
        recorder = M8EventRecorder(
            self.config.output_path,
            episode_id=self.config.episode_id,
            seed=self.config.seed,
            profile_id=self.config.profile_id,
            strategy=self.config.strategy,
            fairness=fairness,
            split=self.config.split,
            overwrite=False,
        )
        recorder.__enter__()
        self._recorder = recorder
        self._fairness = fairness
        start_payload = self._clean(payload)
        for name in required:
            start_payload.pop(name, None)
        recorder.start(**start_payload)
        for event_type, buffered in self._buffered:
            recorder.append(event_type, **buffered)
        self._buffered.clear()

    def record(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot record after dynamic recorder close")
        if not event_type:
            raise ValueError("M8 event_type must be non-empty")
        data = self._clean(payload)
        if event_type == "episode_start":
            self._start(payload)
            return
        recorder = self._recorder
        if recorder is None:
            if event_type in {"destination_switched", "task_terminated", "episode_end"}:
                raise RuntimeError(f"{event_type} arrived before authoritative episode_start")
            self._buffered.append((event_type, data))
            return
        if event_type == "destination_switched":
            step = int(data.pop("step", data.pop("switch_step", 0)))
            old = data.pop("old_destination_xyz")
            new = data.pop("new_destination_xyz")
            generation_before = int(data.pop("generation_before", 1))
            generation_after = int(data.pop("generation_after", 2))
            recorder.destination_switched(
                step=step,
                old_destination_xyz=old,
                new_destination_xyz=new,
                generation_before=generation_before,
                generation_after=generation_after,
                **data,
            )
            return
        if event_type == "task_terminated":
            reason = str(data.pop("reason", ""))
            success = bool(data.pop("success", False))
            recorder.terminate(reason=reason, success=success, **data)
            return
        if event_type == "episode_end":
            if self._pending_end is not None:
                raise RuntimeError("duplicate authoritative M8 episode_end")
            self._pending_end = data
            return
        recorder.append(event_type, **data)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        recorder = self._recorder
        if recorder is None:
            if self._buffered:
                raise RuntimeError("M8 recorder closed without authoritative episode_start")
            return
        try:
            if self._pending_end is None:
                raise RuntimeError("M8 recorder closed without authoritative episode_end")
            end = dict(self._pending_end)
            success = bool(end.pop("success", False))
            completion_reason = str(
                end.pop("completion_reason", end.pop("reason", ""))
            )
            recorder.finish(
                success=success,
                completion_reason=completion_reason,
                **end,
            )
            recorder.__exit__(None, None, None)
        except Exception as exc:
            recorder.__exit__(type(exc), exc, exc.__traceback__)
            raise


def create_dynamic_event_recorder_node(config: DynamicRecorderConfig) -> Any:
    """Create the ROS wrapper after Isaac has initialized its ROS bridge."""

    config.validate()
    try:
        from action_stream_msgs.msg import (
            ActionChunk,
            ExecutorDiagnostics,
            InferenceRequest,
            Observation,
            RobotCommand,
            RuntimeEvent,
        )
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
    except ImportError as exc:  # pragma: no cover - native Isaac/ROS only
        raise RuntimeError(
            "dynamic event recorder requires post-bridge ROS 2 message imports"
        ) from exc

    class DynamicEventRecorderNode(Node):
        def __init__(self) -> None:
            super().__init__("action_stream_m8_event_recorder")
            self._core = DynamicM8RecorderCore(config)
            self._closed = False
            self._request_starts: dict[int, int] = {}
            self._seen_responses: set[int] = set()
            self._last_response_request_id = 0
            reliable = QoSProfile(depth=512, reliability=ReliabilityPolicy.RELIABLE)
            self.create_subscription(
                Observation, "/action_stream/observation", self._on_observation, reliable
            )
            self.create_subscription(
                InferenceRequest,
                "/action_stream/inference_request",
                self._on_request,
                reliable,
            )
            self.create_subscription(
                ActionChunk, "/action_stream/action_chunk", self._on_chunk, reliable
            )
            self.create_subscription(
                RobotCommand, "/action_stream/robot_command", self._on_command, reliable
            )
            self.create_subscription(
                ExecutorDiagnostics,
                "/action_stream/diagnostics",
                self._on_diagnostics,
                reliable,
            )
            self.create_subscription(
                RuntimeEvent, "/action_stream/events", self._on_runtime_event, reliable
            )

        @property
        def fairness(self) -> Mapping[str, Any] | None:
            return self._core.fairness

        @staticmethod
        def _times(message: Any) -> dict[str, int]:
            return {
                "sim_time_ns": _sim_time_ns(message),
                "steady_time_ns": int(message.steady_time_ns),
                "wall_time_ns": int(message.wall_time_ns),
            }

        def _on_observation(self, message: Any) -> None:
            if str(message.episode_id) != config.episode_id:
                return
            self._core.record(
                "observation",
                {
                    **self._times(message),
                    "sim_step": int(message.observation_step),
                    "observation_step": int(message.observation_step),
                    "generation_id": int(message.generation_id),
                    "task_id": str(message.task_id),
                    "robot_state": list(message.robot_state),
                    "task_state": list(message.task_state),
                    "terminated": bool(message.terminated),
                    "success": (
                        len(message.task_state) > 39
                        and float(message.task_state[39]) == 1.0
                    ),
                },
            )

        def _on_request(self, message: Any) -> None:
            if str(message.episode_id) != config.episode_id:
                return
            request_id = int(message.request_id)
            start = int(message.request_steady_time_ns)
            self._request_starts[request_id] = start
            self._core.record(
                "inference_request",
                {
                    "sim_time_ns": _sim_time_ns(message),
                    "steady_time_ns": start,
                    "wall_time_ns": int(message.request_wall_time_ns),
                    "sim_step": int(message.source_observation_step),
                    "request_id": request_id,
                    "request_ordinal": request_id - 1,
                    "generation_id": int(message.generation_id),
                    "source_observation_step": int(message.source_observation_step),
                    "expected_horizon": int(message.expected_horizon),
                    "fault_trace_sha256": config.expected_fault_trace_sha256,
                },
            )

        def _record_sync_wait(self, request_id: int, end_ns: int, reason: str) -> None:
            if config.strategy != "sync_hold":
                return
            start = self._request_starts.pop(request_id, None)
            if start is None:
                return
            self._core.record(
                "sync_wait",
                {
                    "sim_time_ns": 0,
                    "steady_time_ns": end_ns,
                    "wall_time_ns": time.time_ns(),
                    "request_id": request_id,
                    "duration_ns": max(0, end_ns - start),
                    "counts_as_hold": True,
                    "reason": reason,
                },
            )

        def _on_chunk(self, message: Any) -> None:
            if str(message.episode_id) != config.episode_id:
                return
            request_id = int(message.request_id)
            duplicate = request_id in self._seen_responses
            out_of_order = request_id < self._last_response_request_id
            self._seen_responses.add(request_id)
            self._last_response_request_id = max(self._last_response_request_id, request_id)
            start = self._request_starts.get(request_id, int(message.steady_time_ns))
            self._core.record(
                "chunk_arrived",
                {
                    **self._times(message),
                    "sim_step": int(message.source_observation_step),
                    "request_id": request_id,
                    "request_ordinal": request_id - 1,
                    "generation_id": int(message.generation_id),
                    "source_observation_step": int(message.source_observation_step),
                    "latency_ms": max(
                        0.0, (int(message.steady_time_ns) - start) / 1e6
                    ),
                    "duplicate": duplicate,
                    "out_of_order": out_of_order,
                    "actions": [
                        {
                            "target_step": int(action.target_step),
                            "command": list(action.command),
                        }
                        for action in message.actions
                    ],
                },
            )
            if not duplicate:
                self._record_sync_wait(
                    request_id, int(message.steady_time_ns), "response_arrived"
                )

        def _on_command(self, message: Any) -> None:
            if str(message.episode_id) != config.episode_id:
                return
            self._core.record(
                "command_executed",
                {
                    **self._times(message),
                    "sim_step": int(message.actual_target_step),
                    "actual_target_step": int(message.actual_target_step),
                    "source_request_id": int(message.source_request_id),
                    "source_generation_id": int(message.source_generation_id),
                    "source_observation_step": int(message.source_observation_step),
                    "source_target_step": int(message.source_target_step),
                    "command": list(message.command),
                    "hold": bool(message.hold),
                    "reason": str(message.reason),
                },
            )

        def _on_diagnostics(self, message: Any) -> None:
            if str(message.episode_id) != config.episode_id:
                return
            names = (
                "latest_observation_step",
                "latest_executed_target_step",
                "active_generation_id",
                "queue_length",
                "accepted_chunks",
                "rejected_chunks",
                "rejected_previous_episode_chunks",
                "rejected_stale_generation_chunks",
                "rejected_unknown_request_chunks",
                "duplicate_responses",
                "expired_actions_removed",
                "duplicate_actions_removed",
                "generation_invalidated_actions",
                "queue_rebuilds",
                "sync_periodic_replans",
                "sync_periodic_replan_actions_removed",
                "deadline_misses",
                "hold_steps",
                "total_hold_duration_ns",
                "current_hold_duration_ns",
                "current_action_age_steps",
                "executed_source_generation_id",
            )
            payload = {name: int(getattr(message, name)) for name in names}
            payload.update(
                {
                    **self._times(message),
                    "sim_step": int(message.latest_observation_step),
                    "episode_active": bool(message.episode_active),
                    "episode_terminated": bool(message.episode_terminated),
                    "episode_success": bool(message.episode_success),
                    "has_executed_target_step": bool(
                        message.has_executed_target_step
                    ),
                }
            )
            self._core.record("diagnostics", payload)

        def _on_runtime_event(self, message: Any) -> None:
            if str(message.episode_id) != config.episode_id:
                return
            event_type = str(message.event_type)
            detail = _json_detail(message.detail)
            payload = {
                **detail,
                **self._times(message),
                "sim_step": int(
                    message.actual_target_step or message.source_observation_step
                ),
                "reason": str(message.reason),
                "request_id": int(message.request_id),
                "generation_id": int(message.generation_id),
                "source_observation_step": int(message.source_observation_step),
                "source_target_step": int(message.source_target_step),
                "actual_target_step": int(message.actual_target_step),
                "active_generation_id": int(message.active_generation_id),
                "queue_length_before": int(message.queue_length_before),
                "queue_length_after": int(message.queue_length_after),
                "action_count": int(message.action_count),
                "detail": str(message.detail),
            }
            if event_type == "queue_updated":
                payload.update(_queue_detail(message.detail))
            if event_type == "response_dropped":
                # A sync drop does not cancel the C++ in-flight request.  Keep
                # the wait open until terminal drain instead of fabricating a
                # retry/cancellation event.
                pass
            if event_type == "episode_end":
                end_ns = int(message.steady_time_ns)
                for request_id in tuple(self._request_starts):
                    self._record_sync_wait(
                        request_id, end_ns, "episode_ended_while_response_pending"
                    )
            self._core.record(event_type, payload)

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            self._core.close()

    return DynamicEventRecorderNode()
