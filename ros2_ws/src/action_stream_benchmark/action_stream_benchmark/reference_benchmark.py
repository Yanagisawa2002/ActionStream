"""Deterministic reference benchmark for the ROS runtime semantics.

This module is deliberately ROS-free.  It gives CI and replay tooling a small,
fully deterministic plant against which the same request, chunk, queue, and
fault contracts used by the ROS nodes can be exercised.  It is not presented
as an Isaac Sim result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from pathlib import Path
from statistics import median
from typing import Any, Final, Iterable

from action_stream_policy.scripted_policy import (
    ACTION_DIMENSION,
    CHUNK_HORIZON,
    ActionChunkData,
    PolicyObservation,
    PolicyRequestData,
    ReachLiftScriptedPolicy,
)

from .faults import FaultTrace
from .plant import CONTROL_FREQUENCY_HZ, ReachLiftPlant
from .schema import MILESTONE, SCHEMA_VERSION, write_json_atomic, write_jsonl_atomic


CONTROL_PERIOD_NS: Final[int] = int(1_000_000_000 / CONTROL_FREQUENCY_HZ)
DEFAULT_REQUEST_INTERVAL_STEPS: Final[int] = 10


class Strategy(str, Enum):
    SYNC_HOLD = "sync_hold"
    NAIVE_ASYNC = "naive_async"
    ALIGNED_ASYNC = "aligned_async"


@dataclass(frozen=True, slots=True)
class QueuedAction:
    request_id: int
    generation_id: int
    source_observation_step: int
    source_target_step: int
    command: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PendingResponse:
    due_wall_time_ns: int
    request_ordinal: int
    request: PolicyRequestData
    chunk: ActionChunkData
    latency_ms: int
    duplicate: bool = False


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


class ReferenceEpisode:
    """Run one strategy against one immutable seed/profile fault trace."""

    def __init__(
        self,
        *,
        strategy: Strategy | str,
        trace: FaultTrace,
        max_steps: int = 180,
        request_interval_steps: int = DEFAULT_REQUEST_INTERVAL_STEPS,
    ) -> None:
        self.strategy = Strategy(strategy)
        if request_interval_steps <= 0 or request_interval_steps >= CHUNK_HORIZON:
            raise ValueError("request interval must lie in [1, H-1]")
        self.trace = trace
        self.request_interval_steps = request_interval_steps
        self.episode_id = f"m7-{trace.profile.profile_id}-{trace.seed}-{self.strategy.value}"
        self.plant = ReachLiftPlant(max_steps=max_steps)
        self.policy = ReachLiftScriptedPolicy()
        self.queue: list[QueuedAction] = []
        self.pending: list[PendingResponse] = []
        self.events: list[dict[str, Any]] = []
        self.event_index = 0
        self.wall_time_ns = 0
        self.generation_id = 0
        self.request_ordinal = 0
        self.next_request_id = 1
        self.completed_request_ids: set[int] = set()
        self.rejected_request_ids: set[int] = set()
        self.latest_accepted_source_step = -1
        self.last_arrival_ordinal = -1
        self.last_command: tuple[float, ...] = ()
        self.action_ages: list[float] = []
        self.inference_latencies_ms: list[float] = []
        self.counters: dict[str, int] = {
            "inference_requests": 0,
            "returned_responses": 0,
            "dropped_responses": 0,
            "out_of_order_responses": 0,
            "duplicate_responses": 0,
            "accepted_chunks": 0,
            "rejected_chunks": 0,
            "stale_generations_rejected": 0,
            "superseded_sources_rejected": 0,
            "expired_actions_removed": 0,
            "duplicate_target_actions_removed": 0,
            "queue_rebuild_count": 0,
            "deadline_misses": 0,
            "hold_steps": 0,
            "executed_actions": 0,
        }
        self.sync_wait_ns = 0

    def _record(self, event_type: str, **payload: Any) -> None:
        row = {
            "schema_version": SCHEMA_VERSION,
            "milestone": MILESTONE,
            "event_index": self.event_index,
            "event_type": event_type,
            "profile_id": self.trace.profile.profile_id,
            "seed": self.trace.seed,
            "strategy": self.strategy.value,
            "episode_id": self.episode_id,
            "sim_step": self.plant.step_count,
            "sim_time_ns": self.plant.step_count * CONTROL_PERIOD_NS,
            "wall_time_ns": self.wall_time_ns,
            **payload,
        }
        self.events.append(row)
        self.event_index += 1

    def _policy_observation(self) -> PolicyObservation:
        observation = self.plant.observation()
        return PolicyObservation(
            episode_id=observation.episode_id,
            observation_step=observation.observation_step,
            task_id=observation.task_id,
            robot_state=observation.robot_state,
            task_state=observation.task_state,
            terminated=observation.terminated,
        )

    def _submit_request(self) -> bool:
        ordinal = self.request_ordinal
        fault = self.trace.entry(ordinal)
        observation = self._policy_observation()
        request = PolicyRequestData(
            episode_id=self.episode_id,
            request_id=self.next_request_id,
            generation_id=self.generation_id,
            source_observation_step=observation.observation_step,
            expected_horizon=CHUNK_HORIZON,
            observation=observation,
        )
        chunk = self.policy.predict(request)
        self.next_request_id += 1
        self.request_ordinal += 1
        self.counters["inference_requests"] += 1
        self._record(
            "inference_request",
            request_id=request.request_id,
            request_ordinal=ordinal,
            generation_id=request.generation_id,
            source_observation_step=request.source_observation_step,
            expected_horizon=request.expected_horizon,
            trace_sha256=self.trace.sha256,
            fault=asdict(fault),
        )
        if fault.dropped:
            self.counters["dropped_responses"] += 1
            self._record(
                "response_dropped",
                request_id=request.request_id,
                request_ordinal=ordinal,
                generation_id=request.generation_id,
                source_observation_step=request.source_observation_step,
                latency_ms=fault.total_delivery_delay_ms,
            )
            return False

        due_ns = self.wall_time_ns + fault.total_delivery_delay_ms * 1_000_000
        self.pending.append(
            PendingResponse(
                due_wall_time_ns=due_ns,
                request_ordinal=ordinal,
                request=request,
                chunk=chunk,
                latency_ms=fault.total_delivery_delay_ms,
            )
        )
        if fault.duplicate_count:
            self.pending.append(
                PendingResponse(
                    due_wall_time_ns=due_ns + CONTROL_PERIOD_NS,
                    request_ordinal=ordinal,
                    request=request,
                    chunk=chunk,
                    latency_ms=fault.total_delivery_delay_ms + int(1000 / CONTROL_FREQUENCY_HZ),
                    duplicate=True,
                )
            )
        return True

    def _reject_chunk(self, response: PendingResponse, reason: str) -> None:
        self.counters["rejected_chunks"] += 1
        self.rejected_request_ids.add(response.request.request_id)
        if reason == "stale_generation":
            self.counters["stale_generations_rejected"] += 1
        if reason == "superseded_source_observation":
            self.counters["superseded_sources_rejected"] += 1
        self._record(
            "chunk_rejected",
            request_id=response.request.request_id,
            request_ordinal=response.request_ordinal,
            generation_id=response.request.generation_id,
            source_observation_step=response.request.source_observation_step,
            reason=reason,
            queue_length=len(self.queue),
        )

    def _ingest_response(self, response: PendingResponse) -> None:
        request = response.request
        out_of_order = response.request_ordinal < self.last_arrival_ordinal
        self.last_arrival_ordinal = max(self.last_arrival_ordinal, response.request_ordinal)
        self.counters["returned_responses"] += 1
        if out_of_order:
            self.counters["out_of_order_responses"] += 1
        if not response.duplicate:
            self.inference_latencies_ms.append(float(response.latency_ms))
        self._record(
            "chunk_arrived",
            request_id=request.request_id,
            request_ordinal=response.request_ordinal,
            generation_id=request.generation_id,
            source_observation_step=request.source_observation_step,
            latency_ms=response.latency_ms,
            duplicate=response.duplicate,
            out_of_order=out_of_order,
            actions=[
                {"target_step": action.target_step, "command": list(action.command)}
                for action in response.chunk.actions
            ],
        )
        if request.request_id in self.completed_request_ids:
            self.counters["duplicate_responses"] += 1
            self._reject_chunk(response, "duplicate_response")
            return
        self.completed_request_ids.add(request.request_id)
        if request.episode_id != self.episode_id:
            self._reject_chunk(response, "previous_episode")
            return
        if self.strategy is not Strategy.NAIVE_ASYNC:
            if request.generation_id != self.generation_id:
                self._reject_chunk(response, "stale_generation")
                return
            if request.source_observation_step < self.latest_accepted_source_step:
                self._reject_chunk(response, "superseded_source_observation")
                return

        incoming = [
            QueuedAction(
                request_id=request.request_id,
                generation_id=request.generation_id,
                source_observation_step=request.source_observation_step,
                source_target_step=action.target_step,
                command=action.command,
            )
            for action in response.chunk.actions
        ]
        queue_before = len(self.queue)
        if self.strategy is Strategy.NAIVE_ASYNC:
            self.queue.extend(incoming)
            reason = "arrival_order_append"
            expired = 0
            duplicates = 0
        else:
            next_target = self.plant.step_count + 1
            expired = sum(action.source_target_step < next_target for action in incoming)
            valid = [action for action in incoming if action.source_target_step >= next_target]
            by_target: dict[int, QueuedAction] = {}
            duplicates = 0
            for action in valid:
                if action.source_target_step in by_target:
                    duplicates += 1
                    continue
                by_target[action.source_target_step] = action
            valid = [by_target[target] for target in sorted(by_target)]
            self.counters["expired_actions_removed"] += expired
            self.counters["duplicate_target_actions_removed"] += duplicates
            if not valid:
                self._reject_chunk(response, "fully_expired")
                return
            self.queue = valid
            self.latest_accepted_source_step = request.source_observation_step
            self.counters["queue_rebuild_count"] += 1
            reason = (
                "sync_valid_rebuild"
                if self.strategy is Strategy.SYNC_HOLD
                else "aligned_atomic_rebuild"
            )
        self.counters["accepted_chunks"] += 1
        self._record(
            "queue_updated",
            request_id=request.request_id,
            generation_id=request.generation_id,
            source_observation_step=request.source_observation_step,
            reason=reason,
            queue_length_before=queue_before,
            queue_length_after=len(self.queue),
            expired_actions_removed=expired,
            duplicate_target_actions_removed=duplicates,
        )

    def _deliver_due(self) -> None:
        due = [item for item in self.pending if item.due_wall_time_ns <= self.wall_time_ns]
        self.pending = [item for item in self.pending if item.due_wall_time_ns > self.wall_time_ns]
        for response in sorted(
            due,
            key=lambda item: (
                item.due_wall_time_ns,
                item.request_ordinal,
                item.duplicate,
            ),
        ):
            self._ingest_response(response)

    def _wait_for_sync_response(self, *, count_hold: bool) -> None:
        while not self.queue:
            self._deliver_due()
            if self.queue:
                break
            scheduled = self._submit_request()
            current_request_id = self.next_request_id - 1
            if not scheduled:
                delay_ms = self.trace.entry(self.request_ordinal - 1).total_delivery_delay_ms
                retry_wait_ns = max(delay_ms * 1_000_000, CONTROL_PERIOD_NS)
                self.wall_time_ns += retry_wait_ns
                if count_hold:
                    self.sync_wait_ns += retry_wait_ns
                self._record(
                    "sync_wait",
                    duration_ns=retry_wait_ns,
                    counts_as_hold=count_hold,
                    reason="dropped_retry",
                )
                continue
            due_ns = min(
                item.due_wall_time_ns
                for item in self.pending
                if item.request.request_id == current_request_id and not item.duplicate
            )
            wait_ns = max(0, due_ns - self.wall_time_ns)
            self.wall_time_ns = due_ns
            if count_hold:
                self.sync_wait_ns += wait_ns
            self._record(
                "sync_wait",
                duration_ns=wait_ns,
                counts_as_hold=count_hold,
                reason="inference_pending",
            )
            self._deliver_due()

    def _select_command(
        self,
    ) -> tuple[tuple[float, ...], QueuedAction | None, bool, str]:
        actual_target = self.plant.step_count + 1
        selected: QueuedAction | None = None
        if self.strategy is Strategy.NAIVE_ASYNC:
            if self.queue:
                selected = self.queue.pop(0)
        else:
            while self.queue and self.queue[0].source_target_step < actual_target:
                expired = self.queue.pop(0)
                self.counters["expired_actions_removed"] += 1
                self._record(
                    "action_discarded",
                    reason="expired_before_execution",
                    request_id=expired.request_id,
                    generation_id=expired.generation_id,
                    source_observation_step=expired.source_observation_step,
                    source_target_step=expired.source_target_step,
                    actual_target_step=actual_target,
                )
            if self.queue and self.queue[0].source_target_step == actual_target:
                selected = self.queue.pop(0)
        if selected is not None:
            self.last_command = selected.command
            return (
                selected.command,
                selected,
                False,
                (
                    "arrival_order_execution"
                    if self.strategy is Strategy.NAIVE_ASYNC
                    else "target_step_match"
                ),
            )
        observation = self.plant.observation()
        hold = self.last_command or observation.robot_state
        return tuple(hold), None, True, "queue_empty"

    def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        initial = self.plant.reset(self.episode_id, seed=self.trace.seed)
        self.last_command = initial.robot_state
        self._record(
            "episode_start",
            generation_id=self.generation_id,
            trace_sha256=self.trace.sha256,
            action_dimension=ACTION_DIMENSION,
            horizon=CHUNK_HORIZON,
            control_frequency_hz=CONTROL_FREQUENCY_HZ,
        )
        self._record(
            "observation",
            observation_step=initial.observation_step,
            robot_state=list(initial.robot_state),
            task_state=list(initial.task_state),
            terminated=False,
            success=False,
        )

        # A common startup barrier obtains one complete plan before logical time
        # advances.  This is the same paired bootstrap for all three strategies.
        self._wait_for_sync_response(count_hold=self.strategy is Strategy.SYNC_HOLD)

        while not self.plant.terminated:
            step = self.plant.step_count
            if self.strategy is Strategy.SYNC_HOLD and not self.queue:
                self._wait_for_sync_response(count_hold=True)
            elif (
                self.strategy is not Strategy.SYNC_HOLD
                and step > 0
                and step % self.request_interval_steps == 0
            ):
                self._submit_request()

            self._deliver_due()
            command, source, hold, reason = self._select_command()
            actual_target = self.plant.step_count + 1
            if hold:
                self.counters["hold_steps"] += 1
                self.counters["deadline_misses"] += 1
            else:
                self.counters["executed_actions"] += 1
                assert source is not None
                self.action_ages.append(float(actual_target - source.source_observation_step))
            result = self.plant.step(command)
            self.wall_time_ns += CONTROL_PERIOD_NS
            self._record(
                "command_executed",
                actual_target_step=actual_target,
                source_request_id=0 if source is None else source.request_id,
                source_generation_id=0 if source is None else source.generation_id,
                source_observation_step=(0 if source is None else source.source_observation_step),
                source_target_step=0 if source is None else source.source_target_step,
                command=list(command),
                hold=hold,
                reason=reason,
            )
            observation = result.observation
            self._record(
                "observation",
                observation_step=observation.observation_step,
                robot_state=list(observation.robot_state),
                task_state=list(observation.task_state),
                terminated=observation.terminated,
                success=observation.success,
            )

        hold_ns = self.sync_wait_ns + self.counters["hold_steps"] * CONTROL_PERIOD_NS
        metrics: dict[str, Any] = {
            "task_success": self.plant.success,
            "completion_reason": self.plant.termination_reason,
            "simulation_steps": self.plant.step_count,
            "wall_clock_seconds": self.wall_time_ns / 1_000_000_000,
            "simulation_seconds": self.plant.step_count / CONTROL_FREQUENCY_HZ,
            "total_hold_seconds": hold_ns / 1_000_000_000,
            **self.counters,
            "mean_action_age_steps": (
                sum(self.action_ages) / len(self.action_ages) if self.action_ages else 0.0
            ),
            "p50_action_age_steps": _percentile(self.action_ages, 0.50),
            "p95_action_age_steps": _percentile(self.action_ages, 0.95),
            "mean_inference_latency_ms": (
                sum(self.inference_latencies_ms) / len(self.inference_latencies_ms)
                if self.inference_latencies_ms
                else 0.0
            ),
            "p50_inference_latency_ms": _percentile(self.inference_latencies_ms, 0.50),
            "p95_inference_latency_ms": _percentile(self.inference_latencies_ms, 0.95),
            "deadline_miss_rate": (
                self.counters["deadline_misses"] / self.plant.step_count
                if self.plant.step_count
                else 0.0
            ),
            "median_action_age_steps": (median(self.action_ages) if self.action_ages else 0.0),
        }
        self._record(
            "episode_end",
            success=self.plant.success,
            completion_reason=self.plant.termination_reason,
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "milestone": MILESTONE,
            "evidence_class": "deterministic_ros_test_plant_reference",
            "not_isaac_sim_result": True,
            "profile_id": self.trace.profile.profile_id,
            "profile": asdict(self.trace.profile),
            "seed": self.trace.seed,
            "strategy": self.strategy.value,
            "episode_id": self.episode_id,
            "trace_sha256": self.trace.sha256,
            "request_interval_steps": self.request_interval_steps,
            "metrics": metrics,
        }
        return self.events, summary


def run_reference_episode(
    *,
    strategy: Strategy | str,
    trace: FaultTrace,
    event_log_path: Path | str | None = None,
    summary_path: Path | str | None = None,
    max_steps: int = 180,
    request_interval_steps: int = DEFAULT_REQUEST_INTERVAL_STEPS,
) -> dict[str, Any]:
    runner = ReferenceEpisode(
        strategy=strategy,
        trace=trace,
        max_steps=max_steps,
        request_interval_steps=request_interval_steps,
    )
    events, summary = runner.run()
    if event_log_path is not None:
        write_jsonl_atomic(event_log_path, events)
    if summary_path is not None:
        write_json_atomic(summary_path, summary)
    return summary
