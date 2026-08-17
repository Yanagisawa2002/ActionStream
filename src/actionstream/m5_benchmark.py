"""Run the isolated M5-G0 aligned and oracle-pose-gated conditions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from actionstream.lerobot_backend import (
    LeRobotBackend,
    immutable_observation_snapshot,
    thaw_observation_snapshot,
)
from actionstream.m5_protocol import (
    canonical_sha256,
    file_sha256,
    implementation_source_hash,
    validate_manifest,
)
from actionstream.m5_runtime import (
    ActionProvenance,
    M5AlignedActionQueue,
    M5InferenceRequest,
    M5LatestRequestWorker,
    summarize_action_records,
)
from actionstream.m5_scene import (
    DeterministicPlanarShift,
    EntityPose,
    LiberoSceneAdapter,
    OracleGroundTruthPoseDetector,
    PlanarBounds,
)
from actionstream.runtime import InferencePayload, InferenceResult


SHIFT_CONDITIONS = (
    "aligned_shift",
    "aligned_oracle_pose_gate_shift",
)
NO_SHIFT_CONDITIONS = (
    "aligned_no_shift",
    "aligned_oracle_pose_gate_no_shift",
)
ALL_CONDITIONS = frozenset((*SHIFT_CONDITIONS, *NO_SHIFT_CONDITIONS))
GATED_CONDITIONS = frozenset(
    ("aligned_oracle_pose_gate_shift", "aligned_oracle_pose_gate_no_shift")
)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _lerobot_version() -> str:
    return importlib.metadata.version("lerobot")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, *, role: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must be a JSON object: {path}")
    return payload


def _validate_payload_hash(
    payload: Mapping[str, Any],
    *,
    hash_field: str,
    role: str,
) -> str:
    expected = payload.get(hash_field)
    core = {key: value for key, value in payload.items() if key != hash_field}
    actual = canonical_sha256(core)
    if expected != actual:
        raise ValueError(
            f"{role} hash mismatch: expected {expected}, computed {actual}"
        )
    return str(expected)


def _audit_candidate_order(
    config: Mapping[str, Any],
    task_audit: Mapping[str, Any],
) -> list[int]:
    configured = sorted(
        config["candidate_tasks"],
        key=lambda candidate: int(candidate["audit_order"]),
    )
    configured_ids = [int(candidate["task_id"]) for candidate in configured]
    audited = task_audit.get("candidates_inspected")
    if not isinstance(audited, list):
        raise ValueError("Task audit has no candidate list")
    audited_ids = [int(candidate["task_id"]) for candidate in audited]
    if len(configured_ids) != 2 or audited_ids != configured_ids:
        raise ValueError("M5 task audit must match exactly two configured candidates")
    return configured_ids


def _source_hash(root: Path) -> tuple[str, list[dict[str, str]]]:
    return implementation_source_hash(root)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(child) for child in value]
    return value


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        handle.flush()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
        handle.flush()
    temporary.replace(path)


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(payload)
    return rows


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _validated_bundle(path: Path, *, contract_sha256: str) -> dict[str, Any]:
    payload = _read_json_object(path, role="episode bundle")
    expected = payload.get("bundle_sha256")
    core = {key: value for key, value in payload.items() if key != "bundle_sha256"}
    actual = canonical_sha256(core)
    if expected != actual:
        raise ValueError(f"Episode bundle hash mismatch: {path}")
    if payload.get("run_contract_sha256") != contract_sha256:
        raise ValueError(f"Episode bundle belongs to a different run contract: {path}")
    return payload


def _event(
    events: list[dict[str, Any]],
    *,
    episode_id: str,
    condition: str,
    seed: int,
    step: int,
    timestamp: float,
    event_type: str,
    **fields: Any,
) -> None:
    events.append(
        {
            "episode_id": episode_id,
            "condition": condition,
            "seed": seed,
            "step": step,
            "timestamp": timestamp,
            "event_type": event_type,
            **fields,
        }
    )


def _coalesce_missed_control_ticks(
    scheduled: float,
    now: float,
    period: float,
) -> tuple[float, int]:
    if now <= scheduled:
        return scheduled, 0
    missed = max(0, math.floor((now - scheduled) / period))
    return scheduled + missed * period, missed


def _make_worker(backend: LeRobotBackend, delay_ms: int) -> M5LatestRequestWorker:
    def infer(request: M5InferenceRequest) -> InferencePayload:
        worker_observation = thaw_observation_snapshot(request.observation)
        output = backend.infer_action_chunk(
            worker_observation,
            request.task_instruction,
        )
        return InferencePayload(
            actions=output.actions,
            model_inference_latency_seconds=output.model_latency_seconds,
            metadata={
                "raw_shape": list(output.raw_shape),
                "raw_dtype": output.raw_dtype,
                "request_id": request.request_id,
                "request_generation_id": request.request_generation_id,
            },
        )

    return M5LatestRequestWorker(
        infer,
        delivery_delay_seconds=delay_ms / 1000.0,
    )


def _post_episode_result_discards(
    result: InferenceResult,
    request: M5InferenceRequest,
    *,
    discard_step: int,
    discard_timestamp: float,
    current_world_epoch: int,
    reason: str,
    queue_depth: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for action_index, action in enumerate(result.actions):
        provenance = ActionProvenance(
            action=action,
            episode_id=request.episode_id,
            condition=request.condition,
            task_id=request.task_id,
            task_instruction=request.task_instruction,
            seed=request.seed,
            request_id=request.request_id,
            request_generation_id=request.request_generation_id,
            observation_control_step=request.observation_control_step,
            observation_timestamp=request.request_timestamp,
            world_epoch_at_observation=request.world_epoch_at_observation,
            entity_pose_at_observation=request.entity_pose_at_observation,
            policy_result_arrival_step=discard_step,
            policy_result_arrival_timestamp=discard_timestamp,
            original_chunk_action_index=action_index,
        )
        row = provenance.to_dict()
        row.update(
            {
                "record_type": "discarded",
                "source_kind": "discarded",
                "held": False,
                "queue_execution_step": None,
                "execution_timestamp": None,
                "current_world_epoch_at_execution": None,
                "stale": (request.world_epoch_at_observation < current_world_epoch),
                "discarded": True,
                "discard_reason": reason,
                "discard_step": discard_step,
                "discard_timestamp": discard_timestamp,
                "current_world_epoch_at_discard": current_world_epoch,
                "gate_triggered": False,
                "was_queued": False,
                "queue_depth_before_action": None,
                "queue_depth_after_action": None,
                "queue_depth_before_invalidation": queue_depth,
                "queue_depth_after_invalidation": queue_depth,
                "command_origin": None,
            }
        )
        records.append(row)
    return records


def _run_episode(
    backend: LeRobotBackend,
    worker: M5LatestRequestWorker,
    *,
    run_id: str,
    git_commit: str,
    source_sha256: str,
    source_files: list[dict[str, str]],
    config_sha256: str,
    seed_manifest_sha256: str,
    phase: str,
    seed_manifest_name: str,
    task_audit_sha256: str,
    protocol_decision_sha256: str | None,
    no_shift_decision_sha256: str | None,
    calibration_directive_sha256: str | None,
    previous_task_decision_sha256: str | None,
    condition: str,
    task_id: int,
    instruction_expected: str,
    source_entity: str,
    moved_entity: str,
    seed: int,
    initial_state_index: int,
    pair_index: int,
    displacement_magnitude_mm: int,
    injected_delay_ms: int,
    replan_interval_steps: int,
    workspace_bounds: PlanarBounds,
    maximum_robot_base_distance_m: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unsupported M5 condition: {condition}")
    shifted = condition in SHIFT_CONDITIONS
    gated = condition in GATED_CONDITIONS
    if shifted and displacement_magnitude_mm not in {30, 50, 70}:
        raise ValueError(
            "Shifted M5 episodes require a registered 30/50/70 mm magnitude"
        )
    if not shifted and displacement_magnitude_mm != 0:
        raise ValueError("No-shift M5 episodes require zero displacement")

    observation, _, instruction = backend.reset_episode(
        task_id=task_id,
        seed=seed,
        initial_state_index=initial_state_index,
    )
    if instruction != instruction_expected:
        raise RuntimeError(
            f"Task instruction drift: expected {instruction_expected!r}, got {instruction!r}"
        )
    controller_hz = backend.controller_frequency_hz(task_id)
    if not math.isclose(controller_hz, 20.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(
            f"M5-G0 requires the audited 20 Hz controller, got {controller_hz}"
        )
    period = 1.0 / controller_hz
    delay_steps = math.floor(injected_delay_ms / 1000.0 * controller_hz)
    midpoint_offset_steps = max(1, math.floor(delay_steps / 2))

    adapter = LiberoSceneAdapter(
        backend,
        workspace_bounds=workspace_bounds,
        reachability_bounds=workspace_bounds,
        displacement_tolerance_m=0.001,
        maximum_robot_base_distance_m=maximum_robot_base_distance_m,
        support_contact_tokens=("floor",),
    )
    detector = OracleGroundTruthPoseDetector(0.010)
    initial_moved_pose = adapter.read_entity_pose(task_id, moved_entity)
    initial_source_pose = adapter.read_entity_pose(task_id, source_entity)

    episode_id = (
        f"{run_id}-pair{pair_index}-task{task_id}-state{initial_state_index}-"
        f"seed{seed}-{condition}"
    )
    queue = M5AlignedActionQueue()
    queue.reset_episode(episode_id)
    worker.reset_episode(episode_id)
    torch.cuda.reset_peak_memory_stats()

    episode_started = time.monotonic()
    requests: list[dict[str, Any]] = []
    request_objects: dict[str, M5InferenceRequest] = {}
    request_rows: dict[str, dict[str, Any]] = {}
    live_request_ids: set[str] = set()
    events: list[dict[str, Any]] = []
    _event(
        events,
        episode_id=episode_id,
        condition=condition,
        seed=seed,
        step=0,
        timestamp=episode_started,
        event_type="episode_start",
        phase=phase,
        task_id=task_id,
        initial_state_index=initial_state_index,
    )
    extra_action_rows: list[dict[str, Any]] = []
    world_epoch = 0
    request_counter = 0
    active_reference_pose: EntityPose | None = None
    scheduled_shift_step: int | None = None
    shift_request_id: str | None = None
    shift_start_timestamp: float | None = None
    shift_timestamp: float | None = None
    shift_step: int | None = None
    perturbation_result: dict[str, Any] | None = None
    perturbation_valid = not shifted
    invalid_reason: str | None = None
    gate_trigger_timestamp: float | None = None
    queue_clear_timestamp: float | None = None
    first_fresh_action_timestamp: float | None = None
    detector_checks = 0
    detector_max_translation_delta_m = 0.0
    detector_max_rotation_delta_rad = 0.0
    scene_gate_trigger_count = 0
    false_gate_count = 0
    gate_waiting_for_fresh = False
    success = False
    step_count = 0
    termination_reason = "time_limit"
    deadline_misses = 0
    contacted_moved_entity_before_shift = False
    first_moved_entity_contact_step: int | None = None
    max_queue_depth_seen = 0

    def submit_request(
        control_step: int,
        current_observation: Mapping[str, Any],
        *,
        fresh_replan: bool = False,
        allow_schedule: bool = True,
    ) -> M5InferenceRequest:
        nonlocal request_counter, scheduled_shift_step, shift_request_id
        pose = adapter.read_entity_pose(task_id, moved_entity)
        request_id = (
            f"{episode_id}-g{queue.current_request_generation_id}-r{request_counter}"
        )
        request_counter += 1
        timestamp = time.monotonic()
        request = M5InferenceRequest(
            observation=immutable_observation_snapshot(current_observation),
            task_instruction=instruction,
            episode_id=episode_id,
            observation_control_step=control_step,
            request_timestamp=timestamp,
            queue_depth_at_request_steps=queue.queue_length,
            queue_headroom_at_request_steps=queue.queue_length,
            request_id=request_id,
            request_generation_id=queue.current_request_generation_id,
            condition=condition,
            task_id=task_id,
            seed=seed,
            world_epoch_at_observation=world_epoch,
            entity_pose_at_observation=pose.to_dict(),
        )
        replaced_request = worker.submit(request)
        row = {
            **request.provenance_dict(),
            "run_id": run_id,
            "phase": phase,
            "pair_index": pair_index,
            "initial_state_index": initial_state_index,
            "queue_depth_at_request_steps": queue.queue_length,
            "queue_headroom_at_request_steps": queue.queue_length,
            "fresh_replan": fresh_replan,
            "policy_result_arrival_step": None,
            "policy_result_arrival_timestamp": None,
            "policy_result_delivery_timestamp": None,
            "chunk_action_count": None,
            "chunk_accepted": None,
            "chunk_rejection_reason": None,
            "request_status": "submitted",
            "invalidated_by_gate": False,
        }
        requests.append(row)
        request_rows[request_id] = row
        request_objects[request_id] = request
        live_request_ids.add(request_id)
        if replaced_request is not None:
            if not isinstance(replaced_request, M5InferenceRequest):
                raise RuntimeError("M5 worker replaced a non-M5 request")
            replaced_id = replaced_request.request_id
            replaced_row = request_rows.get(replaced_id)
            if replaced_row is None:
                raise RuntimeError("Replaced M5 request has no provenance row")
            replaced_row.update(
                {
                    "chunk_accepted": False,
                    "chunk_rejection_reason": "mailbox_replaced",
                    "request_status": "mailbox_replaced",
                }
            )
            live_request_ids.discard(replaced_id)
            _event(
                events,
                episode_id=episode_id,
                condition=condition,
                seed=seed,
                step=control_step,
                timestamp=timestamp,
                event_type="policy_request_mailbox_replaced",
                request_id=replaced_id,
                replacement_request_id=request_id,
            )
        _event(
            events,
            episode_id=episode_id,
            condition=condition,
            seed=seed,
            step=control_step,
            timestamp=timestamp,
            event_type="fresh_policy_request" if fresh_replan else "policy_request",
            request_id=request_id,
            request_generation_id=request.request_generation_id,
            world_epoch_at_observation=world_epoch,
            queue_depth=queue.queue_length,
        )

        if (
            shifted
            and allow_schedule
            and control_step > 0
            and scheduled_shift_step is None
            and shift_step is None
            and queue.queue_length > 0
        ):
            scheduled_shift_step = control_step + midpoint_offset_steps
            shift_request_id = request_id
            _event(
                events,
                episode_id=episode_id,
                condition=condition,
                seed=seed,
                step=control_step,
                timestamp=timestamp,
                event_type="perturbation_scheduled",
                request_id=request_id,
                scheduled_shift_step=scheduled_shift_step,
                delay_steps=delay_steps,
                midpoint_offset_steps=midpoint_offset_steps,
            )
        return request

    def process_result(
        result: InferenceResult,
        *,
        arrival_step: int,
        arrival_timestamp: float,
        merge: bool,
    ) -> None:
        nonlocal active_reference_pose, gate_waiting_for_fresh
        request_id_value = result.metadata.get("request_id")
        if not isinstance(request_id_value, str) or not request_id_value:
            raise RuntimeError("An M5 inference result lacks its explicit request_id")
        request_id = request_id_value
        if request_id not in request_objects:
            raise RuntimeError(f"Unknown M5 inference result request_id: {request_id}")
        request = request_objects[request_id]
        metadata_generation = result.metadata.get("request_generation_id")
        if int(metadata_generation) != request.request_generation_id:
            raise RuntimeError(
                "M5 result generation metadata disagrees with its request"
            )
        row = request_rows[request_id]
        live_request_ids.discard(request_id)
        row.update(
            {
                "policy_result_arrival_step": arrival_step,
                "policy_result_arrival_timestamp": arrival_timestamp,
                "policy_result_delivery_timestamp": result.delivery_timestamp,
                "chunk_action_count": len(result.actions),
            }
        )
        _event(
            events,
            episode_id=episode_id,
            condition=condition,
            seed=seed,
            step=arrival_step,
            timestamp=arrival_timestamp,
            event_type="policy_result_arrival",
            request_id=request_id,
            request_generation_id=request.request_generation_id,
            current_request_generation_id=queue.current_request_generation_id,
            merge=merge,
        )
        if not merge:
            reason = (
                "late_generation"
                if request.request_generation_id < queue.current_request_generation_id
                else "episode_ended"
            )
            row.update(
                {
                    "chunk_accepted": False,
                    "chunk_rejection_reason": reason,
                    "request_status": "discarded_after_episode",
                }
            )
            extra_action_rows.extend(
                _post_episode_result_discards(
                    result,
                    request,
                    discard_step=arrival_step,
                    discard_timestamp=arrival_timestamp,
                    current_world_epoch=world_epoch,
                    reason=reason,
                    queue_depth=queue.queue_length,
                )
            )
            return

        outcome = queue.replace(
            result,
            request=request,
            current_control_step=arrival_step,
            result_arrival_step=arrival_step,
            result_arrival_timestamp=arrival_timestamp,
            current_world_epoch=world_epoch,
        )
        row.update(
            {
                "chunk_accepted": outcome.accepted,
                "chunk_rejection_reason": None if outcome.accepted else outcome.reason,
                "request_status": "accepted" if outcome.accepted else "discarded",
                "merge_age_steps": outcome.age_steps,
                "temporal_stale_prefix_steps": outcome.dropped_prefix_steps,
                "queue_depth_after_merge": outcome.queue_length,
            }
        )
        _event(
            events,
            episode_id=episode_id,
            condition=condition,
            seed=seed,
            step=arrival_step,
            timestamp=arrival_timestamp,
            event_type="queue_merge" if outcome.accepted else "policy_result_discarded",
            request_id=request_id,
            accepted=outcome.accepted,
            reason=outcome.reason,
            age_steps=outcome.age_steps,
            dropped_prefix_steps=outcome.dropped_prefix_steps,
            queue_depth=outcome.queue_length,
        )
        if outcome.accepted:
            active_reference_pose = EntityPose.from_mapping(
                request.entity_pose_at_observation
            )
            if request.request_generation_id == queue.current_request_generation_id:
                gate_waiting_for_fresh = False

    def ensure_fresh_replan(
        control_step: int,
        current_observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Retry a current-scene request if alignment rejected it as fully stale."""

        if not gate_waiting_for_fresh or queue.has_safe_action:
            return current_observation
        current_generation_live = any(
            request_objects[request_id].request_generation_id
            == queue.current_request_generation_id
            for request_id in live_request_ids
        )
        if current_generation_live:
            return current_observation
        fresh_observation = adapter.fresh_batched_observation(task_id)
        submit_request(
            control_step,
            fresh_observation,
            fresh_replan=True,
            allow_schedule=False,
        )
        return fresh_observation

    submit_request(0, observation, allow_schedule=False)
    if not worker.wait_for_result(timeout=120):
        raise TimeoutError("Timed out waiting for the initial M5 policy result")
    initial_arrival = time.monotonic()
    for result in worker.drain_results():
        process_result(
            result,
            arrival_step=0,
            arrival_timestamp=initial_arrival,
            merge=True,
        )
    if not queue.has_safe_action:
        raise RuntimeError("Initial M5 inference completed without a safe action")

    control_epoch = time.monotonic()
    scheduled = control_epoch
    while step_count < backend.episode_length:
        scheduled, missed = _coalesce_missed_control_ticks(
            scheduled,
            time.monotonic(),
            period,
        )
        deadline_misses += missed
        remaining = scheduled - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

        pre_shift_arrival = time.monotonic()
        for result in worker.drain_results():
            process_result(
                result,
                arrival_step=step_count,
                arrival_timestamp=pre_shift_arrival,
                merge=True,
            )
        observation = thaw_observation_snapshot(
            ensure_fresh_replan(step_count, observation)
        )

        now = time.monotonic()
        if shifted and shift_step is None:
            contact_now = adapter.robot_contacts_entity(task_id, moved_entity)
            contacted_moved_entity_before_shift = (
                contacted_moved_entity_before_shift or contact_now
            )
            if contact_now and first_moved_entity_contact_step is None:
                first_moved_entity_contact_step = step_count
        if shifted and scheduled_shift_step == step_count and shift_step is None:
            if (
                shift_request_id is None
                or request_rows[shift_request_id]["policy_result_arrival_timestamp"]
                is not None
            ):
                perturbation_valid = False
                invalid_reason = "associated_policy_result_available_before_shift"
                termination_reason = "invalid_perturbation"
                _event(
                    events,
                    episode_id=episode_id,
                    condition=condition,
                    seed=seed,
                    step=step_count,
                    timestamp=now,
                    event_type="perturbation_invalid",
                    invalid_reason=invalid_reason,
                    request_id=shift_request_id,
                )
                break
            if contacted_moved_entity_before_shift:
                perturbation_valid = False
                invalid_reason = "moved_entity_contacted_before_shift"
                termination_reason = "invalid_perturbation"
                _event(
                    events,
                    episode_id=episode_id,
                    condition=condition,
                    seed=seed,
                    step=step_count,
                    timestamp=now,
                    event_type="perturbation_invalid",
                    invalid_reason=invalid_reason,
                )
                break
            shift = DeterministicPlanarShift.along_axis(
                moved_entity,
                displacement_magnitude_mm / 1000.0,
                axis_xy=(-1.0, 0.0),
            )
            shift_start_timestamp = now
            physical = shift.apply(
                adapter,
                task_id,
                workspace_bounds=workspace_bounds,
                reachability_bounds=workspace_bounds,
            )
            perturbation_result = physical.to_dict()
            perturbation_valid = physical.valid
            shift_step = step_count
            shift_timestamp = time.monotonic()
            if not physical.valid:
                invalid_reason = physical.invalid_reason
                termination_reason = "invalid_perturbation"
                _event(
                    events,
                    episode_id=episode_id,
                    condition=condition,
                    seed=seed,
                    step=step_count,
                    timestamp=shift_timestamp,
                    event_type="perturbation_invalid",
                    invalid_reason=physical.invalid_reason,
                    invalid_detail=physical.invalid_detail,
                )
                break
            world_epoch += 1
            observation = thaw_observation_snapshot(physical.fresh_observation or {})
            _event(
                events,
                episode_id=episode_id,
                condition=condition,
                seed=seed,
                step=step_count,
                timestamp=shift_timestamp,
                event_type="perturbation",
                request_id=shift_request_id,
                world_epoch=world_epoch,
                shift_started_timestamp=shift_start_timestamp,
                shift_completed_timestamp=shift_timestamp,
                requested_displacement_xy_m=list(physical.requested_displacement_xy_m),
                achieved_displacement_xyz_m=list(
                    physical.achieved_displacement_xyz_m or ()
                ),
            )

        if gated and (queue.has_safe_action or live_request_ids):
            current_pose = adapter.read_entity_pose(task_id, moved_entity)
            references: list[tuple[str, EntityPose]] = []
            if active_reference_pose is not None and queue.has_safe_action:
                references.append(("active_queue", active_reference_pose))
            for request_id in sorted(live_request_ids):
                request = request_objects[request_id]
                if request.request_generation_id == queue.current_request_generation_id:
                    references.append(
                        (
                            request_id,
                            EntityPose.from_mapping(request.entity_pose_at_observation),
                        )
                    )
            detection = None
            changed_reference = None
            for reference_name, reference_pose in references:
                detector_checks += 1
                candidate = detector.detect(reference_pose, current_pose)
                detector_max_translation_delta_m = max(
                    detector_max_translation_delta_m,
                    candidate.translation_delta_m,
                )
                detector_max_rotation_delta_rad = max(
                    detector_max_rotation_delta_rad,
                    candidate.rotation_delta_rad,
                )
                _event(
                    events,
                    episode_id=episode_id,
                    condition=condition,
                    seed=seed,
                    step=step_count,
                    timestamp=time.monotonic(),
                    event_type="scene_detector_check",
                    reference=reference_name,
                    detection=candidate.to_dict(),
                )
                if candidate.changed:
                    detection = candidate
                    changed_reference = reference_name
                    break
            if detection is not None:
                trigger_time = time.monotonic()
                scene_gate_trigger_count += 1
                if not shifted:
                    false_gate_count += 1
                previous_generation = queue.current_request_generation_id
                invalidation = queue.invalidate_scene(
                    current_control_step=step_count,
                    current_world_epoch=world_epoch,
                    invalidation_timestamp=trigger_time,
                )
                gate_trigger_timestamp = trigger_time
                queue_clear_timestamp = trigger_time
                gate_waiting_for_fresh = True
                for request_id in live_request_ids:
                    request = request_objects[request_id]
                    if request.request_generation_id == previous_generation:
                        request_rows[request_id]["invalidated_by_gate"] = True
                cancelled_request = worker.cancel_pending_request()
                if cancelled_request is not None:
                    if not isinstance(cancelled_request, M5InferenceRequest):
                        raise RuntimeError("M5 worker cancelled a non-M5 request")
                    cancelled_id = cancelled_request.request_id
                    cancelled_row = request_rows[cancelled_id]
                    cancelled_row.update(
                        {
                            "chunk_accepted": False,
                            "chunk_rejection_reason": "gate_invalidated_pending",
                            "request_status": "gate_invalidated_pending",
                            "invalidated_by_gate": True,
                        }
                    )
                    live_request_ids.discard(cancelled_id)
                    _event(
                        events,
                        episode_id=episode_id,
                        condition=condition,
                        seed=seed,
                        step=step_count,
                        timestamp=trigger_time,
                        event_type="policy_request_cancelled",
                        request_id=cancelled_id,
                        reason="gate_invalidated_pending",
                    )
                observation = adapter.fresh_batched_observation(task_id)
                active_reference_pose = current_pose
                _event(
                    events,
                    episode_id=episode_id,
                    condition=condition,
                    seed=seed,
                    step=step_count,
                    timestamp=trigger_time,
                    event_type="scene_gate_trigger",
                    changed_reference=changed_reference,
                    detection=detection.to_dict(),
                    previous_request_generation_id=previous_generation,
                    current_request_generation_id=(
                        invalidation.current_request_generation_id
                    ),
                    queue_depth_before=invalidation.queue_depth_before,
                    queue_depth_after=invalidation.queue_depth_after,
                )
                _event(
                    events,
                    episode_id=episode_id,
                    condition=condition,
                    seed=seed,
                    step=step_count,
                    timestamp=trigger_time,
                    event_type="queue_clear",
                    queue_depth_before=invalidation.queue_depth_before,
                    queue_depth_after=0,
                )
                submit_request(
                    step_count,
                    observation,
                    fresh_replan=True,
                    allow_schedule=False,
                )

        arrival = time.monotonic()
        for result in worker.drain_results():
            process_result(
                result,
                arrival_step=step_count,
                arrival_timestamp=arrival,
                merge=True,
            )
        observation = thaw_observation_snapshot(
            ensure_fresh_replan(step_count, observation)
        )

        max_queue_depth_seen = max(max_queue_depth_seen, queue.queue_length)
        current_pose_mapping = adapter.read_entity_pose(
            task_id,
            moved_entity,
        ).to_dict()
        action_record = queue.next_action(
            current_control_step=step_count,
            current_world_epoch=world_epoch,
            execution_timestamp=time.monotonic(),
            current_entity_pose=current_pose_mapping,
        )
        action = action_record.action
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid M5 environment action: {action}")
        if (
            shifted
            and shift_timestamp is not None
            and first_fresh_action_timestamp is None
            and action_record.source_kind == "policy"
            and action_record.world_epoch_at_observation == world_epoch
        ):
            first_fresh_action_timestamp = action_record.execution_timestamp
            _event(
                events,
                episode_id=episode_id,
                condition=condition,
                seed=seed,
                step=step_count,
                timestamp=first_fresh_action_timestamp,
                event_type="first_fresh_action",
                request_id=action_record.request_id,
                request_generation_id=action_record.request_generation_id,
            )

        step_output = backend.step(task_id, action)
        step_count += 1
        observation = step_output.observation
        success = success or step_output.success
        _event(
            events,
            episode_id=episode_id,
            condition=condition,
            seed=seed,
            step=step_count - 1,
            timestamp=action_record.execution_timestamp,
            event_type="action_execution",
            request_id=action_record.request_id,
            request_generation_id=action_record.request_generation_id,
            source_kind=action_record.source_kind,
            stale=action_record.stale,
            queue_depth_before=action_record.queue_depth_before_action,
            queue_depth_after=action_record.queue_depth_after_action,
            world_epoch=world_epoch,
        )
        if success:
            termination_reason = "success"
            break
        if step_output.terminated:
            termination_reason = "terminated"
            break
        if step_output.truncated:
            termination_reason = "truncated"
            break
        if step_count % replan_interval_steps == 0 and not gate_waiting_for_fresh:
            submit_request(step_count, observation)
        scheduled += period

    episode_finished = time.monotonic()
    cancelled_request = worker.cancel_pending_request()
    if cancelled_request is not None:
        if not isinstance(cancelled_request, M5InferenceRequest):
            raise RuntimeError("M5 worker cancelled a non-M5 request")
        cancelled_id = cancelled_request.request_id
        request_rows[cancelled_id].update(
            {
                "chunk_accepted": False,
                "chunk_rejection_reason": "episode_ended_pending_cancelled",
                "request_status": "episode_ended_pending_cancelled",
            }
        )
        live_request_ids.discard(cancelled_id)
    if not worker.wait_idle(timeout=120):
        raise TimeoutError("Timed out draining final M5 inference")
    final_arrival = time.monotonic()
    for result in worker.drain_all_results():
        process_result(
            result,
            arrival_step=step_count,
            arrival_timestamp=final_arrival,
            merge=False,
        )
    for request_id in live_request_ids:
        row = request_rows[request_id]
        if row["request_status"] == "submitted":
            row["request_status"] = "cancelled_or_invalidated"
            row["chunk_accepted"] = False
            row["chunk_rejection_reason"] = (
                "gate_invalidated"
                if row["invalidated_by_gate"]
                else "episode_ended_before_inference"
            )
    queue.discard_remaining_at_episode_end(
        current_control_step=step_count,
        current_world_epoch=world_epoch,
        discard_timestamp=final_arrival,
    )

    if shifted and shift_step is None and invalid_reason is None:
        perturbation_valid = False
        invalid_reason = "perturbation_trigger_not_reached"
        termination_reason = "invalid_perturbation"

    action_rows = [record.to_dict() for record in queue.action_records]
    action_rows.extend(extra_action_rows)
    action_summary = summarize_action_records(action_rows, controller_hz)
    action_summary["maximum_queue_depth"] = max(
        max_queue_depth_seen,
        queue.maximum_queue_depth,
        int(action_summary["maximum_queue_depth"]),
    )
    achieved = (
        None
        if perturbation_result is None
        else perturbation_result.get("achieved_displacement_xyz_m")
    )
    requested = (
        None
        if perturbation_result is None
        else perturbation_result.get("requested_displacement_xy_m")
    )
    time_shift_to_gate = (
        None
        if shift_timestamp is None or gate_trigger_timestamp is None
        else gate_trigger_timestamp - shift_timestamp
    )
    time_shift_to_clear = (
        None
        if shift_timestamp is None or queue_clear_timestamp is None
        else queue_clear_timestamp - shift_timestamp
    )
    time_shift_to_fresh = (
        None
        if shift_timestamp is None or first_fresh_action_timestamp is None
        else first_fresh_action_timestamp - shift_timestamp
    )
    episode = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "run_id": run_id,
        "episode_id": episode_id,
        "pair_index": pair_index,
        "git_commit_at_run": git_commit,
        "implementation_source_sha256": source_sha256,
        "implementation_source_files": source_files,
        "experiment_config_sha256": config_sha256,
        "seed_manifest_sha256": seed_manifest_sha256,
        "seed_manifest_name": seed_manifest_name,
        "phase": phase,
        "task_audit_sha256": task_audit_sha256,
        "protocol_decision_sha256": protocol_decision_sha256,
        "no_shift_decision_sha256": no_shift_decision_sha256,
        "calibration_directive_sha256": calibration_directive_sha256,
        "previous_task_decision_sha256": previous_task_decision_sha256,
        "lerobot_version": _lerobot_version(),
        "model_id": backend.model_id,
        "model_revision_sha": backend.model_revision,
        "suite": backend.suite,
        "task_id": task_id,
        "task_instruction": instruction,
        "source_entity": source_entity,
        "moved_entity": moved_entity,
        "initial_source_entity_pose": initial_source_pose.to_dict(),
        "initial_moved_entity_pose": initial_moved_pose.to_dict(),
        "initial_state_index": initial_state_index,
        "seed": seed,
        "policy_rng_seed": seed,
        "policy_rng_reset_per_episode": True,
        "condition": condition,
        "runtime_mode": "async_aligned",
        "oracle_pose_gate_enabled": gated,
        "injected_delay_ms": injected_delay_ms,
        "controller_frequency_hz": controller_hz,
        "chunk_size_steps": 30,
        "replan_interval_steps": replan_interval_steps,
        "nominal_queue_headroom_steps": 30 - replan_interval_steps,
        "success": success,
        "environment_steps": step_count,
        "simulated_completion_time_seconds": step_count / controller_hz,
        "wall_clock_episode_seconds": episode_finished - episode_started,
        "termination_reason": termination_reason,
        "world_epoch_final": world_epoch,
        "shift_request_id": shift_request_id,
        "shift_step": shift_step,
        "shift_start_timestamp": shift_start_timestamp,
        "shift_completed_timestamp": shift_timestamp,
        "scheduled_shift_step": scheduled_shift_step,
        "displacement_magnitude_mm": displacement_magnitude_mm,
        "requested_displacement_xy_m": requested,
        "achieved_displacement_xyz_m": achieved,
        "perturbation_valid": perturbation_valid,
        "invalid_reason": invalid_reason,
        "perturbation_result": perturbation_result,
        "contacted_moved_entity_before_shift": contacted_moved_entity_before_shift,
        "first_moved_entity_contact_step": first_moved_entity_contact_step,
        "moved_state_recovered": bool(success and shifted and perturbation_valid),
        "pre_shift_target_contact_after_shift": None,
        "pre_shift_target_contact_measurement_available": False,
        "detector_checks": detector_checks,
        "detector_max_translation_delta_m": detector_max_translation_delta_m,
        "detector_max_rotation_delta_rad": detector_max_rotation_delta_rad,
        "scene_gate_trigger_count": scene_gate_trigger_count,
        "false_gate_count": false_gate_count,
        "queue_invalidation_count": queue.queue_invalidation_count,
        "fresh_replan_count": sum(
            bool(request["fresh_replan"]) for request in requests
        ),
        "fresh_replan_chunks_accepted": queue.fresh_replan_count,
        "stale_policy_results_discarded": queue.stale_policy_results_discarded,
        "stale_queued_actions_discarded": queue.stale_queued_actions_discarded,
        "hold_steps_introduced_by_gate": queue.gate_hold_steps,
        "queue_hold_steps": queue.queue_hold_steps,
        "time_from_shift_to_gate_trigger_seconds": time_shift_to_gate,
        "time_from_shift_to_queue_clear_seconds": time_shift_to_clear,
        "time_from_shift_to_first_fresh_action_seconds": time_shift_to_fresh,
        "queue_invalidation_latency_seconds": time_shift_to_clear,
        "control_deadline_misses": deadline_misses,
        "peak_cuda_memory_mib": backend.peak_cuda_memory_mib,
        "request_count": len(requests),
        **action_summary,
    }
    _event(
        events,
        episode_id=episode_id,
        condition=condition,
        seed=seed,
        step=step_count,
        timestamp=episode_finished,
        event_type="episode_end",
        success=success,
        termination_reason=termination_reason,
    )
    return episode, action_rows, requests, events


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[2]
    config_path = args.config.resolve()
    manifest_path = args.seed_manifest.resolve()
    task_audit_path = args.task_audit.resolve()
    manifest = _read_json_object(manifest_path, role="seed manifest")
    validate_manifest(manifest)
    config = _read_json_object(config_path, role="experiment config")
    task_audit = _read_json_object(task_audit_path, role="task audit")
    if config.get("milestone") != "M5-G0" or task_audit.get("milestone") != "M5-G0":
        raise ValueError("Config and task audit must both describe M5-G0")
    candidate_order = _audit_candidate_order(config, task_audit)

    source_sha256, source_files = _source_hash(root)
    git_commit = _git_commit()
    config_sha256 = _sha256(config_path)
    task_audit_sha256 = file_sha256(task_audit_path)
    seed_manifest_sha256 = str(manifest["manifest_sha256"])

    expected_candidate = next(
        (
            candidate
            for candidate in config["candidate_tasks"]
            if int(candidate["task_id"]) == args.task_id
        ),
        None,
    )
    if expected_candidate is None:
        raise ValueError(
            f"Task {args.task_id} is not one of the two audited candidates"
        )
    instruction = str(expected_candidate["instruction"])
    source_entity = str(expected_candidate["source_entity"])
    moved_entity = str(expected_candidate["moved_entity"])
    runtime = config["runtime"]
    delay_ms = int(runtime["injected_delay_ms"])
    replan = int(runtime["replan_interval_steps"])
    bounds_cfg = config["perturbation"]["workspace_bounds_m"]
    bounds = PlanarBounds(
        min_x_m=float(bounds_cfg["x"][0]),
        max_x_m=float(bounds_cfg["x"][1]),
        min_y_m=float(bounds_cfg["y"][0]),
        max_y_m=float(bounds_cfg["y"][1]),
    )
    maximum_robot_base_distance_m = float(bounds_cfg["maximum_robot_base_distance"])

    expected_manifests = {
        "audit": ("runtime_smoke", None),
        "calibration": ("calibration", 5),
        "no_shift": ("no_shift_regression", 10),
        "sealed": ("sealed_evaluation", 30),
    }
    expected_manifest_name, expected_pair_count = expected_manifests[args.phase]
    if manifest.get("manifest_name") != expected_manifest_name:
        raise ValueError(
            f"Phase {args.phase!r} requires manifest {expected_manifest_name!r}, "
            f"got {manifest.get('manifest_name')!r}"
        )
    if (
        expected_pair_count is not None
        and len(manifest["pairs"]) != expected_pair_count
    ):
        raise ValueError(
            f"Phase {args.phase!r} requires exactly {expected_pair_count} pairs, "
            f"got {len(manifest['pairs'])}"
        )

    shift_phase = args.phase in {"audit", "calibration", "sealed"}
    conditions = SHIFT_CONDITIONS if shift_phase else NO_SHIFT_CONDITIONS
    magnitude = args.displacement_mm if shift_phase else 0
    if shift_phase and magnitude not in {30, 50, 70}:
        raise ValueError("Shift phases require --displacement-mm=30, 50, or 70")
    if not shift_phase and args.displacement_mm != 0:
        raise ValueError("No-shift phase requires --displacement-mm=0")

    calibration_directive_sha256: str | None = None
    previous_task_decision_sha256: str | None = None
    protocol_decision_sha256: str | None = None
    no_shift_decision_sha256: str | None = None

    if args.phase == "audit":
        if args.task_id != candidate_order[0] or magnitude != 50:
            raise ValueError(
                "The infrastructure audit is frozen to candidate 1 at 50 mm"
            )
        if any(
            value is not None
            for value in (
                args.calibration_directive,
                args.previous_task_decision,
                args.protocol_decision,
                args.no_shift_decision,
            )
        ):
            raise ValueError(
                "The audit phase does not consume post-calibration decisions"
            )
    elif args.phase == "calibration":
        if args.protocol_decision is not None or args.no_shift_decision is not None:
            raise ValueError(
                "Calibration cannot consume or expose post-calibration evidence"
            )
        if magnitude == 50:
            if args.calibration_directive is not None:
                raise ValueError("The frozen calibration must start at 50 mm")
        else:
            if args.calibration_directive is None:
                raise ValueError(
                    f"{magnitude} mm calibration requires the prior frozen directive"
                )
            directive_path = args.calibration_directive.resolve()
            directive = _read_json_object(
                directive_path,
                role="calibration directive",
            )
            calibration_directive_sha256 = _validate_payload_hash(
                directive,
                hash_field="selection_sha256",
                role="calibration directive",
            )
            if (
                int(directive.get("task_id", -1)) != args.task_id
                or directive.get("status") != f"needs_{magnitude}_mm"
                or int(directive.get("next_magnitude_mm", -1)) != magnitude
            ):
                raise ValueError(
                    "Calibration directive does not authorize this task/magnitude"
                )
        task_order_index = candidate_order.index(args.task_id)
        if task_order_index == 0:
            if args.previous_task_decision is not None:
                raise ValueError(
                    "First task candidate cannot have a prior-task decision"
                )
        else:
            if args.previous_task_decision is None:
                raise ValueError(
                    "Second task candidate requires the first candidate's final NO-GO"
                )
            previous_path = args.previous_task_decision.resolve()
            previous = _read_json_object(previous_path, role="previous task decision")
            previous_task_decision_sha256 = _validate_payload_hash(
                previous,
                hash_field="selection_sha256",
                role="previous task decision",
            )
            if (
                int(previous.get("task_id", -1))
                != candidate_order[task_order_index - 1]
                or previous.get("status") != "candidate_no_go"
            ):
                raise ValueError(
                    "Second task candidate is unauthorized before first-candidate NO-GO"
                )
    else:
        if (
            args.calibration_directive is not None
            or args.previous_task_decision is not None
        ):
            raise ValueError(
                "Post-calibration phases cannot accept calibration directives"
            )
        if args.protocol_decision is None:
            raise ValueError(f"Phase {args.phase!r} requires --protocol-decision")
        protocol_path = args.protocol_decision.resolve()
        protocol = _read_json_object(protocol_path, role="frozen protocol")
        protocol_decision_sha256 = _validate_payload_hash(
            protocol,
            hash_field="protocol_decision_sha256",
            role="frozen protocol",
        )
        if protocol.get("implementation_source_sha256") != source_sha256:
            raise ValueError("Frozen protocol implementation source hash has drifted")
        if protocol.get("config", {}).get("sha256") != config_sha256:
            raise ValueError("Frozen protocol config hash has drifted")
        if protocol.get("task_audit", {}).get("sha256") != task_audit_sha256:
            raise ValueError("Frozen protocol task-audit hash has drifted")
        frozen_manifest = protocol.get("seed_manifests", {}).get(
            expected_manifest_name,
            {},
        )
        if frozen_manifest.get("sha256") != seed_manifest_sha256:
            raise ValueError("Run manifest does not match the frozen protocol")
        if args.task_id != int(protocol["no_shift_task_id"]):
            raise ValueError("Task does not match the frozen protocol")
        if args.phase == "sealed":
            if (
                protocol.get("status") != "selected"
                or protocol.get("sealed_permitted") is not True
                or args.task_id != int(protocol["selected_task_id"])
                or magnitude != int(protocol["selected_displacement_magnitude_mm"])
            ):
                raise ValueError("Sealed run is not authorized by the frozen protocol")
            if args.no_shift_decision is None:
                raise ValueError("Sealed run requires a passing no-shift decision")
            no_shift_path = args.no_shift_decision.resolve()
            no_shift_decision = _read_json_object(
                no_shift_path,
                role="no-shift decision",
            )
            no_shift_decision_sha256 = _validate_payload_hash(
                no_shift_decision,
                hash_field="no_shift_decision_sha256",
                role="no-shift decision",
            )
            if (
                no_shift_decision.get("status") != "pass"
                or no_shift_decision.get("protocol_decision_sha256")
                != protocol_decision_sha256
            ):
                raise ValueError("Sealed run requires the bound passing no-shift gate")
        elif args.no_shift_decision is not None:
            raise ValueError("No-shift phase cannot consume its own future decision")

    output_dir = args.output_dir.resolve()
    run_manifest_path = output_dir / "run_manifest.json"
    attempts_path = output_dir / "attempts.jsonl"
    existing_run_manifest: dict[str, Any] | None = None
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.resume_after_crash:
            raise FileExistsError(
                f"Refusing to overwrite M5 output directory: {output_dir}"
            )
        if not run_manifest_path.exists():
            raise FileNotFoundError(
                "Crash resume requires the original run_manifest.json"
            )
        existing_run_manifest = _read_json_object(
            run_manifest_path,
            role="run manifest",
        )
        existing_contract = existing_run_manifest.get("run_contract")
        if not isinstance(existing_contract, dict):
            raise ValueError("Existing run manifest has no contract")
        run_id = str(existing_contract["run_id"])
        git_commit = str(existing_contract["git_commit_at_start"])
        if args.run_id is not None and args.run_id != run_id:
            raise ValueError("Resume run_id differs from the original contract")
    else:
        if args.resume_after_crash:
            raise FileNotFoundError("No crashed M5 output exists to resume")
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = args.run_id or f"m5g0-{args.phase}-{uuid.uuid4().hex[:8]}"

    run_contract = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "run_id": run_id,
        "phase": args.phase,
        "task_id": args.task_id,
        "displacement_magnitude_mm": magnitude,
        "conditions": list(conditions),
        "pairs": manifest["pairs"],
        "git_commit_at_start": git_commit,
        "implementation_source_sha256": source_sha256,
        "experiment_config_sha256": config_sha256,
        "seed_manifest_name": expected_manifest_name,
        "seed_manifest_sha256": seed_manifest_sha256,
        "task_audit_sha256": task_audit_sha256,
        "protocol_decision_sha256": protocol_decision_sha256,
        "no_shift_decision_sha256": no_shift_decision_sha256,
        "calibration_directive_sha256": calibration_directive_sha256,
        "previous_task_decision_sha256": previous_task_decision_sha256,
    }
    run_contract_sha256 = canonical_sha256(run_contract)
    if existing_run_manifest is not None:
        if (
            existing_run_manifest.get("run_contract_sha256") != run_contract_sha256
            or existing_run_manifest.get("run_contract") != run_contract
        ):
            raise ValueError("Crash resume contract differs from the original run")
        if existing_run_manifest.get("status") == "completed":
            bundle_paths: list[Path] = []
            for pair in manifest["pairs"]:
                pair_index = int(pair["pair_index"])
                pair_conditions = (
                    conditions if pair_index % 2 == 0 else tuple(reversed(conditions))
                )
                bundle_paths.extend(
                    output_dir
                    / "episode_bundles"
                    / f"pair{pair_index:03d}_{condition}.json"
                    for condition in pair_conditions
                )
            episodes: list[dict[str, Any]] = []
            all_actions: list[dict[str, Any]] = []
            all_requests: list[dict[str, Any]] = []
            all_events: list[dict[str, Any]] = []
            for bundle_path in bundle_paths:
                bundle = _validated_bundle(
                    bundle_path,
                    contract_sha256=run_contract_sha256,
                )
                episodes.append(bundle["episode"])
                all_actions.extend(bundle["actions"])
                all_requests.extend(bundle["requests"])
                all_events.extend(bundle["events"])
            aggregates = {
                "episodes.jsonl": episodes,
                "actions.jsonl": all_actions,
                "requests.jsonl": all_requests,
                "events.jsonl": all_events,
            }
            for filename, expected_rows in aggregates.items():
                if _read_jsonl_objects(output_dir / filename) != expected_rows:
                    raise ValueError(
                        "Completed aggregate differs from episode bundles: "
                        f"{filename}"
                    )
            expected_counts = {
                "episode_count": len(episodes),
                "action_record_count": len(all_actions),
                "request_count": len(all_requests),
                "event_count": len(all_events),
            }
            for field, expected_count in expected_counts.items():
                if int(existing_run_manifest.get(field, -1)) != expected_count:
                    raise ValueError(
                        f"Completed run manifest {field} does not match evidence"
                    )
            print(
                f"validated-completed {args.phase} task={args.task_id} "
                f"episodes={len(episodes)} contract={run_contract_sha256}",
                flush=True,
            )
            return episodes
        attempt_count = int(existing_run_manifest.get("attempt_count", 0)) + 1
    else:
        attempt_count = 1
    run_manifest_payload = {
        "schema_version": 1,
        "milestone": "M5-G0",
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "status": "running",
        "attempt_count": attempt_count,
        "last_attempt_started_at_utc": _utc_now(),
        "last_error": None,
    }
    _write_json_atomic(run_manifest_path, run_manifest_payload)
    _append_jsonl(
        attempts_path,
        [
            {
                "attempt": attempt_count,
                "status": "started",
                "timestamp_utc": _utc_now(),
                "run_contract_sha256": run_contract_sha256,
                "resume_after_crash": bool(args.resume_after_crash),
            }
        ],
    )

    bundle_paths: list[Path] = []
    backend: LeRobotBackend | None = None
    worker: M5LatestRequestWorker | None = None
    try:
        backend = LeRobotBackend(
            task_ids=[args.task_id],
            seed=int(manifest["pairs"][0]["seed"]),
            suite=str(config["suite"]),
            episode_length=int(runtime["episode_length_steps"]),
            model_id=str(config["model_id"]),
            model_revision=str(config["model_revision_sha"]),
            device="cuda",
        )
        worker = _make_worker(backend, delay_ms)
        for pair in manifest["pairs"]:
            pair_index = int(pair["pair_index"])
            pair_conditions = (
                conditions if pair_index % 2 == 0 else tuple(reversed(conditions))
            )
            for condition in pair_conditions:
                bundle_path = (
                    output_dir
                    / "episode_bundles"
                    / f"pair{pair_index:03d}_{condition}.json"
                )
                bundle_paths.append(bundle_path)
                if bundle_path.exists():
                    if not args.resume_after_crash:
                        raise FileExistsError(
                            f"Unexpected pre-existing episode bundle: {bundle_path}"
                        )
                    bundle = _validated_bundle(
                        bundle_path,
                        contract_sha256=run_contract_sha256,
                    )
                    episode = bundle["episode"]
                    print(
                        f"resume-skip {condition} task={args.task_id} "
                        f"pair={pair_index} seed={pair['seed']}",
                        flush=True,
                    )
                    continue
                episode, actions, requests, events = _run_episode(
                    backend,
                    worker,
                    run_id=run_id,
                    git_commit=git_commit,
                    source_sha256=source_sha256,
                    source_files=source_files,
                    config_sha256=config_sha256,
                    seed_manifest_sha256=seed_manifest_sha256,
                    phase=args.phase,
                    seed_manifest_name=expected_manifest_name,
                    task_audit_sha256=task_audit_sha256,
                    protocol_decision_sha256=protocol_decision_sha256,
                    no_shift_decision_sha256=no_shift_decision_sha256,
                    calibration_directive_sha256=calibration_directive_sha256,
                    previous_task_decision_sha256=previous_task_decision_sha256,
                    condition=condition,
                    task_id=args.task_id,
                    instruction_expected=instruction,
                    source_entity=source_entity,
                    moved_entity=moved_entity,
                    seed=int(pair["seed"]),
                    initial_state_index=int(pair["initial_state_index"]),
                    pair_index=pair_index,
                    displacement_magnitude_mm=magnitude,
                    injected_delay_ms=delay_ms,
                    replan_interval_steps=replan,
                    workspace_bounds=bounds,
                    maximum_robot_base_distance_m=maximum_robot_base_distance_m,
                )
                bundle_core = {
                    "schema_version": 1,
                    "milestone": "M5-G0",
                    "run_contract_sha256": run_contract_sha256,
                    "episode": episode,
                    "actions": actions,
                    "requests": requests,
                    "events": events,
                }
                bundle = {
                    **bundle_core,
                    "bundle_sha256": canonical_sha256(bundle_core),
                }
                _write_json_atomic(bundle_path, bundle)
                print(
                    f"{condition} task={args.task_id} pair={pair_index} "
                    f"seed={pair['seed']} success={episode['success']} "
                    f"steps={episode['environment_steps']} "
                    f"shift_valid={episode['perturbation_valid']} "
                    f"stale_steps={episode['stale_action_steps']}",
                    flush=True,
                )
        worker.close()
        worker = None
        backend.close()
        backend = None
        episodes: list[dict[str, Any]] = []
        all_actions: list[dict[str, Any]] = []
        all_requests: list[dict[str, Any]] = []
        all_events: list[dict[str, Any]] = []
        for bundle_path in bundle_paths:
            bundle = _validated_bundle(
                bundle_path,
                contract_sha256=run_contract_sha256,
            )
            episodes.append(bundle["episode"])
            all_actions.extend(bundle["actions"])
            all_requests.extend(bundle["requests"])
            all_events.extend(bundle["events"])
        _write_jsonl_atomic(output_dir / "episodes.jsonl", episodes)
        _write_jsonl_atomic(output_dir / "actions.jsonl", all_actions)
        _write_jsonl_atomic(output_dir / "requests.jsonl", all_requests)
        _write_jsonl_atomic(output_dir / "events.jsonl", all_events)
        run_manifest_payload.update(
            {
                "status": "completed",
                "completed_at_utc": _utc_now(),
                "episode_count": len(episodes),
                "action_record_count": len(all_actions),
                "request_count": len(all_requests),
                "event_count": len(all_events),
            }
        )
        _write_json_atomic(run_manifest_path, run_manifest_payload)
        _append_jsonl(
            attempts_path,
            [
                {
                    "attempt": attempt_count,
                    "status": "completed",
                    "timestamp_utc": _utc_now(),
                    "run_contract_sha256": run_contract_sha256,
                    "episode_count": len(episodes),
                }
            ],
        )
    except BaseException as exc:
        try:
            run_manifest_payload.update(
                {
                    "status": "crashed",
                    "failed_at_utc": _utc_now(),
                    "last_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            _write_json_atomic(run_manifest_path, run_manifest_payload)
            _append_jsonl(
                attempts_path,
                [
                    {
                        "attempt": attempt_count,
                        "status": "crashed",
                        "timestamp_utc": _utc_now(),
                        "run_contract_sha256": run_contract_sha256,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                ],
            )
        except Exception:
            pass
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
        finally:
            if backend is not None:
                backend.close()
    return episodes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("audit", "calibration", "no_shift", "sealed"),
        required=True,
    )
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--displacement-mm", type=int, default=0)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--task-audit", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/m5_g0.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--calibration-directive", type=Path)
    parser.add_argument("--previous-task-decision", type=Path)
    parser.add_argument("--protocol-decision", type=Path)
    parser.add_argument("--no-shift-decision", type=Path)
    parser.add_argument("--resume-after-crash", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
