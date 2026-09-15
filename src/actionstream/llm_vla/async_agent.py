"""Paced finite Agent with the production ActionStream engine and bounded retry.

Only public RGB/proprioception enters this module. A retry is an explicitly
bounded attempt after RGB still says incomplete; it does not claim to diagnose
a physical failure. Failure eligibility and recovery success are scored later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
import time

import numpy as np

from .coverage import CoveragePermit, authorize
from .finite_agent import CHECKPOINT_SHA256, TemporalVerifier, validate_chunk
from .grounding import materialize


@dataclass(frozen=True)
class AsyncAgentConfig:
    period_s: float = 0.05
    observation_max_age_s: float = 0.15
    action_source_max_age_s: float = 1.0
    policy_controls: int = 300
    settle_controls: int = 60
    recovery_open_controls: int = 20
    post_stop_controls: int = 40
    max_attempts: int = 1
    max_starvation_controls: int = 20
    request_interval_controls: int = 10
    inference_timeout_s: float = 2.0
    worker_warmup_timeout_s: float = 30.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.endswith("_s"):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(key + " must be finite and positive")
            elif type(value) is not int or value < 1:
                raise ValueError(key + " must be a positive integer")
        if self.max_attempts not in (1, 2):
            raise ValueError("At most one retry is supported")


class ChunkWorker:
    """The existing production engine owns all VLA inference and processor reset."""

    def __init__(self, port, instruction, config, emit):
        import torch
        from actionstream.lerobot_inference import (
            ActionStreamInferenceConfig,
            ActionStreamInferenceEngine,
        )

        self.warmup_done = threading.Event()
        self.warmup_error = None
        self.warmup_timeout = config.worker_warmup_timeout_s
        self.emit = emit

        def infer(data, task):
            observation = data["public"]
            warming = data.get("warmup", False)
            try:
                actions, receipt = port.infer(observation, task)
                validated = np.stack(validate_chunk(actions))
            except BaseException as exc:
                if warming:
                    self.warmup_error = exc
                raise
            finally:
                if warming:
                    self.warmup_done.set()
            if warming:
                return torch.from_numpy(validated)
            emit(
                "async_chunk",
                source_control=data["control"],
                observation=observation.binding(),
                **receipt,
            )
            return torch.from_numpy(validated)

        self.engine = ActionStreamInferenceEngine(
            policy=None,
            preprocessor=None,
            postprocessor=None,
            hw_features={},
            task=instruction,
            device="cpu",
            robot_type="libero",
            infer_chunk=infer,
            reset_provider=port.reset_inference,
            config=ActionStreamInferenceConfig(
                inference_timeout_s=config.inference_timeout_s,
                bounded_hold_steps=0,
                latest_only_fallback=False,
                minimum_request_interval_steps=config.request_interval_controls,
                join_timeout_s=3.0,
                telemetry_jsonl_path=str(port.output / "engine.jsonl")
                if hasattr(port, "output")
                else None,
            ),
        )
        self.publications = {}
        self.next_publication = 0
        self.epoch = 0
        self.engine.start()
        self.engine.resume()

    def warmup(self, observation):
        # CUDA library initialization can be thread-specific. Warm the actual
        # owner, then invalidate every warmup action before refreshing the scene.
        started = time.monotonic()
        self.engine.notify_observation(
            dict(public=observation, control=-1, warmup=True)
        )
        if not self.warmup_done.wait(self.warmup_timeout):
            raise TimeoutError("Inference worker warmup exceeded its startup budget")
        if self.warmup_error is not None:
            raise RuntimeError("Inference worker warmup failed") from self.warmup_error
        self.invalidate()
        offset = self.epoch
        self.emit(
            "worker_warmup_complete",
            wall_s=time.monotonic() - started,
            engine_epoch_offset=offset,
            discarded_all_warmup_actions=True,
        )
        self.resume()
        return offset

    def publish(self, control, observation):
        self.publications[self.next_publication] = (control, observation)
        self.next_publication += 1
        self.engine.notify_observation(dict(public=observation, control=control))

    def take(self):
        packet = self.engine.get_action_with_revision(None)
        if packet is None:
            return None
        source_control, source = self.publications[packet.source_step]
        return dict(
            action=packet.action.cpu().numpy(),
            source_control=source_control,
            source=source,
            current=self.engine.is_action_current(packet),
            epoch=packet.epoch,
            ordinal=packet.request_ordinal,
        )

    def invalidate(self):
        before = self.engine.telemetry.to_dict()
        self.engine.pause()
        self.engine.reset()
        self.epoch += 1
        self.publications.clear()
        self.next_publication = 0
        return before

    def resume(self):
        self.engine.resume()

    def close(self):
        self.engine.stop()
        snapshot = self.engine.telemetry.to_dict()
        if snapshot["failed"]:
            self.emit(
                "worker_close_failed",
                telemetry=snapshot,
                failure_traceback=self.engine.failure_traceback,
            )
            raise RuntimeError("Asynchronous inference worker failed to stop cleanly")
        return snapshot


def execute_async_request(
    request,
    permit,
    contract,
    port_factory,
    predictor,
    emit,
    config=AsyncAgentConfig(),
    *,
    worker_factory=ChunkWorker,
):
    if type(permit) is not CoveragePermit or permit.original != request:
        raise ValueError("Missing or foreign whole-request authorization")
    if authorize(request, permit.raw, contract) != permit:
        raise ValueError("Changed or forged whole-request authorization")
    task = materialize(request, permit.raw, contract).control_spec()
    task.require_accepted()
    # Both the control thread and the inference worker produce journal events.
    event_lock = threading.Lock()
    original_emit = emit

    def emit(event, **values):
        with event_lock:
            original_emit(event, **values)

    record = dict(
        status="RUNNING",
        request_id=request.request_id,
        attempts=1,
        control_steps=0,
        post_stop_controls=0,
        first_claim_control=None,
        checkpoint_sha256=CHECKPOINT_SHA256,
        config=asdict(config),
    )
    emit(
        "authorized", original=asdict(request), permit=asdict(permit), task=asdict(task)
    )
    port, worker = None, None
    started = time.monotonic()
    revision = control = attempt_start = starvation = 0
    recovering_until = None
    phase = "policy"
    next_tick = None
    verifier = TemporalVerifier(predictor, request.request_id, revision)
    last_gripper = -1.0
    try:
        port = port_factory()
        observation = port.start(request.request_id, revision)
        warmup_started = time.monotonic()
        port.warmup(observation, task.canonical_instruction, predictor)
        emit("warmup", wall_s=time.monotonic() - warmup_started)
        # Warmup did not move the simulator; obtain a fresh timestamped image.
        worker = worker_factory(port, task.canonical_instruction, config, emit)
        record["engine_epoch_offset"] = worker.warmup(observation)
        observation = port.refresh(request.request_id, revision)
        while True:
            if next_tick is not None:
                time.sleep(max(0.0, next_tick - time.monotonic()))
            tick_start = time.monotonic()
            next_tick = tick_start + config.period_s  # Never burst to catch up.
            age_before = tick_start - observation.captured_monotonic
            checked = None
            if 0 <= age_before <= config.observation_max_age_s:
                checked = verifier.update(control, observation)
                age_after = time.monotonic() - observation.captured_monotonic
                if age_after > config.observation_max_age_s:
                    checked = None
            if checked is None:
                verifier = TemporalVerifier(predictor, request.request_id, revision)
                emit(
                    "stale_observation",
                    control=control,
                    revision=revision,
                    observation=observation.binding(),
                    age_s=time.monotonic() - observation.captured_monotonic,
                )
            else:
                emit("verification", **checked)
            if checked and checked["confirmed"]:
                record["first_claim_control"] = control
                invalidated = time.monotonic()
                telemetry = worker.invalidate()
                emit(
                    "confirmed_stop",
                    control=control,
                    revision=revision,
                    confirmed_monotonic=invalidated,
                    invalidation_wall_s=time.monotonic() - invalidated,
                    telemetry=telemetry,
                )
                phase = "post_stop"
                # Physical continuation uses new observations, with no policy pulls.
                for index in range(config.post_stop_controls):
                    action = port.pose_hold(observation, -1.0)
                    before = time.monotonic()
                    observation = port.step(
                        action, request.request_id, revision, control + index + 1
                    )
                    record["post_stop_controls"] += 1
                    emit(
                        "dispatch",
                        control=control + index,
                        revision=revision,
                        phase=phase,
                        action=action.tolist(),
                        source=None,
                        started_monotonic=before,
                        dispatched_monotonic=before,
                        work_wall_s=time.monotonic() - before,
                    )
                    time.sleep(max(0.0, before + config.period_s - time.monotonic()))
                record["status"] = "complete"
                break
            elapsed = control - attempt_start
            if phase == "policy" and elapsed >= config.policy_controls:
                telemetry = worker.invalidate()
                phase = "settle"
                emit(
                    "settling_started",
                    control=control,
                    revision=revision,
                    telemetry=telemetry,
                )
            if (
                phase == "settle"
                and elapsed >= config.policy_controls + config.settle_controls
            ):
                if (
                    record["attempts"] < config.max_attempts
                    and checked
                    and checked["decision"] == "incomplete"
                ):
                    revision += 1
                    record["attempts"] += 1
                    verifier = TemporalVerifier(predictor, request.request_id, revision)
                    recovering_until = control + config.recovery_open_controls
                    phase = "recovery_open"
                    emit(
                        "recovery_started",
                        control=control,
                        revision=revision,
                        reason="rgb_incomplete_after_bounded_attempt",
                        attempts=record["attempts"],
                    )
                else:
                    record["status"] = "unconfirmed_horizon"
                    break
            if phase == "recovery_open" and control >= recovering_until:
                phase, attempt_start, starvation = "policy", control, 0
                worker.resume()
                emit("recovery_policy_resumed", control=control, revision=revision)
            packet, source = None, None
            if phase == "policy":
                worker.publish(control, observation)
                packet = worker.take()
                if packet is not None:
                    bound = packet["source"]
                    source_age = time.monotonic() - bound.captured_monotonic
                    if not (
                        packet["current"]
                        and bound.request_id == request.request_id
                        and bound.revision == revision
                        and 0 <= source_age <= config.action_source_max_age_s
                    ):
                        emit(
                            "stale_action_rejected",
                            control=control,
                            source=bound.binding(),
                            age_s=source_age,
                        )
                        packet = None
                if packet is None:
                    starvation += 1
                    if starvation > config.max_starvation_controls:
                        record["status"] = "inference_unavailable"
                        break
                else:
                    starvation = 0
                    source = dict(
                        observation=packet["source"].binding(),
                        control=packet["source_control"],
                        epoch=packet["epoch"],
                        ordinal=packet["ordinal"],
                        age_s=source_age,
                    )
            if packet is not None:
                action = packet["action"]
                last_gripper = float(action[6])
            else:
                if phase != "policy":
                    last_gripper = -1.0
                action = port.pose_hold(observation, last_gripper)
            if (
                action.shape != (7,)
                or not np.isfinite(action).all()
                or abs(action[6]) > 1
            ):
                raise ValueError("Invalid native dispatch action")
            record["control_steps"] = control + 1
            dispatched = time.monotonic()
            observation = port.step(action, request.request_id, revision, control + 1)
            emit(
                "dispatch",
                control=control,
                revision=revision,
                phase=phase,
                action=action.tolist(),
                source=source,
                started_monotonic=tick_start,
                dispatched_monotonic=dispatched,
                work_wall_s=time.monotonic() - tick_start,
            )
            control += 1
    except Exception as exc:
        record.update(status="ERROR", error_type=type(exc).__name__, error=str(exc))
    finally:
        if worker is not None:
            try:
                emit("worker_closed", telemetry=worker.close())
            except Exception as exc:
                record.update(status="ERROR", worker_close_error=str(exc))
        if port is not None:
            try:
                port.close()
            except Exception as exc:
                record.update(status="ERROR", close_error=str(exc))
        record["wall_s"] = time.monotonic() - started
        emit("execution_end", **record)
    return record
