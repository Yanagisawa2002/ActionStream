"""Finite synchronous Agent: whole-request permission and confirmed RGB stopping.

No simulator, private truth, or evaluation labels enter this controller.
Historical controllers and frozen completion modules remain unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
import time

import numpy as np

from .confirmation import ContinuousConfirmation
from .coverage import CoveragePermit, authorize
from .grounding import materialize
from .temporal_completion import camera_rgb, classify

CHECKPOINT_SHA256 = "28482b40e470d932dfe44312d2dbcb7fa732146a2b2f9c7ebf7683b47cdfa7f9"
POLICY_CONTROLS = 300
SETTLE_CONTROLS = 60
POST_STOP_CONTROLS = 40
HOLD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])


class TemporalVerifier:
    """One request/revision owns one history; gaps invalidate accumulated evidence."""

    def __init__(self, predictor, request_id, revision=0):
        self.predictor = predictor
        self.request_id, self.revision = request_id, revision
        self.frames = deque(maxlen=11)
        self.gate = ContinuousConfirmation(10)
        self.last_control = None
        self.last_observation = None
        self.last_capture = None

    def update(self, control, observation):
        if type(control) is not int or control < 0:
            raise ValueError("Control identity must be a nonnegative integer")
        if (
            observation.request_id != self.request_id
            or observation.revision != self.revision
        ):
            raise ValueError("Foreign request or revision in RGB history")
        if self.last_control is not None and control <= self.last_control:
            raise ValueError("Controls must advance strictly")
        if (
            observation.observation_id == self.last_observation
            or not np.isfinite(observation.captured_monotonic)
            or (
                self.last_capture is not None
                and observation.captured_monotonic <= self.last_capture
            )
        ):
            raise ValueError("Reused or out-of-order observation")
        if self.last_control is not None and control != self.last_control + 1:
            self.frames.clear()
        self.last_control = control
        self.last_observation = observation.observation_id
        self.last_capture = observation.captured_monotonic
        self.frames.append(
            camera_rgb(
                {
                    "agentview_image": observation.pixels["image"][0],
                    "robot0_eye_in_hand_image": observation.pixels["image2"][0],
                }
            )
        )
        probabilities, clip_hash = None, None
        decision = "unknown"
        started = time.monotonic()
        if len(self.frames) == 11:
            clip = np.stack([self.frames[i] for i in (0, 5, 10)])
            values = np.asarray(self.predictor.predict(clip[None]))
            if values.shape != (1, 3):
                raise ValueError("Expected one three-class RGB prediction")
            decision = classify(values[0])
            probabilities = values[0].tolist()
            clip_hash = hashlib.sha256(clip.tobytes()).hexdigest()
        return dict(
            control=control,
            observation=observation.binding(),
            probabilities=probabilities,
            decision=decision,
            confirmed=self.gate.update(control, decision),
            clip_sha256=clip_hash,
            history_ready=len(self.frames) == 11,
            inference_wall_s=time.monotonic() - started,
        )


def validate_chunk(actions):
    values = np.asarray(actions)
    if (
        values.shape != (30, 7)
        or not np.isfinite(values).all()
        or (np.abs(values[:, 6]) > 1).any()
    ):
        raise ValueError("Invalid official native action chunk")
    return deque(row.copy() for row in values)


def execute_request(request, permit, contract, port_factory, predictor, emit):
    """Authorize before constructing a port. Only RGB can cause a success claim.

    The port returns observations only, including after native containment. It
    owns physical hold/open mechanics; private scoring runs after execution.
    Unknown clears confirmation and continues within the frozen 360-control cap.
    """
    if type(permit) is not CoveragePermit or permit.original != request:
        raise ValueError("Missing or foreign whole-request authorization")
    if authorize(request, permit.raw, contract) != permit:
        raise ValueError("Changed or forged whole-request authorization")
    task = materialize(request, permit.raw, contract).control_spec()
    task.require_accepted()
    emit(
        "authorized", original=asdict(request), permit=asdict(permit), task=asdict(task)
    )
    record = dict(
        status="RUNNING",
        request_id=request.request_id,
        control_steps=0,
        post_stop_controls=0,
        first_claim_control=None,
        inference_calls=0,
        discarded_actions=0,
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    verifier = TemporalVerifier(predictor, request.request_id)
    pending, port = deque(), None
    started = time.monotonic()
    try:
        port = port_factory()
        observation = port.start(request.request_id, 0)
        for control in range(POLICY_CONTROLS + SETTLE_CONTROLS + 1):
            checked = verifier.update(control, observation)
            emit("verification", **checked)
            if checked["confirmed"]:
                record.update(
                    first_claim_control=control, discarded_actions=len(pending)
                )
                pending.clear()
                emit("confirmed_stop", control=control, **record)
                port.begin_hold()
                for _ in range(POST_STOP_CONTROLS):
                    record["post_stop_controls"] += 1
                    observation = port.step(
                        HOLD.copy(),
                        request.request_id,
                        0,
                        control + record["post_stop_controls"],
                    )
                record["status"] = "complete"
                break
            if control == POLICY_CONTROLS + SETTLE_CONTROLS:
                record["status"] = "unconfirmed_horizon"
                break
            if control < POLICY_CONTROLS:
                if not pending:
                    actions, receipt = port.infer(
                        observation, task.canonical_instruction
                    )
                    pending = validate_chunk(actions)
                    record["inference_calls"] += 1
                    emit("chunk", source_control=control, **receipt)
                action = pending.popleft()
            else:
                if control == POLICY_CONTROLS:
                    pending.clear()
                    port.begin_hold()
                    emit("settling_started", control=control)
                action = HOLD.copy()
            # Reserve each attempted physical call, even if it raises.
            record["control_steps"] = control + 1
            observation = port.step(action, request.request_id, 0, control + 1)
    except Exception as exc:
        record.update(status="ERROR", error_type=type(exc).__name__, error=str(exc))
    finally:
        pending.clear()
        if port is not None:
            try:
                port.close()
            except Exception as exc:
                record.update(status="ERROR", close_error=str(exc))
        record["wall_s"] = time.monotonic() - started
        emit("execution_end", **record)
    return record
