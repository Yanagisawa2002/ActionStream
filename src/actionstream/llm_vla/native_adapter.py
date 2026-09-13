"""Native transport and private evaluator; no truth is returned through PublicPort."""

from __future__ import annotations

import json
from pathlib import Path
import time
import uuid

import numpy as np

from .contracts import CAPABILITY, Observation


class NativePort:
    def __init__(self, backend, episode: dict, directory: Path):
        from actionstream.native_baseline import Journal

        self.backend, self.episode = backend, episode
        self.oracle = Journal(directory / "oracle_trace.jsonl")
        self.directory = directory
        self.last_control_step = 0

    def project(self, raw, request_id, revision):
        return Observation.project(
            raw, request_id, revision, "o_" + uuid.uuid4().hex[:16]
        )

    def start(self, request_id, revision):
        from actionstream.native_baseline import array_stats, validate_observation

        sub_env = self.backend._sub_env(5)
        state = array_stats(sub_env._init_states[self.episode["initial_state_index"]])
        if state["sha256"] != self.episode["expected_init_state_sha256"]:
            raise ValueError("Frozen native reset identity mismatch")
        raw, _, instruction = self.backend.reset_episode(
            task_id=5,
            seed=self.episode["seed"],
            initial_state_index=self.episode["initial_state_index"],
        )
        if instruction != CAPABILITY["canonical_instruction"]:
            raise ValueError("Native task language changed")
        if (
            self.backend.controller_frequency_hz(5) != 20
            or sub_env.num_steps_wait != 10
        ):
            raise ValueError("Native control/reset contract changed")
        if any(robot.controller.use_delta for robot in sub_env._env.env.robots):
            raise ValueError("Native absolute control contract changed")
        validate_observation(raw)
        observation = self.project(raw, request_id, revision)
        self.oracle.emit(
            "reset_identity",
            episode=self.episode,
            init_state=state,
            observation=observation.binding(),
            native_is_success=None,
        )
        return observation

    def infer(self, observation, instruction):
        result = self.backend.infer_action_chunk(
            observation.native_input(), instruction
        )
        if result.raw_shape != (1, 30, 20) or result.raw_dtype != "torch.float32":
            raise ValueError("Frozen VLA raw action contract changed")
        return result.actions, dict(
            model_latency_s=result.model_latency_seconds,
            raw_shape=result.raw_shape,
            raw_dtype=result.raw_dtype,
            processor="unchanged LeRobotBackend official per-timestep postprocessing",
        )

    def step(self, action, request_id, revision, control_step):
        from actionstream.native_baseline import native_success, validate_observation

        if control_step != self.last_control_step + 1:
            raise ValueError("Nonsequential physical control identity")
        self.last_control_step = control_step
        started = time.monotonic()
        result = self.backend.step(5, action)
        self.oracle.emit(
            "step_returned",
            control_step=control_step,
            action=action.tolist(),
            native_step_s=time.monotonic() - started,
            success_repr=repr(result.info.get("is_success")),
            terminated=result.terminated,
            truncated=result.truncated,
        )
        # Private evaluator output is written only to its separate journal.
        success = native_success(result.info)
        validate_observation(result.observation)
        observation = self.project(result.observation, request_id, revision)
        self.oracle.emit(
            "scored_step",
            control_step=control_step,
            observation=observation.binding(),
            native_is_success=success,
            terminated=result.terminated,
            truncated=result.truncated,
        )
        return observation, bool(result.terminated or result.truncated)

    def refresh(self, request_id, revision):
        from actionstream.native_baseline import array_stats

        # Same official formatter as step/render; force fresh camera sensors without
        # advancing simulation or resetting. Privileged state hashes stay private.
        sub_env = self.backend._sub_env(5)
        simulator = sub_env._env.env
        before = array_stats(simulator.sim.get_state().flatten())
        raw = sub_env._format_raw_obs(simulator._get_observations(force_update=True))

        def batch(value):
            return (
                {key: batch(item) for key, item in value.items()}
                if isinstance(value, dict)
                else np.asarray(value)[None]
            )

        observation = self.project(batch(raw), request_id, revision)
        after = array_stats(simulator.sim.get_state().flatten())
        self.oracle.emit(
            "refresh_identity",
            control_step=self.last_control_step,
            before=before,
            after=after,
            observation=observation.binding(),
            reset=False,
        )
        if before["sha256"] != after["sha256"]:
            raise ValueError("Observation refresh changed physical state")
        return observation

    def close(self):
        self.oracle.close()


def score_native(directory: Path, record: dict):
    """After the execution returns, compare claims with the private step journal."""

    def rows(name):
        return [
            json.loads(line) for line in (directory / name).read_text().splitlines()
        ]

    oracle, runtime = rows("oracle_trace.jsonl"), rows("runtime_trace.jsonl")
    returned = [r for r in oracle if r["event"] == "step_returned"]
    truth = {r["control_step"]: r for r in oracle if r["event"] == "scored_step"}
    checks = [r for r in runtime if r["event"] == "checker"]
    complete_log = (
        record["status"] != "ERROR"
        and len(returned) == record["control_steps"]
        and list(truth) == list(range(1, record["control_steps"] + 1))
    )
    claims = []
    for row in checks:
        matched = truth.get(row["control_step"])
        identity = matched is not None and matched["observation"] == row["observation"]
        # A software-interruption refresh gets a new identity at the same physical state.
        refreshes = [
            r
            for r in oracle
            if r["event"] == "refresh_identity"
            and r["control_step"] == row["control_step"]
            and r["observation"] == row["observation"]
            and r["before"]["sha256"] == r["after"]["sha256"]
        ]
        identity = identity or (matched is not None and len(refreshes) == 1)
        complete_log = complete_log and identity
        claims.append(
            dict(
                control_step=row["control_step"],
                checker=row["decision"]["status"],
                native_is_success=matched["native_is_success"] if matched else None,
                identity_match=bool(identity),
            )
        )
    false = sum(
        r["checker"] == "complete" and r["native_is_success"] is not True
        for r in claims
    )
    return dict(
        status="PASS" if complete_log and false == 0 else "FAIL",
        complete_log=bool(complete_log),
        false_complete=false,
        claims=claims,
        physical_controls=len(returned),
        final_native_success=truth[max(truth)]["native_is_success"] if truth else None,
    )
