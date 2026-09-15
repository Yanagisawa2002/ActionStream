"""LIBERO port and private recorder for the finite synchronous Agent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np

from actionstream.completion_labels import StableTruth, simulator_facts
from .contracts import CAPABILITY, Observation
from .temporal_completion import camera_rgb


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def batch(value):
    return (
        {k: batch(v) for k, v in value.items()}
        if isinstance(value, dict)
        else np.asarray(value)[None]
    )


def layout(env):
    values = np.asarray(
        [
            env.sim.data.get_body_xpos(obj.root_body)
            for _, obj in sorted(env.env.objects_dict.items())
        ]
    ).round(4)
    return hashlib.sha256(values.tobytes()).hexdigest()


class SimulationPort:
    """The controller receives RGB/proprioception, never truth or native done flags."""

    def __init__(self, backend, seed, output, excluded_layouts=None):
        self.backend, self.seed, self.output = backend, seed, Path(output)
        self.excluded = excluded_layouts if excluded_layouts is not None else set()
        self.truth = StableTruth()
        self.control = 0
        self.states, self.rgb, self.facts, self.actions, self.bindings = (
            [],
            [],
            [],
            [],
            [],
        )
        self.timing = []
        self.hold_at = []

    def capture(self, raw, request_id, revision):
        observation = Observation.project(
            batch(self.sub._format_raw_obs(raw)), request_id, revision, uuid.uuid4().hex
        )
        facts = simulator_facts(self.env)
        facts["strict_complete"] = self.truth.update(self.control, facts)
        self.facts.append(facts)
        self.states.append(self.env.sim.get_state().flatten().copy())
        self.rgb.append(camera_rgb(raw))
        self.bindings.append(observation.binding())
        return observation

    def start(self, request_id, revision):
        self.sub = self.backend._sub_env(5)
        self.sub.init_states = False
        self.sub._init_states = None
        _, _, instruction = self.backend.reset_episode(
            task_id=5, seed=self.seed, initial_state_index=0
        )
        self.env = self.sub._env
        if instruction != CAPABILITY["canonical_instruction"]:
            raise ValueError("Native instruction differs from authorized skill")
        if (
            self.backend.controller_frequency_hz(5) != 20
            or self.sub.num_steps_wait != 10
        ):
            raise ValueError("Native control/reset frequency changed")
        if self.env.robots[0].controller.use_delta:
            raise ValueError("Policy requires absolute native actions")
        signature = layout(self.env)
        save(
            self.output / "initial_layout.json", dict(seed=self.seed, sha256=signature)
        )
        if signature in self.excluded:
            raise ValueError(
                "Initial layout duplicates a consumed or current trajectory"
            )
        self.excluded.add(signature)
        (self.output / "environment.xml").write_text(self.env.env.model.get_xml())
        raw = self.env.set_init_state(self.env.sim.get_state().flatten())
        return self.capture(raw, request_id, revision)

    def infer(self, observation, instruction):
        result = self.backend.infer_action_chunk(
            observation.native_input(), instruction
        )
        if result.raw_shape != (1, 30, 20) or result.raw_dtype != "torch.float32":
            raise ValueError("Pinned VLA output contract changed")
        return result.actions, dict(
            model_latency_s=result.model_latency_seconds,
            raw_shape=result.raw_shape,
            raw_dtype=result.raw_dtype,
            actions=result.actions.tolist(),
        )

    def begin_hold(self):
        controller = self.env.robots[0].controller
        controller.use_delta = True
        controller.update(force=True)
        controller.reset_goal()
        self.hold_at.append(self.control)

    def step(self, action, request_id, revision, control):
        if control != self.control + 1:
            raise ValueError("Nonconsecutive physical control identity")
        started = time.monotonic()
        raw, _, _, _ = self.env.step(action)
        self.control = control
        self.actions.append(np.asarray(action).tolist())
        self.timing.append(time.monotonic() - started)
        # Containment is recorded privately and does not terminate observation.
        return self.capture(raw, request_id, revision)

    def close(self):
        save(self.output / "private_truth.json", self.facts)
        save(self.output / "actions.json", self.actions)
        save(self.output / "observation_bindings.json", self.bindings)
        save(
            self.output / "port_receipt.json",
            dict(
                control_records=len(self.facts),
                action_records=len(self.actions),
                hold_started_controls=self.hold_at,
                native_step_wall_s=self.timing,
            ),
        )
        if self.states:
            np.savez_compressed(
                self.output / "trajectory.npz",
                states=np.stack(self.states),
                rgb=np.stack(self.rgb),
            )
