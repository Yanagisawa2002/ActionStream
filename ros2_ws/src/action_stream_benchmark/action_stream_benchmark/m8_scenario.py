"""Seed-controlled native-Isaac destination-switch scenario traces."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .m8_protocol import M8_MILESTONE, M8_SCHEMA_VERSION
from .schema import canonical_sha256, read_json, write_json_atomic


def _xyz(name: str, value: Sequence[float]) -> tuple[float, float, float]:
    converted = tuple(float(item) for item in value)
    if len(converted) != 3 or not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{name} must contain exactly three finite values")
    return converted


@dataclass(frozen=True, slots=True)
class ScenarioSpace:
    object_x_range: tuple[float, float] = (0.42, 0.48)
    object_y_range: tuple[float, float] = (-0.04, 0.04)
    object_z_m: float = 0.02575
    zone_negative_y_xyz: tuple[float, float, float] = (0.48, -0.22, 0.02575)
    zone_positive_y_xyz: tuple[float, float, float] = (0.48, 0.22, 0.02575)
    switch_steps: tuple[int, ...] = (90, 100, 110)

    def validate(self) -> None:
        x_range = tuple(float(value) for value in self.object_x_range)
        y_range = tuple(float(value) for value in self.object_y_range)
        if len(x_range) != 2 or x_range[0] > x_range[1]:
            raise ValueError("object_x_range must be ordered")
        if len(y_range) != 2 or y_range[0] > y_range[1]:
            raise ValueError("object_y_range must be ordered")
        if not all(math.isfinite(value) for value in (*x_range, *y_range, self.object_z_m)):
            raise ValueError("scenario bounds must be finite")
        _xyz("zone_negative_y_xyz", self.zone_negative_y_xyz)
        _xyz("zone_positive_y_xyz", self.zone_positive_y_xyz)
        if not self.switch_steps or any(int(step) <= 0 for step in self.switch_steps):
            raise ValueError("switch_steps must be positive")


@dataclass(frozen=True, slots=True)
class ScenarioTrace:
    seed: int
    object_xyz: tuple[float, float, float]
    object_wxyz: tuple[float, float, float, float]
    zone_a_xyz: tuple[float, float, float]
    zone_b_xyz: tuple[float, float, float]
    original_destination_xyz: tuple[float, float, float]
    final_destination_xyz: tuple[float, float, float]
    original_zone_label: str
    final_zone_label: str
    switch_step: int

    def __post_init__(self) -> None:
        if self.seed < 0 or self.switch_step <= 0:
            raise ValueError("seed must be non-negative and switch_step positive")
        object.__setattr__(self, "object_xyz", _xyz("object_xyz", self.object_xyz))
        orientation = tuple(float(value) for value in self.object_wxyz)
        if len(orientation) != 4 or not all(math.isfinite(value) for value in orientation):
            raise ValueError("object_wxyz must contain four finite values")
        object.__setattr__(self, "object_wxyz", orientation)
        object.__setattr__(self, "zone_a_xyz", _xyz("zone_a_xyz", self.zone_a_xyz))
        object.__setattr__(self, "zone_b_xyz", _xyz("zone_b_xyz", self.zone_b_xyz))
        object.__setattr__(self, "original_destination_xyz", _xyz("original_destination_xyz", self.original_destination_xyz))
        object.__setattr__(self, "final_destination_xyz", _xyz("final_destination_xyz", self.final_destination_xyz))
        if self.original_destination_xyz == self.final_destination_xyz:
            raise ValueError("destination switch must change the physical destination")
        if {self.original_zone_label, self.final_zone_label} != {"A", "B"}:
            raise ValueError("original/final zone labels must be A and B")

    @property
    def object_position_xyz(self) -> tuple[float, float, float]:
        """Compatibility alias for recorder/fairness field naming."""

        return self.object_xyz

    def core_payload(self) -> dict[str, Any]:
        return {
            "schema_version": M8_SCHEMA_VERSION,
            "milestone": M8_MILESTONE,
            "task_id": "dynamic_target_pick_place_v1",
            "disturbance_type": "destination_switch",
            "seed": self.seed,
            "object_position_xyz": list(self.object_xyz),
            "object_orientation_wxyz": list(self.object_wxyz),
            "zone_a_xyz": list(self.zone_a_xyz),
            "zone_b_xyz": list(self.zone_b_xyz),
            "original_destination_xyz": list(self.original_destination_xyz),
            "final_destination_xyz": list(self.final_destination_xyz),
            "original_zone_label": self.original_zone_label,
            "final_zone_label": self.final_zone_label,
            "original_destination_id": 0,
            "final_destination_id": 1,
            "switch_step": self.switch_step,
            "initial_generation_id": 1,
            "disturbed_generation_id": 2,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.core_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.core_payload(), "scenario_sha256": self.sha256}


def generate_scenario(seed: int, *, space: ScenarioSpace | None = None) -> ScenarioTrace:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    try:
        from action_stream_isaac.dynamic_task import (
            DynamicTaskConfig,
            scenario_for_seed,
            scenario_payload,
        )
    except ImportError:
        DynamicTaskConfig = None
        scenario_for_seed = None
        scenario_payload = None
    if scenario_for_seed is not None:
        if space is None:
            native = scenario_for_seed(seed)
        else:
            space.validate()
            native = scenario_for_seed(
                seed,
                DynamicTaskConfig(
                    object_x_range_m=space.object_x_range,
                    object_y_range_m=space.object_y_range,
                    cube_side_m=2.0 * space.object_z_m,
                    zone_a_xy_m=space.zone_negative_y_xyz[:2],
                    zone_b_xy_m=space.zone_positive_y_xyz[:2],
                    switch_steps=space.switch_steps,
                ),
            )
        payload = scenario_payload(native)
        return ScenarioTrace(
            seed=int(payload["seed"]),
            object_xyz=tuple(payload["object_position_xyz"]),
            object_wxyz=tuple(payload["object_orientation_wxyz"]),
            zone_a_xyz=tuple(payload["zone_a_xyz"]),
            zone_b_xyz=tuple(payload["zone_b_xyz"]),
            original_destination_xyz=tuple(payload["original_destination_xyz"]),
            final_destination_xyz=tuple(payload["final_destination_xyz"]),
            original_zone_label=str(payload["original_zone_label"]),
            final_zone_label=str(payload["final_zone_label"]),
            switch_step=int(payload["switch_step"]),
        )

    # Import-safe fallback is byte-for-byte the canonical pure dynamic-task
    # algorithm.  Tests exercise equality whenever the Isaac package is present.
    active_space = space or ScenarioSpace()
    active_space.validate()
    digest = hashlib.sha256(f"{M8_MILESTONE}:{seed}".encode("ascii")).digest()
    def unit(offset: int) -> float:
        return int.from_bytes(digest[offset : offset + 8], "big") / float((1 << 64) - 1)
    object_xyz = (
        active_space.object_x_range[0]
        + unit(0) * (active_space.object_x_range[1] - active_space.object_x_range[0]),
        active_space.object_y_range[0]
        + unit(8) * (active_space.object_y_range[1] - active_space.object_y_range[0]),
        active_space.object_z_m,
    )
    negative = _xyz("zone_negative_y_xyz", active_space.zone_negative_y_xyz)
    positive = _xyz("zone_positive_y_xyz", active_space.zone_positive_y_xyz)
    if seed % 2 == 0:
        original, final, original_label, final_label = negative, positive, "A", "B"
    else:
        original, final, original_label, final_label = positive, negative, "B", "A"
    switch_index = int.from_bytes(digest[16:20], "big") % len(active_space.switch_steps)
    return ScenarioTrace(
        seed=seed,
        object_xyz=object_xyz,
        object_wxyz=(1.0, 0.0, 0.0, 0.0),
        zone_a_xyz=negative,
        zone_b_xyz=positive,
        original_destination_xyz=original,
        final_destination_xyz=final,
        original_zone_label=original_label,
        final_zone_label=final_label,
        switch_step=active_space.switch_steps[switch_index],
    )


def write_scenario(path: Path | str, trace: ScenarioTrace) -> None:
    write_json_atomic(path, trace.to_dict())


def scenario_from_mapping(
    payload: Mapping[str, Any],
    *,
    source: str = "<mapping>",
) -> ScenarioTrace:
    """Parse and self-hash-check a scenario from loose or archived JSON."""

    trace = ScenarioTrace(
        seed=int(payload["seed"]),
        object_xyz=tuple(payload.get("object_xyz", payload.get("object_position_xyz"))),
        object_wxyz=tuple(payload.get("object_wxyz", payload.get("object_orientation_wxyz", (1.0, 0.0, 0.0, 0.0)))),
        zone_a_xyz=tuple(payload["zone_a_xyz"]),
        zone_b_xyz=tuple(payload["zone_b_xyz"]),
        original_destination_xyz=tuple(payload["original_destination_xyz"]),
        final_destination_xyz=tuple(payload["final_destination_xyz"]),
        original_zone_label=str(payload["original_zone_label"]),
        final_zone_label=str(payload["final_zone_label"]),
        switch_step=int(payload["switch_step"]),
    )
    if payload.get("milestone") != M8_MILESTONE or payload.get("scenario_sha256") != trace.sha256:
        raise ValueError(f"scenario trace hash or milestone mismatch: {source}")
    return trace


def load_scenario(path: Path | str) -> ScenarioTrace:
    return scenario_from_mapping(read_json(path), source=str(path))


def euclidean_distance(first: Sequence[float], second: Sequence[float]) -> float:
    a = _xyz("first", first)
    b = _xyz("second", second)
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b, strict=True)))


def command_targets_obsolete_destination(
    command_xyz: Sequence[float],
    *,
    obsolete_destination_xyz: Sequence[float],
    final_destination_xyz: Sequence[float],
    margin_m: float = 0.0,
) -> bool:
    if not math.isfinite(margin_m) or margin_m < 0.0:
        raise ValueError("margin_m must be finite and non-negative")
    obsolete = euclidean_distance(command_xyz, obsolete_destination_xyz)
    final = euclidean_distance(command_xyz, final_destination_xyz)
    return obsolete + margin_m < final


def motion_targets_obsolete_destination(
    previous_object_xyz: Sequence[float],
    current_object_xyz: Sequence[float],
    *,
    obsolete_destination_xyz: Sequence[float],
    final_destination_xyz: Sequence[float],
    progress_epsilon_m: float = 1e-4,
) -> bool:
    if not math.isfinite(progress_epsilon_m) or progress_epsilon_m < 0.0:
        raise ValueError("progress_epsilon_m must be finite and non-negative")
    old_before = euclidean_distance(previous_object_xyz, obsolete_destination_xyz)
    old_after = euclidean_distance(current_object_xyz, obsolete_destination_xyz)
    final_before = euclidean_distance(previous_object_xyz, final_destination_xyz)
    final_after = euclidean_distance(current_object_xyz, final_destination_xyz)
    return old_before - old_after > progress_epsilon_m and final_before - final_after <= 0.0


def scenario_pair_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable fields that must match across paired methods."""

    return {
        key: payload.get(key)
        for key in (
            "scenario_sha256",
            "reset_state_sha256",
            "object_position_xyz",
            "original_destination_xyz",
            "final_destination_xyz",
            "switch_step",
        )
    }
