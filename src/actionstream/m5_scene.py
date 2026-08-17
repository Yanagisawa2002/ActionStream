"""Ground-truth scene primitives for the isolated M5 oracle experiment.

This module deliberately contains no queue or benchmark logic.  It provides a
small pose-comparison interface plus the minimum LIBERO state access needed by
the M5 runner.  The detector only compares simulator poses; evaluator labels
and injection scheduling remain outside this API.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np


LOGGER = logging.getLogger(__name__)

ORACLE_TRANSLATION_THRESHOLD_M = 0.010
TASK0_ID = 0
TASK0_TARGET_ENTITY = "basket_1"
TASK0_SOURCE_ENTITY = "alphabet_soup_1"
TASK0_SHIFT_CANDIDATES_M = (0.030, 0.050, 0.070)
TASK0_SHIFT_AXIS = (-1.0, 0.0)


def _finite_tuple(
    value: Sequence[float], length: int, field_name: str
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{field_name} must have shape ({length},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class EntityPose:
    """Immutable simulator pose, with a normalized MuJoCo ``wxyz`` quaternion."""

    entity_name: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.entity_name, str) or not self.entity_name:
            raise ValueError("entity_name must be a non-empty string")
        position = _finite_tuple(self.position, 3, "position")
        quaternion = np.asarray(
            _finite_tuple(self.quaternion, 4, "quaternion"),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("quaternion must have non-zero norm")
        quaternion /= norm
        object.__setattr__(self, "position", position)
        object.__setattr__(
            self,
            "quaternion",
            tuple(float(item) for item in quaternion),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible pose mapping used by M5 provenance records."""

        return {
            "entity_name": self.entity_name,
            "position": list(self.position),
            "quaternion": list(self.quaternion),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EntityPose:
        return cls(
            entity_name=str(value["entity_name"]),
            position=tuple(value["position"]),
            quaternion=tuple(value["quaternion"]),
        )


@dataclass(frozen=True)
class SceneChangeDetection:
    """Result of comparing an action's source-scene pose with the current pose."""

    changed: bool
    translation_delta_m: float
    rotation_delta_rad: float
    reference_pose: EntityPose
    current_pose: EntityPose

    @property
    def detected(self) -> bool:
        """Readable alias for callers that use detector terminology."""

        return self.changed

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "translation_delta_m": self.translation_delta_m,
            "rotation_delta_rad": self.rotation_delta_rad,
            "reference_pose": self.reference_pose.to_dict(),
            "current_pose": self.current_pose.to_dict(),
        }


@runtime_checkable
class SceneChangeDetector(Protocol):
    """Pose-only detector contract used by a scene-gated runtime."""

    def detect(
        self,
        reference_pose: EntityPose,
        current_pose: EntityPose,
    ) -> SceneChangeDetection:
        """Compare current scene state with the pose captured at observation time."""


class OracleGroundTruthPoseDetector:
    """Ideal translation detector backed by exact simulator poses.

    Rotation is measured and returned for auditability, but M5-G0 deliberately
    uses translation alone as the trigger.
    """

    def __init__(
        self, translation_threshold_m: float = ORACLE_TRANSLATION_THRESHOLD_M
    ) -> None:
        threshold = float(translation_threshold_m)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("translation_threshold_m must be finite and positive")
        self._translation_threshold_m = threshold

    @property
    def translation_threshold_m(self) -> float:
        return self._translation_threshold_m

    def detect(
        self,
        reference_pose: EntityPose,
        current_pose: EntityPose,
    ) -> SceneChangeDetection:
        if reference_pose.entity_name != current_pose.entity_name:
            raise ValueError(
                "Cannot compare different entities: "
                f"{reference_pose.entity_name!r} and {current_pose.entity_name!r}"
            )

        reference_position = np.asarray(reference_pose.position, dtype=np.float64)
        current_position = np.asarray(current_pose.position, dtype=np.float64)
        translation_delta_m = float(
            np.linalg.norm(current_position - reference_position)
        )

        reference_quaternion = np.asarray(reference_pose.quaternion, dtype=np.float64)
        current_quaternion = np.asarray(current_pose.quaternion, dtype=np.float64)
        # q and -q encode the same orientation, hence abs(dot).
        quaternion_dot = float(
            np.clip(abs(np.dot(reference_quaternion, current_quaternion)), 0.0, 1.0)
        )
        rotation_delta_rad = float(2.0 * math.acos(quaternion_dot))
        changed = translation_delta_m >= self._translation_threshold_m

        LOGGER.debug(
            "Oracle pose comparison entity=%s translation_delta_m=%.9f "
            "rotation_delta_rad=%.9f changed=%s",
            reference_pose.entity_name,
            translation_delta_m,
            rotation_delta_rad,
            changed,
        )
        return SceneChangeDetection(
            changed=changed,
            translation_delta_m=translation_delta_m,
            rotation_delta_rad=rotation_delta_rad,
            reference_pose=reference_pose,
            current_pose=current_pose,
        )


@dataclass(frozen=True)
class PlanarBounds:
    """Inclusive XY bounds for a physical workspace or reachability envelope."""

    min_x_m: float
    max_x_m: float
    min_y_m: float
    max_y_m: float

    def __post_init__(self) -> None:
        values = (
            float(self.min_x_m),
            float(self.max_x_m),
            float(self.min_y_m),
            float(self.max_y_m),
        )
        if not all(math.isfinite(item) for item in values):
            raise ValueError("Planar bounds must be finite")
        if values[0] > values[1] or values[2] > values[3]:
            raise ValueError("Planar bounds minimum must not exceed maximum")
        object.__setattr__(self, "min_x_m", values[0])
        object.__setattr__(self, "max_x_m", values[1])
        object.__setattr__(self, "min_y_m", values[2])
        object.__setattr__(self, "max_y_m", values[3])

    def contains(self, position_xy: Sequence[float], *, margin_m: float = 0.0) -> bool:
        x, y = _finite_tuple(position_xy, 2, "position_xy")
        margin = float(margin_m)
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("margin_m must be finite and non-negative")
        return (
            self.min_x_m + margin <= x <= self.max_x_m - margin
            and self.min_y_m + margin <= y <= self.max_y_m - margin
        )


@dataclass(frozen=True)
class SourcePoseLift:
    """Current source-object pose and vertical movement from its reference pose."""

    pose: EntityPose
    reference_z_m: float
    vertical_displacement_m: float

    @property
    def lift_m(self) -> float:
        return max(0.0, self.vertical_displacement_m)

    def is_lifted(self, threshold_m: float) -> bool:
        threshold = float(threshold_m)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold_m must be finite and non-negative")
        return self.vertical_displacement_m >= threshold


def _freeze_observation(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_observation(child) for key, child in value.items()}
        )
    if isinstance(value, np.ndarray):
        copied = value.copy()
        copied.setflags(write=False)
        return copied
    if isinstance(value, list | tuple):
        return tuple(_freeze_observation(child) for child in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class PerturbationResult:
    """Auditable result of a requested physical planar shift."""

    valid: bool
    entity_name: str
    requested_displacement_xy_m: tuple[float, float]
    achieved_displacement_xyz_m: tuple[float, float, float] | None
    displacement_error_m: float | None
    initial_pose: EntityPose | None
    attempted_pose: EntityPose | None
    final_pose: EntityPose | None
    fresh_observation: Mapping[str, Any] | None
    invalid_reason: str | None
    invalid_detail: str | None
    non_floor_contacts: tuple[str, ...] = ()
    rolled_back: bool = False

    def __post_init__(self) -> None:
        requested = _finite_tuple(
            self.requested_displacement_xy_m,
            2,
            "requested_displacement_xy_m",
        )
        object.__setattr__(self, "requested_displacement_xy_m", requested)
        if self.achieved_displacement_xyz_m is not None:
            achieved = _finite_tuple(
                self.achieved_displacement_xyz_m,
                3,
                "achieved_displacement_xyz_m",
            )
            object.__setattr__(self, "achieved_displacement_xyz_m", achieved)
        if self.displacement_error_m is not None:
            error = float(self.displacement_error_m)
            if not math.isfinite(error) or error < 0:
                raise ValueError("displacement_error_m must be finite and non-negative")
            object.__setattr__(self, "displacement_error_m", error)
        if self.valid and self.invalid_reason is not None:
            raise ValueError("A valid perturbation cannot have an invalid_reason")
        if not self.valid and not self.invalid_reason:
            raise ValueError("An invalid perturbation must report invalid_reason")
        if self.fresh_observation is not None:
            object.__setattr__(
                self,
                "fresh_observation",
                _freeze_observation(self.fresh_observation),
            )
        object.__setattr__(
            self,
            "non_floor_contacts",
            tuple(sorted(str(item) for item in self.non_floor_contacts)),
        )

    @property
    def reason(self) -> str | None:
        return self.invalid_reason

    @property
    def requested_displacement_m(self) -> float:
        return float(np.linalg.norm(self.requested_displacement_xy_m))

    @property
    def achieved_displacement_m(self) -> float | None:
        if self.achieved_displacement_xyz_m is None:
            return None
        return float(np.linalg.norm(self.achieved_displacement_xyz_m))

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "entity_name": self.entity_name,
            "requested_displacement_xy_m": list(self.requested_displacement_xy_m),
            "achieved_displacement_xyz_m": (
                list(self.achieved_displacement_xyz_m)
                if self.achieved_displacement_xyz_m is not None
                else None
            ),
            "displacement_error_m": self.displacement_error_m,
            "initial_pose": self.initial_pose.to_dict() if self.initial_pose else None,
            "attempted_pose": (
                self.attempted_pose.to_dict() if self.attempted_pose else None
            ),
            "final_pose": self.final_pose.to_dict() if self.final_pose else None,
            "invalid_reason": self.invalid_reason,
            "invalid_detail": self.invalid_detail,
            "non_floor_contacts": list(self.non_floor_contacts),
            "rolled_back": self.rolled_back,
        }


@dataclass(frozen=True)
class DeterministicPlanarShift:
    """A fixed entity/vector pair that can be shared by paired M5 trials."""

    entity_name: str
    displacement_xy_m: tuple[float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.entity_name, str) or not self.entity_name:
            raise ValueError("entity_name must be a non-empty string")
        displacement = _finite_tuple(self.displacement_xy_m, 2, "displacement_xy_m")
        if float(np.linalg.norm(displacement)) <= 0:
            raise ValueError("displacement_xy_m must be non-zero")
        object.__setattr__(self, "displacement_xy_m", displacement)

    @classmethod
    def along_axis(
        cls,
        entity_name: str,
        magnitude_m: float,
        axis_xy: Sequence[float] = TASK0_SHIFT_AXIS,
    ) -> DeterministicPlanarShift:
        magnitude = float(magnitude_m)
        axis = np.asarray(_finite_tuple(axis_xy, 2, "axis_xy"), dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("magnitude_m must be finite and positive")
        if axis_norm <= 0:
            raise ValueError("axis_xy must be non-zero")
        displacement = magnitude * axis / axis_norm
        return cls(entity_name, tuple(float(item) for item in displacement))

    def apply(
        self,
        adapter: LiberoSceneAdapter,
        task_id: int,
        *,
        workspace_bounds: PlanarBounds | None = None,
        reachability_bounds: PlanarBounds | None = None,
    ) -> PerturbationResult:
        return adapter.shift_entity_planar(
            task_id,
            self.entity_name,
            self.displacement_xy_m,
            workspace_bounds=workspace_bounds,
            reachability_bounds=reachability_bounds,
        )


class _InvalidShift(RuntimeError):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


BoundsConfig = PlanarBounds | Mapping[int, PlanarBounds] | None


class LiberoSceneAdapter:
    """Minimal physical-scene adapter around an initialized ``LeRobotBackend``."""

    _TABLE_SIZE_ATTRIBUTES = (
        "living_room_table_full_size",
        "kitchen_table_full_size",
        "study_table_full_size",
        "coffee_table_full_size",
        "table_full_size",
    )

    def __init__(
        self,
        backend: Any,
        *,
        workspace_bounds: BoundsConfig = None,
        reachability_bounds: BoundsConfig = None,
        displacement_tolerance_m: float = 0.001,
        maximum_robot_base_distance_m: float | None = None,
        support_contact_tokens: Sequence[str] = ("floor", "ground", "table"),
    ) -> None:
        tolerance = float(displacement_tolerance_m)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("displacement_tolerance_m must be finite and non-negative")
        tokens = tuple(str(token).casefold() for token in support_contact_tokens)
        if not tokens or any(not token for token in tokens):
            raise ValueError("support_contact_tokens must contain non-empty strings")
        if maximum_robot_base_distance_m is not None:
            maximum_robot_base_distance_m = float(maximum_robot_base_distance_m)
            if (
                not math.isfinite(maximum_robot_base_distance_m)
                or maximum_robot_base_distance_m <= 0
            ):
                raise ValueError(
                    "maximum_robot_base_distance_m must be finite and positive"
                )
        self.backend = backend
        self._workspace_bounds = workspace_bounds
        self._reachability_bounds = reachability_bounds
        self.displacement_tolerance_m = tolerance
        self.maximum_robot_base_distance_m = maximum_robot_base_distance_m
        self.support_contact_tokens = tokens

    def _sub_env(self, task_id: int) -> Any:
        accessor = getattr(self.backend, "_sub_env", None)
        if accessor is None or not callable(accessor):
            raise RuntimeError(
                "Backend does not expose a synchronous LIBERO sub-environment"
            )
        return accessor(int(task_id))

    def _scene(self, task_id: int) -> tuple[Any, Any, Any, Any]:
        sub_env = self._sub_env(task_id)
        offscreen_env = getattr(sub_env, "_env", None)
        if offscreen_env is None:
            raise RuntimeError("Reset the LIBERO task before accessing its scene")
        task_env = getattr(offscreen_env, "env", offscreen_env)
        sim = getattr(task_env, "sim", getattr(offscreen_env, "sim", None))
        if sim is None:
            raise RuntimeError("LIBERO task does not expose its MuJoCo simulation")
        return sub_env, offscreen_env, task_env, sim

    @staticmethod
    def _entity(task_env: Any, entity_name: str) -> Any:
        getter = getattr(task_env, "get_object", None)
        entity = getter(entity_name) if callable(getter) else None
        if entity is None:
            raise ValueError(f"Unknown LIBERO entity {entity_name!r}")
        return entity

    @staticmethod
    def _free_joint(sim: Any, entity: Any, entity_name: str) -> str:
        for joint_name in reversed(tuple(getattr(entity, "joints", ()) or ())):
            try:
                qpos = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64)
                qvel = np.asarray(sim.data.get_joint_qvel(joint_name), dtype=np.float64)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if qpos.shape == (7,) and qvel.shape == (6,):
                return str(joint_name)
        raise ValueError(f"Entity {entity_name!r} does not have a 7D free joint")

    @staticmethod
    def _body_pose(
        task_env: Any, sim: Any, entity: Any, entity_name: str
    ) -> EntityPose:
        body_id: int | None = None
        body_ids = getattr(task_env, "obj_body_id", None)
        if isinstance(body_ids, Mapping) and entity_name in body_ids:
            body_id = int(body_ids[entity_name])
        if body_id is None:
            root_body = getattr(entity, "root_body", None)
            name_to_id = getattr(sim.model, "body_name2id", None)
            if root_body is not None and callable(name_to_id):
                body_id = int(name_to_id(root_body))
        if body_id is None:
            raise ValueError(f"Cannot resolve body for entity {entity_name!r}")

        data = sim.data
        if hasattr(data, "body_xpos") and hasattr(data, "body_xquat"):
            position = np.asarray(data.body_xpos[body_id], dtype=np.float64)
            quaternion = np.asarray(data.body_xquat[body_id], dtype=np.float64)
        else:
            position = np.asarray(
                data.get_body_xpos(getattr(entity, "root_body")), dtype=np.float64
            )
            quaternion = np.asarray(
                data.get_body_xquat(getattr(entity, "root_body")), dtype=np.float64
            )
        return EntityPose(
            entity_name=entity_name,
            position=tuple(float(item) for item in position),
            quaternion=tuple(float(item) for item in quaternion),
        )

    def read_entity_pose(self, task_id: int, entity_name: str) -> EntityPose:
        """Read an entity's current physical body pose from MuJoCo."""

        _, _, task_env, sim = self._scene(task_id)
        entity = self._entity(task_env, entity_name)
        return self._body_pose(task_env, sim, entity, entity_name)

    # Short alias useful in detector wiring.
    entity_pose = read_entity_pose

    def source_pose_and_lift(
        self,
        task_id: int,
        source_entity_name: str,
        *,
        reference_pose: EntityPose | None = None,
        reference_z_m: float | None = None,
    ) -> SourcePoseLift:
        """Read a source object and its vertical displacement from a fixed reference."""

        if (reference_pose is None) == (reference_z_m is None):
            raise ValueError("Provide exactly one of reference_pose or reference_z_m")
        if reference_pose is not None:
            if reference_pose.entity_name != source_entity_name:
                raise ValueError("reference_pose belongs to a different entity")
            reference_z = float(reference_pose.position[2])
        else:
            reference_z = float(reference_z_m)
        if not math.isfinite(reference_z):
            raise ValueError("reference_z_m must be finite")
        pose = self.read_entity_pose(task_id, source_entity_name)
        return SourcePoseLift(
            pose=pose,
            reference_z_m=reference_z,
            vertical_displacement_m=float(pose.position[2] - reference_z),
        )

    read_source_pose_and_lift = source_pose_and_lift

    def robot_contacts_entity(self, task_id: int, entity_name: str) -> bool:
        _, _, task_env, _ = self._scene(task_id)
        entity = self._entity(task_env, entity_name)
        robots = tuple(getattr(task_env, "robots", ()) or ())
        if not robots:
            raise RuntimeError("LIBERO task does not expose a robot")
        robot_model = getattr(robots[0], "robot_model", robots[0])
        checker = getattr(task_env, "check_contact", None)
        if not callable(checker):
            raise RuntimeError("LIBERO task does not expose contact queries")
        return bool(checker(robot_model, entity))

    def entities_in_contact(
        self, task_id: int, first_entity: str, second_entity: str
    ) -> bool:
        _, _, task_env, _ = self._scene(task_id)
        first = self._entity(task_env, first_entity)
        second = self._entity(task_env, second_entity)
        checker = getattr(task_env, "check_contact", None)
        if not callable(checker):
            raise RuntimeError("LIBERO task does not expose contact queries")
        return bool(checker(first, second))

    def _batch_observation(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: self._batch_observation(child) for key, child in value.items()}
        if isinstance(value, np.ndarray):
            return np.expand_dims(value.copy(), axis=0)
        if np.isscalar(value):
            return np.asarray([value])
        return copy.deepcopy(value)

    def fresh_batched_observation(self, task_id: int) -> dict[str, Any]:
        """Render a new normal vector-style observation without stepping the robot."""

        sub_env, offscreen_env, task_env, _ = self._scene(task_id)
        direct_reader = getattr(sub_env, "fresh_batched_observation", None)
        if callable(direct_reader):
            observation = direct_reader()
            if not isinstance(observation, Mapping):
                raise RuntimeError("Fresh observation reader returned a non-mapping")
            return copy.deepcopy(dict(observation))

        post_process = getattr(offscreen_env, "_post_process", None)
        if callable(post_process):
            post_process()
        update_observables = getattr(offscreen_env, "_update_observables", None)
        if callable(update_observables):
            update_observables(force=True)
        raw_reader = getattr(task_env, "_get_observations", None)
        formatter = getattr(sub_env, "_format_raw_obs", None)
        if not callable(raw_reader) or not callable(formatter):
            raise RuntimeError(
                "LIBERO task cannot produce a fresh formatted observation"
            )
        formatted = formatter(raw_reader())
        if not isinstance(formatted, Mapping):
            raise RuntimeError("LIBERO observation formatter returned a non-mapping")
        return self._batch_observation(formatted)

    @staticmethod
    def _configured_bounds(config: BoundsConfig, task_id: int) -> PlanarBounds | None:
        if config is None or isinstance(config, PlanarBounds):
            return config
        try:
            bounds = config[int(task_id)]
        except KeyError as exc:
            raise ValueError(f"No planar bounds configured for task {task_id}") from exc
        if not isinstance(bounds, PlanarBounds):
            raise TypeError("Configured task bounds must be PlanarBounds")
        return bounds

    def _derived_workspace_bounds(self, task_env: Any) -> PlanarBounds:
        offset = np.asarray(getattr(task_env, "workspace_offset", ()), dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Cannot derive workspace bounds: invalid workspace_offset")
        for attribute in self._TABLE_SIZE_ATTRIBUTES:
            if not hasattr(task_env, attribute):
                continue
            size = np.asarray(getattr(task_env, attribute), dtype=np.float64)
            if size.shape == (3,) and np.isfinite(size).all() and np.all(size[:2] > 0):
                return PlanarBounds(
                    min_x_m=float(offset[0] - size[0] / 2),
                    max_x_m=float(offset[0] + size[0] / 2),
                    min_y_m=float(offset[1] - size[1] / 2),
                    max_y_m=float(offset[1] + size[1] / 2),
                )
        raise ValueError("Cannot derive workspace bounds from the LIBERO task")

    def _bounds(
        self,
        task_id: int,
        task_env: Any,
        workspace_override: PlanarBounds | None,
        reachability_override: PlanarBounds | None,
    ) -> tuple[PlanarBounds, PlanarBounds]:
        workspace = (
            workspace_override
            or self._configured_bounds(self._workspace_bounds, task_id)
            or self._derived_workspace_bounds(task_env)
        )
        reachability = (
            reachability_override
            or self._configured_bounds(self._reachability_bounds, task_id)
            or workspace
        )
        return workspace, reachability

    @staticmethod
    def _robot_base_xy(task_env: Any, sim: Any) -> np.ndarray:
        robots = tuple(getattr(task_env, "robots", ()) or ())
        if not robots:
            raise RuntimeError("LIBERO task does not expose a robot")
        robot_model = getattr(robots[0], "robot_model", robots[0])
        root_body = getattr(robot_model, "root_body", None)
        name_to_id = getattr(sim.model, "body_name2id", None)
        if not root_body or not callable(name_to_id):
            raise RuntimeError("LIBERO robot base body cannot be resolved")
        body_id = int(name_to_id(root_body))
        position = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise RuntimeError("LIBERO robot base pose is invalid")
        return position[:2].copy()

    def _non_floor_contacts(
        self, task_env: Any, sim: Any, entity: Any
    ) -> tuple[str, ...]:
        getter = getattr(task_env, "get_contacts", None)
        if callable(getter):
            contacts = {str(item) for item in getter(entity)}
        else:
            own_geoms = set(
                str(item) for item in (getattr(entity, "contact_geoms", ()) or ())
            )
            contacts = set()
            contact_data = getattr(sim.data, "contact", ())
            count = int(getattr(sim.data, "ncon", 0))
            id_to_name = getattr(sim.model, "geom_id2name", None)
            if not callable(id_to_name):
                raise RuntimeError("MuJoCo model cannot resolve contact geometry names")
            for contact in contact_data[:count]:
                first = str(id_to_name(contact.geom1))
                second = str(id_to_name(contact.geom2))
                if first in own_geoms and second not in own_geoms:
                    contacts.add(second)
                elif second in own_geoms and first not in own_geoms:
                    contacts.add(first)
        return tuple(
            sorted(
                name
                for name in contacts
                if not any(
                    token in name.casefold() for token in self.support_contact_tokens
                )
            )
        )

    @staticmethod
    def _invalid_result(
        *,
        entity_name: str,
        requested: tuple[float, float],
        reason: str,
        detail: str,
        achieved: tuple[float, float, float] | None = None,
        error: float | None = None,
        initial: EntityPose | None = None,
        attempted: EntityPose | None = None,
        final: EntityPose | None = None,
        contacts: tuple[str, ...] = (),
        rolled_back: bool = False,
    ) -> PerturbationResult:
        return PerturbationResult(
            valid=False,
            entity_name=entity_name,
            requested_displacement_xy_m=requested,
            achieved_displacement_xyz_m=achieved,
            displacement_error_m=error,
            initial_pose=initial,
            attempted_pose=attempted,
            final_pose=final,
            fresh_observation=None,
            invalid_reason=reason,
            invalid_detail=detail,
            non_floor_contacts=contacts,
            rolled_back=rolled_back,
        )

    def shift_entity_planar(
        self,
        task_id: int,
        entity_name: str,
        displacement_xy_m: Sequence[float],
        *,
        workspace_bounds: PlanarBounds | None = None,
        reachability_bounds: PlanarBounds | None = None,
    ) -> PerturbationResult:
        """Apply and validate one physical free-joint XY translation.

        Invalid requests are explicit results.  Any failure after mutation
        restores both the original 7D qpos and original 6D qvel, then forwards
        the simulator so no partial teleport is left behind.
        """

        try:
            requested = _finite_tuple(displacement_xy_m, 2, "displacement_xy_m")
        except (TypeError, ValueError) as exc:
            return self._invalid_result(
                entity_name=str(entity_name),
                requested=(0.0, 0.0),
                reason="invalid_request",
                detail=str(exc),
            )
        if float(np.linalg.norm(requested)) <= 0:
            return self._invalid_result(
                entity_name=entity_name,
                requested=requested,
                reason="invalid_request",
                detail="displacement_xy_m must be non-zero",
            )

        try:
            _, _, task_env, sim = self._scene(task_id)
            entity = self._entity(task_env, entity_name)
            joint_name = self._free_joint(sim, entity, entity_name)
            original_qpos = np.asarray(
                sim.data.get_joint_qpos(joint_name), dtype=np.float64
            ).copy()
            original_qvel = np.asarray(
                sim.data.get_joint_qvel(joint_name), dtype=np.float64
            ).copy()
            initial_pose = self._body_pose(task_env, sim, entity, entity_name)
            workspace, reachability = self._bounds(
                task_id,
                task_env,
                workspace_bounds,
                reachability_bounds,
            )
        except Exception as exc:
            return self._invalid_result(
                entity_name=entity_name,
                requested=requested,
                reason="scene_access_error",
                detail=f"{type(exc).__name__}: {exc}",
            )

        target_qpos = original_qpos.copy()
        target_qpos[0] += requested[0]
        target_qpos[1] += requested[1]
        target_xy = target_qpos[:2]
        entity_margin = max(
            0.0, float(getattr(entity, "horizontal_radius", 0.0) or 0.0)
        )

        if not workspace.contains(target_xy, margin_m=entity_margin):
            return self._invalid_result(
                entity_name=entity_name,
                requested=requested,
                reason="workspace_bounds",
                detail=(
                    f"Target xy={target_xy.tolist()} with margin={entity_margin:.9f} m "
                    "is outside workspace bounds"
                ),
                initial=initial_pose,
                final=initial_pose,
            )
        if not reachability.contains(target_xy, margin_m=entity_margin):
            return self._invalid_result(
                entity_name=entity_name,
                requested=requested,
                reason="reachability_bounds",
                detail=(
                    f"Target xy={target_xy.tolist()} with margin={entity_margin:.9f} m "
                    "is outside reachability bounds"
                ),
                initial=initial_pose,
                final=initial_pose,
            )
        if self.maximum_robot_base_distance_m is not None:
            try:
                robot_base_xy = self._robot_base_xy(task_env, sim)
            except Exception as exc:
                return self._invalid_result(
                    entity_name=entity_name,
                    requested=requested,
                    reason="reachability_check_error",
                    detail=f"{type(exc).__name__}: {exc}",
                    initial=initial_pose,
                    final=initial_pose,
                )
            robot_base_distance = float(np.linalg.norm(target_xy - robot_base_xy))
            if robot_base_distance > self.maximum_robot_base_distance_m:
                return self._invalid_result(
                    entity_name=entity_name,
                    requested=requested,
                    reason="reachability_distance",
                    detail=(
                        f"Target xy={target_xy.tolist()} is "
                        f"{robot_base_distance:.9f} m from robot base "
                        f"{robot_base_xy.tolist()}, above "
                        f"{self.maximum_robot_base_distance_m:.9f} m"
                    ),
                    initial=initial_pose,
                    final=initial_pose,
                )

        mutated = False
        attempted_pose: EntityPose | None = None
        achieved: tuple[float, float, float] | None = None
        displacement_error: float | None = None
        contacts: tuple[str, ...] = ()
        invalid: _InvalidShift | None = None

        try:
            sim.data.set_joint_qpos(joint_name, target_qpos)
            mutated = True
            sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
            sim.forward()

            attempted_pose = self._body_pose(task_env, sim, entity, entity_name)
            achieved_array = np.asarray(attempted_pose.position) - np.asarray(
                initial_pose.position
            )
            achieved = tuple(float(item) for item in achieved_array)
            displacement_error = float(
                np.linalg.norm(
                    achieved_array[:2] - np.asarray(requested, dtype=np.float64)
                )
            )
            if displacement_error > self.displacement_tolerance_m:
                raise _InvalidShift(
                    "displacement_mismatch",
                    f"Planar displacement error {displacement_error:.9f} m exceeds "
                    f"{self.displacement_tolerance_m:.9f} m",
                )
            if abs(float(achieved_array[2])) > self.displacement_tolerance_m:
                raise _InvalidShift(
                    "height_changed",
                    f"Physical z changed by {achieved_array[2]:.9f} m",
                )
            orientation_dot = abs(
                float(np.dot(initial_pose.quaternion, attempted_pose.quaternion))
            )
            orientation_error = 2.0 * math.acos(
                float(np.clip(orientation_dot, 0.0, 1.0))
            )
            if orientation_error > 1e-7:
                raise _InvalidShift(
                    "orientation_changed",
                    f"Physical orientation changed by {orientation_error:.9f} rad",
                )

            contacts = self._non_floor_contacts(task_env, sim, entity)
            if contacts:
                raise _InvalidShift(
                    "non_floor_collision",
                    f"Entity contacts non-support geometries: {list(contacts)}",
                )

            try:
                observation = self.fresh_batched_observation(task_id)
            except Exception as exc:
                raise _InvalidShift(
                    "fresh_observation_failed",
                    f"{type(exc).__name__}: {exc}",
                ) from exc
            final_pose = self._body_pose(task_env, sim, entity, entity_name)
            return PerturbationResult(
                valid=True,
                entity_name=entity_name,
                requested_displacement_xy_m=requested,
                achieved_displacement_xyz_m=achieved,
                displacement_error_m=displacement_error,
                initial_pose=initial_pose,
                attempted_pose=attempted_pose,
                final_pose=final_pose,
                fresh_observation=observation,
                invalid_reason=None,
                invalid_detail=None,
                non_floor_contacts=(),
                rolled_back=False,
            )
        except _InvalidShift as exc:
            invalid = exc
        except Exception as exc:
            invalid = _InvalidShift("mutation_failed", f"{type(exc).__name__}: {exc}")

        rolled_back = False
        rollback_detail: str | None = None
        if mutated:
            try:
                sim.data.set_joint_qpos(joint_name, original_qpos)
                sim.data.set_joint_qvel(joint_name, original_qvel)
                sim.forward()
                rolled_back = True
            except Exception as exc:
                rollback_detail = f"{type(exc).__name__}: {exc}"
        try:
            final_pose = self._body_pose(task_env, sim, entity, entity_name)
        except Exception:
            final_pose = None

        assert invalid is not None
        reason = invalid.reason
        detail = invalid.detail
        if rollback_detail is not None:
            reason = "rollback_failed"
            detail = f"{detail}; rollback failed: {rollback_detail}"
        return self._invalid_result(
            entity_name=entity_name,
            requested=requested,
            reason=reason,
            detail=detail,
            achieved=achieved,
            error=displacement_error,
            initial=initial_pose,
            attempted=attempted_pose,
            final=final_pose,
            contacts=contacts,
            rolled_back=rolled_back,
        )

    # Naming used by some runner code reads more naturally as "apply".
    apply_planar_shift = shift_entity_planar


__all__ = [
    "DeterministicPlanarShift",
    "EntityPose",
    "LiberoSceneAdapter",
    "ORACLE_TRANSLATION_THRESHOLD_M",
    "OracleGroundTruthPoseDetector",
    "PerturbationResult",
    "PlanarBounds",
    "SceneChangeDetection",
    "SceneChangeDetector",
    "SourcePoseLift",
    "TASK0_ID",
    "TASK0_SHIFT_AXIS",
    "TASK0_SHIFT_CANDIDATES_M",
    "TASK0_SOURCE_ENTITY",
    "TASK0_TARGET_ENTITY",
]
