"""Audited native-Isaac proxy scenes for additional official LIBERO tasks.

The X-VLA policy still observes the pinned official LIBERO renderer.  Isaac is
the sole physics/state source: one task-specific rigid contact proxy is mapped
back to the matching named LIBERO free joint at every policy request.  Static
colored geometry exists only to make the native physics video interpretable;
it is never used as policy input or as the authoritative success predicate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from actionstream.isaac_learned import (
    LIBERO_HAND_TO_EEF_ROTATION_MAT,
    LIBERO_HAND_TO_EEF_TRANSLATION_XYZ,
    LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ,
    isaac_hand_pose_from_policy_eef,
    policy_eef_pose_from_isaac_hand,
)


@dataclass(frozen=True, slots=True)
class CollisionBoxSpec:
    position_xyz: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    half_extents_xyz: tuple[float, float, float]

    def validate(self) -> None:
        for name, values, length in (
            ("position_xyz", self.position_xyz, 3),
            ("orientation_wxyz", self.orientation_wxyz, 4),
            ("half_extents_xyz", self.half_extents_xyz, 3),
        ):
            if len(values) != length or not all(
                math.isfinite(float(value)) for value in values
            ):
                raise ValueError(f"collision box {name} must contain {length} finite values")
        if any(value <= 0.0 or value > 0.25 for value in self.half_extents_xyz):
            raise ValueError("collision box half extents must lie in (0,0.25]")
        quaternion_norm = math.sqrt(sum(value * value for value in self.orientation_wxyz))
        # MuJoCo normalizes XML quaternions at compile time.  The canonical
        # plate asset stores one quaternion rounded to a norm of 1.002875.
        if not math.isclose(quaternion_norm, 1.0, abs_tol=5e-3):
            raise ValueError("collision box orientation must be a unit quaternion")


@dataclass(frozen=True, slots=True)
class LiberoProxyTaskSpec:
    key: str
    suite: str
    task_id: int
    task_family: str
    instruction: str
    initial_state_index: int
    coordinate_translation_xyz: tuple[float, float, float]
    official_reference_eef_xyz: tuple[float, float, float]
    official_object_name: str
    official_object_position_xyz: tuple[float, float, float]
    official_object_orientation_wxyz: tuple[float, float, float, float]
    collision_scale_xyz: tuple[float, float, float]
    reference_root_to_proxy_world_xyz: tuple[float, float, float]
    collision_proxy_source: str
    native_target_kind: str
    official_target_position_xyz: tuple[float, float, float] | None = None
    official_goal_region_xyxy: tuple[float, float, float, float] | None = None
    official_fixture_position_xyz: tuple[float, float, float] | None = None
    compound_collision_boxes: tuple[CollisionBoxSpec, ...] = ()
    object_mass_kg: float | None = None

    def validate(self) -> None:
        if not self.key or not self.suite or not self.instruction:
            raise ValueError("LIBERO proxy task text fields must be non-empty")
        if self.task_id < 0 or self.initial_state_index < 0:
            raise ValueError("LIBERO proxy task indexes must be non-negative")
        for name, value, length in (
            ("coordinate_translation_xyz", self.coordinate_translation_xyz, 3),
            ("official_reference_eef_xyz", self.official_reference_eef_xyz, 3),
            ("official_object_position_xyz", self.official_object_position_xyz, 3),
            ("official_object_orientation_wxyz", self.official_object_orientation_wxyz, 4),
            ("collision_scale_xyz", self.collision_scale_xyz, 3),
            ("reference_root_to_proxy_world_xyz", self.reference_root_to_proxy_world_xyz, 3),
        ):
            if len(value) != length or not all(math.isfinite(float(item)) for item in value):
                raise ValueError(f"{name} must contain {length} finite values")
        if any(value <= 0.0 or value > 0.25 for value in self.collision_scale_xyz):
            raise ValueError("collision_scale_xyz must lie in (0,0.25]")
        if not self.collision_proxy_source.strip():
            raise ValueError("collision_proxy_source must be non-empty")
        for box in self.compound_collision_boxes:
            box.validate()
        if self.object_mass_kg is not None and not (
            math.isfinite(self.object_mass_kg) and 0.0 < self.object_mass_kg <= 10.0
        ):
            raise ValueError("object_mass_kg must be finite and lie in (0,10]")
        if self.native_target_kind == "placement":
            if self.official_target_position_xyz is None:
                raise ValueError("placement proxy task requires a target position")
        elif self.native_target_kind == "push_region":
            if self.official_goal_region_xyxy is None:
                raise ValueError("push proxy task requires a goal region")
            x0, y0, x1, y1 = self.official_goal_region_xyxy
            if not x0 < x1 or not y0 < y1:
                raise ValueError("push goal region must be increasing")
        else:
            raise ValueError(f"unsupported native target kind: {self.native_target_kind}")

    def map_position(self, value: Sequence[float]) -> tuple[float, float, float]:
        position = tuple(float(item) for item in value)
        if len(position) != 3 or not all(math.isfinite(item) for item in position):
            raise ValueError("LIBERO task position must contain three finite values")
        return tuple(
            item + offset
            for item, offset in zip(position, self.coordinate_translation_xyz, strict=True)
        )

    def unmap_position(self, value: Sequence[float]) -> tuple[float, float, float]:
        position = tuple(float(item) for item in value)
        if len(position) != 3 or not all(math.isfinite(item) for item in position):
            raise ValueError("Isaac task position must contain three finite values")
        return tuple(
            item - offset
            for item, offset in zip(position, self.coordinate_translation_xyz, strict=True)
        )


# Audited from pinned hf-libero / LeRobot resets at initial-state 0 and seed
# 2026081700.  LIBERO tabletop tasks use a 0.90 m workspace offset and a Panda
# base at x=-0.66; the native scene maps the tabletop to z=0 and the base x to 0.
LIBERO_PROXY_TASK_SPECS: Mapping[str, LiberoProxyTaskSpec] = {
    "spatial2": LiberoProxyTaskSpec(
        key="spatial2",
        suite="libero_spatial",
        task_id=2,
        task_family="spatial_object_on_object",
        instruction="pick the akita black bowl from table center and place it on the plate",
        initial_state_index=0,
        coordinate_translation_xyz=(0.66, 0.0, -0.90),
        official_reference_eef_xyz=(
            -0.2093685498908786,
            -0.004006983696766443,
            1.1850137440020873,
        ),
        official_object_name="akita_black_bowl_1",
        official_object_position_xyz=(
            -0.07499158181372285,
            0.014961151087266268,
            0.8984070750975921,
        ),
        official_object_orientation_wxyz=(
            0.707106783005648,
            -1.38747552821732e-05,
            1.6673002289947323e-06,
            0.7071067792293566,
        ),
        collision_scale_xyz=(0.024, 0.020, 0.040),
        reference_root_to_proxy_world_xyz=(0.006, 0.039, 0.0226),
        collision_proxy_source=(
            "graspable bowl-rim contact patch centered at the first close pose from the "
            "successful official X-VLA sync capability episode"
        ),
        native_target_kind="placement",
        official_target_position_xyz=(
            0.07160358130313628,
            0.20039036544318572,
            0.9025063385290465,
        ),
    ),
    "goal5": LiberoProxyTaskSpec(
        key="goal5",
        suite="libero_goal",
        task_id=5,
        task_family="planar_contact_push",
        instruction="push the plate to the front of the stove",
        initial_state_index=0,
        coordinate_translation_xyz=(0.66, 0.0, -0.90),
        official_reference_eef_xyz=(
            -0.2132534930122563,
            -0.004300895646637536,
            1.1707250596348218,
        ),
        official_object_name="plate_1",
        official_object_position_xyz=(
            0.05185924290397088,
            -0.028493122928679903,
            0.9025063385290465,
        ),
        official_object_orientation_wxyz=(
            0.7071068099946488,
            -1.3218930710778984e-05,
            5.90520400510994e-06,
            0.7071067522302282,
        ),
        collision_scale_xyz=(
            0.13762122220335729,
            0.1374699120165223,
            0.01894636973086458,
        ),
        reference_root_to_proxy_world_xyz=(0.0, 0.0, 0.0),
        collision_proxy_source=(
            "the ten oriented box collision geoms and compiled body mass from the "
            "canonical LIBERO stable_scanned_objects/plate/plate.xml"
        ),
        native_target_kind="push_region",
        official_goal_region_xyxy=(-0.09, 0.17, -0.01, 0.25),
        official_fixture_position_xyz=(-0.4057642106346003, 0.21998013266492758, 0.905),
        compound_collision_boxes=(
            CollisionBoxSpec(
                position_xyz=(0.0, 0.0, 0.00313),
                orientation_wxyz=(0.499356, 0.49356, -0.50636, 0.50636),
                half_extents_xyz=(0.00222, 0.02851, 0.03057),
            ),
            CollisionBoxSpec(
                position_xyz=(0.0, 0.04828, 0.00696),
                orientation_wxyz=(0.70574, 0.12139, 0.68790, 0.11832),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
            CollisionBoxSpec(
                position_xyz=(0.0, 0.04828, 0.00696),
                orientation_wxyz=(0.70574, 0.12139, 0.68790, 0.11832),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
            CollisionBoxSpec(
                position_xyz=(0.03237, 0.03583, 0.00696),
                orientation_wxyz=(0.70114, 0.36036, 0.59839, -0.14306),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
            CollisionBoxSpec(
                position_xyz=(0.04818, 0.00314, 0.00696),
                orientation_wxyz=(0.59590, 0.55892, 0.41899, -0.39618),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
            CollisionBoxSpec(
                position_xyz=(0.03658, -0.03152, 0.00696),
                orientation_wxyz=(0.67591, -0.40164, 0.59224, 0.17629),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
            CollisionBoxSpec(
                position_xyz=(0.00230, -0.04823, 0.00696),
                orientation_wxyz=(0.13507, 0.69059, -0.10500, -0.70272),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
            CollisionBoxSpec(
                position_xyz=(-0.03334, -0.03492, 0.00696),
                orientation_wxyz=(0.59339, 0.15267, 0.69911, -0.36855),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
            CollisionBoxSpec(
                position_xyz=(-0.04827, -0.00130, 0.00696),
                orientation_wxyz=(0.40827, 0.40746, 0.58825, -0.56679),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
            CollisionBoxSpec(
                position_xyz=(-0.03442, 0.03386, 0.00696),
                orientation_wxyz=(0.60516, -0.15392, 0.68136, 0.38190),
                half_extents_xyz=(0.00254, 0.01913, 0.02851),
            ),
        ),
        object_mass_kg=0.011522081577600004,
    ),
}

for _spec in LIBERO_PROXY_TASK_SPECS.values():
    _spec.validate()


def proxy_task_spec(key: str) -> LiberoProxyTaskSpec:
    try:
        return LIBERO_PROXY_TASK_SPECS[str(key)]
    except KeyError as exc:
        raise ValueError(
            f"unknown LIBERO proxy task {key!r}; available={sorted(LIBERO_PROXY_TASK_SPECS)}"
        ) from exc


def proxy_task_specs_sha256() -> str:
    payload = {key: asdict(value) for key, value in LIBERO_PROXY_TASK_SPECS.items()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rotation_matrix_wxyz(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("proxy orientation must be a finite scalar-first quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("proxy orientation quaternion must be nonzero")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class LiberoProxyTaskScene:
    """Bind one audited LIBERO task to the single native Isaac rigid proxy."""

    def __init__(self, simulation_app: Any, scene: Any, *, spec: LiberoProxyTaskSpec) -> None:
        import omni.usd
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.prims import GeomPrim
        from pxr import Gf, UsdGeom

        spec.validate()
        self._app = simulation_app
        self._scene = scene
        self.spec = spec
        self.instruction = spec.instruction
        self.coordinate_translation_xyz = spec.coordinate_translation_xyz
        self.camera_position_xyz = (1.28, 1.08, 0.92)
        self.camera_target_xyz = (0.53, 0.08, 0.16)
        self.camera_vertical_fov_degrees = 45.0
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac has no active stage for LIBERO proxy task")

        for path in ("/World/ZoneA", "/World/ZoneB", "/World/ActiveDestination"):
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()

        if tuple(scene.object_collision_scale_xyz) != tuple(spec.collision_scale_xyz):
            raise RuntimeError(
                "native object collision scale was not frozen before physics setup: "
                f"expected={spec.collision_scale_xyz}, actual={scene.object_collision_scale_xyz}"
            )
        mapped_reference = spec.map_position(spec.official_object_position_xyz)
        physical_position = tuple(
            root + offset
            for root, offset in zip(
                mapped_reference,
                spec.reference_root_to_proxy_world_xyz,
                strict=True,
            )
        )
        scene.set_object_pose(
            position_xyz=physical_position,
            orientation_wxyz=spec.official_object_orientation_wxyz,
            settle_steps=12,
        )
        settled = scene.measure()
        settled_rotation = _rotation_matrix_wxyz(settled.object_wxyz)
        self._proxy_to_official_root_local_xyz = settled_rotation.T @ (
            np.asarray(mapped_reference, dtype=np.float64)
            - np.asarray(settled.object_xyz, dtype=np.float64)
        )

        target_geometry: dict[str, Any]
        if spec.native_target_kind == "placement":
            assert spec.official_target_position_xyz is not None
            target = spec.map_position(spec.official_target_position_xyz)
            plate_proxy = Cube(
                paths="/World/LiberoProxyTask/PlacementTarget",
                positions=(target[0], target[1], 0.00325),
                sizes=1.0,
                scales=(0.145, 0.145, 0.0065),
                colors=(0.90, 0.90, 0.90),
            )
            GeomPrim(paths=plate_proxy.paths, apply_collision_apis=True)
            target_geometry = {
                "kind": "static_colliding_plate_footprint",
                "center_xyz": [target[0], target[1], 0.00325],
                "scale_xyz": [0.145, 0.145, 0.0065],
                "height_source": (
                    "official predicate sweep: on(bowl,plate) first accepts root z=0.905 m"
                ),
            }
        else:
            assert spec.official_goal_region_xyxy is not None
            x0, y0, x1, y1 = spec.official_goal_region_xyxy
            mapped_min = spec.map_position((x0, y0, 0.90))
            mapped_max = spec.map_position((x1, y1, 0.90))
            center = (
                0.5 * (mapped_min[0] + mapped_max[0]),
                0.5 * (mapped_min[1] + mapped_max[1]),
                0.002,
            )
            scale = (mapped_max[0] - mapped_min[0], mapped_max[1] - mapped_min[1], 0.004)
            Cube(
                paths="/World/LiberoProxyTask/PushGoalRegion",
                positions=center,
                sizes=1.0,
                scales=scale,
                colors=(0.20, 0.75, 0.25),
            )
            target_geometry = {
                "kind": "visual_only_official_goal_region",
                "center_xyz": list(center),
                "scale_xyz": list(scale),
            }
            if spec.official_fixture_position_xyz is not None:
                fixture = spec.map_position(spec.official_fixture_position_xyz)
                Cube(
                    paths="/World/LiberoProxyTask/StoveProxy",
                    positions=(fixture[0], fixture[1], 0.025),
                    sizes=1.0,
                    scales=(0.24, 0.20, 0.05),
                    colors=(0.16, 0.16, 0.18),
                )
                target_geometry["fixture_proxy"] = {
                    "kind": "visual_only_noncolliding_box",
                    "center_xyz": [fixture[0], fixture[1], 0.025],
                    "scale_xyz": [0.24, 0.20, 0.05],
                }

        aligned_home = spec.map_position(spec.official_reference_eef_xyz)
        aligned_hand_position, aligned_hand_axis_angle = isaac_hand_pose_from_policy_eef(
            eef_position_xyz=aligned_home,
            eef_axis_angle_xyz=LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ,
            hand_to_eef_translation_xyz=LIBERO_HAND_TO_EEF_TRANSLATION_XYZ,
            hand_to_eef_rotation_mat=LIBERO_HAND_TO_EEF_ROTATION_MAT,
        )
        scene.set_command((*aligned_hand_position, *aligned_hand_axis_angle, 1.0))
        for _ in range(120):
            scene.step()
        aligned_measurement = scene.measure()
        measured_eef_position, _measured_eef_rotation = policy_eef_pose_from_isaac_hand(
            hand_position_xyz=aligned_measurement.end_effector_xyz,
            hand_orientation_wxyz=aligned_measurement.end_effector_wxyz,
            hand_to_eef_translation_xyz=LIBERO_HAND_TO_EEF_TRANSLATION_XYZ,
            hand_to_eef_rotation_mat=LIBERO_HAND_TO_EEF_ROTATION_MAT,
        )
        home_error = math.dist(measured_eef_position, aligned_home)
        if home_error > 0.03:
            raise RuntimeError(
                f"Isaac failed {spec.key} LIBERO home alignment: position error {home_error:.6f} m"
            )
        for _ in range(4):
            simulation_app.update()
        self.provenance = {
            "schema_version": 1,
            "status": "development_only_not_holdout_frozen",
            "task_spec": asdict(spec),
            "all_task_specs_sha256": proxy_task_specs_sha256(),
            "physics": {
                "source": "native Isaac",
                "dynamic_object_collision_proxy_scale_xyz": list(spec.collision_scale_xyz),
                "dynamic_object_compound_collision_boxes": [
                    asdict(box) for box in spec.compound_collision_boxes
                ],
                "dynamic_object_mass_kg": spec.object_mass_kg,
                "static_target_geometry": target_geometry,
                "static_target_geometry_policy_input": False,
            },
            "proxy_to_official_root_local_xyz": (
                self._proxy_to_official_root_local_xyz.tolist()
            ),
            "aligned_home_measurement": {
                "policy_eef_command_xyz": list(aligned_home),
                "isaac_hand_command_xyz": aligned_hand_position.tolist(),
                "measured_policy_eef_xyz": measured_eef_position.tolist(),
                "position_error_m": home_error,
            },
        }

    def sync(self, _measurement: Any) -> None:
        """The native rigid proxy is already the visible moving object."""

    def official_object_states(self, measurement: Any) -> dict[str, Any]:
        rotation = _rotation_matrix_wxyz(measurement.object_wxyz)
        mapped_isaac_position = (
            np.asarray(measurement.object_xyz, dtype=np.float64)
            + rotation @ self._proxy_to_official_root_local_xyz
        )
        return {
            self.spec.official_object_name: {
                "position_xyz": list(self.spec.unmap_position(mapped_isaac_position)),
                "orientation_wxyz": [float(value) for value in measurement.object_wxyz],
                "linear_velocity_xyz": [
                    float(value) for value in measurement.object_linear_velocity_xyz
                ],
                "angular_velocity_xyz": [
                    float(value) for value in measurement.object_angular_velocity_xyz
                ],
            }
        }

    def task_success(self, measurement: Any) -> bool:
        """Return a descriptive native check; official LIBERO remains authoritative."""

        state = self.official_object_states(measurement)[self.spec.official_object_name]
        x, y, _z = state["position_xyz"]
        if self.spec.native_target_kind == "placement":
            assert self.spec.official_target_position_xyz is not None
            target = self.spec.official_target_position_xyz
            return math.hypot(x - target[0], y - target[1]) <= 0.075
        assert self.spec.official_goal_region_xyxy is not None
        x0, y0, x1, y1 = self.spec.official_goal_region_xyxy
        return x0 <= x <= x1 and y0 <= y <= y1
