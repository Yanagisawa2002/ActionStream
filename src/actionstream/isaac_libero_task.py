"""Development-only LIBERO object-task scene for native Isaac.

The learned X-VLA checkpoint was trained with LIBERO cameras, geometry, and
world coordinates.  This module recreates one audited LIBERO Object task in
Isaac without importing either simulator at module import time.  It remains a
development candidate until task-capable behavior is established and a
separate paired protocol is frozen.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

from actionstream.isaac_learned import (
    LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ,
    LIBERO_REFERENCE_EEF_WORLD_QUATERNION_WXYZ,
    LIBERO_REFERENCE_EEF_XYZ,
)


LIBERO_TASK_ID = 0
LIBERO_TASK_FAMILY = "pick_object_into_container"
LIBERO_TASK_INSTRUCTION = "pick up the alphabet soup and place it in the basket"
LIBERO_TABLE_Z_M = 0.0
LIBERO_ROBOT_BASE_XYZ = (-0.6, 0.0, 0.0)
ISAAC_ROBOT_BASE_XYZ = (0.0, 0.0, 0.0)
LIBERO_TO_ISAAC_TASK_TRANSLATION_XYZ = tuple(
    isaac - libero
    for isaac, libero in zip(
        ISAAC_ROBOT_BASE_XYZ, LIBERO_ROBOT_BASE_XYZ, strict=True
    )
)

# Audited from hf-libero's canonical libero_object task 0, initial-state 0.
# Quaternions are scalar-first MuJoCo world quaternions.
LIBERO_BODY_POSES: dict[str, tuple[tuple[float, float, float], tuple[float, float, float, float]]] = {
    "alphabet_soup": (
        (-0.1190942554819824, -0.2397812171983273, 0.038392910369975934),
        (-0.0022599053613085035, 0.0022578122619941114, 0.7077060816721801, 0.7064997529545018),
    ),
    "basket": (
        (0.014835114894043312, 0.25211744175080475, -0.0045823453265843445),
        (0.7071047512840226, -0.0016942118246894126, 0.00169940206642960189, 0.7071047522480977),
    ),
    "salad_dressing": (
        (0.05000070616610814, -0.09999129744571159, 0.07270544927976626),
        (0.5000063321833649, 0.4999936570715062, 0.5000059110846429, 0.49999409951040047),
    ),
    "cream_cheese": (
        (-0.15127780484515138, 0.05836388222182275, 0.008921477418607878),
        (0.0, 0.0, 0.0, 1.0),
    ),
    "milk": (
        (0.10437762559352011, -0.19602996396059144, 0.06973547626138664),
        (0.49999991071252853, 0.5000000894751193, 0.4999999310315965, 0.5000000687807303),
    ),
    "tomato_sauce": (
        (0.1502390974120275, 0.030060769941393967, 0.038392596419837076),
        (-0.0022615597956423157, 0.0022625459859602127, 0.7076760632212669, 0.7065297982226948),
    ),
    "butter": (
        (-0.2046235291582687, -0.07527223776823, 0.008695477418599518),
        (5.773097206961297e-24, -2.7148202519327063e-17, 1.874776026872469e-17, 1.0),
    ),
}

# Audited agentview camera. MuJoCo cameras look along local -Z; the third
# matrix column is consequently negated to obtain a look direction.
LIBERO_AGENTVIEW_POSITION = (0.8965773716836134, 5.216182733499864e-07, 0.65)
LIBERO_AGENTVIEW_XMAT = (
    (-1.7233905013069872e-06, -0.5287697435529835, 0.8487653140297038),
    (0.9999999999985034, -7.823149652530503e-07, 1.5430955956635258e-06),
    (-1.5194045527300304e-07, 0.848765314031093, 0.5287697435535403),
)
LIBERO_AGENTVIEW_FOVY_DEGREES = 45.0

# Audited from the same canonical hf-libero reset.  This is the MuJoCo
# eye-in-hand camera pose expressed in right_hand-local USD camera axes
# (scalar-first quaternion).  Isaac's panda_hand frame is located at its EEF,
# while LIBERO's right_hand frame is about 9.7 cm above gripper0_eef.  The
# runtime transports the audited camera-to-EEF transform with the measured EEF
# pose instead of attaching the camera to two semantically different link
# frames.  The Isaac rendering mount adds 8 cm of
# vertical clearance and 8 cm of forward clearance: bounded development calibration
# showed that the source offset puts the wider Isaac panda_hand mesh across
# roughly 40% of the policy image, whereas the compensated pose retains the
# hand cue while exposing substantially more of the task surface.
# This compensation is kept explicit so it cannot be mistaken for the canonical
# LIBERO measurement.
LIBERO_WRIST_CAMERA_SOURCE_PARENT = "robot0_right_hand"
LIBERO_WRIST_CAMERA_LOCAL_TRANSLATION_XYZ = (0.05, 0.0, 0.0)
LIBERO_WRIST_CAMERA_LOCAL_QUATERNION_WXYZ = (
    5.2288855e-05,
    0.70710677925,
    0.70710677925,
    -5.23898294e-05,
)
LIBERO_WRIST_CAMERA_FOVY_DEGREES = 75.0
LIBERO_WRIST_CAMERA_CLIPPING_RANGE_M = (0.01, 100.0)
LIBERO_WRIST_CAMERA_AUDITED_EEF_OFFSET_WORLD_XYZ = (
    0.05555217762730828,
    -0.00008014101709790017,
    0.0938939274870507,
)
ISAAC_WRIST_CAMERA_CLEARANCE_COMPENSATION_WORLD_XYZ = (0.08, 0.0, 0.08)
LIBERO_WRIST_CAMERA_EEF_OFFSET_WORLD_XYZ = tuple(
    audited + compensation
    for audited, compensation in zip(
        LIBERO_WRIST_CAMERA_AUDITED_EEF_OFFSET_WORLD_XYZ,
        ISAAC_WRIST_CAMERA_CLEARANCE_COMPENSATION_WORLD_XYZ,
        strict=True,
    )
)
LIBERO_WRIST_CAMERA_WORLD_QUATERNION_WXYZ = (
    0.7064636239671014,
    -0.020447464044321184,
    0.0206601587393956,
    -0.7071521809810737,
)
ISAAC_WRIST_CAMERA_PARENT_PATH = "/World"
ISAAC_TASK_SURFACE_RENDER_CLEARANCE_M = 0.001
LIBERO_WRIST_CAMERA_SETTLE_UPDATES = 4

# The task is staged on LIBERO's living-room table, not a flat brown proxy.
# These values are audited from
# ``scenes/libero_living_room_tabletop_base_style.xml`` and the source OBJ.
# The OBJ's dominant tabletop plane is placed at the normalized task surface
# (Z=0), while preserving the XML-authored 1.5 scale, child translation, and
# Z rotation. The audited maximum Z is retained to document the raised edge
# that invalidated the first capture candidate.
LIBERO_TABLE_VISUAL_SOURCE_POSITION_XYZ = (-0.25, 0.25, 0.0)
LIBERO_TABLE_VISUAL_SOURCE_ORIENTATION_WXYZ = (
    0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865476,
)
LIBERO_TABLE_VISUAL_SOURCE_TABLETOP_Z_M = 0.267306
LIBERO_TABLE_VISUAL_SOURCE_MAX_Z_M = 0.299668


@dataclass(frozen=True, slots=True)
class LiberoVisualAsset:
    name: str
    obj_relative_path: str
    auxiliary_relative_paths: tuple[str, ...]
    scale: float


VISUAL_ASSETS: tuple[LiberoVisualAsset, ...] = (
    LiberoVisualAsset(
        "alphabet_soup",
        "stable_hope_objects/alphabet_soup/textured.obj",
        (
            "stable_hope_objects/alphabet_soup/textured.mtl",
            "stable_hope_objects/alphabet_soup/texture_map.png",
            "stable_hope_objects/alphabet_soup/alphabet_soup.xml",
        ),
        0.01,
    ),
    LiberoVisualAsset(
        "basket",
        "stable_scanned_objects/basket/basket.obj",
        (
            "stable_scanned_objects/basket/basket.mtl",
            "stable_scanned_objects/basket/texture.png",
            "stable_scanned_objects/basket/basket.xml",
        ),
        1.0,
    ),
    LiberoVisualAsset(
        "salad_dressing",
        "stable_hope_objects/salad_dressing/textured.obj",
        (
            "stable_hope_objects/salad_dressing/textured.mtl",
            "stable_hope_objects/salad_dressing/texture_map.png",
            "stable_hope_objects/salad_dressing/salad_dressing.xml",
        ),
        0.01,
    ),
    LiberoVisualAsset(
        "cream_cheese",
        "stable_hope_objects/cream_cheese/cream_cheese.obj",
        (
            "stable_hope_objects/cream_cheese/cream_cheese.mtl",
            "stable_hope_objects/cream_cheese/texture_map.png",
            "stable_hope_objects/cream_cheese/cream_cheese.xml",
        ),
        0.008,
    ),
    LiberoVisualAsset(
        "milk",
        "stable_hope_objects/milk/textured.obj",
        (
            "stable_hope_objects/milk/textured.mtl",
            "stable_hope_objects/milk/texture_map.png",
            "stable_hope_objects/milk/milk.xml",
        ),
        0.0075,
    ),
    LiberoVisualAsset(
        "tomato_sauce",
        "stable_hope_objects/tomato_sauce/textured.obj",
        (
            "stable_hope_objects/tomato_sauce/textured.mtl",
            "stable_hope_objects/tomato_sauce/texture_map.png",
            "stable_hope_objects/tomato_sauce/tomato_sauce.xml",
        ),
        0.01,
    ),
    LiberoVisualAsset(
        "butter",
        "stable_hope_objects/butter/butter.obj",
        (
            "stable_hope_objects/butter/butter.mtl",
            "stable_hope_objects/butter/texture_map.png",
            "stable_hope_objects/butter/butter.xml",
        ),
        0.0075,
    ),
)

LIBERO_TABLE_VISUAL_ASSET = LiberoVisualAsset(
    "living_room_table_visual",
    "scenes/living_room_table/living_room_table.obj",
    (
        "scenes/living_room_table/living_room_table.mtl",
        "scenes/living_room_table/living_room_table_texture.png",
        "scenes/living_room_table/living_room_table.xml",
        "scenes/libero_living_room_tabletop_base_style.xml",
    ),
    1.5,
)


def _diffuse_only_table_mtl(source: str) -> tuple[str, tuple[str, ...]]:
    """Remove the one MTL dependency absent from the canonical asset package."""

    kept: list[str] = []
    removed: list[str] = []
    for line in source.splitlines():
        if line.lstrip().lower().startswith("map_bump "):
            removed.append(line.strip())
        else:
            kept.append(line)
    if len(removed) != 1:
        raise ValueError(
            "expected exactly one missing living-room table map_Bump directive, "
            f"found {len(removed)}"
        )
    return "\n".join(kept) + "\n", tuple(removed)


def _prepare_table_converter_source(
    *, assets_root: Path, cache_directory: Path
) -> tuple[Path, dict[str, Any]]:
    """Make a deterministic diffuse-only OBJ bundle for Isaac conversion."""

    source_obj = assets_root / LIBERO_TABLE_VISUAL_ASSET.obj_relative_path
    source_mtl = assets_root / LIBERO_TABLE_VISUAL_ASSET.auxiliary_relative_paths[0]
    source_texture = assets_root / LIBERO_TABLE_VISUAL_ASSET.auxiliary_relative_paths[1]
    source_mtl_sha256 = _sha256_file(source_mtl)
    key = f"{_sha256_file(source_obj)[:12]}_{source_mtl_sha256[:12]}"
    prepared = cache_directory / "prepared_sources" / f"living_room_table_{key}"
    prepared.mkdir(parents=True, exist_ok=True)
    prepared_obj = prepared / source_obj.name
    prepared_mtl = prepared / source_mtl.name
    prepared_texture = prepared / source_texture.name
    shutil.copy2(source_obj, prepared_obj)
    shutil.copy2(source_texture, prepared_texture)
    sanitized, removed = _diffuse_only_table_mtl(
        source_mtl.read_text(encoding="utf-8")
    )
    prepared_mtl.write_text(sanitized, encoding="utf-8")
    provenance = {
        "reason": "canonical asset package omits MTL map_Bump target; official XML uses the diffuse texture",
        "removed_mtl_directives": list(removed),
        "source_mtl_sha256": source_mtl_sha256,
        "prepared_mtl_sha256": _sha256_file(prepared_mtl),
        "prepared_obj_sha256": _sha256_file(prepared_obj),
        "prepared_texture_sha256": _sha256_file(prepared_texture),
    }
    return prepared_obj, provenance


def libero_to_isaac_position(position: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in position)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("LIBERO position must contain three finite values")
    return tuple(
        value + offset
        for value, offset in zip(
            values, LIBERO_TO_ISAAC_TASK_TRANSLATION_XYZ, strict=True
        )
    )


def isaac_to_libero_position(position: Sequence[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in position)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Isaac position must contain three finite values")
    return tuple(
        value - offset
        for value, offset in zip(
            values, LIBERO_TO_ISAAC_TASK_TRANSLATION_XYZ, strict=True
        )
    )


def _normalized_quaternion_wxyz(
    value: Sequence[float], *, name: str
) -> tuple[float, float, float, float]:
    quaternion = tuple(float(component) for component in value)
    if len(quaternion) != 4 or not all(math.isfinite(component) for component in quaternion):
        raise ValueError(f"{name} must contain four finite values")
    norm = math.sqrt(sum(component * component for component in quaternion))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return tuple(component / norm for component in quaternion)


def _quaternion_multiply_wxyz(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = _normalized_quaternion_wxyz(left, name="left quaternion")
    rw, rx, ry, rz = _normalized_quaternion_wxyz(right, name="right quaternion")
    return _normalized_quaternion_wxyz(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        name="quaternion product",
    )


def _rotate_vector_wxyz(
    quaternion_wxyz: Sequence[float], vector_xyz: Sequence[float]
) -> tuple[float, float, float]:
    w, x, y, z = _normalized_quaternion_wxyz(
        quaternion_wxyz, name="rotation quaternion"
    )
    vector = tuple(float(component) for component in vector_xyz)
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError("rotation vector must contain three finite values")
    # Expanded q * [0,v] * conjugate(q), without normalizing the vector term.
    vx, vy, vz = vector
    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y - z * w) * vy
        + 2 * (x * z + y * w) * vz,
        2 * (x * y + z * w) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z - x * w) * vz,
        2 * (x * z - y * w) * vx
        + 2 * (y * z + x * w) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


def transported_wrist_camera_pose(
    *,
    end_effector_xyz: Sequence[float],
    end_effector_wxyz: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Transport the audited reset camera transform with the measured EEF pose."""

    eef = tuple(float(component) for component in end_effector_xyz)
    if len(eef) != 3 or not all(math.isfinite(component) for component in eef):
        raise ValueError("wrist-camera EEF position must contain three finite values")
    current = _normalized_quaternion_wxyz(
        end_effector_wxyz, name="wrist-camera EEF orientation"
    )
    reference = _normalized_quaternion_wxyz(
        LIBERO_REFERENCE_EEF_WORLD_QUATERNION_WXYZ,
        name="reference EEF orientation",
    )
    reference_inverse = (reference[0], -reference[1], -reference[2], -reference[3])
    delta = _quaternion_multiply_wxyz(current, reference_inverse)
    transported_offset = _rotate_vector_wxyz(
        delta, LIBERO_WRIST_CAMERA_EEF_OFFSET_WORLD_XYZ
    )
    position = tuple(
        component + offset
        for component, offset in zip(eef, transported_offset, strict=True)
    )
    orientation = _quaternion_multiply_wxyz(
        delta, LIBERO_WRIST_CAMERA_WORLD_QUATERNION_WXYZ
    )
    return position, orientation


def isaac_agentview_pose() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    position = libero_to_isaac_position(LIBERO_AGENTVIEW_POSITION)
    forward = tuple(-LIBERO_AGENTVIEW_XMAT[row][2] for row in range(3))
    target = tuple(position[index] + forward[index] for index in range(3))
    return position, target


def task0_scene_payload() -> dict[str, Any]:
    camera_position, camera_target = isaac_agentview_pose()
    return {
        "schema_version": 1,
        "status": "development_only_not_holdout_frozen",
        "suite": "libero_object",
        "task_id": LIBERO_TASK_ID,
        "task_family": LIBERO_TASK_FAMILY,
        "instruction": LIBERO_TASK_INSTRUCTION,
        "initial_state_index": 0,
        "coordinate_mapping": "robot-base-anchored world translation; axes unchanged",
        "translation_xyz": list(LIBERO_TO_ISAAC_TASK_TRANSLATION_XYZ),
        "libero_robot_base_xyz": list(LIBERO_ROBOT_BASE_XYZ),
        "isaac_robot_base_xyz": list(ISAAC_ROBOT_BASE_XYZ),
        "isaac_aligned_home_eef_xyz": list(
            libero_to_isaac_position(LIBERO_REFERENCE_EEF_XYZ)
        ),
        "isaac_aligned_home_eef_axis_angle_xyz": list(
            LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ
        ),
        "table_top_z_m": libero_to_isaac_position((0.0, 0.0, LIBERO_TABLE_Z_M))[2],
        "body_poses": {
            name: {
                "position_xyz": list(libero_to_isaac_position(pose[0])),
                "orientation_wxyz": list(pose[1]),
            }
            for name, pose in LIBERO_BODY_POSES.items()
        },
        "agentview": {
            "position_xyz": list(camera_position),
            "target_xyz": list(camera_target),
            "vertical_fov_degrees": LIBERO_AGENTVIEW_FOVY_DEGREES,
            "source": "hf-libero canonical task reset camera transform",
        },
        "table_visual": {
            "source_asset": LIBERO_TABLE_VISUAL_ASSET.obj_relative_path,
            "source_position_xyz": list(LIBERO_TABLE_VISUAL_SOURCE_POSITION_XYZ),
            "source_orientation_wxyz": list(
                LIBERO_TABLE_VISUAL_SOURCE_ORIENTATION_WXYZ
            ),
            "source_scale": LIBERO_TABLE_VISUAL_ASSET.scale,
            "source_tabletop_z_m": LIBERO_TABLE_VISUAL_SOURCE_TABLETOP_Z_M,
            "source_max_z_m": LIBERO_TABLE_VISUAL_SOURCE_MAX_Z_M,
            "normalization": "dominant source tabletop plane mapped to Isaac task surface Z",
        },
        "wrist_camera": {
            "source_parent": LIBERO_WRIST_CAMERA_SOURCE_PARENT,
            "source_local_translation_xyz": list(
                LIBERO_WRIST_CAMERA_LOCAL_TRANSLATION_XYZ
            ),
            "source_local_orientation_wxyz": list(
                LIBERO_WRIST_CAMERA_LOCAL_QUATERNION_WXYZ
            ),
            "isaac_parent_prim_path": ISAAC_WRIST_CAMERA_PARENT_PATH,
            "audited_camera_to_eef_offset_world_xyz": list(
                LIBERO_WRIST_CAMERA_AUDITED_EEF_OFFSET_WORLD_XYZ
            ),
            "isaac_clearance_compensation_world_xyz": list(
                ISAAC_WRIST_CAMERA_CLEARANCE_COMPENSATION_WORLD_XYZ
            ),
            "isaac_eef_offset_world_xyz": list(
                LIBERO_WRIST_CAMERA_EEF_OFFSET_WORLD_XYZ
            ),
            "isaac_world_orientation_wxyz": list(
                LIBERO_WRIST_CAMERA_WORLD_QUATERNION_WXYZ
            ),
            "vertical_fov_degrees": LIBERO_WRIST_CAMERA_FOVY_DEGREES,
            "clipping_range_m": list(LIBERO_WRIST_CAMERA_CLIPPING_RANGE_M),
            "camera_axes": "usd",
            "static_settle_updates_before_capture": LIBERO_WRIST_CAMERA_SETTLE_UPDATES,
            "source": "hf-libero canonical eye_in_hand camera transform",
            "mount_adapter": (
                "audited reference camera-to-EEF transform plus bounded Isaac "
                "mesh-clearance compensation, rigidly transported by measured "
                "EEF translation and orientation"
            ),
        },
        "physics": {
            "target_collision": "existing native Isaac dynamic cube proxy",
            "basket_collision": "five static box proxies",
            "distractors": "visual-only development geometry",
            "task_surface": "native ground collision plus large visual task surface",
            "task_surface_render_clearance_m": ISAAC_TASK_SURFACE_RENDER_CLEARANCE_M,
        },
    }


def task0_scene_sha256() -> str:
    encoded = json.dumps(
        task0_scene_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assets_root(root: Path) -> dict[str, str]:
    resolved = root.expanduser().resolve()
    hashes: dict[str, str] = {}
    for spec in (*VISUAL_ASSETS, LIBERO_TABLE_VISUAL_ASSET):
        for relative in (spec.obj_relative_path, *spec.auxiliary_relative_paths):
            path = resolved / relative
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError(f"missing non-empty canonical LIBERO asset: {path}")
            hashes[relative] = _sha256_file(path)
    return hashes


def _wait_for_async_task(simulation_app: Any, awaitable: Any, *, timeout_seconds: float) -> Any:
    task = asyncio.ensure_future(awaitable)
    deadline = time.monotonic() + timeout_seconds
    while not task.done():
        if not simulation_app.is_running():
            raise RuntimeError("Isaac stopped during LIBERO asset conversion")
        if time.monotonic() >= deadline:
            task.cancel()
            raise TimeoutError("LIBERO asset conversion timed out")
        simulation_app.update()
    return task.result()


class LiberoWristCamera:
    """A real panda-hand-mounted viewport camera at the audited LIBERO pose."""

    def __init__(
        self,
        simulation_app: Any,
        *,
        output_directory: Path,
        width: int,
        height: int,
        timeout_seconds: float,
    ) -> None:
        import omni.kit.app
        import omni.usd

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in (
            "omni.kit.renderer.capture",
            "omni.kit.viewport.utility",
        ):
            manager.set_extension_enabled_immediate(extension, True)
            simulation_app.update()
            if not manager.is_extension_enabled(extension):
                raise RuntimeError(
                    f"Isaac wrist-camera extension {extension!r} is unavailable"
                )

        from isaacsim.core.rendering_manager import RenderingManager
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from pxr import Gf, Sdf, UsdGeom

        self.frame_directory = output_directory
        self.frame_directory.mkdir(parents=True, exist_ok=False)
        self._timeout_seconds = float(timeout_seconds)
        self._frame_paths: list[Path] = []
        self._RenderingManager = RenderingManager
        self._capture_viewport_to_file = capture_viewport_to_file

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac has no active stage for the wrist camera")
        camera_path = f"{ISAAC_WRIST_CAMERA_PARENT_PATH}/LiberoWristCamera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        xformable = UsdGeom.Xformable(camera.GetPrim())
        aligned_eef = libero_to_isaac_position(LIBERO_REFERENCE_EEF_XYZ)
        initial_position = tuple(
            eef + offset
            for eef, offset in zip(
                aligned_eef, LIBERO_WRIST_CAMERA_EEF_OFFSET_WORLD_XYZ, strict=True
            )
        )
        self._translate_op = xformable.AddTranslateOp()
        self._translate_op.Set(Gf.Vec3d(*initial_position))
        wrist_quaternion = LIBERO_WRIST_CAMERA_WORLD_QUATERNION_WXYZ
        self._orient_op = xformable.AddOrientOp()
        self._orient_op.Set(
            Gf.Quatf(float(wrist_quaternion[0]), Gf.Vec3f(*wrist_quaternion[1:]))
        )
        horizontal_aperture = float(camera.GetHorizontalApertureAttr().Get())
        vertical_aperture = horizontal_aperture * float(height) / float(width)
        camera.GetVerticalApertureAttr().Set(vertical_aperture)
        focal_length = 0.5 * vertical_aperture / math.tan(
            math.radians(LIBERO_WRIST_CAMERA_FOVY_DEGREES) / 2.0
        )
        camera.GetFocalLengthAttr().Set(focal_length)
        camera.GetClippingRangeAttr().Set(
            Gf.Vec2f(*LIBERO_WRIST_CAMERA_CLIPPING_RANGE_M)
        )
        self._camera_path = Sdf.Path(camera_path)
        self._viewport = get_active_viewport()
        if self._viewport is None or self._viewport.stage is None:
            raise RuntimeError("Isaac has no active viewport for the wrist camera")
        for _ in range(LIBERO_WRIST_CAMERA_SETTLE_UPDATES):
            self._RenderingManager.render()
            simulation_app.update()

    def capture(
        self,
        simulation_app: Any,
        *,
        end_effector_xyz: Sequence[float],
        end_effector_wxyz: Sequence[float],
    ) -> Path:
        from pxr import Gf

        camera_position, camera_orientation = transported_wrist_camera_pose(
            end_effector_xyz=end_effector_xyz,
            end_effector_wxyz=end_effector_wxyz,
        )
        self._translate_op.Set(Gf.Vec3d(*camera_position))
        self._orient_op.Set(
            Gf.Quatf(
                float(camera_orientation[0]), Gf.Vec3f(*camera_orientation[1:])
            )
        )
        path = self.frame_directory / f"frame_{len(self._frame_paths):06d}.png"
        previous_camera_path = self._viewport.camera_path
        self._viewport.camera_path = self._camera_path
        try:
            for _ in range(LIBERO_WRIST_CAMERA_SETTLE_UPDATES):
                self._RenderingManager.render()
                simulation_app.update()
            helper = self._capture_viewport_to_file(
                self._viewport, file_path=str(path), is_hdr=False
            )
            task = asyncio.ensure_future(helper.wait_for_result(completion_frames=0))
            deadline = time.monotonic() + self._timeout_seconds
            while not task.done() or not path.is_file() or path.stat().st_size <= 0:
                if task.cancelled():
                    raise RuntimeError("Isaac wrist-camera capture was cancelled")
                if task.done() and task.exception() is not None:
                    raise RuntimeError(
                        "Isaac wrist-camera capture failed"
                    ) from task.exception()
                if not simulation_app.is_running():
                    raise RuntimeError("Isaac stopped during wrist-camera capture")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Isaac wrist-camera capture timed out")
                self._RenderingManager.render()
                simulation_app.update()
        finally:
            self._viewport.camera_path = previous_camera_path
            self._RenderingManager.render()
            simulation_app.update()
        self._frame_paths.append(path)
        return path

    def close(self) -> None:
        return None


class LiberoObjectTask0Scene:
    """Add canonical task visuals, task plane, and basket proxies to one scene."""

    def __init__(
        self,
        simulation_app: Any,
        scene: Any,
        *,
        assets_root: Path,
        cache_directory: Path,
        conversion_timeout_seconds: float = 180.0,
    ) -> None:
        import omni.kit.asset_converter
        import omni.usd
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.prims import GeomPrim
        from pxr import Gf, UsdGeom

        self._app = simulation_app
        self._scene = scene
        self._stage = omni.usd.get_context().get_stage()
        if self._stage is None:
            raise RuntimeError("Isaac has no active stage for LIBERO task decoration")
        self._assets_root = assets_root.expanduser().resolve()
        self.asset_source_sha256 = validate_assets_root(self._assets_root)
        cache = cache_directory.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("omni.kit.asset_converter", True)
        simulation_app.update()
        context = omni.kit.asset_converter.AssetConverterContext()
        context.ignore_materials = False
        context.ignore_cameras = True
        context.single_mesh = False
        context.smooth_normals = True
        context.use_meter_as_world_unit = True
        converter = omni.kit.asset_converter.get_instance()
        converted: dict[str, Path] = {}
        self.table_material_preparation: dict[str, Any] | None = None
        for spec in (*VISUAL_ASSETS, LIBERO_TABLE_VISUAL_ASSET):
            source = self._assets_root / spec.obj_relative_path
            fingerprint = self.asset_source_sha256[spec.obj_relative_path][:12]
            if spec == LIBERO_TABLE_VISUAL_ASSET:
                source, self.table_material_preparation = (
                    _prepare_table_converter_source(
                        assets_root=self._assets_root,
                        cache_directory=cache,
                    )
                )
                fingerprint += (
                    "_" + self.table_material_preparation["prepared_mtl_sha256"][:12]
                )
            output = cache / f"{spec.name}_{fingerprint}.usd"
            if not output.is_file() or output.stat().st_size <= 0:
                conversion = converter.create_converter_task(
                    str(source), str(output), lambda _progress, _total: None, context
                )
                success = _wait_for_async_task(
                    simulation_app,
                    conversion.wait_until_finished(),
                    timeout_seconds=conversion_timeout_seconds,
                )
                if not success or not output.is_file() or output.stat().st_size <= 0:
                    raise RuntimeError(
                        "failed to convert canonical LIBERO asset "
                        f"{source}: status={conversion.get_status()}, "
                        f"error={conversion.get_error_message()}"
                    )
            converted[spec.name] = output
        self.converted_asset_sha256 = {
            name: _sha256_file(path) for name, path in converted.items()
        }

        # Hide the old colored smoke-only visualization. Collision APIs remain
        # active, while the learned policy sees the canonical target asset.
        for path in (
            "/World/DynamicObject",
            "/World/ZoneA",
            "/World/ZoneB",
            "/World/ActiveDestination",
        ):
            prim = self._stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()

        table_top_z = task0_scene_payload()["table_top_z_m"]
        table_position = list(
            libero_to_isaac_position(LIBERO_TABLE_VISUAL_SOURCE_POSITION_XYZ)
        )
        table_position[2] = (
            table_top_z
            + ISAAC_TASK_SURFACE_RENDER_CLEARANCE_M
            - LIBERO_TABLE_VISUAL_ASSET.scale
            * LIBERO_TABLE_VISUAL_SOURCE_TABLETOP_Z_M
        )
        table_root = self._stage.DefinePrim(
            "/World/LiberoTask0/TableVisual", "Xform"
        )
        table_xform = UsdGeom.Xformable(table_root)
        table_xform.AddTranslateOp().Set(Gf.Vec3d(*table_position))
        table_quaternion = LIBERO_TABLE_VISUAL_SOURCE_ORIENTATION_WXYZ
        table_xform.AddOrientOp().Set(
            Gf.Quatf(float(table_quaternion[0]), Gf.Vec3f(*table_quaternion[1:]))
        )
        table_xform.AddScaleOp().Set(
            Gf.Vec3f(*([LIBERO_TABLE_VISUAL_ASSET.scale] * 3))
        )
        table_asset = self._stage.DefinePrim(
            "/World/LiberoTask0/TableVisual/Asset", "Xform"
        )
        table_asset.GetReferences().AddReference(
            str(converted[LIBERO_TABLE_VISUAL_ASSET.name])
        )

        # The canonical basket mesh is visual. Five simple static colliders give
        # Isaac real containment physics without pretending to be a converted
        # MuJoCo collision model.
        basket_xyz = libero_to_isaac_position(LIBERO_BODY_POSES["basket"][0])
        basket_boxes = (
            ("Base", (basket_xyz[0], basket_xyz[1], table_top_z + 0.006), (0.15, 0.15, 0.012)),
            ("Left", (basket_xyz[0], basket_xyz[1] - 0.071, table_top_z + 0.075), (0.15, 0.012, 0.15)),
            ("Right", (basket_xyz[0], basket_xyz[1] + 0.071, table_top_z + 0.075), (0.15, 0.012, 0.15)),
            ("Front", (basket_xyz[0] - 0.071, basket_xyz[1], table_top_z + 0.075), (0.012, 0.15, 0.15)),
            ("Back", (basket_xyz[0] + 0.071, basket_xyz[1], table_top_z + 0.075), (0.012, 0.15, 0.15)),
        )
        for name, position, scale in basket_boxes:
            proxy = Cube(
                paths=f"/World/LiberoTask0/BasketCollision{name}",
                positions=position,
                sizes=1.0,
                scales=scale,
                colors=[0.42, 0.31, 0.18],
            )
            GeomPrim(paths=proxy.paths, apply_collision_apis=True)

        self._visual_ops: dict[str, tuple[Any, Any]] = {}
        asset_by_name = {spec.name: spec for spec in VISUAL_ASSETS}
        for name, (position, quaternion) in LIBERO_BODY_POSES.items():
            root_path = f"/World/LiberoTask0/Visuals/{name}"
            prim = self._stage.DefinePrim(root_path, "Xform")
            xformable = UsdGeom.Xformable(prim)
            translate_op = xformable.AddTranslateOp()
            orient_op = xformable.AddOrientOp()
            xformable.AddScaleOp().Set(Gf.Vec3f(*([asset_by_name[name].scale] * 3)))
            translate_op.Set(Gf.Vec3d(*libero_to_isaac_position(position)))
            orient_op.Set(Gf.Quatf(float(quaternion[0]), Gf.Vec3f(*quaternion[1:])))
            # Converter-authored USD can carry xform ops on its default prim.
            # Keep that reference on a child so it cannot collide with the
            # audited world-pose ops owned by this parent.
            asset_prim = self._stage.DefinePrim(f"{root_path}/Asset", "Xform")
            asset_prim.GetReferences().AddReference(str(converted[name]))
            self._visual_ops[name] = (translate_op, orient_op)

        target_position, target_quaternion = LIBERO_BODY_POSES["alphabet_soup"]
        scene.set_object_pose(
            position_xyz=libero_to_isaac_position(target_position),
            orientation_wxyz=target_quaternion,
            settle_steps=12,
        )
        settled_object = scene.measure()
        self._target_visual_z_offset = (
            libero_to_isaac_position(target_position)[2]
            - float(settled_object.object_xyz[2])
        )

        # The two Panda assets use the same base/world axes but different
        # default joint homes. Move Isaac to the canonical LIBERO EEF home so
        # policy state, policy action, task geometry, and camera share one
        # base-anchored coordinate map.
        aligned_home = libero_to_isaac_position(LIBERO_REFERENCE_EEF_XYZ)
        scene.set_command(
            (*aligned_home, *LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ, 1.0)
        )
        for _ in range(120):
            scene.step()
        aligned_measurement = scene.measure()
        home_error = math.dist(aligned_measurement.end_effector_xyz, aligned_home)
        if home_error > 0.03:
            raise RuntimeError(
                f"Isaac failed LIBERO home alignment: position error {home_error:.6f} m"
            )
        self.aligned_home_measurement = {
            "command_xyz": list(aligned_home),
            "command_axis_angle_xyz": list(LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ),
            "measured_xyz": list(aligned_measurement.end_effector_xyz),
            "measured_orientation_wxyz": list(
                aligned_measurement.end_effector_wxyz
            ),
            "position_error_m": home_error,
        }
        self.sync(aligned_measurement)
        for _ in range(4):
            simulation_app.update()

        self.provenance = {
            "task_scene": task0_scene_payload(),
            "task_scene_sha256": task0_scene_sha256(),
            "assets_root": str(self._assets_root),
            "asset_source_sha256": self.asset_source_sha256,
            "converted_asset_sha256": self.converted_asset_sha256,
            "converted_asset_cache": str(cache),
            "table_material_preparation": self.table_material_preparation,
            "table_visual_transform": {
                "position_xyz": table_position,
                "orientation_wxyz": list(
                    LIBERO_TABLE_VISUAL_SOURCE_ORIENTATION_WXYZ
                ),
                "scale": LIBERO_TABLE_VISUAL_ASSET.scale,
            },
            "aligned_home_measurement": self.aligned_home_measurement,
        }

    def sync(self, measurement: Any) -> None:
        from pxr import Gf

        translate_op, orient_op = self._visual_ops["alphabet_soup"]
        visual_position = tuple(float(value) for value in measurement.object_xyz)
        translate_op.Set(
            Gf.Vec3d(
                visual_position[0],
                visual_position[1],
                visual_position[2] + self._target_visual_z_offset,
            )
        )
        orient_op.Set(
            Gf.Quatf(
                float(measurement.object_wxyz[0]),
                Gf.Vec3f(*measurement.object_wxyz[1:]),
            )
        )

    def official_object_states(self, measurement: Any) -> dict[str, Any]:
        """Map the current Isaac target visual back to its official free joint."""

        visual_isaac_position = (
            float(measurement.object_xyz[0]),
            float(measurement.object_xyz[1]),
            float(measurement.object_xyz[2]) + self._target_visual_z_offset,
        )
        return {
            "alphabet_soup_1": {
                "position_xyz": list(isaac_to_libero_position(visual_isaac_position)),
                "orientation_wxyz": [
                    float(component) for component in measurement.object_wxyz
                ],
            }
        }

    @staticmethod
    def task_success(measurement: Any) -> bool:
        basket = libero_to_isaac_position(LIBERO_BODY_POSES["basket"][0])
        x, y, z = (float(value) for value in measurement.object_xyz)
        table_top = task0_scene_payload()["table_top_z_m"]
        return (
            abs(x - basket[0]) <= 0.06108
            and abs(y - basket[1]) <= 0.06108
            and table_top <= z <= table_top + 0.16
        )
