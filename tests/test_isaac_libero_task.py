from __future__ import annotations

import math

import pytest

from actionstream.isaac_learned import (
    LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ,
    LIBERO_REFERENCE_EEF_WORLD_QUATERNION_WXYZ,
    LIBERO_REFERENCE_EEF_XYZ,
)
from actionstream.isaac_libero_task import (
    ISAAC_TASK_SURFACE_RENDER_CLEARANCE_M,
    ISAAC_WRIST_CAMERA_CLEARANCE_COMPENSATION_WORLD_XYZ,
    LIBERO_AGENTVIEW_FOVY_DEGREES,
    LIBERO_BODY_POSES,
    LIBERO_TABLE_Z_M,
    LIBERO_TABLE_VISUAL_ASSET,
    LIBERO_TABLE_VISUAL_SOURCE_MAX_Z_M,
    LIBERO_TABLE_VISUAL_SOURCE_ORIENTATION_WXYZ,
    LIBERO_TABLE_VISUAL_SOURCE_POSITION_XYZ,
    LIBERO_TABLE_VISUAL_SOURCE_TABLETOP_Z_M,
    LIBERO_TO_ISAAC_TASK_TRANSLATION_XYZ,
    LIBERO_WRIST_CAMERA_FOVY_DEGREES,
    LIBERO_WRIST_CAMERA_AUDITED_EEF_OFFSET_WORLD_XYZ,
    LIBERO_WRIST_CAMERA_CLIPPING_RANGE_M,
    LIBERO_WRIST_CAMERA_EEF_OFFSET_WORLD_XYZ,
    LIBERO_WRIST_CAMERA_LOCAL_QUATERNION_WXYZ,
    LIBERO_WRIST_CAMERA_LOCAL_TRANSLATION_XYZ,
    LIBERO_WRIST_CAMERA_SETTLE_UPDATES,
    LIBERO_WRIST_CAMERA_WORLD_QUATERNION_WXYZ,
    VISUAL_ASSETS,
    _diffuse_only_table_mtl,
    isaac_agentview_pose,
    isaac_to_libero_position,
    libero_to_isaac_position,
    task0_scene_payload,
    task0_scene_sha256,
    transported_wrist_camera_pose,
    validate_assets_root,
)


def test_libero_task_geometry_uses_robot_base_anchored_translation() -> None:
    mapped_origin = libero_to_isaac_position((0.0, 0.0, LIBERO_TABLE_Z_M))
    assert mapped_origin == pytest.approx(LIBERO_TO_ISAAC_TASK_TRANSLATION_XYZ)
    assert mapped_origin == pytest.approx((0.6, 0.0, 0.0))

    target = libero_to_isaac_position(LIBERO_BODY_POSES["alphabet_soup"][0])
    assert 0.25 <= target[0] <= 0.70
    assert -0.35 <= target[1] <= 0.35
    assert target[2] > mapped_origin[2]

    mapped_home = libero_to_isaac_position(LIBERO_REFERENCE_EEF_XYZ)
    assert mapped_home == pytest.approx((0.4474475362, -0.00627673, 0.2484475446))
    assert isaac_to_libero_position(mapped_home) == pytest.approx(
        LIBERO_REFERENCE_EEF_XYZ
    )


def test_agentview_preserves_audited_forward_direction_and_fov() -> None:
    position, target = isaac_agentview_pose()
    direction = tuple(target[index] - position[index] for index in range(3))
    assert math.sqrt(sum(value * value for value in direction)) == pytest.approx(1.0)
    assert direction[0] < 0.0
    assert direction[2] < 0.0
    assert LIBERO_AGENTVIEW_FOVY_DEGREES == 45.0


def test_scene_payload_covers_every_visual_asset() -> None:
    names = {spec.name for spec in VISUAL_ASSETS}
    assert names == set(LIBERO_BODY_POSES)
    payload = task0_scene_payload()
    assert set(payload["body_poses"]) == names
    assert payload["status"] == "development_only_not_holdout_frozen"
    assert payload["isaac_aligned_home_eef_axis_angle_xyz"] == pytest.approx(
        LIBERO_REFERENCE_EEF_AXIS_ANGLE_XYZ
    )
    assert len(task0_scene_sha256()) == 64


def test_table_visual_preserves_official_living_room_asset_contract() -> None:
    payload = task0_scene_payload()["table_visual"]
    assert LIBERO_TABLE_VISUAL_ASSET.obj_relative_path == (
        "scenes/living_room_table/living_room_table.obj"
    )
    assert LIBERO_TABLE_VISUAL_ASSET.scale == pytest.approx(1.5)
    assert LIBERO_TABLE_VISUAL_SOURCE_POSITION_XYZ == pytest.approx((-0.25, 0.25, 0.0))
    assert LIBERO_TABLE_VISUAL_SOURCE_ORIENTATION_WXYZ == pytest.approx(
        (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    )
    assert LIBERO_TABLE_VISUAL_SOURCE_MAX_Z_M == pytest.approx(0.299668)
    assert LIBERO_TABLE_VISUAL_SOURCE_TABLETOP_Z_M == pytest.approx(0.267306)
    assert payload["source_asset"] == LIBERO_TABLE_VISUAL_ASSET.obj_relative_path
    assert payload["normalization"] == (
        "dominant source tabletop plane mapped to Isaac task surface Z"
    )


def test_missing_table_bump_map_is_removed_without_changing_diffuse_map() -> None:
    sanitized, removed = _diffuse_only_table_mtl(
        "newmtl table\nmap_Kd living_room_table_texture.png\n"
        "map_Bump -bm 1.000 missing.jpg\n"
    )
    assert sanitized == "newmtl table\nmap_Kd living_room_table_texture.png\n"
    assert removed == ("map_Bump -bm 1.000 missing.jpg",)


def test_wrist_camera_contract_is_normalized_and_hand_local() -> None:
    assert LIBERO_WRIST_CAMERA_LOCAL_TRANSLATION_XYZ == pytest.approx((0.05, 0.0, 0.0))
    assert math.sqrt(
        sum(value * value for value in LIBERO_WRIST_CAMERA_LOCAL_QUATERNION_WXYZ)
    ) == pytest.approx(1.0)
    assert LIBERO_WRIST_CAMERA_AUDITED_EEF_OFFSET_WORLD_XYZ == pytest.approx(
        (0.0555521776, -0.0000801410, 0.0938939275)
    )
    assert ISAAC_WRIST_CAMERA_CLEARANCE_COMPENSATION_WORLD_XYZ == pytest.approx(
        (0.08, 0.0, 0.08)
    )
    assert LIBERO_WRIST_CAMERA_EEF_OFFSET_WORLD_XYZ == pytest.approx(
        (0.1355521776, -0.0000801410, 0.1738939275)
    )
    assert math.sqrt(
        sum(value * value for value in LIBERO_WRIST_CAMERA_WORLD_QUATERNION_WXYZ)
    ) == pytest.approx(1.0)
    assert LIBERO_WRIST_CAMERA_FOVY_DEGREES == 75.0
    assert LIBERO_WRIST_CAMERA_CLIPPING_RANGE_M == pytest.approx((0.01, 100.0))
    payload = task0_scene_payload()["wrist_camera"]
    assert payload["camera_axes"] == "usd"
    assert payload["isaac_parent_prim_path"] == "/World"
    assert payload["audited_camera_to_eef_offset_world_xyz"] == pytest.approx(
        (0.0555521776, -0.0000801410, 0.0938939275)
    )
    assert payload["isaac_clearance_compensation_world_xyz"] == pytest.approx(
        (0.08, 0.0, 0.08)
    )
    assert payload["static_settle_updates_before_capture"] == 4
    assert LIBERO_WRIST_CAMERA_SETTLE_UPDATES == 4
    assert task0_scene_payload()["physics"][
        "task_surface_render_clearance_m"
    ] == pytest.approx(ISAAC_TASK_SURFACE_RENDER_CLEARANCE_M)


def test_wrist_camera_reference_pose_is_preserved_by_rigid_transport() -> None:
    eef = libero_to_isaac_position(LIBERO_REFERENCE_EEF_XYZ)
    position, orientation = transported_wrist_camera_pose(
        end_effector_xyz=eef,
        end_effector_wxyz=LIBERO_REFERENCE_EEF_WORLD_QUATERNION_WXYZ,
    )
    assert position == pytest.approx(
        tuple(
            value + offset
            for value, offset in zip(
                eef, LIBERO_WRIST_CAMERA_EEF_OFFSET_WORLD_XYZ, strict=True
            )
        )
    )
    assert orientation == pytest.approx(
        LIBERO_WRIST_CAMERA_WORLD_QUATERNION_WXYZ
    )


def test_asset_validation_fails_closed(tmp_path) -> None:
    with pytest.raises(ValueError, match="canonical LIBERO asset"):
        validate_assets_root(tmp_path)
