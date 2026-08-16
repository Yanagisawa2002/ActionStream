from __future__ import annotations

import numpy as np
import pytest

from actionstream.isaac_libero_proxy_tasks import (
    LIBERO_PROXY_TASK_SPECS,
    _rotation_matrix_wxyz,
    proxy_task_spec,
    proxy_task_specs_sha256,
)


def test_proxy_tasks_are_distinct_suites_and_task_families() -> None:
    spatial = proxy_task_spec("spatial2")
    goal = proxy_task_spec("goal5")
    assert spatial.suite == "libero_spatial"
    assert goal.suite == "libero_goal"
    assert spatial.task_family != goal.task_family
    assert spatial.official_object_name == "akita_black_bowl_1"
    assert goal.official_object_name == "plate_1"
    assert len(proxy_task_specs_sha256()) == 64


@pytest.mark.parametrize("key", sorted(LIBERO_PROXY_TASK_SPECS))
def test_proxy_coordinate_mapping_round_trips_and_places_home_in_workspace(key: str) -> None:
    spec = proxy_task_spec(key)
    mapped = spec.map_position(spec.official_object_position_xyz)
    assert spec.unmap_position(mapped) == pytest.approx(
        spec.official_object_position_xyz
    )
    home = spec.map_position(spec.official_reference_eef_xyz)
    assert 0.25 <= home[0] <= 0.85
    assert -0.35 <= home[1] <= 0.35
    assert 0.02 <= home[2] <= 0.60


def test_goal5_region_maps_to_a_positive_native_rectangle() -> None:
    spec = proxy_task_spec("goal5")
    assert spec.official_goal_region_xyxy is not None
    x0, y0, x1, y1 = spec.official_goal_region_xyxy
    native_min = spec.map_position((x0, y0, 0.90))
    native_max = spec.map_position((x1, y1, 0.90))
    assert native_min == pytest.approx((0.57, 0.17, 0.0))
    assert native_max == pytest.approx((0.65, 0.25, 0.0))


def test_goal5_proxy_uses_the_frozen_official_plate_collision_envelope() -> None:
    spec = proxy_task_spec("goal5")
    assert spec.collision_scale_xyz == pytest.approx(
        (0.13762122220335729, 0.1374699120165223, 0.01894636973086458)
    )
    mapped_root = spec.map_position(spec.official_object_position_xyz)
    proxy_center = tuple(
        root + offset
        for root, offset in zip(
            mapped_root, spec.reference_root_to_proxy_world_xyz, strict=True
        )
    )
    assert proxy_center == pytest.approx(
        (0.7115115904151907, -0.027675710927846743, 0.00946633852904651)
    )
    assert "ten box collision geoms" in spec.collision_proxy_source


def test_spatial2_proxy_is_an_explicit_graspable_rim_contact_patch() -> None:
    spec = proxy_task_spec("spatial2")
    assert spec.collision_scale_xyz == pytest.approx((0.024, 0.020, 0.040))
    mapped_root = spec.map_position(spec.official_object_position_xyz)
    proxy_center = tuple(
        root + offset
        for root, offset in zip(
            mapped_root, spec.reference_root_to_proxy_world_xyz, strict=True
        )
    )
    assert proxy_center == pytest.approx(
        (0.5910084, 0.0539612, 0.0210071), abs=1e-7
    )
    rotation = _rotation_matrix_wxyz(spec.official_object_orientation_wxyz)
    proxy_to_root_local = rotation.T @ (
        np.asarray(mapped_root) - np.asarray(proxy_center)
    )
    reconstructed = np.asarray(proxy_center) + rotation @ proxy_to_root_local
    assert reconstructed == pytest.approx(mapped_root)
