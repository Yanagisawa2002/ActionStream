from __future__ import annotations

import pytest

from actionstream.isaac_libero_proxy_tasks import (
    LIBERO_PROXY_TASK_SPECS,
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
