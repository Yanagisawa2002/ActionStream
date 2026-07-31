from __future__ import annotations

import inspect
import os
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from actionstream.m5_scene import (
    DeterministicPlanarShift,
    EntityPose,
    LiberoSceneAdapter,
    OracleGroundTruthPoseDetector,
    PlanarBounds,
    SceneChangeDetector,
    TASK0_SOURCE_ENTITY,
    TASK0_TARGET_ENTITY,
)


class FakeEntity:
    def __init__(self, name: str, body_id: int, radius: float = 0.02) -> None:
        self.name = name
        self.root_body = f"{name}_body"
        self.joints = [f"{name}_free_joint"]
        self.contact_geoms = [f"{name}_collision"]
        self.body_id = body_id
        self.horizontal_radius = radius


class FakeModel:
    def __init__(self, entities: dict[str, FakeEntity]) -> None:
        self._body_ids = {
            entity.root_body: entity.body_id for entity in entities.values()
        }
        self._body_ids["robot_base"] = len(entities)

    def body_name2id(self, name: str) -> int:
        return self._body_ids[name]


class FakeData:
    def __init__(self, entities: dict[str, FakeEntity]) -> None:
        self._qpos = {
            TASK0_TARGET_ENTITY: np.array([0.10, 0.20, 0.45, 1.0, 0.0, 0.0, 0.0]),
            TASK0_SOURCE_ENTITY: np.array([-0.10, 0.00, 0.44, 1.0, 0.0, 0.0, 0.0]),
        }
        self._qvel = {
            TASK0_TARGET_ENTITY: np.arange(1.0, 7.0),
            TASK0_SOURCE_ENTITY: np.zeros(6),
        }
        self._joint_to_entity = {
            entity.joints[0]: name for name, entity in entities.items()
        }
        self.body_xpos = np.zeros((len(entities) + 1, 3), dtype=np.float64)
        self.body_xpos[len(entities)] = (-0.6, 0.0, 0.0)
        self.body_xquat = np.zeros((len(entities) + 1, 4), dtype=np.float64)
        self.contact = []
        self.ncon = 0

    def get_joint_qpos(self, joint_name: str) -> np.ndarray:
        return self._qpos[self._joint_to_entity[joint_name]]

    def set_joint_qpos(self, joint_name: str, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (7,):
            raise ValueError("free-joint qpos must be 7D")
        self._qpos[self._joint_to_entity[joint_name]][:] = value

    def get_joint_qvel(self, joint_name: str) -> np.ndarray:
        return self._qvel[self._joint_to_entity[joint_name]]

    def set_joint_qvel(self, joint_name: str, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (6,):
            raise ValueError("free-joint qvel must be 6D")
        self._qvel[self._joint_to_entity[joint_name]][:] = value


class FakeSim:
    def __init__(
        self,
        entities: dict[str, FakeEntity],
        *,
        physical_displacement_scale: float = 1.0,
    ) -> None:
        self.entities = entities
        self.model = FakeModel(entities)
        self.data = FakeData(entities)
        self.forward_calls = 0
        self.physical_displacement_scale = physical_displacement_scale
        self._initial_positions = {
            name: self.data._qpos[name][:3].copy() for name in entities
        }
        self.forward()

    def forward(self) -> None:
        self.forward_calls += 1
        for name, entity in self.entities.items():
            qpos = self.data._qpos[name]
            initial = self._initial_positions[name]
            physical_position = initial + self.physical_displacement_scale * (
                qpos[:3] - initial
            )
            self.data.body_xpos[entity.body_id] = physical_position
            self.data.body_xquat[entity.body_id] = qpos[3:]


class FakeTaskEnv:
    workspace_offset = np.array([0.0, 0.0, 0.41])
    living_room_table_full_size = np.array([0.70, 1.60, 0.024])

    def __init__(
        self,
        *,
        physical_displacement_scale: float = 1.0,
        contacts: set[str] | None = None,
        fail_observation: bool = False,
    ) -> None:
        self.entities = {
            TASK0_TARGET_ENTITY: FakeEntity(TASK0_TARGET_ENTITY, 0),
            TASK0_SOURCE_ENTITY: FakeEntity(TASK0_SOURCE_ENTITY, 1),
        }
        self.sim = FakeSim(
            self.entities,
            physical_displacement_scale=physical_displacement_scale,
        )
        self.obj_body_id = {
            name: entity.body_id for name, entity in self.entities.items()
        }
        self.contacts = set(contacts or ())
        self.fail_observation = fail_observation
        self.robot_model = SimpleNamespace(
            name="robot",
            root_body="robot_base",
            contact_geoms=["robot_collision"],
        )
        self.robots = [SimpleNamespace(robot_model=self.robot_model)]
        self.contact_pairs: set[frozenset[str]] = set()

    def get_object(self, entity_name: str) -> FakeEntity | None:
        return self.entities.get(entity_name)

    def get_contacts(self, entity: FakeEntity) -> set[str]:
        return set(self.contacts)

    @staticmethod
    def _contact_name(value: object) -> str:
        return str(getattr(value, "name"))

    def check_contact(self, first: object, second: object) -> bool:
        pair = frozenset((self._contact_name(first), self._contact_name(second)))
        return pair in self.contact_pairs

    def _get_observations(self) -> dict[str, np.ndarray]:
        if self.fail_observation:
            raise RuntimeError("renderer unavailable")
        x_value = self.sim.data.body_xpos[0, 0]
        return {
            "agentview_image": np.full((2, 2, 3), x_value, dtype=np.float32),
        }


class FakeOffscreenEnv:
    def __init__(self, task_env: FakeTaskEnv) -> None:
        self.env = task_env
        self.post_process_calls = 0
        self.update_calls = 0

    def _post_process(self) -> None:
        self.post_process_calls += 1

    def _update_observables(self, *, force: bool = False) -> None:
        assert force
        self.update_calls += 1


class FakeSubEnv:
    def __init__(self, task_env: FakeTaskEnv) -> None:
        self._env = FakeOffscreenEnv(task_env)

    @staticmethod
    def _format_raw_obs(raw: dict[str, np.ndarray]) -> dict[str, object]:
        return {"pixels": {"image": raw["agentview_image"]}}


class FakeBackend:
    def __init__(self, task_env: FakeTaskEnv) -> None:
        self.sub_env = FakeSubEnv(task_env)

    def _sub_env(self, task_id: int) -> FakeSubEnv:
        if task_id != 0:
            raise ValueError("Task was not configured")
        return self.sub_env


def make_adapter(**task_kwargs: object) -> tuple[LiberoSceneAdapter, FakeTaskEnv]:
    task_env = FakeTaskEnv(**task_kwargs)
    return LiberoSceneAdapter(FakeBackend(task_env)), task_env


def test_entity_pose_is_immutable_normalized_and_json_compatible() -> None:
    pose = EntityPose("basket_1", (1, 2, 3), (2, 0, 0, 0))
    assert pose.position == (1.0, 2.0, 3.0)
    assert pose.quaternion == (1.0, 0.0, 0.0, 0.0)
    assert EntityPose.from_mapping(pose.to_dict()) == pose
    with pytest.raises(FrozenInstanceError):
        pose.position = (0.0, 0.0, 0.0)  # type: ignore[misc]


def test_oracle_detector_has_fixed_translation_trigger_and_logs_rotation() -> None:
    detector = OracleGroundTruthPoseDetector()
    reference = EntityPose("basket_1", (0, 0, 0), (1, 0, 0, 0))

    below = detector.detect(
        reference,
        EntityPose("basket_1", (0.009, 0, 0), (0, 0, 0, 1)),
    )
    assert not below.changed
    assert below.rotation_delta_rad == pytest.approx(np.pi)

    at_threshold = detector.detect(
        reference,
        EntityPose("basket_1", (0.010, 0, 0), (-1, 0, 0, 0)),
    )
    assert at_threshold.changed
    assert at_threshold.translation_delta_m == pytest.approx(0.010)
    # Quaternion sign ambiguity must not look like rotation.
    assert at_threshold.rotation_delta_rad == pytest.approx(0.0)


def test_detector_contract_is_pose_only() -> None:
    detector = OracleGroundTruthPoseDetector()
    assert isinstance(detector, SceneChangeDetector)
    parameters = inspect.signature(detector.detect).parameters
    assert tuple(parameters) == ("reference_pose", "current_pose")


def test_adapter_reads_entity_source_lift_and_contacts() -> None:
    adapter, task_env = make_adapter()
    source_reference = adapter.read_entity_pose(0, TASK0_SOURCE_ENTITY)
    source_joint = task_env.entities[TASK0_SOURCE_ENTITY].joints[0]
    lifted_qpos = task_env.sim.data.get_joint_qpos(source_joint).copy()
    lifted_qpos[2] += 0.025
    task_env.sim.data.set_joint_qpos(source_joint, lifted_qpos)
    task_env.sim.forward()

    lift = adapter.read_source_pose_and_lift(
        0,
        TASK0_SOURCE_ENTITY,
        reference_pose=source_reference,
    )
    assert lift.lift_m == pytest.approx(0.025)
    assert lift.is_lifted(0.020)

    task_env.contact_pairs.add(frozenset(("robot", TASK0_TARGET_ENTITY)))
    task_env.contact_pairs.add(frozenset((TASK0_SOURCE_ENTITY, TASK0_TARGET_ENTITY)))
    assert adapter.robot_contacts_entity(0, TASK0_TARGET_ENTITY)
    assert adapter.entities_in_contact(0, TASK0_SOURCE_ENTITY, TASK0_TARGET_ENTITY)


def test_planar_shift_is_exact_physical_and_returns_fresh_batch() -> None:
    adapter, task_env = make_adapter(contacts={"living_room_table_top"})
    initial = adapter.read_entity_pose(0, TASK0_TARGET_ENTITY)
    shift = DeterministicPlanarShift.along_axis(
        TASK0_TARGET_ENTITY,
        0.050,
    )
    result = shift.apply(adapter, 0)

    assert result.valid
    assert result.invalid_reason is None
    assert result.requested_displacement_xy_m == pytest.approx((-0.050, 0.0))
    assert result.achieved_displacement_xyz_m == pytest.approx((-0.050, 0.0, 0.0))
    assert result.displacement_error_m == pytest.approx(0.0, abs=1e-12)
    assert result.final_pose is not None
    assert result.final_pose.position[2] == initial.position[2]
    assert result.final_pose.quaternion == initial.quaternion

    target_joint = task_env.entities[TASK0_TARGET_ENTITY].joints[0]
    np.testing.assert_array_equal(
        task_env.sim.data.get_joint_qvel(target_joint),
        np.zeros(6),
    )
    assert result.fresh_observation is not None
    assert result.fresh_observation["pixels"]["image"].shape == (1, 2, 2, 3)
    assert not result.fresh_observation["pixels"]["image"].flags.writeable
    assert task_env.sim.forward_calls >= 2
    assert adapter.backend.sub_env._env.post_process_calls == 1
    assert adapter.backend.sub_env._env.update_calls == 1


def test_fixed_shift_is_deterministic_from_identical_states() -> None:
    shift = DeterministicPlanarShift.along_axis(TASK0_TARGET_ENTITY, 0.070)
    first_adapter, _ = make_adapter()
    second_adapter, _ = make_adapter()
    first = shift.apply(first_adapter, 0)
    second = shift.apply(second_adapter, 0)
    assert first.valid and second.valid
    assert first.to_dict() == second.to_dict()


def test_out_of_workspace_shift_is_explicit_and_does_not_mutate() -> None:
    adapter, task_env = make_adapter()
    before = adapter.read_entity_pose(0, TASK0_TARGET_ENTITY)
    forward_calls = task_env.sim.forward_calls
    result = adapter.shift_entity_planar(
        0,
        TASK0_TARGET_ENTITY,
        (-0.050, 0.0),
        workspace_bounds=PlanarBounds(0.09, 0.20, 0.10, 0.30),
        reachability_bounds=PlanarBounds(-1.0, 1.0, -1.0, 1.0),
    )

    assert not result.valid
    assert result.invalid_reason == "workspace_bounds"
    assert not result.rolled_back
    assert result.final_pose == before
    assert adapter.read_entity_pose(0, TASK0_TARGET_ENTITY) == before
    assert task_env.sim.forward_calls == forward_calls


def test_reachability_failure_is_reported_separately() -> None:
    adapter, _ = make_adapter()
    result = adapter.shift_entity_planar(
        0,
        TASK0_TARGET_ENTITY,
        (-0.050, 0.0),
        workspace_bounds=PlanarBounds(-1.0, 1.0, -1.0, 1.0),
        reachability_bounds=PlanarBounds(0.09, 0.20, 0.10, 0.30),
    )
    assert not result.valid
    assert result.invalid_reason == "reachability_bounds"


def test_robot_base_distance_reachability_is_enforced() -> None:
    task_env = FakeTaskEnv()
    adapter = LiberoSceneAdapter(
        FakeBackend(task_env),
        maximum_robot_base_distance_m=0.5,
    )
    before = adapter.read_entity_pose(0, TASK0_TARGET_ENTITY)
    result = adapter.shift_entity_planar(
        0,
        TASK0_TARGET_ENTITY,
        (-0.050, 0.0),
        workspace_bounds=PlanarBounds(-1.0, 1.0, -1.0, 1.0),
        reachability_bounds=PlanarBounds(-1.0, 1.0, -1.0, 1.0),
    )
    assert not result.valid
    assert result.invalid_reason == "reachability_distance"
    assert adapter.read_entity_pose(0, TASK0_TARGET_ENTITY) == before


def test_non_floor_collision_is_invalid_and_rolls_back_exact_state() -> None:
    adapter, task_env = make_adapter(
        contacts={"living_room_table_top", "alphabet_soup_1_collision"}
    )
    target_joint = task_env.entities[TASK0_TARGET_ENTITY].joints[0]
    original_qpos = task_env.sim.data.get_joint_qpos(target_joint).copy()
    original_qvel = task_env.sim.data.get_joint_qvel(target_joint).copy()
    initial_pose = adapter.read_entity_pose(0, TASK0_TARGET_ENTITY)

    result = adapter.shift_entity_planar(0, TASK0_TARGET_ENTITY, (-0.030, 0.0))

    assert not result.valid
    assert result.invalid_reason == "non_floor_collision"
    assert result.non_floor_contacts == ("alphabet_soup_1_collision",)
    assert result.rolled_back
    assert result.initial_pose == initial_pose
    assert result.attempted_pose != initial_pose
    assert result.final_pose == initial_pose
    np.testing.assert_array_equal(
        task_env.sim.data.get_joint_qpos(target_joint),
        original_qpos,
    )
    np.testing.assert_array_equal(
        task_env.sim.data.get_joint_qvel(target_joint),
        original_qvel,
    )


def test_displacement_mismatch_reports_error_and_rolls_back() -> None:
    adapter, _ = make_adapter(physical_displacement_scale=0.5)
    initial_pose = adapter.read_entity_pose(0, TASK0_TARGET_ENTITY)

    result = adapter.shift_entity_planar(0, TASK0_TARGET_ENTITY, (-0.030, 0.0))

    assert not result.valid
    assert result.invalid_reason == "displacement_mismatch"
    assert result.achieved_displacement_xyz_m == pytest.approx((-0.015, 0.0, 0.0))
    assert result.displacement_error_m == pytest.approx(0.015)
    assert result.rolled_back
    assert adapter.read_entity_pose(0, TASK0_TARGET_ENTITY) == initial_pose


def test_fresh_observation_failure_is_invalid_and_rolls_back() -> None:
    adapter, _ = make_adapter(fail_observation=True)
    initial_pose = adapter.read_entity_pose(0, TASK0_TARGET_ENTITY)
    result = adapter.shift_entity_planar(0, TASK0_TARGET_ENTITY, (-0.030, 0.0))
    assert not result.valid
    assert result.invalid_reason == "fresh_observation_failed"
    assert "renderer unavailable" in (result.invalid_detail or "")
    assert result.rolled_back
    assert adapter.read_entity_pose(0, TASK0_TARGET_ENTITY) == initial_pose


@pytest.mark.skipif(
    os.environ.get("ACTIONSTREAM_RUN_LIBERO_SCENE_INTEGRATION") != "1",
    reason="Set ACTIONSTREAM_RUN_LIBERO_SCENE_INTEGRATION=1 for the local no-policy LIBERO check",
)
def test_local_libero_scene_integration_without_policy_loading() -> None:
    """Optional physical smoke: construct only the LIBERO env and move basket_1."""

    from actionstream.libero_config import ensure_isolated_libero_config
    from lerobot.envs import make_env, make_env_config

    ensure_isolated_libero_config()
    config = make_env_config(
        "libero",
        task="libero_object",
        task_ids=[0],
        control_mode="absolute",
        episode_length=20,
        max_parallel_tasks=1,
    )
    envs = make_env(config, n_envs=1, use_async_envs=False)
    vector_env = envs["libero_object"][0]
    sub_env = vector_env.envs[0]
    sub_env.init_state_id = 0
    vector_env.reset(seed=[142])

    class EnvOnlyBackend:
        @staticmethod
        def _sub_env(task_id: int) -> object:
            assert task_id == 0
            return sub_env

    try:
        adapter = LiberoSceneAdapter(EnvOnlyBackend())
        # Frozen task-0 basket corridor, audited independently from the outcome:
        # enough room for the basket footprint and all 30/50/70 mm -x candidates.
        task0_bounds = PlanarBounds(-0.25, 0.20, 0.12, 0.38)
        result = adapter.shift_entity_planar(
            0,
            TASK0_TARGET_ENTITY,
            (-0.030, 0.0),
            workspace_bounds=task0_bounds,
            reachability_bounds=task0_bounds,
        )
        assert result.valid, result.to_dict()
        assert result.achieved_displacement_xyz_m == pytest.approx(
            (-0.030, 0.0, 0.0),
            abs=0.001,
        )
    finally:
        vector_env.close()
