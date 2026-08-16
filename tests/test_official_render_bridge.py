from __future__ import annotations

import numpy as np
import pytest

from actionstream.official_render_bridge import (
    apply_isaac_state_to_official_renderer,
    validate_unbatched_bridge_state,
)


def _state() -> dict:
    return {
        "eef": {
            "mat": np.eye(3).tolist(),
            "pos": [0.1, 0.2, 0.3],
            "quat": [0.0, 0.0, 0.0, 1.0],
        },
        "gripper": {"qpos": [0.04, -0.04], "qvel": [0.0, 0.0]},
        "joints": {
            "pos": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "vel": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        },
    }


class _Data:
    def __init__(self) -> None:
        self.qpos = np.zeros(20, dtype=np.float64)
        self.qvel = np.zeros(20, dtype=np.float64)


class _Sim:
    def __init__(self) -> None:
        self.data = _Data()
        self.forwarded = False

    def forward(self) -> None:
        self.forwarded = True


class _Robot:
    _ref_joint_pos_indexes = np.arange(7)
    _ref_joint_vel_indexes = np.arange(7)
    _ref_gripper_joint_pos_indexes = np.asarray([7, 8])
    _ref_gripper_joint_vel_indexes = np.asarray([7, 8])


class _Inner:
    def __init__(self) -> None:
        self.forced = False

    def _update_observables(self, *, force: bool) -> None:
        self.forced = force

    def _get_observations(self) -> dict:
        return {
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
            "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((8, 8, 3), dtype=np.uint8),
        }


class _Wrapped:
    def __init__(self) -> None:
        self.sim = _Sim()
        self.robots = [_Robot()]
        self.env = _Inner()


class _SubEnv:
    def __init__(self) -> None:
        self._env = _Wrapped()

    def _format_raw_obs(self, raw: dict) -> dict:
        return {
            "pixels": {
                "image": raw["agentview_image"],
                "image2": raw["robot0_eye_in_hand_image"],
            }
        }


def test_bridge_writes_exact_robot_state_and_returns_two_images() -> None:
    env = _SubEnv()
    result = apply_isaac_state_to_official_renderer(
        env, _state(), pose_mode="joint_replay"
    )
    assert env._env.sim.forwarded
    assert env._env.env.forced
    assert env._env.sim.data.qpos[:7] == pytest.approx(_state()["joints"]["pos"])
    assert env._env.sim.data.qvel[:7] == pytest.approx(_state()["joints"]["vel"])
    assert env._env.sim.data.qpos[7:9] == pytest.approx(_state()["gripper"]["qpos"])
    assert set(result.observation["pixels"]) == {"image", "image2"}
    assert result.provenance["physics_steps_after_write"] == 0
    assert result.provenance["writeback_exact_at_1e-12"]
    assert result.provenance["requested_vs_rendered_eef_position_l2_m"] == pytest.approx(0.0)


def test_bridge_rejects_nonfinite_state() -> None:
    state = _state()
    state["joints"]["pos"][2] = float("nan")
    with pytest.raises(ValueError, match="joints.pos"):
        validate_unbatched_bridge_state(state)


def test_bridge_requires_a_reset_official_environment() -> None:
    class Empty:
        _env = None

    with pytest.raises(RuntimeError, match="has not been reset"):
        apply_isaac_state_to_official_renderer(Empty(), _state())


def test_bridge_rejects_unknown_pose_mode() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        apply_isaac_state_to_official_renderer(
            _SubEnv(), _state(), pose_mode="not-a-mode"
        )
