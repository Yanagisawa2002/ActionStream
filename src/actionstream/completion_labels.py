"""Offline completion truth: released, supported, slow and continuously stable."""

from collections import deque
import numpy as np

DEFAULT = dict(window_steps=10, linear_speed_max=0.03, angular_speed_max=0.3)


def instantaneous(facts, config=DEFAULT):
    values = [facts["linear_speed"], facts["angular_speed"]]
    return bool(
        np.isfinite(values).all()
        and facts["inside"]
        and not facts["finger_contact"]
        and facts["basket_contact"]
        and 0 <= values[0] <= config["linear_speed_max"]
        and 0 <= values[1] <= config["angular_speed_max"]
    )


class StableTruth:
    def __init__(self, config=None):
        self.config = dict(DEFAULT if config is None else config)
        self.rows = deque(maxlen=self.config["window_steps"] + 1)
        self.last_step = None

    def update(self, step, facts):
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError("Truth history must contain consecutive physical controls")
        self.last_step = step
        self.rows.append(instantaneous(facts, self.config))
        return len(self.rows) == self.rows.maxlen and all(self.rows)


def simulator_facts(env):
    """Privileged evaluator only; never call this from the RGB model."""
    world = env.env
    target = world.objects_dict["tomato_sauce_1"]
    basket = world.objects_dict["basket_1"]
    gripper = world.robots[0].gripper
    pads = (
        gripper.important_geoms["left_fingerpad"]
        + gripper.important_geoms["right_fingerpad"]
    )
    velocity = np.asarray(env.sim.data.get_joint_qvel(target.joints[0]), dtype=float)
    return dict(
        inside=bool(env.check_success()),
        finger_contact=bool(world.check_contact(pads, target.contact_geoms)),
        bilateral_grasp=bool(world._check_grasp(gripper, target.contact_geoms)),
        basket_contact=bool(
            world.check_contact(target.contact_geoms, basket.contact_geoms)
        ),
        linear_speed=float(np.linalg.norm(velocity[:3])),
        angular_speed=float(np.linalg.norm(velocity[3:])),
    )
