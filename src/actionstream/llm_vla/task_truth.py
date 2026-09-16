"""Privileged recording/scoring only. Never imported by online task completion."""

from collections import deque
import math

import numpy as np

PROFILES = {
    "object_tomato_basket": dict(
        target="tomato_sauce_1",
        support="basket_1",
        bddl_sha256="b1f4bb69d256a05f693838de46182bb0d3de0de68eec350df1b5b1e76f6c136d",
    ),
    "spatial_bowl_stove_plate": dict(
        target="akita_black_bowl_1",
        support="plate_1",
        bddl_sha256="dd11837141a30d874ebbbfaae20e91ecc67ec85ab3872ceb4cff3abfcbef7cb0",
    ),
    "goal_wine_cabinet": dict(
        target="wine_bottle_1",
        support="wooden_cabinet_1",
        bddl_sha256="2ef6f403a8451e216af44158c1a6c975cf8e5058ad02d78367ac0be443d096fe",
    ),
}


def instantaneous(facts):
    linear, angular = facts["linear_speed"], facts["angular_speed"]
    return bool(
        facts["goal_satisfied"]
        and facts["support_contact"]
        and not facts["finger_contact"]
        and math.isfinite(linear)
        and 0 <= linear <= 0.03
        and math.isfinite(angular)
        and 0 <= angular <= 0.3
    )


class StableTaskTruth:
    def __init__(self):
        self.window = deque(maxlen=11)
        self.last = None

    def update(self, control, facts):
        if self.last is not None and control != self.last + 1:
            raise ValueError("Truth history must advance by one control")
        self.last = control
        self.window.append(instantaneous(facts))
        return len(self.window) == 11 and all(self.window)


def simulator_facts(env, task):
    profile = PROFILES[task.key]
    world = env.env
    target = world.objects_dict[profile["target"]]
    support = world.get_object(profile["support"])
    if support is None:
        raise ValueError("Missing task-specific support body")
    gripper = world.robots[0].gripper
    pads = (
        gripper.important_geoms["left_fingerpad"]
        + gripper.important_geoms["right_fingerpad"]
    )
    velocity = np.asarray(env.sim.data.get_joint_qvel(target.joints[0]), dtype=float)
    if velocity.shape != (6,):
        raise ValueError("Target must have a six-velocity free joint")
    return dict(
        task_key=task.key,
        task_sha256=task.sha256,
        goal_satisfied=bool(env.check_success()),
        finger_contact=bool(world.check_contact(pads, target.contact_geoms)),
        support_contact=bool(
            world.check_contact(target.contact_geoms, support.contact_geoms)
        ),
        linear_speed=float(np.linalg.norm(velocity[:3])),
        angular_speed=float(np.linalg.norm(velocity[3:])),
    )


def target_visibility(env, task):
    """Offline segmentation proxy, evaluated only after native collection stops."""
    import mujoco

    target = env.env.objects_dict[PROFILES[task.key]["target"]]
    gids = [
        env.sim.model.geom_name2id(g)
        for g in target.visual_geoms + target.contact_geoms
    ]
    ctx = env.sim._render_context_offscreen
    result = []
    for camera in ("agentview", "robot0_eye_in_hand"):
        ctx.gl_ctx.make_current()
        ctx.render(
            192, 192, camera_id=env.sim.model.camera_name2id(camera), segmentation=True
        )
        raw = np.empty((192, 192, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(
            rgb=raw, depth=None, viewport=mujoco.MjrRect(0, 0, 192, 192), con=ctx.con
        )
        colors = raw.astype(np.int32)
        ids = colors[:, :, 0] + colors[:, :, 1] * 256 + colors[:, :, 2] * 65536
        wanted = [
            g.segid + 1
            for g in list(ctx.scn.geoms)[: ctx.scn.ngeom]
            if g.objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
            and g.objid in gids
            and g.segid >= 0
        ]
        result.append(int(np.isin(ids, wanted).sum()))
    return result
