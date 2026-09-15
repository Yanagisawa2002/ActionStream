"""Native asynchronous port; all simulator access stays on the control thread."""

from __future__ import annotations

import numpy as np

from actionstream.completion_labels import StableTruth
from .finite_native import SimulationPort
from .temporal_completion import camera_rgb


class AsyncSimulationPort(SimulationPort):
    def reset_inference(self):
        self.backend.reset_runtime()

    def warmup(self, observation, instruction, predictor):
        # Reported separately from paced dispatch. No physical action is executed.
        self.infer(observation, instruction)
        rgb = camera_rgb(
            {
                "agentview_image": observation.pixels["image"][0],
                "robot0_eye_in_hand_image": observation.pixels["image2"][0],
            }
        )
        predictor.predict(np.stack([rgb, rgb, rgb])[None])

    def refresh(self, request_id, revision):
        if self.control != 0 or self.actions:
            raise ValueError("Only the unmoved warmup observation may be refreshed")
        raw = self.env.set_init_state(self.env.sim.get_state().flatten())
        self.states.clear()
        self.rgb.clear()
        self.facts.clear()
        self.bindings.clear()
        self.truth = StableTruth()
        return self.capture(raw, request_id, revision)

    def pose_hold(self, observation, gripper):
        from scipy.spatial.transform import Rotation

        position = np.asarray(observation.robot_state["eef"]["pos"])[0]
        matrix = np.asarray(observation.robot_state["eef"]["mat"])[0]
        rotation = Rotation.from_matrix(matrix.copy()).as_rotvec()
        # Keep the native controller in absolute mode throughout all phases.
        return np.concatenate([position, rotation, [gripper]]).astype(np.float32)
