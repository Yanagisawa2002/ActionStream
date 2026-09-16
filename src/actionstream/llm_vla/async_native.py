"""Native asynchronous port; all simulator access stays on the control thread."""

from __future__ import annotations

import numpy as np

from .finite_native import SimulationPort
from .async_process import ProcessInferenceMixin
from .temporal_completion import camera_rgb


class AsyncSimulationPort(SimulationPort):
    batch_postprocessing = True

    def reset_inference(self):
        import torch

        if not hasattr(self, "inference_stream"):
            self.inference_stream = torch.cuda.Stream()
        self.backend.reset_runtime()

    def infer(self, observation, instruction):
        # Warmup runs before the worker exists. Subsequent model operations use
        # a worker-owned stream so the RGB verifier need not queue behind VLA.
        if not hasattr(self, "inference_stream"):
            return super().infer(observation, instruction)
        import torch

        with torch.cuda.stream(self.inference_stream):
            return super().infer(observation, instruction)

    def warmup(self, observation, instruction, predictor):
        # Reported separately from paced dispatch. No physical action is executed.
        # VLA warmup is performed on the actual ActionStream owner thread.
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
        self.truth = self.new_truth()
        return self.capture(raw, request_id, revision)

    def pose_hold(self, observation, gripper):
        from scipy.spatial.transform import Rotation

        position = np.asarray(observation.robot_state["eef"]["pos"])[0]
        matrix = np.asarray(observation.robot_state["eef"]["mat"])[0]
        rotation = Rotation.from_matrix(matrix.copy()).as_rotvec()
        # Keep the native controller in absolute mode throughout all phases.
        return np.concatenate([position, rotation, [gripper]]).astype(np.float32)


class IsolatedAsyncSimulationPort(ProcessInferenceMixin, AsyncSimulationPort):
    """Native VLA process isolation with unchanged control and RGB semantics."""
