"""A spawned process owns native VLA work; only public observations cross IPC.

The ActionStream worker remains the single request/reset caller. CUDA is never
forked. Reset preserves the process and RNG stream, so startup warmup survives
queue invalidation. This does not change the engine's advisory deadline claim.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import traceback


def _serve(connection, assets, seed, batch_postprocessing):
    backend = None
    try:
        import torch
        from .agent_cli import backend_factory

        torch.set_num_threads(8)
        backend = backend_factory(Path(assets), seed)
        # Match the parent's episode reset, after loading all modules/weights.
        backend._set_seed(seed)
        backend.reset_runtime()
        connection.send(("ready", os.getpid()))
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                backend.reset_runtime()
                connection.send(("reset", None))
            elif command == "infer":
                observation, instruction = payload
                result = backend.infer_action_chunk(
                    observation, instruction, batch_postprocessing=batch_postprocessing
                )
                connection.send(("result", result))
            else:
                raise ValueError("Unknown inference owner command")
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if backend is not None:
            backend.close()
        connection.close()


class ProcessInferenceMixin:
    """Used only with the single-owner native simulation port."""

    inference_process_target = staticmethod(_serve)

    def _receive(self, expected, timeout=30.0):
        if not self.inference_connection.poll(timeout):
            raise TimeoutError("Native inference owner did not respond")
        kind, payload = self.inference_connection.recv()
        if kind != expected:
            raise RuntimeError(f"Native inference owner {kind}: {payload}")
        return payload

    def reset_inference(self):
        if not hasattr(self, "inference_process"):
            context = mp.get_context("spawn")
            self.inference_connection, child = context.Pipe()
            self.inference_process = context.Process(
                target=self.inference_process_target,
                args=(
                    child,
                    str(Path(self.backend.model_id).parent),
                    self.seed,
                    self.batch_postprocessing,
                ),
                name="ActionStreamNativeVLA",
            )
            self.inference_process.start()
            child.close()
            self.inference_owner_pid = self._receive("ready")
            if self.inference_owner_pid != self.inference_process.pid:
                raise RuntimeError("Inference owner identity mismatch")
        self.inference_connection.send(("reset", None))
        self._receive("reset")

    def infer(self, observation, instruction):
        self.inference_connection.send(
            ("infer", (observation.native_input(), instruction))
        )
        result = self._receive("result")
        if result.raw_shape != (1, 30, 20) or result.raw_dtype != "torch.float32":
            raise ValueError("Pinned VLA output contract changed")
        return result.actions, dict(
            model_latency_s=result.model_latency_seconds,
            raw_shape=result.raw_shape,
            raw_dtype=result.raw_dtype,
            actions=result.actions.tolist(),
            inference_owner="spawned_process",
            inference_pid=self.inference_owner_pid,
        )

    def close(self):
        process = getattr(self, "inference_process", None)
        try:
            if process is not None:
                try:
                    if process.is_alive():
                        self.inference_connection.send(("close", None))
                        process.join(3)
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(3)
                    if process.is_alive():
                        process.kill()
                        process.join(3)
                    self.inference_connection.close()
                    if process.is_alive():
                        raise RuntimeError("Native inference owner failed to stop")
        finally:
            super().close()
