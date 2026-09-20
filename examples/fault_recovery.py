"""CPU-only real-socket disconnect/recovery demo."""

from __future__ import annotations

import torch

from actionstream.inference_transport import InferenceTransportError
from actionstream.rpc_transport import (
    RpcFaultProfile,
    RpcInferenceServer,
    TcpInferenceTransport,
)


class StatefulWorker:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, observation: dict, _task: str) -> torch.Tensor:
        self.calls += 1
        return torch.tensor(
            [[float(observation["value"]), float(self.calls)]],
            dtype=torch.float32,
        )

    def reset(self) -> None:
        self.calls = 0


def main() -> None:
    worker = StatefulWorker()
    faults = RpcFaultProfile(disconnect_before_infer_every_n=2)
    with RpcInferenceServer(worker, reset=worker.reset, fault_profile=faults) as server:
        client = TcpInferenceTransport(server.host, server.port)
        try:
            client.reset()
            first = client.infer({"value": 1}, "cpu-demo", timeout_s=2.0)
            print("request 1:", first.tolist())

            try:
                client.infer({"value": 2}, "cpu-demo", timeout_s=2.0)
            except InferenceTransportError as exc:
                print("request 2: injected disconnect ->", type(exc).__name__)
            else:
                raise RuntimeError("the second request should have disconnected")

            # Reconnection alone is not treated as policy-state reset. Explicitly
            # confirm remote state before continuing the next logical episode.
            client.reset()
            recovered = client.infer({"value": 3}, "cpu-demo", timeout_s=2.0)
            print("request 3 after acknowledged reset:", recovered.tolist())
            print("client reconnects:", client.telemetry().reconnects)
            print("server executor starts:", server.telemetry()["executor_starts"])
        finally:
            client.close()


if __name__ == "__main__":
    main()
