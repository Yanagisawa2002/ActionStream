"""Run a stateful CPU-only ActionStream RPC worker on loopback."""

from __future__ import annotations

import argparse

import torch

from actionstream.rpc_transport import RpcInferenceServer


class DummyWorker:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()

    worker = DummyWorker()
    server = RpcInferenceServer(
        worker,
        host=args.host,
        port=args.port,
        reset=worker.reset,
    )
    print(f"ActionStream dummy RPC worker listening on {server.host}:{server.port}")
    print("This example has no authentication; keep it on a trusted/loopback network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
