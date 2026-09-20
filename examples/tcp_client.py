"""Connect to the CPU-only ActionStream example worker."""

from __future__ import annotations

import argparse

from actionstream.rpc_transport import TcpInferenceTransport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()

    client = TcpInferenceTransport(args.host, args.port)
    try:
        client.reset()
        for value in (1, 2, 3):
            actions = client.infer(
                {"value": value},
                "cpu-demo",
                timeout_s=2.0,
            )
            print(f"value={value} -> {actions.tolist()}")
        print("telemetry:", client.telemetry().to_dict())
    finally:
        client.close()


if __name__ == "__main__":
    main()
