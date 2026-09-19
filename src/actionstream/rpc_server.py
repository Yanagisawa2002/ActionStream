"""Serve an ActionStream inference worker over the versioned TCP RPC protocol."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from actionstream.rpc_transport import RpcFaultProfile, RpcInferenceServer, load_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factory", required=True, help="Trusted module:factory worker"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--telemetry-jsonl", type=Path)
    parser.add_argument("--response-delay-ms", type=float, default=0.0)
    parser.add_argument("--response-jitter-ms", type=float, default=0.0)
    parser.add_argument("--stall-every-n", type=int, default=0)
    parser.add_argument("--stall-ms", type=float, default=0.0)
    parser.add_argument("--disconnect-before-infer-every-n", type=int, default=0)
    parser.add_argument("--drop-response-every-n", type=int, default=0)
    parser.add_argument("--server-error-every-n", type=int, default=0)
    parser.add_argument("--fault-seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    worker, reset = load_worker(args.factory)
    faults = RpcFaultProfile(
        response_delay_s=args.response_delay_ms / 1000,
        response_jitter_s=args.response_jitter_ms / 1000,
        stall_every_n=args.stall_every_n,
        stall_s=args.stall_ms / 1000,
        disconnect_before_infer_every_n=args.disconnect_before_infer_every_n,
        drop_response_every_n=args.drop_response_every_n,
        server_error_every_n=args.server_error_every_n,
        seed=args.fault_seed,
    )
    server = RpcInferenceServer(
        worker,
        reset=reset,
        host=args.host,
        port=args.port,
        fault_profile=faults,
        telemetry_jsonl_path=args.telemetry_jsonl,
    )
    stopping = threading.Event()

    def stop(signum, frame):
        stopping.set()
        server.close()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    receipt = {
        "schema_version": 1,
        "host": server.host,
        "port": server.port,
        "factory": args.factory,
        "fault_profile": faults.__dict__,
        "execution_architecture": "single_persistent_inference_executor",
    }
    if args.ready_file is not None:
        args.ready_file.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    finally:
        server.close()
        print(
            json.dumps({"server_telemetry": server.telemetry()}, sort_keys=True),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
