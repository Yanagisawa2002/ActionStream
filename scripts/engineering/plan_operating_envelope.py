"""Expand the draft ActionStream operating-envelope experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def expand(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    repeat_count = int(config["repeats"])
    executor_modes = config["tcp_executor_modes"]

    for transport in config["transport_modes"]:
        disconnects = config["tcp_disconnect_every_n"] if transport == "tcp" else [0]
        executors = executor_modes if transport == "tcp" else [
            {"name": "not_applicable", "implemented": True}
        ]
        for delay_ms in config["delay_ms"]:
            for startup in config["startup_states"]:
                for disconnect_every_n in disconnects:
                    for executor in executors:
                        for repeat in range(repeat_count):
                            executable = bool(executor["implemented"])
                            rows.append(
                                {
                                    "row_id": len(rows),
                                    "transport": transport,
                                    "delay_ms": int(delay_ms),
                                    "startup": startup,
                                    "disconnect_every_n": int(disconnect_every_n),
                                    "executor_mode": executor["name"],
                                    "repeat": repeat,
                                    "executable_now": executable,
                                    "blocked_reason": None
                                    if executable
                                    else "comparison mechanism not implemented",
                                }
                            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/operating_envelope_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = expand(config)
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)

    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    executable = sum(row["executable_now"] for row in rows)
    blocked = len(rows) - executable
    print(
        f"planned_rows={len(rows)} executable_now={executable} blocked={blocked}",
        file=__import__("sys").stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
