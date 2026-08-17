#!/usr/bin/env python3
"""Expand a frozen ActionStream/Arena split into one-seed-per-process jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionstream.arena_protocol import load_arena_protocol


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol-runtime-path",
        help="Protocol path as mounted inside the Arena Docker container",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Development inspection only"
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    protocol = load_arena_protocol(args.protocol)
    cells = list(protocol.cells(args.split))
    if args.limit is not None:
        if args.split == "holdout":
            raise ValueError("--limit is forbidden for the formal holdout")
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        cells = cells[: args.limit]

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs_dir = output / "jobs"
    jobs_dir.mkdir()
    runtime_path = args.protocol_runtime_path or str(protocol.path)
    rows = []
    for index, cell in enumerate(cells):
        job = protocol.arena_job_for(cell)
        job["jobs"][0]["policy_config_dict"]["protocol_path"] = runtime_path
        path = jobs_dir / f"{index:04d}__{cell.cell_id}.json"
        path.write_text(
            json.dumps(job, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows.append(
            {
                "cell_id": cell.cell_id,
                "pair_id": cell.pair_id,
                "reset_seed": cell.reset_seed,
                "job_path": str(path.relative_to(output)),
                "arena_cli_seed": cell.reset_seed,
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment_id": protocol.raw["experiment_id"],
        "split": args.split,
        "protocol_path": str(protocol.path),
        "protocol_runtime_path": runtime_path,
        "protocol_sha256": protocol.sha256,
        "arena_commit": protocol.raw["arena_source"]["commit"],
        "lerobot_commit": protocol.raw["lerobot_source"]["commit"],
        "cell_count": len(rows),
        "one_reset_seed_per_process": True,
        "cells": rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: manifest[key] for key in ("split", "cell_count", "protocol_sha256")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
