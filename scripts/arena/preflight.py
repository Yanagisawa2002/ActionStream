#!/usr/bin/env python3
"""Fail-closed preflight for the pinned ActionStream/Arena integration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

from actionstream.arena_protocol import load_arena_protocol


def _git_head(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arena-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_arena_protocol(args.protocol)
    arena_root = args.arena_root.expanduser().resolve()
    checks = {
        "arena_checkout_exists": (arena_root / ".git").is_dir(),
        "arena_commit_matches": _git_head(arena_root)
        == protocol.raw["arena_source"]["commit"],
        "isaaclab_importable": importlib.util.find_spec("isaaclab") is not None,
        "isaaclab_arena_importable": importlib.util.find_spec("isaaclab_arena")
        is not None,
        "actionstream_arena_plugin_importable": (
            importlib.util.find_spec("isaaclab_arena_actionstream") is not None
        ),
        "cuda_importable": importlib.util.find_spec("torch") is not None,
    }
    status = "ready_for_simulator_smoke" if all(checks.values()) else "blocked"
    receipt = {
        "schema_version": 1,
        "status": status,
        "evidence_level": "structural",
        "protocol_sha256": protocol.sha256,
        "arena_expected_commit": protocol.raw["arena_source"]["commit"],
        "arena_observed_commit": _git_head(arena_root),
        "checks": checks,
        "learned_policy_result_available": False,
    }
    output = args.receipt.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if status == "ready_for_simulator_smoke" else 2


if __name__ == "__main__":
    raise SystemExit(main())
