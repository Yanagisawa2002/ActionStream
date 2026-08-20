#!/usr/bin/env python3
"""Generate the audited SmolVLA official-RTC sync-gate report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionstream.smolvla_rtc_gate_report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-archive-sha256", required=True)
    args = parser.parse_args()
    summary = write_report(
        args.v2_root.resolve(),
        args.v3_root.resolve(),
        args.output_dir.resolve(),
        raw_archive_sha256=args.raw_archive_sha256,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
