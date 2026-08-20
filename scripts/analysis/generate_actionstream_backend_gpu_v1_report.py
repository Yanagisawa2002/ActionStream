#!/usr/bin/env python3
"""Generate the audited ActionStream LeRobot GPU benchmark v1 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from actionstream.backend_gpu_report import write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = write_report(args.input_root.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
