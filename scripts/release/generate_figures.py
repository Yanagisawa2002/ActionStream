"""Generate every ActionStream v1.0.0 release figure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    scripts = (
        "plot_m4_pressure.py",
        "plot_m3_latency.py",
        "plot_calibration.py",
    )
    for script in scripts:
        subprocess.run(
            [sys.executable, str(script_dir / script)],
            cwd=script_dir,
            check=True,
        )


if __name__ == "__main__":
    main()
