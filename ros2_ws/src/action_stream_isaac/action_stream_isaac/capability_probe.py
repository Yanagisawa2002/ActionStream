"""Import-free capability probe for the pinned Isaac Sim/ROS 2 runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Final, Sequence

try:
    from .task_logic import ISAAC_PIP_VERSION, ISAAC_SIM_RELEASE
except ImportError:  # Allows direct execution before colcon installs the package.
    from task_logic import ISAAC_PIP_VERSION, ISAAC_SIM_RELEASE

# Isaac Sim 6.0.1's installed Compatibility Checker is authoritative for the
# matching wheel build and reports these minima.  The online requirements page
# can publish newer recommendations independently of a pinned build.
WINDOWS_MIN_DRIVER: Final = "537.58"
LINUX_MIN_DRIVER: Final = "535.161"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    observed: str
    required: str
    blocker: bool = True


@dataclass(frozen=True)
class CapabilityReport:
    ready: bool
    classification: str
    isaac_sim_release: str
    isaac_pip_version: str
    ros_imports_deferred_to_kit: bool
    checks: tuple[Check, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "classification": self.classification,
            "isaac_sim_release": self.isaac_sim_release,
            "isaac_pip_version": self.isaac_pip_version,
            "ros_imports_deferred_to_kit": self.ros_imports_deferred_to_kit,
            "checks": [asdict(check) for check in self.checks],
            "fallback": "none",
        }


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts)


def _version_at_least(observed: str, required: str) -> bool:
    left = _version_tuple(observed)
    right = _version_tuple(required)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_importable(name: str) -> tuple[bool, str]:
    """Probe binary-backed modules in a child process and retain a short error."""

    command = [
        sys.executable,
        "-c",
        f"import importlib; importlib.import_module({name!r})",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"probe failed: {type(exc).__name__}"
    if completed.returncode == 0:
        return True, "import succeeded"
    detail = (completed.stderr or completed.stdout).strip().splitlines()
    return False, detail[-1][:400] if detail else f"import exited {completed.returncode}"


def _os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _is_wsl() -> bool:
    release = platform.release().lower()
    proc_version = Path("/proc/version")
    text = (
        proc_version.read_text(encoding="utf-8", errors="replace").lower()
        if proc_version.is_file()
        else ""
    )
    return "microsoft" in release or "microsoft" in text or bool(os.environ.get("WSL_DISTRO_NAME"))


def _supported_host() -> tuple[bool, str]:
    if _is_wsl():
        return False, f"WSL2 ({platform.platform()})"
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        return (
            platform.machine().upper() in {"AMD64", "X86_64"} and build >= 22000,
            f"Windows build {build} {platform.machine()}",
        )
    if sys.platform.startswith("linux"):
        release = _os_release()
        observed = f"{release.get('ID', 'unknown')} {release.get('VERSION_ID', 'unknown')}"
        return (
            platform.machine() == "x86_64"
            and release.get("ID") == "ubuntu"
            and release.get("VERSION_ID") in {"22.04", "24.04"},
            observed,
        )
    return False, platform.platform()


def _gpu_facts() -> tuple[str | None, str | None, str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None, None, None
    command = [
        executable,
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    first = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def probe(*, require_ros_imports: bool = True) -> CapabilityReport:
    """Return a deterministic readiness report without importing Isaac or ROS.

    ``require_ros_imports=False`` is reserved for the adapter's pre-Kit
    preflight.  On the supported Windows pip distribution, the Isaac ROS bridge
    loads its bundled ROS DLLs only after ``SimulationApp`` starts.  Importing
    ``rclpy`` in a child process before that point can therefore fail even when
    the in-Kit bridge is usable.  The adapter defers both ROS imports to its
    post-bridge validation rather than preloading private DLLs.
    """

    checks: list[Check] = []
    supported, observed_host = _supported_host()
    checks.append(
        Check(
            "supported_host",
            supported,
            observed_host,
            "native Windows 11 x86_64 or Ubuntu 22.04/24.04 x86_64",
        )
    )
    checks.append(
        Check(
            "python_version",
            sys.version_info[:2] == (3, 12),
            platform.python_version(),
            "Python 3.12.x",
        )
    )

    isaac_version = _distribution_version("isaacsim")
    checks.append(
        Check(
            "isaacsim_distribution",
            isaac_version == ISAAC_PIP_VERSION,
            isaac_version or "not installed",
            f"isaacsim=={ISAAC_PIP_VERSION} (Isaac Sim release {ISAAC_SIM_RELEASE})",
        )
    )
    app_version = _distribution_version("isaacsim-app")
    checks.append(
        Check(
            "isaacsim_app_distribution",
            app_version == ISAAC_PIP_VERSION,
            app_version or "not installed",
            f"isaacsim-app=={ISAAC_PIP_VERSION}",
        )
    )
    robot_version = _distribution_version("isaacsim-robot")
    checks.append(
        Check(
            "franka_experimental_distribution",
            robot_version == ISAAC_PIP_VERSION,
            robot_version or "not installed",
            f"isaacsim-robot=={ISAAC_PIP_VERSION}; Franka API is verified "
            "after SimulationApp starts",
        )
    )
    if require_ros_imports:
        rclpy_ok, rclpy_observed = _module_importable("rclpy")
        checks.append(
            Check(
                "rclpy",
                rclpy_ok,
                rclpy_observed,
                "ROS 2 Jazzy rclpy built for Python 3.12",
            )
        )
        messages_ok, messages_observed = _module_importable("action_stream_msgs.msg")
        checks.append(
            Check(
                "action_stream_msgs",
                messages_ok,
                messages_observed,
                "built and sourced action_stream_msgs",
            )
        )
    ros_distro = os.environ.get("ROS_DISTRO", "unset")
    checks.append(Check("ros_distro", ros_distro == "jazzy", ros_distro, "jazzy"))
    ros2_executable = shutil.which("ros2")
    checks.append(
        Check(
            "ros2_cli",
            ros2_executable is not None,
            ros2_executable or "not on PATH",
            "sourced ROS 2 Jazzy ros2 executable",
        )
    )

    gpu_name, driver_version, memory_mib = _gpu_facts()
    try:
        memory_sufficient = float(memory_mib) >= 16_384.0 if memory_mib else False
    except ValueError:
        memory_sufficient = False
    checks.append(
        Check(
            "rtx_gpu",
            gpu_name is not None and "RTX" in gpu_name.upper() and memory_sufficient,
            f"{gpu_name or 'not detected'}; VRAM={memory_mib or 'unknown'} MiB",
            "NVIDIA RTX GPU with at least 16 GB VRAM",
        )
    )
    minimum_driver = WINDOWS_MIN_DRIVER if sys.platform == "win32" else LINUX_MIN_DRIVER
    checks.append(
        Check(
            "nvidia_driver_compatibility_checker_floor",
            driver_version is not None and _version_at_least(driver_version, minimum_driver),
            driver_version or "not detected",
            f">={minimum_driver}",
        )
    )

    eula = os.environ.get("OMNI_KIT_ACCEPT_EULA", "unset")
    checks.append(
        Check(
            "headless_eula_acceptance",
            eula.strip().upper() in {"1", "Y", "YES"},
            eula,
            "OMNI_KIT_ACCEPT_EULA=YES",
        )
    )

    ready = all(check.passed for check in checks if check.blocker)
    failed_names = {check.name for check in checks if check.blocker and not check.passed}
    if ready:
        classification = (
            "READY_FOR_ISAAC_ROS2" if require_ros_imports else "READY_FOR_ISAAC_ROS2_PREFLIGHT"
        )
    elif {
        "isaacsim_distribution",
        "isaacsim_app_distribution",
        "franka_experimental_distribution",
    } & failed_names:
        classification = "NO_GO_ISAAC_UNAVAILABLE"
    elif "supported_host" in failed_names:
        classification = "NO_GO_UNSUPPORTED_HOST"
    else:
        classification = "NO_GO_ENVIRONMENT_BLOCKED"
    return CapabilityReport(
        ready=ready,
        classification=classification,
        isaac_sim_release=ISAAC_SIM_RELEASE,
        isaac_pip_version=ISAAC_PIP_VERSION,
        ros_imports_deferred_to_kit=not require_ros_imports,
        checks=tuple(checks),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--defer-ros-imports-to-kit",
        action="store_true",
        help=(
            "run the adapter's pre-Kit checks and defer rclpy/generated-message "
            "imports until SimulationApp has enabled the Isaac ROS bridge"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = probe(require_ros_imports=not args.defer_ros_imports_to_kit)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"classification: {report.classification}")
        print(f"ready: {str(report.ready).lower()}")
        for check in report.checks:
            marker = "PASS" if check.passed else "FAIL"
            print(f"[{marker}] {check.name}: observed={check.observed}; required={check.required}")
        print("fallback: none")
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
