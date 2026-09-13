"""Run one native acceptance phase with an owned, cumulative 20-minute budget."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

LIMIT_SECONDS = 1200


def save(path: Path, value: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def isolated_environment(store: Path) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        PYTHONNOUSERSITE="1",
        TMPDIR=str(store / "tmp"),
        XDG_CACHE_HOME=str(store / "cache"),
        HF_HOME=str(store / "cache/huggingface"),
        TORCH_HOME=str(store / "cache/torch"),
        NUMBA_CACHE_DIR=str(store / "cache/numba"),
        CUDA_CACHE_PATH=str(store / "cache/cuda"),
        LIBERO_CONFIG_PATH=str(store / "cache/libero-config"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        WANDB_MODE="disabled",
        DO_NOT_TRACK="1",
        TOKENIZERS_PARALLELISM="false",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        PYTHONUNBUFFERED="1",
        PATH=str(store / ".venv/bin") + ":/usr/bin:/bin:/usr/lib/wsl/lib",
    )
    return env


def gpu_inventory() -> dict:
    binary = "/usr/lib/wsl/lib/nvidia-smi"
    processes = subprocess.check_output(
        [
            binary,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if processes:
        raise RuntimeError(f"Existing GPU compute load; do not compete: {processes}")
    raw = subprocess.check_output(
        [
            binary,
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    if len(raw.splitlines()) != 1:
        raise RuntimeError("Expected the single authorized GPU")
    name, driver, total, used, free, utilization = [
        part.strip() for part in raw.split(",")
    ]
    if "4090" not in name or int(free) < 12 * 1024 or int(utilization) > 10:
        raise RuntimeError(f"GPU is unavailable or busy: {raw}")
    return dict(
        name=name,
        driver=driver,
        total_MiB=int(total),
        used_MiB=int(used),
        free_MiB=int(free),
        utilization_percent=int(utilization),
        compute_processes=processes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("smoke", "development"))
    parser.add_argument("--smoke-receipt", type=Path)
    args = parser.parse_args()
    store = args.store.resolve()
    owner = json.loads((store / "OWNER.json").read_text())
    if owner.get("scope") != "DISPATCH_002 native development baseline":
        raise RuntimeError("Store is not owned by this dispatch")
    if args.output.exists():
        raise RuntimeError("Refusing to overwrite an earlier native run")
    for drive, reserve in (("c", 15), ("d", 30)):
        if shutil.disk_usage("/mnt/" + drive).free < reserve * 1024**3:
            raise RuntimeError(f"Host {drive}: reserve reached")
    ledger_path = store / "runs/budget.json"
    lock = (store / "runs/budget.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {
            "owner": owner,
            "limit_s": LIMIT_SECONDS,
            "measurement": "sum of native child wall times, including import/load/render/inference/cleanup",
            "runs": [],
        }
    )
    if ledger["owner"] != owner or ledger["limit_s"] != LIMIT_SECONDS:
        raise RuntimeError("Budget ledger ownership or limit changed")
    if any(row["status"] == "RUNNING" for row in ledger["runs"]):
        raise RuntimeError(
            "An unfinished owned child requires explicit recovery before another launch"
        )
    if args.phase == "development" and any(
        row["phase"] == "development" for row in ledger["runs"]
    ):
        raise RuntimeError("Only one development batch is authorized")
    consumed = sum(row["charged_wall_s"] for row in ledger["runs"])
    allowance = min(
        600 if args.phase == "smoke" else LIMIT_SECONDS, LIMIT_SECONDS - consumed
    )
    if allowance < 10:
        raise RuntimeError("Cumulative native run budget exhausted")
    inventory = gpu_inventory()
    command = [
        str(store / ".venv/bin/python"),
        "-m",
        "actionstream.native_baseline",
        "--store",
        str(store),
        "--config",
        str(args.config.resolve()),
        "--output",
        str(args.output.resolve()),
        "--phase",
        args.phase,
    ]
    if args.smoke_receipt:
        command += ["--smoke-receipt", str(args.smoke_receipt.resolve())]
    log_path = args.output.with_suffix(".console.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(
        phase=args.phase,
        status="RUNNING",
        command=command,
        started_utc=datetime.now(timezone.utc).isoformat(),
        allowance_s=allowance,
        charged_wall_s=allowance,
        gpu_before=inventory,
        log=str(log_path),
    )
    ledger["runs"].append(row)
    save(ledger_path, ledger)
    started = time.monotonic()
    child = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with log_path.open("x") as log:
            child = subprocess.Popen(
                command,
                cwd=store,
                env=isolated_environment(store),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            row["pid"] = row["process_group"] = child.pid
            save(ledger_path, ledger)
            try:
                child.wait(timeout=allowance - 5)
                row["status"] = "COMPLETED" if child.returncode == 0 else "ERROR"
            except subprocess.TimeoutExpired:
                row["status"] = "BUDGET_TIMEOUT"
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=1)
            row["returncode"] = child.returncode
    except BaseException as exc:
        row.update(status="SUPERVISOR_ERROR", error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=1)
            row.update(status="SUPERVISOR_INTERRUPTED", returncode=child.returncode)
        row["charged_wall_s"] = time.monotonic() - started
        row["finished_utc"] = datetime.now(timezone.utc).isoformat()
        ledger["consumed_wall_s"] = sum(
            item["charged_wall_s"] for item in ledger["runs"]
        )
        save(ledger_path, ledger)
        save(args.output.with_suffix(".supervisor.json"), row)
        print(json.dumps(row, indent=2), flush=True)
    return 0 if row["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
