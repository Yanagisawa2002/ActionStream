"""One owned DISPATCH_003 GPU child per phase, sharing a sealed 1,200s ledger."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from run_native_baseline import gpu_inventory, isolated_environment, save


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "evidence", "package"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dispatch", choices=("003", "004"), default="003")
    parser.add_argument(
        "--phase", choices=("language", "comparison", "shadow", "native"), required=True
    )
    args = parser.parse_args()
    store, evidence = args.store.resolve(), args.evidence.resolve()
    repair = args.dispatch == "004"
    if args.phase == ("language" if repair else "comparison"):
        raise ValueError("This text phase belongs to another dispatch")
    owner = json.loads((store / "OWNER.json").read_text())
    if owner["scope"] != (
        "DISPATCH_004 grounded language repair"
        if repair
        else "DISPATCH_003 finite LLM VLA integration"
    ):
        raise ValueError("Not the authorized new store")
    output = evidence / args.phase
    if output.exists():
        raise ValueError("No retries or overwrites in the frozen trial")
    for drive, minimum in (("c", 15), ("d", 30)):
        if shutil.disk_usage("/mnt/" + drive).free < minimum * 1024**3:
            raise ValueError("Host drive reserve would be violated")
    if (
        sum(p.stat().st_size for p in store.rglob("*") if p.is_file())
        > (1 if repair else 8) * 1024**3
    ):
        raise ValueError("New-store storage budget exceeded")
    frozen = json.loads((evidence / "frozen_protocol.json").read_text())
    for row in frozen["files"]:
        windows = row["path"].replace("\\", "/")
        path = Path("/mnt/" + windows[0].lower() + windows[2:])
        if digest(path) != row["sha256"]:
            raise ValueError(f"Frozen protocol identity changed: {path}")
    requirements = {
        "language": [],
        "shadow": ["language"],
        "native": ["language", "shadow"],
    }
    if repair:
        requirements = {
            "comparison": [],
            "shadow": [],
            "native": ["language", "shadow"],
        }
    for required in requirements[args.phase]:
        gate = json.loads((evidence / (required + "_score.json")).read_text())
        receipt_phase = "comparison" if repair and required == "language" else required
        if gate["status"] != "PASS" or gate["source_sha256"]["receipt"] != digest(
            evidence / receipt_phase / "run_receipt.json"
        ):
            raise ValueError("Prior real gate did not pass with intact evidence")
    model_store = store.parent / "dispatch003-llm-vla" if repair else store
    asset = json.loads((model_store / "asset_receipt.json").read_text())
    model = model_store / "assets/qwen3-vl-8964489"
    for row in asset["files"]:
        if digest(model / row["name"]) != row["sha256"]:
            raise ValueError("Pinned model asset changed")
    ledger_path = store / ("runs/gpu_budget_dispatch" + args.dispatch + ".json")
    lock = ledger_path.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else dict(
            owner=owner,
            limit_s=1200,
            runs=[],
            measurement="Sum of child wall seconds including imports/load/inference/native/cleanup/failures",
        )
    )
    if ledger["owner"] != owner or ledger["limit_s"] != 1200:
        raise ValueError("Budget ownership mismatch")
    if any(
        r["status"] == "RUNNING" or r["phase"] == args.phase for r in ledger["runs"]
    ):
        raise ValueError("No overlapping, unfinished or repeated phase is allowed")
    allowance = 1200 - sum(r["charged_wall_s"] for r in ledger["runs"])
    if allowance < 10:
        raise ValueError("GPU child budget exhausted")
    inventory = gpu_inventory()
    env = isolated_environment(store)
    env["PATH"] = str(store.parent / ".venv/bin") + ":/usr/bin:/bin:/usr/lib/wsl/lib"
    env["PYTHONPATH"] = str(args.package.resolve())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    module = "native_runner" if args.phase == "native" else "gate_runner"
    if repair:
        module = {
            "comparison": "repair_runner",
            "shadow": "gate_runner",
            "native": "grounded_native",
        }[args.phase]
    command = [
        str(store.parent / ".venv/bin/python"),
        "-m",
        "actionstream.llm_vla." + module,
        "--config",
        str(evidence / "frozen_config.json"),
        "--model",
        str(model),
        "--output",
        str(output),
    ]
    if args.phase == "native":
        command += ["--base-store", str(store.parent), "--evidence", str(evidence)]
    else:
        command += [
            "--inputs",
            str(evidence / "model_inputs" / (args.phase + ".json")),
        ]
        if args.phase != "comparison":
            command += ["--phase", args.phase]
        if repair and args.phase == "shadow":
            command += ["--diagnostic-only"]
    row = dict(
        phase=args.phase,
        status="RUNNING",
        allowance_s=allowance,
        charged_wall_s=allowance,
        gpu_before=inventory,
        command=command,
        started_utc=datetime.now(timezone.utc).isoformat(),
        package_sources={
            str(p.relative_to(args.package)): digest(p)
            for p in args.package.rglob("*.py")
        },
    )
    ledger["runs"].append(row)
    save(ledger_path, ledger)
    child = None
    started = time.monotonic()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Owned supervisor interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with output.with_suffix(".console.log").open("x") as log:
            child = subprocess.Popen(
                command,
                cwd=store,
                env=env,
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
        ledger["consumed_wall_s"] = sum(r["charged_wall_s"] for r in ledger["runs"])
        save(ledger_path, ledger)
        save(output.with_suffix(".supervisor.json"), row)
        print(
            json.dumps(
                {key: value for key, value in row.items() if key != "package_sources"},
                indent=2,
            ),
            flush=True,
        )
    return 0 if row["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
