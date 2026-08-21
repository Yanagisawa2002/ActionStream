"""Resume H1-Final after two receipt-only validator failures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESUME_V1_PATH = ROOT / "scripts/experiments/resume_transport_h1_final.py"
RESUME_V1_CONFIG = ROOT / "configs/actionstream_transport_h1_final_resume.json"
RESUME_V2_CONFIG = ROOT / "configs/actionstream_transport_h1_final_resume_v2.json"


def _load_v1() -> Any:
    spec = importlib.util.spec_from_file_location(
        "resume_transport_h1_final_frozen_v1", RESUME_V1_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen resume wrapper: {RESUME_V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = _load_v1()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cell(cell_dir: Path, runtime: str) -> dict[str, Any]:
    """Validate only fields present in the inspected current-baselines schema."""
    receipt_path = cell_dir / "run_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"Missing child receipt: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    selection = receipt.get("selection", {})
    if selection.get("runtimes") != [runtime]:
        raise RuntimeError(f"Child process mixed runtimes: {selection}")
    if selection.get("profiles") != ["fixed_0950_transport_h1_final"]:
        raise RuntimeError(f"Child process used the wrong profile: {selection}")
    gpu = receipt.get("system_gpu") or {}
    steady = (gpu.get("phases") or {}).get("steady_state") or {}
    if int(gpu.get("sample_count", 0)) <= 0:
        raise RuntimeError(f"No nvidia-smi samples for {cell_dir}")
    if int(steady.get("resident_sample_count", 0)) <= 0:
        raise RuntimeError(f"No steady process GPU residency for {cell_dir}")
    if steady.get("process_gpu_memory_mib_max") is None:
        raise RuntimeError(f"Missing process VRAM residency for {cell_dir}")
    records = [
        json.loads(line)
        for line in (cell_dir / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if len(records) != 5:
        raise RuntimeError(f"Expected five H1-Final rows in {cell_dir}")
    if any(row.get("worker_warmup") is None for row in records):
        raise RuntimeError(f"Missing actual-worker warmup provenance: {cell_dir}")
    if any(row.get("inference_requests_per_second") is None for row in records):
        raise RuntimeError(f"Missing true request/s: {cell_dir}")
    trace_count = len(list((cell_dir / "traces").glob("*.json")))
    telemetry_count = len(list((cell_dir / "traces").glob("*.telemetry.jsonl")))
    video_count = len(list((cell_dir / "videos").glob("*.mp4")))
    if trace_count != 5 or telemetry_count != 5 or video_count != 1:
        raise RuntimeError(
            f"Incomplete cell artifacts in {cell_dir}: traces={trace_count}, "
            f"telemetry={telemetry_count}, videos={video_count}"
        )
    return {"receipt": receipt, "records": records}


def _validate_v2_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "frozen_structural_resume_v2_before_paired_result":
        raise ValueError("Resume v2 is not frozen before paired results")
    frozen = raw.get("frozen_artifacts", {})
    checks = {
        "resume_v1_config_sha256": _sha256(RESUME_V1_CONFIG),
        "resume_v1_orchestrator_sha256": _sha256(RESUME_V1_PATH),
        "resume_v2_orchestrator_sha256": _sha256(Path(__file__).resolve()),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "actual": actual}
        for key, actual in checks.items()
        if frozen.get(key) != actual
    }
    if mismatches:
        raise RuntimeError(f"Structural resume v2 hash mismatch: {mismatches}")
    for key in ("first_process_log", "validator_only_log"):
        item = frozen[key]
        path_value = ROOT / item["path"]
        if _sha256(path_value) != item["sha256"]:
            raise RuntimeError(f"Frozen {key} changed before resume v2")
    if not all(raw.get("unchanged_scientific_contract", {}).values()):
        raise ValueError("Resume v2 changed the scientific contract")
    return raw


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_v2_config(args.resume_config_v2.resolve())
    V1._validate_cell = _validate_cell
    v1_args = argparse.Namespace(
        resume_config=RESUME_V1_CONFIG,
        protocol=args.protocol,
        lerobot_root=args.lerobot_root,
        output_root=args.output_root,
    )
    receipt = V1.run(v1_args)
    receipt.update(
        {
            "status": "completed_after_structural_resume_v2",
            "process_sessions_reported": 3,
            "gpu_process_sessions_with_cuda_work": 2,
            "validator_only_process_sessions": 1,
            "gpu_instance_power_cycles_for_holdout": 1,
            "resume_v1_config_path": str(RESUME_V1_CONFIG),
            "resume_v1_config_sha256": _sha256(RESUME_V1_CONFIG),
            "resume_v2_config_path": str(args.resume_config_v2.resolve()),
            "resume_v2_config_sha256": _sha256(args.resume_config_v2.resolve()),
            "structural_failures_before_full_matrix": 2,
        }
    )
    output_root = args.output_root.resolve()
    V1._write_json(output_root / "orchestrator_receipt.json", receipt)
    artifacts = {
        str(path.relative_to(output_root).as_posix()): _sha256(path)
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    V1._write_json(
        output_root / "artifact_manifest.json",
        {"schema_version": 1, "artifacts": artifacts},
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-config-v2", type=Path, default=RESUME_V2_CONFIG)
    parser.add_argument("--protocol", type=Path, action="append", required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
