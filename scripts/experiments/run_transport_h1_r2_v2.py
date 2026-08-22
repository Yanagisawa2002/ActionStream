"""Resume H1-R2 after a preflight-only venv symlink resolution failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "scripts/experiments/run_transport_h1_r2.py"
REGISTRY_PATH = ROOT / "configs/actionstream_transport_h1_r2_registry.json"
RESUME_CONFIG_PATH = ROOT / "configs/actionstream_transport_h1_r2_resume_v2.json"
FAILURE_RECEIPT_PATH = (
    ROOT / "reports/actionstream_transport_h1_r2/preflight_failure_v1.json"
)


def _load_v1() -> Any:
    spec = importlib.util.spec_from_file_location(
        "transport_h1_r2_frozen_v1", V1_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen H1-R2 runner: {V1_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = _load_v1()
_V1_PREFLIGHT = V1._exact_import_preflight


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _venv_preflight(
    *, python: Path, lerobot_root: Path, environment: dict[str, str]
) -> dict[str, Any]:
    del python
    return _V1_PREFLIGHT(
        python=Path(sys.executable),
        lerobot_root=lerobot_root,
        environment=environment,
    )


def _validate_resume_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "frozen_preflight_resume_before_formal_root":
        raise ValueError("H1-R2 v2 is not frozen before formal output creation")
    frozen = raw.get("frozen_artifacts", {})
    checks = {
        "registry_sha256": _sha256(REGISTRY_PATH),
        "v1_orchestrator_sha256": _sha256(V1_PATH),
        "v2_orchestrator_sha256": _sha256(Path(__file__).resolve()),
        "preflight_failure_receipt_sha256": _sha256(FAILURE_RECEIPT_PATH),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "actual": actual}
        for key, actual in checks.items()
        if frozen.get(key) != actual
    }
    if mismatches:
        raise RuntimeError(f"H1-R2 v2 frozen hash mismatch: {mismatches}")
    failure = raw.get("preflight_failure", {})
    if failure.get("formal_output_root_created") is not False:
        raise ValueError("H1-R2 v2 cannot resume after formal output creation")
    if int(failure.get("scored_episodes_started", -1)) != 0:
        raise ValueError("H1-R2 v2 cannot resume consumed identities")
    if int(failure.get("gpu_child_processes_started", -1)) != 0:
        raise ValueError("H1-R2 v2 is only for the preflight-only failure")
    if not all(raw.get("unchanged_scientific_contract", {}).values()):
        raise ValueError("H1-R2 v2 changed the scientific contract")
    remote_log = ROOT / str(failure["execution_log_relative_path"])
    if remote_log.is_file() and _sha256(remote_log) != failure["execution_log_sha256"]:
        raise RuntimeError("H1-R2 preflight failure log changed")
    return raw


def run(args: argparse.Namespace) -> dict[str, Any]:
    resume = _validate_resume_config(args.resume_config.resolve())
    V1._exact_import_preflight = _venv_preflight
    v1_args = argparse.Namespace(
        protocol=args.protocol,
        lerobot_root=args.lerobot_root,
        output_root=args.output_root,
        capture=args.capture,
    )
    receipt = V1.run(v1_args)
    receipt.update(
        {
            "status": "completed_after_preflight_resume_v2",
            "execution_orchestrator_v1_sha256": _sha256(V1_PATH),
            "execution_orchestrator_v2_sha256": _sha256(Path(__file__).resolve()),
            "resume_config_path": str(args.resume_config.resolve()),
            "resume_config_sha256": _sha256(args.resume_config.resolve()),
            "preflight_only_failed_sessions": 1,
            "gpu_child_processes_before_resume": 0,
            "duplicated_scored_episodes": 0,
            "resume_reason": resume["preflight_failure"]["reason"],
        }
    )
    output_root = args.output_root.resolve()
    V1._write_json(output_root / "orchestrator_receipt.json", receipt)
    artifacts = {
        str(artifact.relative_to(output_root).as_posix()): _sha256(artifact)
        for artifact in output_root.rglob("*")
        if artifact.is_file() and artifact.name != "artifact_manifest.json"
    }
    V1._write_json(
        output_root / "artifact_manifest.json",
        {"schema_version": 1, "artifacts": artifacts},
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-config", type=Path, default=RESUME_CONFIG_PATH)
    parser.add_argument("--protocol", type=Path, action="append", required=True)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--capture", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
