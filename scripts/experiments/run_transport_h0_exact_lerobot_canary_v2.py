"""Resume H0 after a preflight-only venv symlink resolution failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "scripts/experiments/run_transport_h0_exact_lerobot_canary.py"
V1_CONFIG = ROOT / "configs/actionstream_transport_h0_exact_lerobot_canary.json"
V2_CONFIG = ROOT / "configs/actionstream_transport_h0_exact_lerobot_canary_v2.json"


def _load_v1() -> Any:
    spec = importlib.util.spec_from_file_location(
        "transport_h0_canary_frozen_v1", V1_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen H0 canary: {V1_PATH}")
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


def _validate_v2_config(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("status") != "frozen_preflight_resume_before_gpu_child":
        raise ValueError("H0 v2 resume is not frozen before a GPU child")
    frozen = raw.get("frozen_artifacts", {})
    checks = {
        "v1_config_sha256": _sha256(V1_CONFIG),
        "v1_orchestrator_sha256": _sha256(V1_PATH),
        "v2_orchestrator_sha256": _sha256(Path(__file__).resolve()),
    }
    mismatches = {
        key: {"expected": frozen.get(key), "actual": actual}
        for key, actual in checks.items()
        if frozen.get(key) != actual
    }
    if mismatches:
        raise RuntimeError(f"H0 v2 frozen hash mismatch: {mismatches}")
    log = frozen["preflight_failure_log"]
    if _sha256(ROOT / log["path"]) != log["sha256"]:
        raise RuntimeError("H0 preflight-only failure log changed")
    if not all(raw.get("unchanged_scientific_contract", {}).values()):
        raise ValueError("H0 v2 changed the scientific contract")
    return raw


def run(args: argparse.Namespace) -> dict[str, Any]:
    resume = _validate_v2_config(args.resume_config_v2.resolve())
    V1._exact_import_preflight = _venv_preflight
    v1_args = argparse.Namespace(
        protocol=V1_CONFIG,
        lerobot_root=args.lerobot_root,
        output_root=args.output_root,
    )
    receipt = V1.run(v1_args)
    receipt.update(
        {
            "status": "PASS_AFTER_PREFLIGHT_RESUME",
            "execution_orchestrator_v1_sha256": _sha256(V1_PATH),
            "execution_orchestrator_v2_sha256": _sha256(Path(__file__).resolve()),
            "resume_config_v2_path": str(args.resume_config_v2.resolve()),
            "resume_config_v2_sha256": _sha256(args.resume_config_v2.resolve()),
            "preflight_only_process_sessions": 1,
            "gpu_child_processes_started": 1,
            "duplicated_development_episodes": 0,
            "formal_h1_reopened": False,
            "resume_reason": resume["preflight_failure"],
        }
    )
    V1._write_json(args.output_root.resolve() / "h0_canary_receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-config-v2", type=Path, default=V2_CONFIG)
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(run(_parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
