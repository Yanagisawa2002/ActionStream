"""Deterministic, auditable seed manifests for the M5-G0 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
IMPLEMENTATION_SOURCE_PATHS = (
    "configs/m5_g0.json",
    "src/actionstream/runtime.py",
    "src/actionstream/lerobot_backend.py",
    "src/actionstream/processors.py",
    "src/actionstream/libero_config.py",
    "src/actionstream/m5_protocol.py",
    "src/actionstream/m5_scene.py",
    "src/actionstream/m5_runtime.py",
    "src/actionstream/m5_benchmark.py",
    "src/actionstream/m5_calibration.py",
    "src/actionstream/m5_freeze.py",
    "src/actionstream/m5_no_shift.py",
    "src/actionstream/m5_analysis.py",
    "src/actionstream/m5_report.py",
    "src/actionstream/m5_validation.py",
    "scripts/run_m5_g0.sh",
    "scripts/run_m5_g0_smoke.sh",
)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_source_hash(
    repository_root: Path,
) -> tuple[str, list[dict[str, str]]]:
    """Hash every behavior-critical source used by the frozen M5 protocol."""

    entries = [
        {
            "path": relative,
            "sha256": file_sha256(repository_root / relative),
        }
        for relative in IMPLEMENTATION_SOURCE_PATHS
    ]
    return canonical_sha256(entries), entries


def _manifest(name: str, pairs: list[dict[str, int]]) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "milestone": "M5-G0",
        "manifest_name": name,
        "pairs": pairs,
    }
    return {**core, "manifest_sha256": canonical_sha256(core)}


def build_seed_manifests() -> dict[str, dict[str, Any]]:
    """Return the frozen seed/state pools without consulting experiment results."""

    calibration = [
        {"pair_index": index, "seed": 51001 + index, "initial_state_index": state}
        for index, state in enumerate((0, 2, 4, 6, 8))
    ]
    no_shift = [
        {"pair_index": index, "seed": 52001 + index, "initial_state_index": state}
        for index, state in enumerate(range(10, 20))
    ]
    sealed = [
        {"pair_index": index, "seed": 53001 + index, "initial_state_index": state}
        for index, state in enumerate(range(20, 50))
    ]
    return {
        "calibration_seed_manifest.json": _manifest("calibration", calibration),
        "no_shift_seed_manifest.json": _manifest("no_shift_regression", no_shift),
        "sealed_seed_manifest.json": _manifest("sealed_evaluation", sealed),
    }


def validate_manifest(payload: dict[str, Any]) -> None:
    expected = payload.get("manifest_sha256")
    core = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = canonical_sha256(core)
    if expected != actual:
        raise ValueError(
            f"Seed manifest hash mismatch: expected {expected}, computed {actual}"
        )
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Seed manifest must contain at least one pair")
    keys = [(int(pair["seed"]), int(pair["initial_state_index"])) for pair in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("Seed manifest contains duplicate seed/state pairs")


def write_seed_manifests(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, payload in build_seed_manifests().items():
        path = output_dir / filename
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            validate_manifest(existing)
            if existing != payload:
                raise FileExistsError(
                    f"Refusing to change frozen seed manifest: {path}"
                )
        else:
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def validate_disjoint(manifests: list[dict[str, Any]]) -> None:
    seen: dict[int, str] = {}
    for manifest in manifests:
        validate_manifest(manifest)
        name = str(manifest["manifest_name"])
        for pair in manifest["pairs"]:
            seed = int(pair["seed"])
            if seed in seen:
                raise ValueError(
                    f"Seed {seed} appears in both {seen[seed]!r} and {name!r}"
                )
            seen[seed] = name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that will contain the three immutable seed manifests.",
    )
    args = parser.parse_args()
    paths = write_seed_manifests(args.output_dir)
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    validate_disjoint(manifests)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
