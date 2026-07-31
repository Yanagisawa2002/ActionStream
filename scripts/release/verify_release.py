"""Verify v1.0.0 assets without mutating frozen benchmark evidence."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = ROOT / "release" / "v1.0.0"
MANIFEST_PATH = RELEASE_ROOT / "frozen_git_objects.json"
EXPECTED_BASELINE = "d84cb64e9e48b681e083828b24df5a38709c78c8"
EXPECTED_FROZEN_PATHS = frozenset(
    {
        "outputs/m3",
        "outputs/m4",
        "outputs/summary",
        "scripts/bootstrap_wsl.sh",
        "scripts/run_custom_sync_parity.sh",
        "scripts/run_custom_sync_smoke.sh",
        "scripts/run_m3_matrix.sh",
        "scripts/run_m3_smoke.sh",
        "scripts/run_m4_calibration.sh",
        "scripts/run_m4_pressure.sh",
        "scripts/run_m4_sync_hold.sh",
        "scripts/run_official_baseline.sh",
        "scripts/run_official_smoke.sh",
        "scripts/run_preflight.sh",
        "src/actionstream/benchmark.py",
        "src/actionstream/lerobot_backend.py",
        "src/actionstream/libero_config.py",
        "src/actionstream/m4_analysis.py",
        "src/actionstream/m4_calibration.py",
        "src/actionstream/m4_report.py",
        "src/actionstream/preflight.py",
        "src/actionstream/processors.py",
        "src/actionstream/results.py",
        "src/actionstream/runtime.py",
    }
)

TEXT_SUFFIXES = {
    ".cff",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PRIVATE_PATH_PATTERNS = (
    re.compile(
        r"(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+(?:[\\/]|$)"
    ),
    re.compile(r"/mnt/[a-z]/Users/[A-Za-z0-9._-]+(?:/|$)"),
    re.compile(r"/home/[A-Za-z0-9._-]+(?:/|$)"),
)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_manifest_path(value: str) -> Path:
    if Path(value).is_absolute():
        raise AssertionError(f"Manifest path must be relative: {value}")
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise AssertionError(f"Manifest path escapes repository: {value}") from error
    return candidate


def verify_frozen_objects() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    baseline = manifest["baseline_commit"]
    if baseline != EXPECTED_BASELINE:
        raise AssertionError(
            f"Frozen baseline changed: {baseline} != {EXPECTED_BASELINE}"
        )
    actual_paths = frozenset(manifest["objects"])
    if actual_paths != EXPECTED_FROZEN_PATHS:
        missing = sorted(EXPECTED_FROZEN_PATHS - actual_paths)
        unexpected = sorted(actual_paths - EXPECTED_FROZEN_PATHS)
        raise AssertionError(
            f"Frozen path set changed; missing={missing}, unexpected={unexpected}"
        )
    for path, expected_object in manifest["objects"].items():
        baseline_object = git("rev-parse", f"{baseline}:{path}").stdout.strip()
        if baseline_object != expected_object:
            raise AssertionError(
                f"Frozen manifest mismatch for {path}: "
                f"{baseline_object} != {expected_object}"
            )
        diff = git("diff", "--quiet", baseline, "--", path, check=False)
        if diff.returncode != 0:
            raise AssertionError(f"Frozen path changed relative to {baseline}: {path}")
        untracked = git(
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            path,
        ).stdout.strip()
        if untracked:
            raise AssertionError(f"Untracked content exists under frozen path: {path}")


def verify_versions() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if project["project"]["version"] != "1.0.0":
        raise AssertionError("pyproject version is not 1.0.0")
    init_text = (ROOT / "src" / "actionstream" / "__init__.py").read_text(
        encoding="utf-8"
    )
    if '__version__ = "1.0.0"' not in init_text:
        raise AssertionError("package __version__ is not 1.0.0")


def iter_maintained_text_files():
    roots = (
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
        ROOT / "pyproject.toml",
        ROOT / "requirements-release.txt",
        ROOT / "scripts",
        RELEASE_ROOT,
    )
    for path in roots:
        if path.is_file():
            yield path
            continue
        for child in path.rglob("*"):
            if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
                yield child


def verify_portable_paths() -> None:
    violations: list[str] = []
    for path in iter_maintained_text_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in PRIVATE_PATH_PATTERNS):
                violations.append(f"{path.relative_to(ROOT)}:{line_number}")
    if violations:
        raise AssertionError(
            "Host-specific paths remain in maintained release files: "
            + ", ".join(violations)
        )


def verify_report() -> None:
    report_path = ROOT / "outputs" / "m4" / "report" / "m4_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "validated" or report["validation"]["passed"] is not True:
        raise AssertionError("Frozen M4 report is not validated")


def verify_asset_manifest() -> None:
    path = RELEASE_ROOT / "asset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["release"] != "v1.0.0":
        raise AssertionError("Asset manifest release is not v1.0.0")
    for section in ("inputs", "outputs", "artifacts"):
        for item in manifest[section]:
            asset_path = resolve_manifest_path(item["path"])
            if not asset_path.is_file():
                raise AssertionError(f"Manifest asset is missing: {item['path']}")
            actual = sha256(asset_path)
            if actual != item["sha256"]:
                raise AssertionError(
                    f"Manifest hash mismatch for {item['path']}: "
                    f"{actual} != {item['sha256']}"
                )


def verify_evidence_summary() -> None:
    path = RELEASE_ROOT / "evidence_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary["release"] != "v1.0.0":
        raise AssertionError("Evidence summary release is not v1.0.0")
    source = resolve_manifest_path(summary["source"]["path"])
    if sha256(source) != summary["source"]["sha256"]:
        raise AssertionError("Evidence summary source hash does not match")
    if summary["source"]["status"] != "validated":
        raise AssertionError("Evidence summary source is not validated")


def verify_assets() -> None:
    required = (
        RELEASE_ROOT / "figures" / "m4_pressure_headline.pdf",
        RELEASE_ROOT / "figures" / "m4_pressure_headline.png",
        RELEASE_ROOT / "figures" / "m3_latency_effects.pdf",
        RELEASE_ROOT / "figures" / "m3_latency_effects.png",
        RELEASE_ROOT / "figures" / "queue_pressure_calibration.pdf",
        RELEASE_ROOT / "figures" / "queue_pressure_calibration.png",
        RELEASE_ROOT / "media" / "actionstream_v1_demo.mp4",
        RELEASE_ROOT / "media" / "actionstream_v1_demo_poster.png",
        RELEASE_ROOT / "asset_manifest.json",
        RELEASE_ROOT / "evidence_summary.json",
        RELEASE_ROOT / "RELEASE_NOTES.md",
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty release asset: {path}")

    for path in RELEASE_ROOT.glob("figures/*.pdf"):
        if not path.read_bytes().startswith(b"%PDF"):
            raise AssertionError(f"Invalid PDF signature: {path}")
    png_signature = b"\x89PNG\r\n\x1a\n"
    for path in list(RELEASE_ROOT.glob("figures/*.png")) + [
        RELEASE_ROOT / "media" / "actionstream_v1_demo_poster.png"
    ]:
        if not path.read_bytes().startswith(png_signature):
            raise AssertionError(f"Invalid PNG signature: {path}")
    video_head = (RELEASE_ROOT / "media" / "actionstream_v1_demo.mp4").read_bytes()[
        :64
    ]
    if b"ftyp" not in video_head:
        raise AssertionError("Demo video is not an MP4 container")


def main() -> None:
    verify_frozen_objects()
    verify_versions()
    verify_portable_paths()
    verify_report()
    verify_asset_manifest()
    verify_evidence_summary()
    verify_assets()
    print("Release verification passed.")
    print(f"Frozen runtime and M3/M4 evidence match {EXPECTED_BASELINE[:7]}.")
    print("Maintained release files contain no host-specific user paths.")


if __name__ == "__main__":
    main()
