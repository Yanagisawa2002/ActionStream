"""Audit technical and legal gates for an ActionStream public snapshot."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "release" / "public_release_manifest.json"

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    "github_fine_grained_token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "credentialed_url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
}


def _git_lines(*args: str) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [line for line in completed.stdout.splitlines() if line]


def _tracked_files() -> list[Path]:
    return [ROOT / line for line in _git_lines("ls-files")]


def _scan_secrets(paths: list[Path]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file() or path.stat().st_size > 1_048_576:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(
                    {"path": path.relative_to(ROOT).as_posix(), "pattern": name}
                )
    return findings


def audit(*, allow_missing_license: bool) -> tuple[dict, bool]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    limits = manifest["repository_limits"]
    allowlist = [
        re.compile(pattern) for pattern in manifest["public_output_allowlist_regex"]
    ]
    tracked = _tracked_files()

    tracked_rows = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
        }
        for path in tracked
        if path.is_file()
    ]
    output_rows = [row for row in tracked_rows if row["path"].startswith("outputs/")]
    oversized = [
        row for row in tracked_rows if row["bytes"] > limits["max_tracked_file_bytes"]
    ]
    disallowed_outputs = [
        row["path"]
        for row in output_rows
        if not any(pattern.search(row["path"]) for pattern in allowlist)
    ]
    secrets = _scan_secrets(tracked)

    technical_checks = {
        "uv_lock_present": (ROOT / "uv.lock").is_file(),
        "ci_workflow_present": (ROOT / ".github" / "workflows" / "ci.yml").is_file(),
        "tracked_tree_within_limit": (
            sum(row["bytes"] for row in tracked_rows)
            <= limits["max_tracked_tree_bytes"]
        ),
        "tracked_outputs_within_limit": (
            sum(row["bytes"] for row in output_rows)
            <= limits["max_tracked_outputs_bytes"]
        ),
        "no_oversized_tracked_files": not oversized,
        "output_allowlist_clean": not disallowed_outputs,
        "credential_scan_clean": not secrets,
    }
    license_present = (ROOT / "LICENSE").is_file()
    legal_check_passed = license_present or allow_missing_license
    passed = all(technical_checks.values()) and legal_check_passed

    report = {
        "status": "PASS" if passed else "FAIL",
        "technical_checks": technical_checks,
        "legal": {
            "license_present": license_present,
            "missing_license_allowed_for_ci": allow_missing_license,
            "public_release_ready": license_present,
        },
        "tracked": {
            "files": len(tracked_rows),
            "bytes": sum(row["bytes"] for row in tracked_rows),
            "outputs_files": len(output_rows),
            "outputs_bytes": sum(row["bytes"] for row in output_rows),
            "oversized": oversized,
            "disallowed_outputs": disallowed_outputs,
        },
        "credential_findings": secrets,
    }
    return report, passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-missing-license",
        action="store_true",
        help="Pass technical CI while retaining an explicit legal release blocker.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report, passed = audit(allow_missing_license=args.allow_missing_license)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
