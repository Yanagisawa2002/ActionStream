"""Real one-pass inference over sealed public inputs; never reads scorer files."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .contracts import CheckDecision, TaskSpec
from .qwen import LocalQwen, checker_messages, parser_messages


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("language", "shadow"), required=True)
    parser.add_argument("--diagnostic-only", action="store_true")
    for name in ("config", "inputs", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cases = json.loads(args.inputs.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="RUNNING",
        execution_mode="DIAGNOSTIC_ONLY" if args.diagnostic_only else "GATED_PREFLIGHT",
        phase=args.phase,
        pid=os.getpid(),
        executable=sys.executable,
        config_sha256=sha256(args.config),
        input_sha256=sha256(args.inputs),
        expected_cases=len(cases),
        cases=[],
        model_load={"status": "NOT_RUN"},
    )
    path = args.output / "run_receipt.json"
    save(path, receipt)
    started = time.monotonic()
    model = None
    try:
        receipt["model_load"]["status"] = "RUNNING"
        save(path, receipt)
        model = LocalQwen(args.model, config["model"])
        receipt["model_load"] = dict(status="PASS", **model.receipt)
        save(path, receipt)
        for ordinal, case in enumerate(cases):
            if args.phase == "language":
                if set(case) != {"request_id", "text"}:
                    raise ValueError("Unexpected model-facing language metadata")
                identity = case["request_id"]
                messages = parser_messages(
                    config["parser_system_prompt"], identity, case["text"]
                )
            else:
                from PIL import Image

                if set(case) != {"observation_id", "views"}:
                    raise ValueError("Unexpected model-facing shadow metadata")
                identity = case["observation_id"]
                images = []
                for view in case["views"]:
                    source = (args.inputs.parent / view["path"]).resolve()
                    if (
                        not source.is_relative_to(args.inputs.parent.resolve())
                        or sha256(source) != view["sha256"]
                    ):
                        raise ValueError("Shadow frame identity mismatch")
                    with Image.open(source) as image:
                        images.append(image.convert("RGB"))
                messages = checker_messages(
                    config["checker_system_prompt"], identity, images
                )
            row = dict(
                ordinal=ordinal,
                identity=identity,
                input=case,
                **model.generate(messages),
            )
            if row["status"] == "COMPLETED":
                contract = TaskSpec if args.phase == "language" else CheckDecision
                try:
                    row["normalized"] = asdict(
                        contract.parse(row["raw_output"], identity)
                    )
                    row["schema_status"] = "VALID"
                except (ValueError, TypeError) as exc:
                    row.update(schema_status="EXPLICIT_FAILURE", schema_error=str(exc))
            else:
                row["schema_status"] = "EXPLICIT_FAILURE"
            receipt["cases"].append(row)
            save(path, receipt)
            print(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "ordinal",
                            "identity",
                            "status",
                            "schema_status",
                            "normalized",
                            "wall_s",
                        )
                    }
                ),
                flush=True,
            )
            if row.get("error_type") == "TimeoutError":
                raise TimeoutError("Stop the phase after a model timeout; no retries")
        receipt["status"] = "COMPLETED"
        return 0
    except BaseException as exc:
        if receipt["model_load"]["status"] == "RUNNING":
            receipt["model_load"]["status"] = "ERROR"
        receipt.update(
            status="ERROR",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(receipt["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        if model is not None:
            receipt["peak_torch_allocated_MiB"] = (
                model.torch.cuda.max_memory_allocated() / 1024**2
            )
            model.close()
        receipt["wall_s"] = time.monotonic() - started
        save(path, receipt)


if __name__ == "__main__":
    raise SystemExit(main())
