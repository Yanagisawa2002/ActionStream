"""One frozen paired v1/v2 text comparison and seen-set v2 regression."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import traceback

from .contracts import TaskSpec
from .gate_runner import save, sha256
from .grounding import OriginalRequest, adjudicate
from .qwen import LocalQwen, parser_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "inputs", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    regression_path = args.inputs.parent / "regression.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    if len(inputs) != 48 or len(regression) != 24:
        raise ValueError("Frozen case counts differ")
    schedule = [("new", version, item) for item in inputs for version in ("v1", "v2")]
    schedule += [("regression", "v2", item) for item in regression]
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="RUNNING",
        phase="comparison",
        pid=os.getpid(),
        expected_calls=120,
        config_sha256=sha256(args.config),
        input_sha256=sha256(args.inputs),
        regression_sha256=sha256(regression_path),
        cases=[],
        model_load={"status": "RUNNING"},
    )
    path = args.output / "run_receipt.json"
    save(path, receipt)
    model = None
    started = time.monotonic()
    try:
        model = LocalQwen(args.model, config["model"])
        receipt["model_load"] = dict(status="PASS", **model.receipt)
        save(path, receipt)
        for ordinal, (dataset, version, item) in enumerate(schedule):
            if set(item) != {"request_id", "text"}:
                raise ValueError("Unexpected model-facing metadata")
            system = (
                config["v1_parser_system_prompt"]
                if version == "v1"
                else config["parser_system_prompt"]
            )
            row = dict(
                ordinal=ordinal,
                dataset=dataset,
                version=version,
                identity=item["request_id"],
                original_sha256=OriginalRequest(
                    item["request_id"], item["text"]
                ).text_sha256,
                input=item,
                **model.generate(
                    parser_messages(system, item["request_id"], item["text"])
                ),
            )
            if row["status"] != "COMPLETED":
                row.update(
                    schema_status="EXPLICIT_FAILURE",
                    decision="explicit_failure",
                    reason="model_error",
                )
            elif version == "v1":
                try:
                    task = TaskSpec.parse(row["raw_output"], row["identity"])
                    row.update(
                        schema_status="VALID",
                        decision=task.decision,
                        normalized=asdict(task),
                    )
                except (ValueError, TypeError) as exc:
                    row.update(
                        schema_status="EXPLICIT_FAILURE",
                        decision="explicit_failure",
                        reason="malformed",
                        error=str(exc),
                    )
            else:
                verdict = adjudicate(
                    OriginalRequest(item["request_id"], item["text"]),
                    row["raw_output"],
                    config["contract"],
                )
                row.update(verdict)
            receipt["cases"].append(row)
            save(path, receipt)
            print(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in (
                            "ordinal",
                            "dataset",
                            "version",
                            "identity",
                            "status",
                            "schema_status",
                            "decision",
                            "reason",
                            "wall_s",
                        )
                    }
                ),
                flush=True,
            )
            if row.get("error_type") == "TimeoutError":
                raise TimeoutError(
                    "One-shot text phase stops after timeout; no retries"
                )
        receipt["status"] = "COMPLETED"
        return 0
    except BaseException as exc:
        if model is None:
            receipt["model_load"]["status"] = "ERROR"
        receipt.update(
            status="ERROR",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        print(receipt["traceback"], flush=True)
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
