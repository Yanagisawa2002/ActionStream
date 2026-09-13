"""Conditional native integration, entered only after both independent real gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

from .contracts import CheckDecision, TaskSpec
from .gate_runner import save, sha256
from .integration import execute
from .qwen import LocalQwen, checker_messages, parser_messages


def main(*, protocol_version="v1"):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model", "output", "base-store", "evidence"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if protocol_version == "v2":
        from .repair_scoring import require_joint_gates

        require_joint_gates(args.evidence)
    else:
        for phase in ("language", "shadow"):
            gate = json.loads((args.evidence / (phase + "_score.json")).read_text())
            if gate["status"] != "PASS" or gate["source_sha256"]["receipt"] != sha256(
                args.evidence / phase / "run_receipt.json"
            ):
                raise ValueError(
                    "Both real gates must pass before native model imports"
                )
    protocol = json.loads(
        (args.evidence / "scorer_only/native_protocol.json").read_text()
    )
    args.output.mkdir(exist_ok=False)
    receipt = dict(
        status="RUNNING",
        episodes=[dict(id=e["id"], status="NOT_RUN") for e in protocol],
    )
    path = args.output / "run_receipt.json"
    save(path, receipt)
    model = backend = None
    started = time.monotonic()
    try:
        model = LocalQwen(args.model, config["model"])
        receipt["qwen_model"] = model.receipt
        for index, episode in enumerate(protocol):
            if index == 2 and any(
                e.get("independent_score", {}).get("status") != "PASS"
                for e in receipt["episodes"][:2]
            ):
                receipt["stop_reason"] = (
                    "Two complete native logs required before interruption"
                )
                break
            directory = args.output / episode["id"]
            directory.mkdir()
            receipt["episodes"][index]["status"] = "RUNNING"
            save(path, receipt)
            parsed = model.generate(
                parser_messages(
                    config["parser_system_prompt"],
                    episode["request_id"],
                    episode["text"],
                )
            )
            save(
                directory / "parser_call.json",
                dict(input_text=episode["text"], **parsed),
            )
            if parsed["status"] != "COMPLETED":
                raise ValueError("Native task parser did not complete")
            authorization = original = None
            if protocol_version == "v2":
                from .grounding import OriginalRequest, adjudicate, materialize

                original = OriginalRequest(episode["request_id"], episode["text"])
                verdict = adjudicate(original, parsed["raw_output"], config["contract"])
                save(directory / "authorization_verdict.json", verdict)
                if verdict["decision"] != "accept":
                    receipt["episodes"][index].update(
                        status="REJECTED",
                        decision=verdict["decision"],
                        reason=verdict["reason"],
                    )
                    continue
                authorization = materialize(
                    original, parsed["raw_output"], config["contract"]
                )
                spec = authorization.control_spec()
            else:
                spec = TaskSpec.parse(parsed["raw_output"], episode["request_id"])
            save(directory / "taskspec.json", asdict(spec))
            if spec.decision != "accept":
                receipt["episodes"][index].update(
                    status="REJECTED", decision=spec.decision
                )
                continue
            if backend is None:
                from actionstream.lerobot_backend import LeRobotBackend
                from actionstream.native_baseline import verify_assets, verify_egl

                native_config = json.loads(
                    (
                        args.evidence.parent
                        / "embodied_implementation_20260913/source/configs/native_baseline_development.json"
                    ).read_text()
                )
                receipt["native_assets"] = verify_assets(native_config, args.base_store)
                receipt["egl"] = verify_egl(args.output)
                backend = LeRobotBackend(
                    task_ids=[5],
                    seed=episode["seed"],
                    suite="libero_object",
                    episode_length=300,
                    model_id=str(args.base_store / "assets/xvla-12e8783"),
                    model_revision=native_config["model_revision"],
                    device="cuda",
                )
            from actionstream.native_baseline import Journal
            from .native_adapter import NativePort, score_native

            journal = Journal(directory / "runtime_trace.jsonl")
            port = NativePort(backend, episode, directory)

            def checker(observation):
                images = observation.checker_images()
                snapshot = directory / observation.observation_id
                snapshot.mkdir()
                for view, image in enumerate(images):
                    image.save(snapshot / f"view{view}.png")
                call = model.generate(
                    checker_messages(
                        config["checker_system_prompt"],
                        observation.observation_id,
                        images,
                    )
                )
                save(
                    snapshot / "checker_call.json",
                    dict(
                        binding=observation.binding(),
                        transformed_frame_sha256=[
                            sha256(snapshot / f"view{view}.png") for view in range(2)
                        ],
                        **call,
                    ),
                )
                if call["status"] != "COMPLETED":
                    raise ValueError("Checker model call failed")
                decision = CheckDecision.parse(
                    call["raw_output"], observation.observation_id
                )
                return decision, {
                    "call_wall_s": call["wall_s"],
                    "raw_output": call["raw_output"],
                }

            try:
                if protocol_version == "v2":
                    from .grounding import execute_authorized

                    result = execute_authorized(
                        original,
                        authorization,
                        config["contract"],
                        port,
                        checker,
                        journal.emit,
                        interruption_step=67 if episode["interruption"] else None,
                    )
                else:
                    result = execute(
                        spec,
                        port,
                        checker,
                        journal.emit,
                        interruption_step=67 if episode["interruption"] else None,
                    )
            finally:
                port.close()
                journal.close()
            save(directory / "episode_receipt.json", result)
            evaluation = score_native(directory, result)
            save(directory / "independent_score.json", evaluation)
            receipt["episodes"][index].update(**result, independent_score=evaluation)
            save(path, receipt)
            if evaluation["status"] != "PASS":
                receipt["stop_reason"] = (
                    "Native error, false-complete or trace identity failure"
                )
                break
        receipt["status"] = (
            "STOPPED_BY_GATE" if "stop_reason" in receipt else "COMPLETED"
        )
        return 0
    except BaseException as exc:
        for row in receipt["episodes"]:
            if row["status"] == "RUNNING":
                row.update(
                    status="ERROR", error_type=type(exc).__name__, error=str(exc)
                )
        receipt.update(
            status="ERROR",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                receipt.update(status="ERROR", native_close_error=repr(exc))
        if model is not None:
            receipt["peak_torch_allocated_MiB"] = (
                model.torch.cuda.max_memory_allocated() / 1024**2
            )
            try:
                model.close()
            except Exception as exc:
                receipt.update(status="ERROR", qwen_close_error=repr(exc))
        receipt["wall_s"] = time.monotonic() - started
        save(path, receipt)


if __name__ == "__main__":
    raise SystemExit(main())
