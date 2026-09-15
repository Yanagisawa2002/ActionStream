"""Run one finite language-to-VLA Agent request in LIBERO simulation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .coverage import adjudicate_covered, authorize
from .finite_agent import CHECKPOINT_SHA256, execute_request
from .finite_native import SimulationPort, digest, save
from .grounding import OriginalRequest


def parse_request(llm, config, original):
    from .qwen import parser_messages

    call = llm.generate(
        parser_messages(
            config["parser_system_prompt"], original.request_id, original.text
        )
    )
    verdict = (
        adjudicate_covered(original, call["raw_output"], config["contract"])
        if call["status"] == "COMPLETED"
        else dict(
            decision="error", schema_status="EXPLICIT_FAILURE", reason="model_error"
        )
    )
    return dict(original=asdict(original), call=call, verdict=verdict)


def verify_assets(assets, manifest, checkpoint):
    if digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("This Agent requires the frozen adapted checkpoint")
    for row in manifest["files"]:
        if digest(Path(assets) / row["destination"]) != row["sha256"]:
            raise ValueError("Runtime asset mismatch: " + row["destination"])


def backend_factory(assets, seed):
    from actionstream.lerobot_backend import LeRobotBackend

    return LeRobotBackend(
        task_ids=[5],
        seed=seed,
        suite="libero_object",
        episode_length=400,
        model_id=str(Path(assets) / "xvla"),
        model_revision="12e8783e996944f5c97e490d37d4c145484ed70a",
        device="cuda",
    )


def execute_parsed(parsed, config, seed, output, port_factory, predictor):
    """Shared application boundary used by the CLI and heldout runner."""
    original = OriginalRequest(**parsed["original"])
    # Re-adjudicate from the immutable text/raw response, never trust stored accept flags.
    if parsed["call"]["status"] != "COMPLETED":
        result = dict(
            status="LANGUAGE_ERROR",
            request_id=original.request_id,
            backend_created=False,
        )
        save(output / "outcome.json", result)
        return result
    verdict = adjudicate_covered(
        original, parsed["call"]["raw_output"], config["contract"]
    )
    if verdict["decision"] != "accept":
        result = dict(
            status="LANGUAGE_BLOCKED",
            request_id=original.request_id,
            decision=verdict["decision"],
            backend_created=False,
        )
        save(output / "outcome.json", result)
        return result
    permit = authorize(original, parsed["call"]["raw_output"], config["contract"])
    with (output / "runtime.jsonl").open("x") as journal:

        def emit(event, **values):
            journal.write(
                json.dumps(dict(event=event, **values), allow_nan=False) + "\n"
            )
            journal.flush()

        result = execute_request(
            original, permit, config["contract"], port_factory, predictor, emit
        )
    save(output / "outcome.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--seed", type=int, required=True)
    for name in ("assets", "checkpoint", "language-config", "asset-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.language_config.read_text())
    verify_assets(
        args.assets, json.loads(args.asset_manifest.read_text()), args.checkpoint
    )
    from actionstream.libero_config import ensure_isolated_libero_config
    import os

    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(args.output / "libero-config")
    import torch
    from .qwen import LocalQwen
    from .temporal_completion import TemporalPredictor

    torch.set_num_threads(8)
    llm = LocalQwen(args.assets / "qwen", config["model"])
    try:
        parsed = parse_request(
            llm, config, OriginalRequest("cli-request", args.request)
        )
        save(args.output / "parser_call.json", parsed)
    finally:
        llm.close()
    backend = None

    def make_port():
        nonlocal backend
        backend = backend_factory(args.assets, args.seed)
        return SimulationPort(backend, args.seed, args.output)

    try:
        predictor = TemporalPredictor(args.checkpoint, CHECKPOINT_SHA256)
        result = execute_parsed(
            parsed, config, args.seed, args.output, make_port, predictor
        )
        evaluated_success = False
        if (args.output / "private_truth.json").exists() and result[
            "status"
        ] != "ERROR":
            from .finite_scoring import score_episode

            score = score_episode(args.output)
            save(args.output / "independent_score.json", score)
            evaluated_success = (
                result["status"] == "complete"
                and score["integrity_passed"]
                and not score["premature_stop"]
                and score["post_stop_stable"] is True
            )
            result = dict(result, independent_score=score)
        print(json.dumps(result))
        return 0 if evaluated_success else 2
    finally:
        if backend is not None:
            backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
