"""Collect development episodes or run a task-conditioned integration candidate.

No acceptance or heldout-task mode exists until development gates are frozen.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from .async_agent import AsyncAgentConfig
from .finite_native import digest, save
from .grounding import OriginalRequest
from .multitask_agent import execute_task
from .multitask_language import parse_request, authorize_task, validate_permit
from .task_completion import TaskPredictor, encode_task


class CollectionPredictor:
    """Never claims success/recovery: observe a bounded VLA attempt to its horizon."""

    checkpoint_sha256 = None

    def __init__(self, task, condition):
        self.task, self.condition = task, condition

    def validate_binding(self, task):
        if self.task != task:
            raise ValueError("Foreign collection task")
        self.condition.validate(task)

    def predict(self, clips):
        return np.tile([0.0, 0.0, 1.0], (len(clips), 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect", "evaluate"), required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--recovery", action="store_true")
    parser.add_argument("--force-open", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--consumed-manifest", type=Path, required=True)
    for name in (
        "assets",
        "asset-manifest",
        "libero-assets",
        "libero-asset-manifest",
        "language-config",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "collect" and args.recovery:
        parser.error(
            "Collection uses one bounded attempt; recovery requires the learned RGB model"
        )
    if args.mode == "evaluate" and (
        not args.checkpoint or not args.checkpoint_sha256 or args.split != "validation"
    ):
        parser.error(
            "Candidate evaluation requires a hashed checkpoint and validation split"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "purpose.json", dict(mode=args.mode, split=args.split))
    from actionstream.delivery import environment, verify

    runtime = environment(require_cuda=True)
    save(args.output / "environment.json", runtime)
    if runtime["status"] != "PASS":
        raise RuntimeError("Runtime differs from the pinned CUDA environment")
    for name, manifest, root in (
        ("model_assets", args.asset_manifest, args.assets),
        ("libero_assets", args.libero_asset_manifest, args.libero_assets),
    ):
        receipt = verify(json.loads(manifest.read_text()), root)
        save(args.output / (name + ".json"), receipt)
        if receipt["status"] != "PASS":
            raise RuntimeError("Asset verification failed: " + name)
    from actionstream.libero_config import ensure_isolated_libero_config
    from .qwen import LocalQwen
    import torch

    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(
        args.output / "libero-config", assets_dir=args.libero_assets
    )
    torch.set_num_threads(8)
    config = json.loads(args.language_config.read_text())
    llm = LocalQwen(args.assets / "qwen", config["model"])
    try:
        original = OriginalRequest("cli-request", args.request)
        parsed = parse_request(llm, original)
        save(args.output / "parser_call.json", parsed)
        if parsed["verdict"]["decision"] != "accept":
            save(
                args.output / "outcome.json",
                dict(status="LANGUAGE_BLOCKED", backend_created=False),
            )
            return 2
        permit = authorize_task(original, parsed["call"]["raw_output"])
        task = validate_permit(original, permit)
        condition = encode_task(llm, task)
        save(args.output / "condition.json", asdict(condition))
    finally:
        llm.close()
    predictor = (
        CollectionPredictor(task, condition)
        if args.mode == "collect"
        else TaskPredictor(args.checkpoint, args.checkpoint_sha256, task, condition)
    )
    consumed = json.loads(args.consumed_manifest.read_text())
    save(args.output / "consumed_manifest.json", consumed)
    # Unscoped legacy hashes are excluded conservatively from every task.
    excluded = {
        row["layout_sha256"]
        for row in consumed["episodes"]
        if row.get("task_key") in (None, task.key)
    }
    backend = port = None
    from .multitask_native import backend_factory, TaskSimulationPort

    def make_port(authorized_task):
        nonlocal backend, port
        backend = backend_factory(args.assets, args.seed, authorized_task)
        port = TaskSimulationPort(
            backend,
            args.seed,
            args.output,
            authorized_task,
            excluded,
            forced_open_until=300 if args.force_open else 0,
        )
        return port

    try:
        with (args.output / "runtime.jsonl").open("x") as journal:

            def emit(event, **values):
                journal.write(
                    json.dumps(dict(event=event, **values), allow_nan=False) + "\n"
                )
                journal.flush()

            outcome = execute_task(
                original,
                permit,
                make_port,
                predictor,
                emit,
                AsyncAgentConfig(max_attempts=2 if args.recovery else 1),
            )
        save(args.output / "outcome.json", outcome)
        if outcome["status"] == "ERROR":
            return 2
        if args.mode == "collect":
            if outcome["status"] != "unconfirmed_horizon":
                raise RuntimeError(
                    "Incomplete collection; the bounded horizon was not reached"
                )
            from .task_truth import target_visibility

            # Recorder is closed; all privileged replay happens after online execution.
            visible = []
            for state in port.states:
                port.env.set_init_state(state)
                visible.append(target_visibility(port.env, task))
            save(args.output / "visibility.json", visible)
            from .multitask_data import episode_entry

            save(
                args.output / "dataset_entry.json",
                episode_entry(args.output, args.split),
            )
            print(json.dumps(dict(status="COLLECTED", task=task.key, acceptance=False)))
            return 0
        from .multitask_scoring import score_episode

        score = score_episode(args.output)
        save(args.output / "independent_score.json", score)
        print(
            json.dumps(
                dict(status="CANDIDATE_EVALUATED", acceptance=False, score=score)
            )
        )
        return 0 if score["safely_completed"] else 2
    finally:
        if backend is not None:
            backend.close()
        save(
            args.output / "files.json",
            {
                path.name: dict(bytes=path.stat().st_size, sha256=digest(path))
                for path in sorted(args.output.iterdir())
                if path.is_file() and path.name != "files.json"
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
