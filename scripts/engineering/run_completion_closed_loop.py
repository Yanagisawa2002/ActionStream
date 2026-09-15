"""Explicit diagnostic simulation: real Qwen -> X-VLA -> frozen RGB completion.

This experiment has its own protocol and does not rewrite historical release gates.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

from actionstream.llm_vla.completion import FrozenCompletion, native_rgb
from actionstream.llm_vla.coverage import adjudicate_covered, authorize, execute_covered
from actionstream.llm_vla.grounding import OriginalRequest
from actionstream.llm_vla.qwen import LocalQwen, parser_messages
from evaluate_visual_completion import digest


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["base", "output", "protocol", "language-config", "asset-manifest"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    config = json.loads(args.language_config.read_text())
    assert protocol["closed_loop"]["max_control_steps"] == 300
    assert protocol["closed_loop"]["controls_between_checks"] == 30
    assert protocol["closed_loop"]["unknown_behavior"] == "stop"
    assert protocol["thresholds"] == {"incomplete": 0.1, "complete": 0.9}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_bytes(args.protocol.read_bytes())
    (args.output / "language_config.json").write_bytes(
        args.language_config.read_bytes()
    )
    manifest = json.loads(args.asset_manifest.read_text())
    for entry in manifest["files"]:
        if digest(args.base / "assets" / entry["destination"]) != entry["sha256"]:
            raise ValueError("Pinned runtime asset mismatch: " + entry["destination"])
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    from actionstream.libero_config import ensure_isolated_libero_config

    ensure_isolated_libero_config(args.output / "libero-config")
    import torch

    torch.set_num_threads(8)
    receipt = dict(
        status="RUNNING",
        mode="DIAGNOSTIC_SIMULATION",
        release_gates_changed=False,
        protocol_sha256=digest(args.protocol),
        runner_sha256=digest(__file__),
        asset_manifest_sha256=digest(args.asset_manifest),
        language_config_sha256=digest(args.language_config),
        packages={
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "torchvision",
                "transformers",
                "lerobot",
                "hf-libero",
                "mujoco",
                "robosuite",
                "numpy",
                "Pillow",
            ]
        },
        episodes=[],
    )
    save(args.output / "run_receipt.json", receipt)
    llm = backend = None
    started = time.monotonic()
    try:
        llm = LocalQwen(args.base / "assets/qwen", config["model"])
        receipt["language_model"] = llm.receipt
        requests = []
        for i, text in enumerate(protocol["closed_loop"]["instructions"]):
            directory = args.output / f"episode_{i + 1}"
            directory.mkdir()
            original = OriginalRequest(f"completion_diag_{i + 1}", text)
            parsed = llm.generate(
                parser_messages(
                    config["parser_system_prompt"], original.request_id, text
                )
            )
            save(directory / "parser_call.json", dict(input_text=text, **parsed))
            verdict = (
                adjudicate_covered(original, parsed["raw_output"], config["contract"])
                if parsed["status"] == "COMPLETED"
                else dict(decision="error")
            )
            save(directory / "authorization.json", verdict)
            record = dict(
                id=directory.name,
                instruction=text,
                language_decision=verdict["decision"],
                status="PENDING"
                if verdict["decision"] == "accept"
                else "LANGUAGE_REJECTED",
            )
            receipt["episodes"].append(record)
            requests.append((original, parsed, verdict, directory, record))
        llm.close()
        llm = None
        checker = FrozenCompletion(args.base / "runs/completion-v1/best.pt")
        from actionstream.native_baseline import Journal, array_stats
        from actionstream.llm_vla.native_adapter import NativePort, score_native
        from PIL import Image

        # Independent evaluator extension, never included in the public observation.
        class ScoredPort(NativePort):
            def step(self, *values):
                result = super().step(*values)
                simulator = self.backend._sub_env(5)._env.env
                target = simulator.objects_dict["tomato_sauce_1"]
                held = bool(
                    simulator._check_grasp(
                        simulator.robots[0].gripper, target.contact_geoms
                    )
                )
                self.oracle.emit(
                    "release_audit",
                    control_step=self.last_control_step,
                    gripper_contact_grasp=held,
                )
                return result

        for i, (original, parsed, verdict, directory, record) in enumerate(requests):
            if verdict["decision"] != "accept":
                continue
            permit = authorize(original, parsed["raw_output"], config["contract"])
            port = None

            def factory():
                nonlocal backend, port
                if backend is None:
                    from actionstream.lerobot_backend import LeRobotBackend

                    backend = LeRobotBackend(
                        task_ids=[5],
                        seed=protocol["closed_loop"]["seeds"][i],
                        suite="libero_object",
                        episode_length=300,
                        model_id=str(args.base / "assets/xvla"),
                        model_revision="12e8783e996944f5c97e490d37d4c145484ed70a",
                        device="cuda",
                    )
                state_index = protocol["closed_loop"]["initial_state_indices"][i]
                episode = dict(
                    id=directory.name,
                    initial_state_index=state_index,
                    seed=protocol["closed_loop"]["seeds"][i],
                    expected_init_state_sha256=array_stats(
                        backend._sub_env(5)._init_states[state_index]
                    )["sha256"],
                )
                save(directory / "reset_contract.json", episode)
                port = ScoredPort(backend, episode, directory)
                return port

            def rgb_checker(observation):
                frames = native_rgb(observation.pixels)
                snapshot = directory / observation.observation_id
                snapshot.mkdir()
                for view in range(2):
                    Image.fromarray(frames[view]).save(snapshot / f"view{view}.png")
                return checker(observation)

            journal = Journal(directory / "runtime_trace.jsonl")
            try:
                result = execute_covered(
                    original,
                    permit,
                    config["contract"],
                    factory,
                    rgb_checker,
                    journal.emit,
                )
            finally:
                if port is not None:
                    port.close()
                journal.close()
            save(directory / "episode_receipt.json", result)
            score = score_native(directory, result)
            oracle = [
                json.loads(line)
                for line in (directory / "oracle_trace.jsonl").read_text().splitlines()
            ]
            held = {
                r["control_step"]: r["gripper_contact_grasp"]
                for r in oracle
                if r["event"] == "release_audit"
            }
            score["complete_while_held"] = sum(
                c["checker"] == "complete" and held.get(c["control_step"], True)
                for c in score["claims"]
            )
            score["independently_confirmed_completion"] = bool(
                result["checker_complete"]
                and score["final_native_success"] is True
                and score["false_complete"] == 0
                and score["complete_while_held"] == 0
                and score["complete_log"]
            )
            save(directory / "independent_score.json", score)
            record.update(**result, independent_score=score)
            print(json.dumps(record), flush=True)
            save(args.output / "run_receipt.json", receipt)
        receipt["status"] = "COMPLETED"
        receipt["all_three_confirmed"] = len(receipt["episodes"]) == 3 and all(
            e.get("independent_score", {}).get(
                "independently_confirmed_completion", False
            )
            for e in receipt["episodes"]
        )
        receipt["acceptance"] = "PASS" if receipt["all_three_confirmed"] else "FAIL"
    except BaseException as exc:
        receipt.update(
            status="ERROR", error=repr(exc), traceback=traceback.format_exc()
        )
        raise
    finally:
        if llm is not None:
            llm.close()
        if backend is not None:
            backend.close()
        receipt["elapsed_seconds"] = time.monotonic() - started
        save(args.output / "run_receipt.json", receipt)


if __name__ == "__main__":
    main()
