"""Independently restore one retained task episode; do not load either model."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from actionstream.llm_vla.finite_native import digest, save
from actionstream.llm_vla.multitask_scoring import score_episode
from actionstream.llm_vla.task_registry import development_task
from actionstream.llm_vla.task_truth import PROFILES, StableTaskTruth, simulator_facts


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--libero-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, entry in read(args.episode / "files.json").items():
        path = args.episode / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError("Episode file changed: " + name)
    identity = read(args.episode / "task.json")
    task = development_task(identity["task"]["key"])
    scored = score_episode(args.episode)
    save(args.output / "score.json", scored)
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    from actionstream.libero_config import ensure_isolated_libero_config

    ensure_isolated_libero_config(
        args.output / "libero-config", assets_dir=args.libero_assets
    )
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = (
        Path(get_libero_path("bddl_files"))
        / task.suite
        / (task.canonical_instruction.replace(" ", "_") + ".bddl")
    )
    if digest(bddl) != PROFILES[task.key]["bddl_sha256"]:
        raise ValueError("Native task definition changed")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=360, camera_widths=360
    )
    recorded = read(args.episode / "private_truth.json")
    restored, changed = [], []
    try:
        env.reset()
        env.reset_from_xml_string((args.episode / "environment.xml").read_text())
        truth = StableTaskTruth()
        with np.load(args.episode / "trajectory.npz", allow_pickle=False) as data:
            if len(data["states"]) != len(recorded):
                raise ValueError("Physical state coverage changed")
            for control, state in enumerate(data["states"]):
                env.set_state(state)
                env.sim.forward()
                facts = simulator_facts(env, task)
                facts["strict_complete"] = truth.update(control, facts)
                restored.append(facts)
                fields = [
                    key
                    for key in (
                        "goal_satisfied",
                        "support_contact",
                        "finger_contact",
                        "strict_complete",
                    )
                    if facts[key] != recorded[control][key]
                ]
                fields += [
                    key
                    for key in ("linear_speed", "angular_speed")
                    if not math.isclose(
                        facts[key], recorded[control][key], rel_tol=1e-10, abs_tol=1e-12
                    )
                ]
                if fields:
                    changed.append(
                        dict(
                            control=control,
                            fields=fields,
                            recorded=recorded[control],
                            restored=facts,
                        )
                    )
    finally:
        env.close()
    strict_changes = sum("strict_complete" in row["fields"] for row in changed)
    result = dict(
        status="PASS" if scored["integrity_passed"] and strict_changes == 0 else "FAIL",
        task_key=task.key,
        states=len(restored),
        changed_strict=strict_changes,
        changes=changed,
        source_sha256=digest(__file__),
        trajectory_sha256=digest(args.episode / "trajectory.npz"),
        episode_manifest_sha256=digest(args.episode / "files.json"),
        scope="physical label restoration and retained RGB identity; not model acceptance",
    )
    save(args.output / "restored_truth.json", restored)
    save(args.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "changes"}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
