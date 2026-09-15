"""Restore every saved async physical state without running either learned model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from actionstream.completion_labels import StableTruth, simulator_facts
from actionstream.delivery import digest
from actionstream.libero_config import ensure_isolated_libero_config
from actionstream.llm_vla.async_scoring import score_episode
from actionstream.llm_vla.finite_native import save


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    for phase in ("normal", "recovery"):
        assert read(args.run / phase / "result.json")["requested"] == 10
    protocol = read(args.run / "protocol.json")
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(
        args.output / "libero-config", assets_dir=args.store / "libero-assets"
    )
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()["libero_object"]().get_task(5)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=360,
        camera_widths=360,
    )
    rows = []
    try:
        for directory in sorted(args.run.glob("*/episode-*")):
            phase = directory.parent.name
            if phase not in ("normal", "recovery"):
                continue
            identity = phase + "/" + directory.name
            manifest = read(directory / "files.json")
            for name, entry in manifest.items():
                path = directory / name
                assert path.stat().st_size == entry["bytes"], str(path)
                assert digest(path) == entry["sha256"], str(path)
            fault = (
                protocol["fault"]["controls_exclusive"] if phase == "recovery" else 0
            )
            rescored = score_episode(directory, forced_open_until=fault)
            assert rescored == read(directory / "independent_score.json"), identity
            env.reset()
            env.reset_from_xml_string((directory / "environment.xml").read_text())
            facts = read(directory / "private_truth.json")
            events = [
                json.loads(line)
                for line in (directory / "runtime.jsonl").read_text().splitlines()
            ]
            checks = [
                e for e in events if e["event"] == "verification" and e["history_ready"]
            ]
            truth, restored, changed = StableTruth(), [], []
            with np.load(directory / "trajectory.npz", allow_pickle=False) as data:
                states, rgb = data["states"], data["rgb"]
                assert len(states) == len(rgb) == len(facts)
                for row in checks:
                    c = row["control"]
                    clip = rgb[np.array([c - 10, c - 5, c])]
                    assert (
                        hashlib.sha256(clip.tobytes()).hexdigest() == row["clip_sha256"]
                    )
                for control, state in enumerate(states):
                    env.set_state(state)
                    env.sim.forward()
                    fact = simulator_facts(env)
                    fact["strict_complete"] = truth.update(control, fact)
                    restored.append(fact)
                    fields = [
                        k
                        for k in (
                            "inside",
                            "finger_contact",
                            "basket_contact",
                            "strict_complete",
                        )
                        if fact[k] != facts[control][k]
                    ]
                    fields += [
                        k
                        for k in ("linear_speed", "angular_speed")
                        if not math.isclose(
                            fact[k], facts[control][k], rel_tol=1e-10, abs_tol=1e-12
                        )
                    ]
                    if fields:
                        changed.append(
                            dict(
                                control=control,
                                fields=fields,
                                recorded=facts[control],
                                restored=fact,
                            )
                        )
            save(args.output / (phase + "-" + directory.name + ".json"), restored)
            rows.append(
                dict(
                    id=identity,
                    states=len(facts),
                    clips_checked=len(checks),
                    trajectory_sha256=digest(directory / "trajectory.npz"),
                    scoring_matches=True,
                    changed=changed,
                )
            )
            save(args.output / "progress.json", rows)
            print(
                json.dumps(
                    dict(
                        id=identity,
                        states=len(facts),
                        changed_strict=sum(
                            "strict_complete" in r["fields"] for r in changed
                        ),
                    )
                ),
                flush=True,
            )
    finally:
        env.close()
    strict = sum(
        "strict_complete" in r["fields"] for row in rows for r in row["changed"]
    )
    result = dict(
        status="PASS" if len(rows) == 20 and strict == 0 else "FAIL",
        source_sha256=digest(Path(__file__)),
        freeze_sha256=digest(args.run / "freeze.json"),
        episodes=len(rows),
        states=sum(row["states"] for row in rows),
        clips_checked=sum(row["clips_checked"] for row in rows),
        changed_strict=strict,
        changed_facts=sum(len(row["changed"]) for row in rows),
        records=rows,
    )
    save(args.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
