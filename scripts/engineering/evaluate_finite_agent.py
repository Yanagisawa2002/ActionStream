"""Freeze, then run new language requests through the same public Agent as its CLI."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from actionstream.llm_vla.agent_cli import (
    backend_factory,
    execute_parsed,
    parse_request,
    verify_assets,
)
from actionstream.llm_vla.finite_agent import CHECKPOINT_SHA256
from actionstream.llm_vla.finite_native import SimulationPort, digest, layout, save
from actionstream.llm_vla.finite_scoring import score_episode, score_run
from actionstream.llm_vla.grounding import OriginalRequest

ROOT = Path(__file__).resolve().parents[2]
PRIOR = (
    "completion-v2/fresh_holdout/manifest.json",
    "development-controls/collection/manifest.json",
    "completion-acceptance-v3/collection/manifest.json",
)


def read(path):
    return json.loads(Path(path).read_text())


def source_paths():
    return sorted(
        list((ROOT / "src/actionstream").rglob("*.py"))
        + [
            Path(__file__).resolve(),
            ROOT / "configs/finite_agent_language.json",
            ROOT / "configs/finite_agent_acceptance_v1.json",
            ROOT / "configs/completion_runtime_assets.json",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        ]
    )


def verify_freeze(args):
    frozen = read(args.output / "freeze.json")
    if frozen["protocol_sha256"] != digest(args.protocol):
        raise ValueError("Frozen protocol changed")
    for relative, expected in frozen["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    if digest(args.output / "frozen.pt") != CHECKPOINT_SHA256:
        raise ValueError("Frozen completion checkpoint changed")
    for relative, expected in frozen["prior_manifests"].items():
        if digest(args.base / relative) != expected:
            raise ValueError("Consumed data manifest changed")
    return frozen


def freeze(args, protocol):
    args.output.mkdir(parents=True, exist_ok=False)
    if protocol["checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("Unexpected completion candidate")
    checkpoint = args.base / "completion-acceptance-v3/frozen.pt"
    verify_assets(
        args.base / "assets", read(ROOT / protocol["assets_manifest"]), checkpoint
    )
    shutil.copyfile(checkpoint, args.output / "frozen.pt")
    shutil.copyfile(args.protocol, args.output / "protocol.json")
    shutil.copyfile(
        ROOT / protocol["language_config"], args.output / "language_config.json"
    )
    sources = {str(p.relative_to(ROOT).as_posix()): digest(p) for p in source_paths()}
    for name in sources:
        dest = args.output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    save(
        args.output / "freeze.json",
        dict(
            status="FROZEN_BEFORE_NEW_LANGUAGE_OR_LAYOUTS",
            protocol_sha256=digest(args.protocol),
            checkpoint_sha256=CHECKPOINT_SHA256,
            source_sha256=sources,
            prior_manifests={name: digest(args.base / name) for name in PRIOR},
            original_demonstrations_sha256=digest(args.base / "data/tomato-xet.hdf5"),
            frozen_unix=time.time(),
            new_data_seen=False,
        ),
    )
    import torch
    from actionstream.llm_vla.temporal_completion import TemporalPredictor

    torch.set_num_threads(8)
    model = TemporalPredictor(args.output / "frozen.pt", CHECKPOINT_SHA256)
    with np.load(
        args.base / "development-controls/collection/validation_2026091808.npz",
        allow_pickle=False,
    ) as d:
        clip = d["rgb"][np.array([0, 5, 10])][None]
        probabilities = model.predict(clip)
    if probabilities.shape != (1, 3) or not np.isfinite(probabilities).all():
        raise ValueError("Consumed-clip inference smoke failed")
    save(
        args.output / "smoke.json",
        dict(
            status="PASS",
            source="previously consumed development validation_2026091808 controls 0/5/10",
            probabilities=probabilities.tolist(),
            checkpoint_sha256=CHECKPOINT_SHA256,
            completed_unix=time.time(),
        ),
    )
    print("FINITE_AGENT_FROZEN_AND_SMOKE_PASS", flush=True)


def excluded_layouts(args, backend):
    import h5py

    excluded = set()
    for relative in PRIOR:
        for entry in read(args.base / relative)["entries"]:
            excluded.add(
                entry.get("initial_layout_sha256")
                or entry["metadata"]["initial_layout_sha256"]
            )
    sub = backend._sub_env(5)
    sub._ensure_env()
    with h5py.File(args.base / "data/tomato-xet.hdf5") as data:
        for key in data["data"]:
            sub._env.set_init_state(data["data"][key]["states"][0])
            excluded.add(layout(sub._env))
    save(args.output / "excluded_layouts.json", sorted(excluded))
    return excluded


def run(args, protocol):
    verify_freeze(args)
    if read(args.output / "smoke.json")["status"] != "PASS":
        raise ValueError("Frozen model smoke must precede language/model acceptance")
    (args.output / "run").mkdir(exist_ok=False)
    from actionstream.libero_config import ensure_isolated_libero_config

    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(args.output / "libero-config")
    import torch
    from actionstream.llm_vla.qwen import LocalQwen
    from actionstream.llm_vla.temporal_completion import TemporalPredictor

    torch.set_num_threads(8)
    config = read(args.output / "language_config.json")
    parsed, language, episodes = {}, [], []
    receipt = dict(
        status="RUNNING",
        started_unix=time.time(),
        freeze_sha256=digest(args.output / "freeze.json"),
        packages={
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "torchvision",
                "transformers",
                "lerobot",
                "hf-libero",
                "mujoco",
                "robosuite",
                "numpy",
                "Pillow",
            )
        },
    )
    save(args.output / "run_receipt.json", receipt)
    llm = backend = None
    try:
        llm = LocalQwen(args.base / "assets/qwen", config["model"])
        save(args.output / "language_model.json", llm.receipt)
        for case in protocol["requests"]:
            directory = args.output / "run" / case["id"]
            directory.mkdir()
            original = OriginalRequest(case["id"], case["text"])
            result = parse_request(llm, config, original)
            save(directory / "parser_call.json", result)
            parsed[case["id"]] = result
            language.append(
                dict(
                    id=case["id"],
                    decision=result["verdict"]["decision"],
                    model_status=result["call"]["status"],
                    schema_status=result["verdict"]["schema_status"],
                )
            )
            save(args.output / "language_progress.json", language)
            print(json.dumps(dict(phase="language", **language[-1])), flush=True)
        llm.close()
        llm = None
        model = TemporalPredictor(args.output / "frozen.pt", CHECKPOINT_SHA256)
        excluded = None
        for case in protocol["requests"]:
            if "seed" not in case:
                continue
            directory = args.output / "run" / case["id"]

            def make_port():
                nonlocal backend, excluded
                # Called only after the public Agent revalidates the whole request.
                if backend is None:
                    backend = backend_factory(args.base / "assets", case["seed"])
                    excluded = excluded_layouts(args, backend)
                return SimulationPort(backend, case["seed"], directory, excluded)

            result = execute_parsed(
                parsed[case["id"]], config, case["seed"], directory, make_port, model
            )
            if result["status"] in ("LANGUAGE_BLOCKED", "LANGUAGE_ERROR"):
                print(
                    json.dumps(dict(phase="execution", id=case["id"], **result)),
                    flush=True,
                )
                continue
            if result["status"] == "ERROR":
                raise RuntimeError(f"Agent execution failed at {case['id']}: {result}")
            score = score_episode(directory)
            save(directory / "independent_score.json", score)
            episodes.append(
                dict(id=case["id"], seed=case["seed"], outcome=result, score=score)
            )
            save(args.output / "execution_progress.json", episodes)
            print(
                json.dumps(dict(phase="execution", id=case["id"], **score)), flush=True
            )
        verify_freeze(args)
        verdict = score_run(protocol, language, episodes)
        save(args.output / "verdict.json", verdict)
        receipt.update(
            status="COMPLETED", completed_unix=time.time(), verdict=verdict["status"]
        )
        print(json.dumps(verdict), flush=True)
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
        save(args.output / "run_receipt.json", receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    for name in ("base", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        globals()[args.phase](args, read(args.protocol))
    except BaseException as exc:
        if args.output.exists():
            save(
                args.output / f"failure-{args.phase}-{time.time_ns()}.json",
                dict(error=repr(exc), traceback=traceback.format_exc()),
            )
        raise


if __name__ == "__main__":
    main()
