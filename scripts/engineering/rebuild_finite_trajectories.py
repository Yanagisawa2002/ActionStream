"""Reconstruct lost historical trajectories only when their original bytes match.

This uses consumed seeds and retained parser calls, not new acceptance data.
New run logs stay separate from restored blobs. Hash mismatches remain failures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from actionstream.delivery import digest, environment, verify
from actionstream.llm_vla.agent_cli import (
    backend_factory,
    execute_parsed,
    verify_assets,
)
from actionstream.llm_vla.finite_agent import CHECKPOINT_SHA256
from actionstream.llm_vla.finite_native import SimulationPort, save
from actionstream.llm_vla.finite_scoring import score_episode

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/finite_agent_20260915"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    restored = args.output / "restored-blobs"
    restored.mkdir()
    shutil.copyfile(Path(__file__), args.output / "reconstruction_runner.py")
    runtime = environment(require_cuda=True)
    save(args.output / "environment.json", runtime)
    if runtime["status"] != "PASS":
        raise RuntimeError("Runtime lock mismatch")
    verify_assets(
        args.store / "assets",
        read(ROOT / "configs/completion_runtime_assets.json"),
        args.store / "frozen.pt",
    )
    shutil.copyfile(args.store / "frozen.pt", restored / "frozen.pt")
    manifest = read(REPORT / "external_blobs.json")
    config = read(REPORT / "language_config.json")
    protocol = read(REPORT / "protocol.json")
    save(args.output / "original-external-blobs.json", manifest)
    save(
        args.output / "provenance.json",
        dict(
            source_commit=args.source_commit,
            runner_sha256=digest(Path(__file__)),
            started_unix=time.time(),
            original_manifest_sha256=digest(REPORT / "external_blobs.json"),
            original_records_sha256=digest(REPORT / "raw_records.tar.gz"),
            method="Consumed-seed deterministic rerun; only exact original size and SHA-256 are restored",
            source_sha256={
                str(p.relative_to(ROOT).as_posix()): digest(p)
                for p in sorted((ROOT / "src").rglob("*.py"))
            },
        ),
    )
    from actionstream.libero_config import ensure_isolated_libero_config
    from actionstream.llm_vla.temporal_completion import TemporalPredictor
    import torch

    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(
        args.output / "libero-config", assets_dir=args.store / "libero-assets"
    )
    torch.set_num_threads(8)
    predictor = TemporalPredictor(args.store / "frozen.pt", CHECKPOINT_SHA256)
    backend = backend_factory(args.store / "assets", protocol["requests"][0]["seed"])
    rows = []
    try:
        with tarfile.open(REPORT / "raw_records.tar.gz") as archive:
            for case in protocol["requests"]:
                if case.get("seed") is None:
                    continue
                name, seed = case["id"], case["seed"]
                output = args.output / "new-runs" / name
                output.mkdir(parents=True)
                parsed = json.load(archive.extractfile(f"run/{name}/parser_call.json"))
                save(output / "retained_parser_call.json", parsed)
                result = execute_parsed(
                    parsed,
                    config,
                    seed,
                    output,
                    lambda: SimulationPort(backend, seed, output),
                    predictor,
                )
                score = score_episode(output)
                save(output / "independent_score.json", score)
                source = output / "trajectory.npz"
                key = f"run/{name}/trajectory.npz"
                expected = manifest[key]
                actual = dict(bytes=source.stat().st_size, sha256=digest(source))
                matched = actual == expected
                if matched:
                    target = restored / key
                    target.parent.mkdir(parents=True)
                    os.link(source, target)
                rows.append(
                    dict(
                        id=name,
                        seed=seed,
                        original=expected,
                        reconstructed=actual,
                        byte_identical=matched,
                        outcome=result["status"],
                        score=score,
                    )
                )
                save(
                    args.output / "progress.json",
                    dict(
                        completed=len(rows),
                        matched=sum(r["byte_identical"] for r in rows),
                        episodes=rows,
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            completed=len(rows),
                            id=name,
                            byte_identical=matched,
                            actual=actual,
                        )
                    ),
                    flush=True,
                )
    finally:
        backend.close()
    receipt = verify(manifest, restored)
    receipt.update(
        source_commit=args.source_commit,
        completed_unix=time.time(),
        reconstructed_trajectories=len(rows),
        method="Byte-identical reconstruction, not recovery from backup",
    )
    save(args.output / "restored-verification.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
