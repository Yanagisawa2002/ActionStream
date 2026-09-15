"""Freeze and evaluate paced dispatch and real simulated gripper-failure recovery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from actionstream.delivery import digest, environment, verify
from actionstream.llm_vla.async_agent import AsyncAgentConfig, execute_async_request
from actionstream.llm_vla.async_native import AsyncSimulationPort
from actionstream.llm_vla.async_scoring import score_episode
from actionstream.llm_vla.agent_cli import backend_factory, parse_request, verify_assets
from actionstream.llm_vla.coverage import authorize
from actionstream.llm_vla.finite_agent import CHECKPOINT_SHA256
from actionstream.llm_vla.finite_native import save
from actionstream.llm_vla.grounding import OriginalRequest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/async_agent_acceptance_v1.json"


def read(path):
    return json.loads(Path(path).read_text())


def historical_layouts():
    with tarfile.open(
        ROOT / "reports/finite_agent_20260915/raw_records.tar.gz"
    ) as archive:
        layouts = set(json.load(archive.extractfile("excluded_layouts.json")))
        for name in archive.getnames():
            if name.endswith("/initial_layout.json"):
                layouts.add(json.load(archive.extractfile(name))["sha256"])
    if len(layouts) != 122:
        raise ValueError("Historical layout exclusion coverage changed")
    return layouts


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    files = sorted((ROOT / "src").rglob("*.py")) + [
        Path(__file__).resolve(),
        PROTOCOL,
        ROOT / "configs/finite_agent_language.json",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / "configs/completion_runtime_assets.json",
        ROOT / "configs/libero_runtime_assets.json",
    ]
    sources = {str(path.relative_to(ROOT).as_posix()): digest(path) for path in files}
    for name in sources:
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    save(args.output / "protocol.json", read(PROTOCOL))
    save(args.output / "excluded_layouts.json", sorted(historical_layouts()))
    save(
        args.output / "freeze.json",
        dict(
            source_commit=args.source_commit,
            source_sha256=sources,
            frozen_unix=time.time(),
            new_data_seen=False,
            checkpoint_sha256=CHECKPOINT_SHA256,
            exclusions_sha256=digest(args.output / "excluded_layouts.json"),
            protocol_sha256=digest(args.output / "protocol.json"),
        ),
    )


def check_freeze(args):
    frozen = read(args.output / "freeze.json")
    for name, expected in frozen["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen source changed: " + name)
    if digest(args.output / "excluded_layouts.json") != frozen["exclusions_sha256"]:
        raise ValueError("Frozen layout exclusions changed")
    if digest(args.output / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    return frozen


class FaultPort(AsyncSimulationPort):
    def __init__(self, *args, forced_open_until=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.forced_open_until = forced_open_until

    def step(self, action, request_id, revision, control):
        physical = action.copy()
        if control <= self.forced_open_until:
            physical[-1] = -1.0
        return super().step(physical, request_id, revision, control)


def run(args):
    frozen = check_freeze(args)
    protocol = read(args.output / "protocol.json")
    mode = args.phase
    directory = args.output / mode
    directory.mkdir(exist_ok=False)
    started = time.time()
    runtime = environment(require_cuda=True)
    save(directory / "environment.json", runtime)
    if runtime["status"] != "PASS":
        raise RuntimeError("Runtime lock mismatch")
    verify_assets(
        args.store / "assets",
        read(ROOT / "configs/completion_runtime_assets.json"),
        args.store / "frozen.pt",
    )
    if (
        verify(
            read(ROOT / "configs/libero_runtime_assets.json"),
            args.store / "libero-assets",
        )["status"]
        != "PASS"
    ):
        raise RuntimeError("Incomplete LIBERO asset delivery")
    if mode != "development":
        delivery = read(args.delivery_receipt)
        if delivery["status"] != "PASS":
            raise RuntimeError(
                "Reproducible current delivery must pass before acceptance"
            )
    from actionstream.libero_config import ensure_isolated_libero_config
    from actionstream.llm_vla.qwen import LocalQwen
    from actionstream.llm_vla.temporal_completion import TemporalPredictor
    import torch

    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    ensure_isolated_libero_config(
        directory / "libero-config", assets_dir=args.store / "libero-assets"
    )
    torch.set_num_threads(8)
    language = read(ROOT / "configs/finite_agent_language.json")
    seeds = (
        [2026092000]
        if mode == "development"
        else protocol["normal_seeds" if mode == "normal" else "recovery_seeds"]
    )
    llm = LocalQwen(args.store / "assets/qwen", language["model"])
    try:
        parsed = [
            parse_request(
                llm,
                language,
                OriginalRequest(f"{mode}-{i:02d}", protocol["instruction"]),
            )
            for i in range(len(seeds))
        ]
    finally:
        llm.close()
    save(directory / "parser_calls.json", parsed)
    predictor = TemporalPredictor(args.store / "frozen.pt", CHECKPOINT_SHA256)
    backend = backend_factory(args.store / "assets", seeds[0])
    excluded = (
        set()
        if mode == "development"
        else set(read(args.output / "excluded_layouts.json"))
    )
    # Include any previously evaluated mode in the current-batch exclusion set.
    for path in args.output.glob("*/episode-*/initial_layout.json"):
        excluded.add(read(path)["sha256"])
    episodes = []
    try:
        for index, (seed, call) in enumerate(zip(seeds, parsed)):
            check_freeze(args)
            output = directory / f"episode-{index:02d}"
            output.mkdir()
            save(output / "parser_call.json", call)
            request = OriginalRequest(**call["original"])
            if call["verdict"]["decision"] != "accept":
                episodes.append(dict(seed=seed, status="LANGUAGE_BLOCKED", score=None))
                save(output / "outcome.json", episodes[-1])
                continue
            permit = authorize(
                request, call["call"]["raw_output"], language["contract"]
            )
            fault = protocol["fault"]["controls_exclusive"] if mode == "recovery" else 0
            config = AsyncAgentConfig(
                **protocol["config"], max_attempts=2 if mode == "recovery" else 1
            )
            with (output / "runtime.jsonl").open("x") as journal:

                def emit(event, **values):
                    journal.write(
                        json.dumps(dict(event=event, **values), allow_nan=False) + "\n"
                    )
                    journal.flush()

                outcome = execute_async_request(
                    request,
                    permit,
                    language["contract"],
                    lambda: FaultPort(
                        backend, seed, output, excluded, forced_open_until=fault
                    ),
                    predictor,
                    emit,
                    config,
                )
            save(output / "outcome.json", outcome)
            try:
                score = score_episode(output, forced_open_until=fault)
            except Exception as exc:
                score = dict(
                    integrity_passed=False,
                    integrity_errors=[str(exc)],
                    safely_completed=False,
                )
            save(output / "independent_score.json", score)
            blobs = {
                str(path.relative_to(output)): dict(
                    bytes=path.stat().st_size, sha256=digest(path)
                )
                for path in output.iterdir()
                if path.is_file()
            }
            save(output / "files.json", blobs)
            episodes.append(dict(seed=seed, status=outcome["status"], score=score))
            save(
                directory / "progress.json",
                dict(completed=len(episodes), requested=len(seeds), episodes=episodes),
            )
            print(
                json.dumps(
                    dict(
                        mode=mode,
                        completed=len(episodes),
                        requested=len(seeds),
                        outcome=outcome["status"],
                        score=score,
                    )
                ),
                flush=True,
            )
    finally:
        backend.close()
    gates = protocol["gates"]
    scores = [e["score"] for e in episodes if e["score"] is not None]
    valid = len(scores) == len(seeds) and all(s["integrity_passed"] for s in scores)
    safe = sum(s["safely_completed"] for s in scores)
    common = valid and all(
        not s.get("premature_stop", True)
        and not s.get("missed_completed_event", True)
        and (
            s.get("confirmation_delay_s") is None
            or s["confirmation_delay_s"] <= gates["max_confirmation_delay_s"]
        )
        and (
            s.get("confirmation_wall_delay_s") is None
            or 0 <= s["confirmation_wall_delay_s"] <= gates["max_confirmation_delay_s"]
        )
        for s in scores
    )
    passed = common and safe >= gates["normal_safe_completions_min"]
    if mode == "normal":
        passed = passed and all(
            s["timing"]["period_p95_s"] <= gates["normal_period_p95_s_max"]
            and s["timing"]["period_max_s"] <= gates["normal_period_max_s_max"]
            and s["timing"]["deadline_misses"] / s["timing"]["controls"]
            <= gates["normal_deadline_miss_fraction_max"]
            for s in scores
        )
    if mode == "recovery":
        passed = (
            common
            and sum(s["recovery"]["physical_failure_eligible"] for s in scores)
            >= gates["recovery_eligible_min"]
        )
        passed = (
            passed
            and sum(s["recovery"]["recovered_from_failure"] for s in scores)
            >= gates["recovered_min"]
        )
    result = dict(
        status="GO" if passed else "NO-GO",
        phase=mode,
        requested=len(seeds),
        safely_completed=safe,
        source_commit=frozen["source_commit"],
        freeze_sha256=digest(args.output / "freeze.json"),
        started_unix=started,
        completed_unix=time.time(),
        episodes=episodes,
    )
    if mode == "development":
        result["status"] = "DEVELOPMENT_ONLY"
    save(directory / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("freeze", "development", "normal", "recovery")
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--delivery-receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args)
    else:
        print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
