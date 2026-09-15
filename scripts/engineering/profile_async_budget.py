"""Profile consumed async seeds only; timing spans never select control actions.

Nested CPU spans overlap and must not be summed. CUDA event time is stream
elapsed time (including contention), not host waiting or exclusive GPU usage.
"""

from __future__ import annotations

import argparse
from functools import wraps
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import numpy as np

from actionstream.delivery import digest, environment
from actionstream.llm_vla import finite_agent, finite_native, temporal_completion
from actionstream.llm_vla.agent_cli import backend_factory
from actionstream.llm_vla.async_agent import AsyncAgentConfig, execute_async_request
from actionstream.llm_vla.async_scoring import score_episode
from actionstream.llm_vla.coverage import authorize
from actionstream.llm_vla.finite_agent import CHECKPOINT_SHA256
from actionstream.llm_vla.finite_native import save
from actionstream.llm_vla.grounding import OriginalRequest
from actionstream.libero_config import ensure_isolated_libero_config

from evaluate_async_agent import FaultPort

ROOT = Path(__file__).resolve().parents[2]
CONSUMED = {*range(2026092100, 2026092110), *range(2026092200, 2026092210)}


class Spans:
    def __init__(self):
        self.rows = []
        self.gpu = []

    def wrap(self, label, function):
        @wraps(function)
        def timed(*args, **kwargs):
            start = time.monotonic()
            try:
                return function(*args, **kwargs)
            finally:
                self.rows.append(
                    dict(
                        label=label,
                        start=start,
                        end=time.monotonic(),
                        thread=threading.get_ident(),
                    )
                )

        return timed

    def gpu_forward(self, model):
        import torch

        original = model.forward

        def forward(*args, **kwargs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall = time.monotonic()
            begin.record()
            result = self.wrap("rgb_forward_submit", original)(*args, **kwargs)
            end.record()
            self.gpu.append((wall, begin, end))
            return result

        model.forward = forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--fault", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--switch-interval", type=float)
    parser.add_argument("--check-postprocessor-parity", action="store_true")
    parser.add_argument("--isolate-inference", action="store_true")
    args = parser.parse_args()
    if not set(args.seeds) <= CONSUMED:
        raise ValueError("Development profiler accepts previously consumed seeds only")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "runner.py")
    save(
        args.output / "provenance.json",
        dict(
            purpose="DEVELOPMENT_ONLY",
            seeds=args.seeds,
            fault=args.fault,
            started_unix=time.time(),
            source_sha256={
                p.relative_to(ROOT).as_posix(): digest(p)
                for p in [
                    *sorted((ROOT / "src").rglob("*.py")),
                    Path(__file__).resolve(),
                ]
            },
            environment=environment(require_cuda=True),
        ),
    )
    provenance = json.loads((args.output / "provenance.json").read_text())
    for name, expected in provenance["source_sha256"].items():
        target = args.output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if digest(target) != expected:
            raise RuntimeError("Development source changed while snapshotting")
    ensure_isolated_libero_config(
        args.output / "libero-config", assets_dir=args.store / "libero-assets"
    )
    import torch

    torch.set_num_threads(args.torch_threads)
    if args.switch_interval is not None:
        sys.setswitchinterval(args.switch_interval)
    save(
        args.output / "threadpools.json",
        dict(
            torch_threads=torch.get_num_threads(),
            python_switch_interval_s=sys.getswitchinterval(),
            environment={
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        ),
    )
    spans = Spans()
    import actionstream.lerobot_backend as backend_module

    backend_module.LeRobotBackend.prepare_observation = spans.wrap(
        "vla_prepare", backend_module.LeRobotBackend.prepare_observation
    )
    reference = backend_module.postprocess_action_chunk_per_timestep
    backend_module.postprocess_action_chunk_per_timestep = spans.wrap(
        "vla_postprocess", reference
    )
    if hasattr(backend_module, "postprocess_xvla_libero_chunk_batched"):
        batched = backend_module.postprocess_xvla_libero_chunk_batched
        parity_done = False

        def checked_batched(raw, **kwargs):
            nonlocal parity_done
            result = batched(raw, **kwargs)
            if args.check_postprocessor_parity and not parity_done:
                expected = reference(raw, **kwargs)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                np.savez_compressed(
                    args.output / "postprocessor-parity.npz",
                    raw=raw.detach().cpu().numpy(),
                    expected=expected.cpu().numpy(),
                    batched=result.cpu().numpy(),
                )
                save(
                    args.output / "postprocessor-parity.json",
                    dict(
                        status="PASS",
                        exact=True,
                        source="first actual worker warmup chunk, before physical controls",
                    ),
                )
                parity_done = True
            return result

        backend_module.postprocess_xvla_libero_chunk_batched = spans.wrap(
            "vla_postprocess", checked_batched
        )
    for module in (finite_agent, finite_native, temporal_completion):
        if hasattr(module, "camera_rgb"):
            module.camera_rgb = spans.wrap("image_preprocess", module.camera_rgb)
    finite_native.simulator_facts = spans.wrap(
        "private_truth", finite_native.simulator_facts
    )
    finite_agent.TemporalVerifier.update = spans.wrap(
        "verifier_total", finite_agent.TemporalVerifier.update
    )
    finite_native.SimulationPort.capture = spans.wrap(
        "capture_total", finite_native.SimulationPort.capture
    )
    predictor = temporal_completion.TemporalPredictor(
        args.store / "frozen.pt", CHECKPOINT_SHA256
    )

    # Same operations and order as TemporalPredictor.predict. Split host submit
    # from blocking transfers; no extra synchronize is inserted in the loop.
    def predict(clips):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cpu = spans.wrap(
                "rgb_input_copy", lambda: torch.from_numpy(np.asarray(clips).copy())
            )()
            device = spans.wrap("rgb_host_to_device", cpu.cuda)()
            output = predictor.model(device)
            output = spans.wrap(
                "rgb_output_submit", lambda: output.float().softmax(-1)
            )()
            return spans.wrap("rgb_readback_wait", lambda: output.cpu().numpy())()

    predictor.predict = predict
    predictor.predict = spans.wrap("rgb_predict_total", predictor.predict)
    spans.gpu_forward(predictor.model)
    backend = backend_factory(args.store / "assets", args.seeds[0])
    language = json.loads((ROOT / "configs/finite_agent_language.json").read_text())
    # Replay the actual retained Qwen output; no model decisions are fabricated.
    original_calls = args.store / "async-acceptance-v2/normal/parser_calls.json"
    parsed = json.loads(original_calls.read_text())[0]
    request = OriginalRequest(**parsed["original"])
    permit = authorize(request, parsed["call"]["raw_output"], language["contract"])
    save(
        args.output / "parser_provenance.json",
        dict(path=str(original_calls), sha256=digest(original_calls), parsed=parsed),
    )
    scores = []
    try:
        for seed in args.seeds:
            directory = args.output / str(seed)
            directory.mkdir()
            spans.rows.clear()
            spans.gpu.clear()

            from actionstream.llm_vla.async_process import ProcessInferenceMixin

            base = (
                type("ProfileBase", (ProcessInferenceMixin, FaultPort), {})
                if args.isolate_inference
                else FaultPort
            )

            class ProfilePort(base):
                def start(self, *values):
                    result = super().start(*values)
                    step = getattr(self.env.step, "__wrapped__", self.env.step)
                    self.env.step = spans.wrap("simulator_step", step)
                    target = self.env.env
                    for name, label in (
                        ("_pre_action", "sim_pre_action"),
                        ("_post_action", "sim_post_action"),
                        ("_get_observations", "sim_observations"),
                        ("_update_observables", "sim_sensor_update"),
                    ):
                        if hasattr(target, name):
                            original = getattr(target, name)
                            setattr(
                                target,
                                name,
                                spans.wrap(
                                    label, getattr(original, "__wrapped__", original)
                                ),
                            )
                    renderer = getattr(
                        self.env.sim.render, "__wrapped__", self.env.sim.render
                    )
                    try:
                        self.env.sim.render = spans.wrap("sim_render", renderer)
                    except (AttributeError, TypeError):
                        pass  # Some simulator builds expose a read-only extension type.
                    return result

            port = ProfilePort(
                backend, seed, directory, set(), forced_open_until=args.fault
            )
            port.infer = spans.wrap("vla_worker", port.infer)
            with (directory / "runtime.jsonl").open("x") as journal:

                def emit(event, **values):
                    journal.write(
                        json.dumps(dict(event=event, **values), allow_nan=False) + "\n"
                    )
                    journal.flush()

                emit = spans.wrap("journal_write", emit)
                outcome = execute_async_request(
                    request,
                    permit,
                    language["contract"],
                    lambda: port,
                    predictor,
                    emit,
                    AsyncAgentConfig(max_attempts=2 if args.fault else 1),
                )
            save(directory / "outcome.json", outcome)
            torch.cuda.synchronize()
            save(
                directory / "profile.json",
                dict(
                    cpu=spans.rows,
                    gpu=[
                        dict(
                            start=wall, stream_elapsed_s=begin.elapsed_time(end) / 1000
                        )
                        for wall, begin, end in spans.gpu
                    ],
                ),
            )
            score = score_episode(directory, forced_open_until=args.fault)
            save(directory / "independent_score.json", score)
            scores.append(dict(seed=seed, outcome=outcome, score=score))
            save(args.output / "progress.json", scores)
            print(
                json.dumps(dict(seed=seed, status=outcome["status"], score=score)),
                flush=True,
            )
    finally:
        backend.close()
    save(args.output / "result.json", dict(status="DEVELOPMENT_ONLY", episodes=scores))


if __name__ == "__main__":
    main()
