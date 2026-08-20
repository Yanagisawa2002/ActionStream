from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from actionstream.gpu_measurement import NvidiaSmiMonitor


def test_nvidia_smi_monitor_records_process_residency_and_phase(
    monkeypatch, tmp_path: Path
) -> None:
    responses = iter(
        [
            SimpleNamespace(stdout="0, GPU-test, 73, 12345, 32607\n"),
            SimpleNamespace(stdout=f"{os.getpid()}, GPU-test, 12258\n"),
        ]
    )
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: next(responses))
    monitor = NvidiaSmiMonitor(tmp_path / "gpu.jsonl")
    monitor.set_phase("steady_state", runtime="aligned")
    sample = monitor._sample()
    monitor._samples.append(sample)
    monitor._write(sample)

    assert sample["process_resident"] is True
    assert sample["process_gpu_memory_mib"] == 12258
    assert sample["gpu_utilization_percent"] == 73
    assert sample["phase"] == "steady_state"
    summary = monitor.summary()["phases"]["steady_state"]
    assert summary["resident_sample_count"] == 1
    assert summary["process_gpu_memory_mib_max"] == 12258
