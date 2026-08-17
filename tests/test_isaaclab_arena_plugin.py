from __future__ import annotations

import importlib
import sys
import threading
import time
import types
from pathlib import Path

import pytest
import torch

from actionstream.arena_protocol import load_arena_protocol


ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SRC = ROOT / "integrations" / "isaaclab_arena" / "src"
PROTOCOL_PATH = ROOT / "configs" / "isaaclab_arena_frozen_v1.json"


class _PolicyBase:
    config_class = None

    def __init__(self, config):
        self.config = config

    def set_task_description(self, task_description):
        self.task_description = task_description
        return task_description


class _FakeProvider:
    def __init__(self, _config, _model):
        self.ready = threading.Event()
        self.reset_count = 0
        self.closed = False

    @property
    def supports_rtc(self):
        return False

    def reset(self):
        self.reset_count += 1

    def predict_targets(self, snapshot, task, *, rtc):
        assert task == "move the cube"
        assert rtc is False
        position = snapshot.policy["eef_pos"] + torch.tensor([0.01, 0.0, 0.0])
        target = torch.cat(
            (
                position,
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(snapshot.num_envs, 1),
                torch.zeros((snapshot.num_envs, 1)),
            ),
            dim=1,
        )
        self.ready.set()
        return target.unsqueeze(0).repeat(6, 1, 1)

    def close(self):
        self.closed = True


def _install_fake_arena(monkeypatch: pytest.MonkeyPatch):
    arena = types.ModuleType("isaaclab_arena")
    policy_package = types.ModuleType("isaaclab_arena.policy")
    policy_base = types.ModuleType("isaaclab_arena.policy.policy_base")
    policy_base.PolicyBase = _PolicyBase
    monkeypatch.setitem(sys.modules, "isaaclab_arena", arena)
    monkeypatch.setitem(sys.modules, "isaaclab_arena.policy", policy_package)
    monkeypatch.setitem(sys.modules, "isaaclab_arena.policy.policy_base", policy_base)

    provider = types.ModuleType("actionstream_test_arena_provider")
    provider.create = lambda config, model: _FakeProvider(config, model)
    monkeypatch.setitem(sys.modules, "actionstream_test_arena_provider", provider)
    monkeypatch.syspath_prepend(str(PLUGIN_SRC))
    for name in list(sys.modules):
        if name == "isaaclab_arena_actionstream" or name.startswith(
            "isaaclab_arena_actionstream."
        ):
            del sys.modules[name]
    return importlib.import_module("isaaclab_arena_actionstream.policy")


def _observation(num_envs: int = 8):
    return {
        "policy": {
            "eef_pos": torch.tensor([[0.40, 0.0, 0.30]]).repeat(num_envs, 1),
            "eef_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
            "gripper_pos": torch.zeros((num_envs, 1)),
            "joint_pos": torch.zeros((num_envs, 7)),
        },
        "camera_obs": {
            "external_camera_rgb": torch.zeros((num_envs, 4, 6, 3), dtype=torch.uint8),
            "external_camera_2_rgb": torch.ones((num_envs, 4, 6, 3), dtype=torch.uint8),
            "wrist_camera_rgb": torch.full((num_envs, 4, 6, 3), 2, dtype=torch.uint8),
        },
    }


class _Env:
    unwrapped = types.SimpleNamespace(device="cpu")


def _config(module, tmp_path: Path, *, runtime: str = "actionstream_guarded"):
    protocol = load_arena_protocol(PROTOCOL_PATH)
    cell = next(
        cell
        for cell in protocol.cells("development")
        if cell.task_family == "pick_and_place"
        and cell.model == "xvla"
        and cell.runtime == runtime
    )
    return module.ActionStreamArenaPolicyConfig(
        protocol_path=str(PROTOCOL_PATH),
        protocol_sha256=protocol.sha256,
        split=cell.split,
        task_family=cell.task_family,
        reset_seed=cell.reset_seed,
        network_trace_id=cell.network_trace,
        model=cell.model,
        runtime=cell.runtime,
        num_envs=cell.num_envs,
        provider_factory="actionstream_test_arena_provider:create",
        device="cpu",
        telemetry_directory=str(tmp_path),
    )


def test_arena_adapter_drives_vector_actions_and_closes_worker(monkeypatch, tmp_path):
    module = _install_fake_arena(monkeypatch)
    policy = module.ActionStreamArenaPolicy(_config(module, tmp_path))
    policy.set_task_description("move the cube")

    first = policy.get_action(_Env(), _observation())
    assert first.shape == (8, 7)
    assert torch.allclose(first[:, :6], torch.zeros((8, 6)))
    assert policy._provider.ready.wait(timeout=2.0)
    deadline = time.monotonic() + 2.0
    while policy._engine.telemetry.chunks_accepted == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    second = policy.get_action(_Env(), _observation())
    assert second.shape == (8, 7)
    assert torch.all(second[:, 0] > 0)

    policy.reset(env_ids=torch.tensor([0, 3]))
    assert policy._partial_reset_count == 1
    provider = policy._provider
    policy.shutdown_remote()
    assert provider.closed
    lines = policy._telemetry_path.read_text(encoding="utf-8").splitlines()
    assert any('"event":"start"' in line for line in lines)
    assert any('"event":"action"' in line for line in lines)
    assert any('"event":"stop"' in line for line in lines)


def test_arena_adapter_refuses_to_impersonate_upstream_baseline(monkeypatch, tmp_path):
    module = _install_fake_arena(monkeypatch)
    config = _config(module, tmp_path, runtime="sync")
    with pytest.raises(NotImplementedError, match="will not impersonate"):
        module.ActionStreamArenaPolicy(config)
