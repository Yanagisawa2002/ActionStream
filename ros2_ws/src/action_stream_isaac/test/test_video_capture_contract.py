"""Import-safe contract tests for optional Isaac viewport video capture."""

from types import SimpleNamespace

import pytest

from action_stream_isaac.isaac_adapter import _load_video_capture_config


def _args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "video_output": None,
        "headless": False,
        "auto_start_episode": "m7-video-test",
        "exit_after_auto_episode": True,
        "video_fps": 20,
        "video_width": 1280,
        "video_height": 720,
        "video_finalize_timeout_seconds": 120.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_video_capture_is_optional() -> None:
    assert _load_video_capture_config(_args()) is None


def test_video_capture_config_is_resolved_without_creating_output(tmp_path) -> None:
    output = tmp_path / "demo.mp4"
    config = _load_video_capture_config(_args(video_output=output))
    assert config is not None
    assert config.output_path == output.resolve()
    assert config.fps == 20
    assert not output.exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"auto_start_episode": ""}, "requires --auto-start-episode"),
        ({"video_fps": 0}, "--video-fps"),
        ({"video_width": 32}, "--video-width/--video-height"),
        ({"video_finalize_timeout_seconds": 0.0}, "--video-finalize-timeout-seconds"),
    ],
)
def test_video_capture_rejects_invalid_runtime_contract(
    tmp_path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        _load_video_capture_config(_args(video_output=tmp_path / "demo.mp4", **overrides))


def test_video_capture_refuses_wrong_extension_and_overwrite(tmp_path) -> None:
    with pytest.raises(SystemExit, match="must end in .mp4"):
        _load_video_capture_config(_args(video_output=tmp_path / "demo.avi"))

    output = tmp_path / "demo.mp4"
    output.write_bytes(b"existing")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        _load_video_capture_config(_args(video_output=output))

    output.unlink()
    (tmp_path / "demo_frames").mkdir()
    with pytest.raises(SystemExit, match="refusing to reuse stale capture frames"):
        _load_video_capture_config(_args(video_output=output))
