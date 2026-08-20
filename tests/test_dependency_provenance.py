from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from actionstream.preflight import (
    EXPECTED_LEROBOT_COMMIT,
    EXPECTED_LEROBOT_URL,
    _verify_lerobot_install,
)


def test_locked_lerobot_install_exposes_exact_vcs_commit() -> None:
    provenance = _verify_lerobot_install()
    assert provenance["version"] == "0.6.2"
    assert provenance["commit"] == EXPECTED_LEROBOT_COMMIT
    assert provenance["url"] == EXPECTED_LEROBOT_URL


def test_lerobot_provenance_rejects_an_unpinned_commit(monkeypatch) -> None:
    direct_url = {
        "url": EXPECTED_LEROBOT_URL,
        "vcs_info": {"commit_id": "0" * 40},
    }
    fake = SimpleNamespace(
        version="0.6.2",
        read_text=lambda _name: json.dumps(direct_url),
    )
    monkeypatch.setattr("actionstream.preflight.importlib.metadata.distribution", lambda _name: fake)

    with pytest.raises(RuntimeError, match="Expected LeRobot commit"):
        _verify_lerobot_install()
