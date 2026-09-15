"""Create an isolated, canonical hf-libero configuration for ActionStream."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG_DIR = Path("~/.cache/actionstream/libero-config").expanduser()


def _canonical_benchmark_root() -> Path:
    """Locate hf-libero without importing ``libero.libero`` and triggering its prompt."""
    package_spec = importlib.util.find_spec("libero")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise RuntimeError(
            "hf-libero is not installed in the active Python environment"
        )

    package_root = Path(next(iter(package_spec.submodule_search_locations))).resolve()
    benchmark_root = package_root / "libero"
    required = (benchmark_root / "bddl_files", benchmark_root / "init_files")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(f"hf-libero canonical files are missing: {missing}")
    return benchmark_root


def ensure_isolated_libero_config(
    config_dir: Path | str | None = None, *, assets_dir: Path | str | None = None
) -> Path:
    """Write a noninteractive config that cannot fall back to ``~/.libero``."""
    selected = (
        Path(config_dir or os.environ.get("LIBERO_CONFIG_PATH") or DEFAULT_CONFIG_DIR)
        .expanduser()
        .resolve()
    )
    global_default = Path("~/.libero").expanduser().resolve()
    if selected == global_default:
        raise RuntimeError(
            "ActionStream refuses to use or overwrite the global ~/.libero configuration; "
            "select an isolated LIBERO_CONFIG_PATH"
        )

    loaded = sys.modules.get("libero.libero")
    if loaded is not None:
        loaded_dir = Path(getattr(loaded, "libero_config_path", "")).resolve()
        if loaded_dir != selected:
            raise RuntimeError(
                "libero.libero was imported before the isolated configuration was selected: "
                f"loaded={loaded_dir}, requested={selected}"
            )

    os.environ["LIBERO_CONFIG_PATH"] = str(selected)
    benchmark_root = _canonical_benchmark_root()
    assets_dir = assets_dir or os.environ.get("ACTIONSTREAM_LIBERO_ASSETS")
    assets = (
        Path(assets_dir).expanduser().resolve()
        if assets_dir
        else benchmark_root / "assets"
    )
    if assets_dir and not assets.is_dir():
        raise RuntimeError(f"Selected LIBERO assets directory is missing: {assets}")
    if loaded is not None and assets_dir:
        cached = getattr(loaded, "_assets_path_cache", None)
        if cached is not None and Path(cached).resolve() != assets:
            raise RuntimeError("Cannot change LIBERO assets after initialization")
    if assets_dir:
        # Backend initialization calls this function again in the same process.
        os.environ["ACTIONSTREAM_LIBERO_ASSETS"] = str(assets)
    datasets_dir = selected.parent / "datasets-unused"
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(datasets_dir),
        "assets": str(assets),
    }

    config_file = selected / "config.yaml"
    serialized = yaml.safe_dump(config, sort_keys=True)
    ownership_marker = selected / ".actionstream-owned"
    if (
        config_file.exists()
        and config_file.read_text(encoding="utf-8") != serialized
        and not ownership_marker.exists()
    ):
        raise RuntimeError(
            f"Refusing to overwrite an existing non-ActionStream LIBERO config: {config_file}"
        )

    selected.mkdir(parents=True, exist_ok=True)
    if (
        not config_file.exists()
        or config_file.read_text(encoding="utf-8") != serialized
    ):
        config_file.write_text(serialized, encoding="utf-8")
    ownership_marker.write_text(
        "ActionStream isolated LIBERO config\n", encoding="utf-8"
    )
    if assets_dir:
        # hf-libero 0.1.4's get_assets_path ignores config.yaml and would fetch an
        # unpinned snapshot. Bind its cache before importing environment modules.
        # This changes only this process, never installed package files.
        libero = importlib.import_module("libero.libero")
        libero._assets_path_cache = str(assets)
    return config_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--assets-dir", type=Path, default=None)
    args = parser.parse_args()
    path = ensure_isolated_libero_config(args.config_dir, assets_dir=args.assets_dir)
    print(path)


if __name__ == "__main__":
    main()
