"""ActionStream benchmark and LeRobot inference-backend package."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("actionstream")
except PackageNotFoundError:  # source tree imported before installation
    __version__ = "1.1.0rc1"
