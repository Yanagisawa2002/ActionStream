"""Real v2 native entrypoint; joint gates run before any model/environment load."""

from .native_runner import main


def run():
    return main(protocol_version="v2")


if __name__ == "__main__":
    raise SystemExit(run())
