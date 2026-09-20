"""Source-checkout entry point for AutoYKT."""

from __future__ import annotations

from pathlib import Path
import sys


_SOURCE_DIRECTORY = Path(__file__).resolve().parent / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIRECTORY))

from autoykt.cli import main  # pylint: disable=wrong-import-position


if __name__ == "__main__":
    raise SystemExit(main())
