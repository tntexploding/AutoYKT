"""Compatibility entry point for interactive calibration."""

from pathlib import Path
import sys


_SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIRECTORY))

from autoykt.calibration import Calibrator  # pylint: disable=wrong-import-position


if __name__ == "__main__":
    Calibrator().run()
