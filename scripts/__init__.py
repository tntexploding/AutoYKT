"""Source-checkout utility entry points."""

import sys
from pathlib import Path


_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
_SOURCE_DIRECTORY = _ROOT_DIRECTORY / "src"
if str(_SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIRECTORY))
