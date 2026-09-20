"""Image I/O and template safety regressions."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from autoykt.monitor.detector import MultiTemplateDetector
from autoykt.monitor.image_utils import read_image, write_png


class ImageRegressionTest(unittest.TestCase):
    """Private Windows paths may contain non-ASCII characters."""

    def test_unicode_image_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "课程截图.png"
            frame = np.random.default_rng(9).integers(
                0, 256, (12, 12, 3), dtype=np.uint8
            )
            write_png(path, frame)
            np.testing.assert_array_equal(read_image(path), frame)

    def test_flat_template_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flat.png"
            write_png(path, np.full((8, 8, 3), (40, 80, 120), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "spatial detail"):
                MultiTemplateDetector([{"name": "flat", "path": str(path)}])
