"""Tests for debounced template appearance semantics."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from autoykt.monitor.detector import MultiTemplateDetector, ImageTemplateMatcher


class MultiTemplateDetectorTest(unittest.TestCase):

    def test_scaled_state_location_is_unique_across_display_scales(self):
        template = np.random.default_rng(8).integers(
            0, 256, (10, 20, 3), dtype=np.uint8
        )
        frame = np.zeros((150, 200, 3), dtype=np.uint8)
        scaled = cv2.resize(template, (30, 15))
        frame[40:55, 60:90] = scaled
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "notice.png"
            cv2.imwrite(str(path), template)
            matcher = ImageTemplateMatcher(
                [("notice", str(path), 0.99)], scales=(1, 1.5, 2), unique=True
            )
            match = matcher.best(frame)
            assert match is not None
            self.assertEqual(match.center, (75, 47))
            frame[80:95, 120:150] = scaled
            self.assertIsNone(matcher.best(frame))

    def test_conflicting_types_block_then_debounce_again_after_resolution(self):
        random = np.random.default_rng(13)
        first, second = [
            random.integers(0, 256, (10, 10, 3), dtype=np.uint8)
            for _ in range(2)
        ]
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[2:12, 2:12] = first
        frame[22:32, 22:32] = second
        with tempfile.TemporaryDirectory() as directory:
            rules = []
            for kind, pixels in (("single", first), ("multiple", second)):
                path = Path(directory) / (kind + ".png")
                cv2.imwrite(str(path), pixels)
                rules.append(
                    {
                        "name": kind,
                        "path": str(path),
                        "threshold": 0.99,
                        "question_type": kind,
                        "consecutive_hits": 2,
                    }
                )
            detector = MultiTemplateDetector(rules)
            for _ in range(3):
                observation = detector.observe(frame)
                self.assertTrue(observation.ambiguous)
                self.assertIsNone(observation.triggered)
            frame[22:32, 22:32] = 0
            self.assertIsNone(detector.observe(frame).triggered)
            observation = detector.observe(frame)
            self.assertFalse(observation.ambiguous)
            assert observation.triggered is not None
            self.assertEqual(observation.triggered.question_type, "single")

    def test_fires_once_until_template_disappears(self) -> None:
        random = np.random.default_rng(42)
        template = random.integers(0, 256, (10, 10, 3), dtype=np.uint8)
        visible = random.integers(0, 256, (40, 40, 3), dtype=np.uint8)
        visible[12:22, 8:18] = template
        absent = random.integers(0, 256, (40, 40, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.png"
            self.assertTrue(cv2.imwrite(str(path), template))
            detector = MultiTemplateDetector(
                [
                    {
                        "name": "question",
                        "path": str(path),
                        "threshold": 0.99,
                        "consecutive_hits": 2,
                        "action": "answer",
                    }
                ]
            )
            self.assertIsNone(detector.observe(visible).triggered)
            self.assertIsNotNone(detector.observe(visible).triggered)
            self.assertIsNone(detector.observe(visible).triggered)
            self.assertFalse(detector.observe(absent).present)
            self.assertIsNone(detector.observe(visible).triggered)
            self.assertIsNotNone(detector.observe(visible).triggered)


if __name__ == "__main__":
    unittest.main()
