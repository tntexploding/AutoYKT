"""Tests for safe, incremental private configuration calibration."""

# These tests intentionally exercise the calibrator's pure edit helpers.
# pylint: disable=protected-access

from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any
import unittest

import yaml

from autoykt.calibration import Calibrator
from autoykt.core.config import AppConfig


def _example_document() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "config.example.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError("example configuration must be a mapping")
    return document


def _calibrator(
    document: dict[str, Any],
    config_path: Path,
) -> Calibrator:
    calibrator = Calibrator.__new__(Calibrator)
    calibrator._document = deepcopy(document)
    calibrator._legacy = False
    calibrator._image_only = False
    calibrator._profile_id = "example_yuketang"
    calibrator._monitor_index = 1
    calibrator._config_path = config_path
    calibrator._regions = {"detection": [10, 10, 200, 100]}
    calibrator._fallback_positions = {}
    calibrator._option_template_paths = {}
    calibrator._trigger_template_path = None
    calibrator._success_template_path = None
    calibrator._window_target = None
    calibrator._source_geometry = None
    calibrator._guard_template_path = None
    return calibrator


class CalibratorTest(unittest.TestCase):
    """Exercise edits without opening an MSS or OpenCV window."""

    def test_option_cycle_includes_f_and_g_without_overwriting_a(self):
        with tempfile.TemporaryDirectory() as directory:
            calibrator = _calibrator(
                _example_document(), Path(directory) / "config.yaml"
            )
            calibrator._current_option = "A"
            observed = []
            for _ in range(7):
                observed.append(calibrator._current_option)
                calibrator._advance_option()
            self.assertEqual(observed, list("ABCDEFG"))
            calibrator._current_option = "Z"
            calibrator._advance_option()
            self.assertEqual(calibrator._current_option, "A")

    def test_color_crops_save_rgb_and_preserve_private_credentials(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            document = _example_document()
            document["answering"]["providers"][0]["api_key"] = "private-test"
            path = Path(directory) / "config.yaml"
            calibrator = _calibrator(document, path)
            frame = np.full((40, 60, 3), 153, dtype=np.uint8)
            frame[:20, :20] = (245, 150, 80)
            calibrator._original_frame = frame
            calibrator._regions.update(
                selected_rgb=[0, 0, 20, 20],
                unselected_rgb=[20, 0, 20, 20],
                button_size=[0, 0, 30, 30],
            )
            calibrator._save()
            saved = yaml.safe_load(path.read_text(encoding="utf-8"))
            style = saved["pages"][0]["answer_style"]
            self.assertTrue(style["match_grayscale"])
            self.assertEqual(
                style["button_colors"]["selected_rgb"], [80, 150, 245]
            )
            self.assertEqual(
                style["button_colors"]["unselected_rgb"], [153] * 3
            )
            self.assertEqual(style["button_colors"]["button_size"], [30, 30])
            self.assertEqual(
                saved["answering"]["providers"][0]["api_key"], "private-test"
            )
            calibrator._regions = {"button_size": [0, 0, 35, 35]}
            calibrator._save()
            updated = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(
                updated["pages"][0]["answer_style"]["button_colors"][
                    "selected_rgb"
                ],
                [80, 150, 245],
            )

    def test_partial_calibration_preserves_option_placeholders(self) -> None:
        """Saving one region must keep options for a later session."""
        with tempfile.TemporaryDirectory() as directory:
            calibrator = _calibrator(
                _example_document(),
                Path(directory) / "config.yaml",
            )
            calibrator._save_v2()
            profile = calibrator._document["pages"][0]
            templates = profile["answer_style"]["option_templates"]
            self.assertEqual(set(templates), {"A", "B", "C", "D"})
            AppConfig.model_validate(calibrator._document)

    def test_fallback_calibration_can_replace_missing_templates(self) -> None:
        """Fallback points can replace missing option image placeholders."""
        with tempfile.TemporaryDirectory() as directory:
            calibrator = _calibrator(
                _example_document(),
                Path(directory) / "config.yaml",
            )
            calibrator._fallback_positions = {
                "A": [100, 100],
                "B": [100, 200],
                "C": [100, 300],
                "D": [100, 400],
            }
            calibrator._save_v2()
            profile = calibrator._document["pages"][0]
            self.assertEqual(
                profile["answer_style"]["option_templates"],
                {},
            )
            AppConfig.model_validate(calibrator._document)

    def test_question_update_preserves_custom_verification_regions(
        self,
    ) -> None:
        """Question edits must not overwrite separately calibrated regions."""
        with tempfile.TemporaryDirectory() as directory:
            document = _example_document()
            regions = document["pages"][0]["regions"]
            regions["verification"] = [500, 500, 200, 100]
            regions["rearm"] = [600, 600, 200, 100]
            calibrator = _calibrator(
                document,
                Path(directory) / "config.yaml",
            )
            calibrator._regions["question"] = [20, 20, 300, 200]
            calibrator._save_v2()
            saved = calibrator._document["pages"][0]["regions"]
            self.assertEqual(saved["verification"], [500, 500, 200, 100])
            self.assertEqual(saved["rearm"], [600, 600, 200, 100])

    def test_success_template_is_written_to_verification(self) -> None:
        """A calibrated success crop becomes explicit verification evidence."""
        with tempfile.TemporaryDirectory() as directory:
            calibrator = _calibrator(
                _example_document(),
                Path(directory) / "config.yaml",
            )
            calibrator._success_template_path = "templates/success.png"
            calibrator._save_v2()
            verification = calibrator._document["pages"][0]["verification"]
            self.assertEqual(
                verification["success_templates"][0]["path"],
                "templates/success.png",
            )


class CalibrationRegressionTest(unittest.TestCase):
    """One page's calibration preserves other pages and unedited options."""

    def test_partial_fallback_keeps_other_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document = _example_document()
            style = document["pages"][0]["answer_style"]
            style["fallback_positions"] = {"B": [50, 60]}
            calibrator = _calibrator(document, Path(directory) / "config.yaml")
            calibrator._fallback_positions = {"A": [10, 20]}
            calibrator._save_v2()
            style = calibrator._document["pages"][0]["answer_style"]
            self.assertEqual(
                style["fallback_positions"], {"A": [10, 20], "B": [50, 60]}
            )
            self.assertEqual(set(style["option_templates"]), {"B", "C", "D"})

    def test_calibration_keeps_all_profiles_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calibrator = _calibrator(
                _example_document(), Path(directory) / "config.yaml"
            )
            calibrator._save_v2()
            self.assertEqual(
                calibrator._document["runtime"]["active_profiles"], []
            )

    def test_profiles_write_templates_to_separate_directories(self) -> None:
        from unittest.mock import patch
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            document = _example_document()
            other = deepcopy(document["pages"][0])
            other["id"] = "another_page"
            document["pages"].append(other)
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            with patch("autoykt.calibration.mss.mss") as screen:
                screen.return_value.monitors = [{}, {}]
                first = Calibrator(path, "example_yuketang")
                second = Calibrator(path, "another_page")
            first._original_frame = np.zeros((8, 8, 3), dtype=np.uint8)
            second._original_frame = np.full((8, 8, 3), 255, dtype=np.uint8)
            for calibrator in (first, second):
                calibrator._write_crop(
                    (0, 0),
                    (8, 8),
                    calibrator._template_directory / "question.png",
                )
            self.assertNotEqual(
                first._template_directory, second._template_directory
            )
            self.assertNotEqual(
                (first._template_directory / "question.png").read_bytes(),
                (second._template_directory / "question.png").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
