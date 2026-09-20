"""Headless calibration and variable-layout button recognition regressions."""

from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from autoykt.calibration import Calibrator, ImageSelections
from autoykt.cli import main
from autoykt.core.config import ButtonColorsConfig, CalibrationSampleConfig, load_config
from autoykt.monitor.calibration_checks import calibration_sample_issues
from autoykt.monitor.detector import OptionTemplateDetector
from autoykt.monitor.image_inspection import inspect_image
from autoykt.monitor.image_utils import read_image, write_png
from tests.test_calibration import _example_document


BLUE = (240, 145, 75)
GRAY = (145, 145, 145)


def _button(key, color):
    frame = np.full((60, 60, 3), color, dtype=np.uint8)
    cv2.putText(
        frame,
        key,
        (15, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


class ImageCalibrationTest(unittest.TestCase):
    """Use synthetic pixels so no classroom content enters the test suite."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "config.yaml"
        self.document = _example_document()
        self.document["pages"][0]["answer_style"][
            "option_match_threshold"
        ] = 0.94
        self.path.write_text(yaml.safe_dump(self.document), encoding="utf-8")
        self.image_path = self.root / "source.png"
        self.frame = np.full((500, 720, 3), 255, dtype=np.uint8)
        for label, x in (("MULTI", 20), ("Course", 220), ("Done", 515)):
            cv2.putText(
                self.frame,
                label,
                (x, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
        self.positions = dict(
            zip(
                "ABCDE",
                [(100, 110), (360, 100), (100, 230), (360, 250), (100, 360)],
            )
        )
        for key, (x, y) in self.positions.items():
            self.frame[y : y + 60, x : x + 60] = _button(
                key, BLUE if key in "ABC" else GRAY
            )
        write_png(self.image_path, self.frame)
        self.selections = ImageSelections.model_validate(
            {
                "regions": {
                    "detection": [0, 0, 200, 75],
                    "question": [0, 80, 700, 400],
                    "answers": [0, 80, 700, 400],
                    "verification": [500, 0, 200, 75],
                    "rearm": [0, 80, 700, 400],
                },
                "trigger": [15, 12, 150, 42],
                "success": [510, 12, 140, 42],
                "page_guard": [215, 12, 150, 42],
                "options": {
                    key: [x, y, 60, 60]
                    for key, (x, y) in self.positions.items()
                },
                "selected_sample": "A",
                "unselected_sample": "D",
            }
        )

    def _calibrate(self):
        with (
            patch("autoykt.calibration.mss.mss") as screen,
            redirect_stdout(StringIO()),
        ):
            Calibrator(
                self.path, "example_yuketang", from_image=self.image_path
            ).apply_selections(self.selections)
        screen.assert_not_called()
        config = load_config(self.path)
        style = config.pages[0].answer_style
        self.detector = OptionTemplateDetector(
            {
                key: str(config.resolve_path(path))
                for key, path in style.option_templates.items()
            },
            style.option_match_threshold,
            match_grayscale=style.match_grayscale,
            button_colors=style.button_colors,
        )
        return config

    def test_calibration_records_all_five_and_keeps_profile_disabled(self):
        self.document["pages"][0]["enabled"] = True
        other = deepcopy(self.document["pages"][0])
        other["id"] = "other"
        self.document["pages"].append(other)
        self.path.write_text(yaml.safe_dump(self.document), encoding="utf-8")
        config = self._calibrate()
        self.assertFalse(config.pages[0].enabled)
        self.assertEqual(
            config.pages[0].answer_style.option_keys, frozenset("ABCDE")
        )
        self.assertFalse(config.pages[0].answer_style.fallback_positions)
        self.assertEqual(config.pages[1].id, "other")
        self.assertEqual(
            yaml.safe_load(self.path.read_text())["pages"][1], other
        )
        matches = self.detector.detect(self.frame)
        self.assertEqual(set(matches), set("ABCDE"))
        for key, (x, y) in self.positions.items():
            self.assertEqual(matches[key]["center"], (x + 30, y + 30))
            self.assertEqual(matches[key]["selected"], key in "ABC")
            self.assertFalse(matches[key]["ambiguous"])

    def test_gray_and_blue_work_at_new_positions_and_variable_counts(self):
        self._calibrate()
        for letters in ("EAC", "DB", "BCADE"):
            for selected in (False, True):
                with self.subTest(letters=letters, selected=selected):
                    frame = np.full_like(self.frame, 255)
                    expected = {}
                    for index, key in enumerate(letters):
                        x, y = 35 + index * 125, 150 + (index % 2) * 105
                        frame[y : y + 60, x : x + 60] = _button(
                            key, BLUE if selected else GRAY
                        )
                        expected[key] = (x + 30, y + 30)
                    matches = self.detector.detect(frame)
                    self.assertEqual(set(matches), set(letters))
                    for key, center in expected.items():
                        self.assertEqual(matches[key]["center"], center)
                        self.assertEqual(matches[key]["selected"], selected)
                        self.assertFalse(matches[key]["ambiguous"])

    def test_unknown_background_is_not_treated_as_unselected(self):
        self._calibrate()
        frame = np.full_like(self.frame, 255)
        frame[100:160, 100:160] = _button("A", (75, 170, 90))
        self.assertIsNone(self.detector.detect(frame)["A"]["selected"])

    def test_duplicate_letter_has_no_trusted_selection_state(self):
        self._calibrate()
        frame = np.full_like(self.frame, 255)
        frame[100:160, 100:160] = _button("E", BLUE)
        frame[300:360, 100:160] = _button("E", GRAY)
        match = self.detector.detect(frame)["E"]
        self.assertTrue(match["ambiguous"])
        self.assertIsNone(match["selected"])

    def test_two_templates_cannot_claim_one_button(self):
        config = self._calibrate()
        path = str(
            config.resolve_path(
                config.pages[0].answer_style.option_templates["A"]
            )
        )
        detector = OptionTemplateDetector(
            {"A": path, "B": path}, match_grayscale=True
        )
        frame = np.full_like(self.frame, 255)
        frame[100:160, 100:160] = _button("A", GRAY)
        matches = detector.detect(frame)
        self.assertTrue(all(match["ambiguous"] for match in matches.values()))

    def test_completed_inspection_has_no_desktop_or_model_calls(self):
        config = self._calibrate()
        with (
            patch("autoykt.calibration.mss.mss") as screen,
            patch("autoykt.monitor.clicker.Clicker.click_point") as mouse,
            patch("autoykt.monitor.screen_watcher.AnswerCoordinator") as models,
        ):
            report = inspect_image(config, "example_yuketang", self.image_path)
        for mocked in (screen, mouse, models):
            mocked.assert_not_called()
        data = json.loads(report.read_text(encoding="utf-8"))
        inspection = next(
            step for step in data["steps"] if step["step"] == "inspection"
        )
        self.assertEqual(inspection["page_state"], "completed")
        self.assertTrue(inspection["page_identity_matches"])
        self.assertEqual(inspection["missing_options"], [])
        self.assertEqual(inspection["options"]["B"]["center"], [390, 130])
        self.assertEqual(inspection["options"]["E"]["state"], "unselected")
        self.assertFalse(
            any(
                step["step"] in ("plan", "click_attempt")
                for step in data["steps"]
            )
        )
        self.assertEqual(
            read_image(report.parent / "annotated.png").shape, self.frame.shape
        )

    def test_labeled_samples_validate_state_options_and_type_without_input(
        self,
    ):
        config = self._calibrate()
        profile = config.pages[0]
        sample = CalibrationSampleConfig(
            path=str(self.image_path),
            expected_state="completed",
            question_type=profile.triggers[0].question_type,
            expected_options=list("ABCDE"),
            selected_options=list("ABC"),
        )
        profile.calibration_samples = [sample]
        with (
            patch("autoykt.monitor.screen_capture.ScreenCapture") as screen,
            patch("autoykt.monitor.operations.WindowGuard") as window,
            patch("autoykt.monitor.screen_watcher.AnswerCoordinator") as models,
        ):
            self.assertEqual(calibration_sample_issues(config, profile), [])
        for mocked in (screen, window, models):
            mocked.assert_not_called()
        sample.expected_state = "question"
        sample.question_type = (
            "multiple" if sample.question_type == "single" else "single"
        )
        sample.expected_options = list("ABCDEF")
        issues = "\n".join(calibration_sample_issues(config, profile))
        self.assertIn("expected question, got completed", issues)
        self.assertIn("trigger, got", issues)
        self.assertIn("expected options", issues)

    def test_labeled_samples_reject_missing_image_and_wrong_selection(self):
        config = self._calibrate()
        profile = config.pages[0]
        profile.calibration_samples = [
            CalibrationSampleConfig(
                path=str(self.image_path),
                expected_state="completed",
                expected_options=list("ABCDE"),
                selected_options=[],
            )
        ]
        self.assertIn(
            "selected options disagree",
            "\n".join(calibration_sample_issues(config, profile)),
        )
        profile.calibration_samples[0].path = str(self.root / "missing.png")
        self.assertTrue(calibration_sample_issues(config, profile))

    def test_missing_completion_marker_is_not_completed(self):
        config = self._calibrate()
        self.frame[:75, 500:700] = 255
        write_png(self.image_path, self.frame)
        report = inspect_image(config, "example_yuketang", self.image_path)
        data = json.loads(report.read_text(encoding="utf-8"))
        result = next(
            step for step in data["steps"] if step["step"] == "inspection"
        )
        self.assertEqual(result["page_state"], "question")
        self.assertFalse(result["success_visible"])

    def test_crop_outside_image_is_rejected_before_writing(self):
        data = self.selections.model_dump(mode="json")
        data["options"]["E"] = [-1, 20, 60, 60]
        invalid = ImageSelections.model_validate(data)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "outside the image"):
            Calibrator(
                self.path, "example_yuketang", from_image=self.image_path
            ).apply_selections(invalid)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse((self.root / "templates").exists())

    def test_identical_state_colors_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            ButtonColorsConfig(
                selected_rgb=(100, 100, 100), unselected_rgb=(110, 110, 110)
            )

    def test_command_line_calibration_and_inspection_are_offline(self):
        selections = self.root / "selections.json"
        selections.write_text(
            self.selections.model_dump_json(), encoding="utf-8"
        )
        common = ["--config", str(self.path), "--profile", "example_yuketang"]
        with (
            patch("autoykt.calibration.mss.mss") as screen,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(
                main(
                    [
                        "calibrate",
                        *common,
                        "--from-image",
                        str(self.image_path),
                        "--selections",
                        str(selections),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    ["inspect-image", *common, "--image", str(self.image_path)]
                ),
                0,
            )
        screen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
