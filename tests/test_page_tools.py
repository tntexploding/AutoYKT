"""Page-state recording and offline calibration with synthetic pixels."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import yaml

from autoykt.calibration import Calibrator
from autoykt.cli import main
from autoykt.core.config import AppConfig, AnswerStyleConfig, WindowTargetConfig
from autoykt.monitor.page_tools import capture_state, preview_option
from tests.test_calibration import _example_document


class PageToolsTest(unittest.TestCase):
    """Setup tools work without a quiz server, model key, OCR, or mouse."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = AppConfig.model_validate(_example_document())
        self.config.bind_source(self.root / "config.yaml", False)
        self.profile = self.config.pages[0]
        self.profile.answer_style = AnswerStyleConfig(
            fallback_positions={"B": (30, 40)}
        )

    def _write_config(self):
        self.config.source_path.write_text(
            yaml.safe_dump(self.config.model_dump(mode="json")),
            encoding="utf-8",
        )

    def _capture(self):
        device = Mock()
        device.__enter__ = Mock(return_value=device)
        device.__exit__ = Mock(return_value=False)
        device.surface_region = {
            "left": 80,
            "top": 90,
            "width": 160,
            "height": 120,
        }
        device.grab_frame.return_value = np.random.default_rng(3).integers(
            0, 256, (120, 160, 3), dtype=np.uint8
        )
        device.absolute_from_monitor.side_effect = lambda point: (
            80 + point[0],
            90 + point[1],
        )
        return device

    def test_capture_and_offline_calibration_round_trip(self):
        self.profile.answer_style.fallback_coordinate_space = "window"
        self.profile.target_window = WindowTargetConfig(
            title_pattern="Quiz", client_size=(160, 120)
        )
        self._write_config()
        device = self._capture()
        with (
            patch(
                "autoykt.monitor.page_tools.ScreenCapture", return_value=device
            ),
            patch("autoykt.monitor.operations.WindowGuard"),
        ):
            report_path = capture_state(self.config, self.profile.id, "success")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "captured")
        self.assertTrue((report_path.parent / "full.png").is_file())
        with patch("autoykt.calibration.mss.mss") as screen:
            calibrator = Calibrator(
                self.config.source_path, self.profile.id, from_state=report_path
            )
            calibrator._refresh_frame()
            screen.assert_not_called()
        np.testing.assert_array_equal(
            calibrator._original_frame, device.grab_frame.return_value
        )
        calibrator._regions["submit"] = [20, 30, 40, 20]
        calibrator._regions["page_guard"] = [2, 2, 6, 6]
        calibrator._guard_template_path = "templates/page.png"
        calibrator._save_v2()
        profile = calibrator._document["pages"][0]
        self.assertEqual(profile["submit_target"]["coordinate_space"], "window")
        self.assertEqual(profile["page_guard"]["region"], [2, 2, 6, 6])
        AppConfig.model_validate(calibrator._document)

    def test_offline_calibration_rejects_another_profile(self):
        self._write_config()
        with patch(
            "autoykt.monitor.page_tools.ScreenCapture",
            return_value=self._capture(),
        ):
            path = capture_state(self.config, self.profile.id, "question")
        document = json.loads(path.read_text(encoding="utf-8"))
        document["profile_id"] = "another_page"
        path.write_text(json.dumps(document), encoding="utf-8")
        with patch("autoykt.calibration.mss.mss") as screen:
            with self.assertRaisesRegex(ValueError, "does not match"):
                Calibrator(
                    self.config.source_path, self.profile.id, from_state=path
                )
            screen.assert_not_called()

    def test_manual_preview_needs_no_models(self):
        with (
            patch(
                "autoykt.monitor.page_tools.ScreenCapture",
                return_value=self._capture(),
            ),
            patch("autoykt.monitor.screen_watcher.AnswerCoordinator") as models,
        ):
            path = preview_option(self.config, self.profile.id, "b")
        models.assert_not_called()
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "preview_only")
        self.assertFalse(
            any(step["step"] == "click_attempt" for step in report["steps"])
        )

    def test_run_switches_are_in_memory_and_reach_scheduler(self):
        self.profile.enabled = True
        self.config.runtime.dry_run = False
        self._write_config()
        before = self.config.source_path.read_bytes()
        run = AsyncMock(return_value=0)
        with (
            patch("autoykt.cli._load_from_argument", return_value=self.config),
            patch("autoykt.cli._run_automation", run),
        ):
            result = main(
                ["run", "--profile", self.profile.id, "--once", "--dry-run"]
            )
        self.assertEqual(result, 0)
        self.assertTrue(self.config.runtime.dry_run)
        self.assertEqual(self.config.runtime.active_profiles, [self.profile.id])
        run.assert_awaited_once_with(self.config, False, once=True)
        self.assertEqual(self.config.source_path.read_bytes(), before)
