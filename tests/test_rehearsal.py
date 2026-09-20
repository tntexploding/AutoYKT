"""Replay production selection and verification without accessing a desktop."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from autoykt.agent.models import ConsensusAnswer
from autoykt.core.config import ClickTargetConfig, PageFlowConfig, WindowTargetConfig
from autoykt.monitor.image_utils import write_png
from autoykt.monitor.rehearsal import run_rehearsal
from tests import test_image_calibration as calibration_fixture
from tests.test_image_calibration import _button, GRAY


class _Answers:

    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    async def answer(
        self,
        image_path,
        allowed_options,
        question_text="",
        knowledge_context="",
        *,
        timeout_seconds=None,
        multiple=False,
    ):
        self.calls.append((allowed_options, multiple))
        answer = next(self.answers)
        return ConsensusAnswer(
            answer, answer is not None, 1.0, votes={answer: 3} if answer else {}
        )


class RehearsalTest(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.fixture = calibration_fixture.ImageCalibrationTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.config = self.fixture._calibrate()
        self.config.knowledge.question_ocr_enabled = False
        self.config.pages[0].triggers[0].question_type = "multiple"
        self.config.pages[0].rearm.minimum_change_ratio = 0.02

    async def test_two_layouts_select_submit_verify_and_rearm_without_duplicates(
        self,
    ):
        second = self.fixture.frame.copy()
        second[80:480, :700] = 255
        for key, (x, y) in zip("ABC", [(70, 140), (350, 340), (360, 190)]):
            second[y : y + 60, x : x + 60] = _button(key, GRAY)
        second_path = self.fixture.root / "second.png"
        write_png(second_path, second)
        answers = _Answers(["B,D", "A,C"])
        self.config.pages.append(
            self.config.pages[0].model_copy(
                update={"id": "formal_live"}, deep=True
            )
        )
        self.config.runtime.active_profiles = ["formal_live"]
        before = self.config.model_dump()
        with (
            patch("autoykt.monitor.profile_runtime.ScreenCapture") as capture,
            patch("autoykt.monitor.clicker.Clicker.click_point") as click,
        ):
            report = await run_rehearsal(
                self.config,
                "example_yuketang",
                [self.fixture.image_path, second_path],
                self.fixture.root / "rehearsal",
                answerer=answers,
            )
        data = json.loads(report.read_text(encoding="utf-8"))
        self.assertTrue(data["passed"], data)
        self.assertEqual(data["actual_desktop_clicks"], 0)
        self.assertEqual(data["actual_website_submissions"], 0)
        self.assertEqual(
            answers.calls,
            [(frozenset("ABCDE"), True), (frozenset("ABC"), True)],
        )
        self.assertEqual(
            [q["virtual_submissions"] for q in data["questions"]],
            [[["B", "D"]], [["A", "C"]]],
        )
        self.assertTrue(all(q["no_duplicate_input"] for q in data["questions"]))
        self.assertEqual(self.config.model_dump(), before)
        capture.assert_not_called()
        click.assert_not_called()

    async def test_bound_window_controls_convert_without_touching_source_or_desktop(
        self,
    ):
        profile = self.config.pages[0]
        profile.target_window = WindowTargetConfig(
            title_pattern="Example classroom", client_size=(720, 500)
        )
        profile.entry_target = ClickTargetConfig(
            point=(10, 10), coordinate_space="window"
        )
        profile.submit_target = ClickTargetConfig(
            point=(600, 40), coordinate_space="window"
        )
        assert profile.page_guard is not None
        profile.page_flow = PageFlowConfig.model_validate(
            {
                "before_question": [
                    {
                        "name": "notice",
                        "when": profile.page_guard.model_dump(),
                        "target": {
                            "point": [10, 10],
                            "coordinate_space": "window",
                        },
                    }
                ],
            }
        )
        before = self.config.model_dump()
        with patch("autoykt.monitor.operations.WindowGuard") as native_window:
            report = await run_rehearsal(
                self.config,
                profile.id,
                [self.fixture.image_path],
                self.fixture.root / "bound-rehearsal",
                answerer=_Answers(["B,D"]),
            )
        self.assertTrue(json.loads(report.read_text())["passed"])
        self.assertEqual(self.config.model_dump(), before)
        native_window.assert_not_called()

    async def test_undetected_next_page_does_not_report_previous_success(self):
        report = await run_rehearsal(
            self.config,
            "example_yuketang",
            [self.fixture.image_path, self.fixture.image_path],
            self.fixture.root / "unchanged-rehearsal",
            answerer=_Answers(["B,D"]),
        )
        data = json.loads(report.read_text())
        self.assertFalse(data["passed"])
        self.assertEqual(data["questions"][1]["outcome"], "not_detected")
        self.assertIsNone(data["questions"][1]["operation_report"])
        self.assertFalse(data["questions"][1]["virtual_clicks"])

    async def test_no_consensus_never_selects_or_submits(self):
        report = await run_rehearsal(
            self.config,
            "example_yuketang",
            [self.fixture.image_path],
            self.fixture.root / "failed-rehearsal",
            answerer=_Answers([None]),
        )
        data = json.loads(report.read_text(encoding="utf-8"))
        self.assertFalse(data["passed"])
        self.assertEqual(data["questions"][0]["virtual_clicks"], [])
        self.assertEqual(data["questions"][0]["virtual_submissions"], [])
