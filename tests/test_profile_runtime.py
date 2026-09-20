"""Integration-style tests for one visual answer cycle."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock

import cv2
import numpy as np

from autoykt.agent.models import ConsensusAnswer
from autoykt.core.config import AppConfig, Region
from autoykt.core.event_bus import EventBus, EventType
from autoykt.core.state import WorkflowState
from autoykt.monitor.profile_runtime import PageAutomation
from autoykt.monitor.image_utils import read_image


class _FakeClicker:

    def __init__(self) -> None:
        self.points: list[tuple[int, int]] = []

    def click_point(self, x: int, y: int) -> bool:
        self.points.append((x, y))
        return True


class _FakeAnswerer:

    def __init__(self, result: ConsensusAnswer) -> None:
        self._result = result
        self.calls = 0

    async def answer(
        self,
        image_path: Path,
        allowed_options: frozenset[str],
        question_text: str = "",
        knowledge_context: str = "",
        *,
        timeout_seconds: float | None = None,
        multiple: bool = False,
    ) -> ConsensusAnswer:
        del (
            image_path,
            allowed_options,
            question_text,
            knowledge_context,
            timeout_seconds,
            multiple,
        )
        self.calls += 1
        return self._result


class _FakeCapture:

    def __init__(
        self,
        region: Region,
        output_directory: Path,
        trigger_frame: np.ndarray,
        clicker: _FakeClicker,
        question_changes: bool = False,
    ) -> None:
        self._region = region
        self._output_directory = output_directory
        self._trigger_frame = trigger_frame
        self._clicker = clicker
        self._question_changes = question_changes
        self._question_grabs = 0
        self.closed = False

    def grab_frame(self) -> np.ndarray:
        width, height = self._region[2], self._region[3]
        if width == 20:
            return self._trigger_frame.copy()
        if width == 30:
            self._question_grabs += 1
            if self._question_changes and self._question_grabs > 2:
                return np.full((height, width, 3), 255, dtype=np.uint8)
        if width == 50 and self._clicker.points:
            return np.full((height, width, 3), 255, dtype=np.uint8)
        return np.zeros((height, width, 3), dtype=np.uint8)

    def save_screenshot(self, frame: np.ndarray, prefix: str) -> Path:
        path = self._output_directory / f"{prefix}.png"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"could not write test image: {path}")
        return path

    def absolute_from_frame(self, point: tuple[int, int]) -> tuple[int, int]:
        return point[0] + 100, point[1] + 200

    def absolute_from_monitor(self, point: tuple[int, int]) -> tuple[int, int]:
        return point[0] + 1000, point[1] + 2000

    def close(self) -> None:
        self.closed = True


def _configuration(template_path: Path, source_path: Path) -> AppConfig:
    document = {
        "version": 2,
        "runtime": {
            "active_profiles": ["test"],
            "poll_interval_seconds": 0.01,
            "dry_run": False,
        },
        "answering": {
            "providers": [
                {
                    "name": "fake",
                    "api_key_env": "FAKE_API_KEY",
                    "models": ["vision"],
                }
            ],
            "minimum_responses": 1,
            "minimum_agreement": 1,
            "auto_apply": True,
        },
        "knowledge": {"enabled": False, "question_ocr_enabled": False},
        "pages": [
            {
                "id": "test",
                "display_name": "Test",
                "monitor_index": 1,
                "regions": {
                    "detection": [0, 0, 20, 20],
                    "question": [0, 0, 30, 30],
                    "answers": [0, 0, 40, 40],
                    "verification": [0, 0, 50, 50],
                    "rearm": [0, 0, 60, 60],
                },
                "triggers": [
                    {
                        "name": "question",
                        "path": str(template_path),
                        "threshold": 0.99,
                        "consecutive_hits": 1,
                        "action": "answer",
                    }
                ],
                "answer_style": {
                    "option_templates": {},
                    "fallback_positions": {"B": [321, 654]},
                    "fallback_coordinate_space": "screen",
                },
                "question_ready": {
                    "delay_seconds": 0,
                    "timeout_seconds": 0.1,
                    "stable_frames": 1,
                },
                "verification": {
                    "success_templates": [],
                    "timeout_seconds": 0.1,
                    "poll_interval_seconds": 0.001,
                    "stable_hits": 1,
                    "minimum_change_ratio": 0.01,
                },
                "rearm": {
                    "timeout_seconds": 1,
                    "stable_hits": 1,
                    "minimum_change_ratio": 0.1,
                },
            }
        ],
    }
    config = AppConfig.model_validate(document)
    config.bind_source(source_path, migrated_from_legacy=False)
    return config


class PageAutomationTest(unittest.IsolatedAsyncioTestCase):
    """Exercise accepted, rejected, and detect-only answer paths."""

    async def test_applies_only_actionable_consensus_and_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path, frame = _trigger_image(root)
            config = _configuration(template_path, root / "config.yaml")
            clicker = _FakeClicker()
            answerer = _FakeAnswerer(
                ConsensusAnswer(
                    option="B",
                    actionable=True,
                    agreement_ratio=1.0,
                    votes={"B": 2},
                    reason="consensus reached",
                )
            )
            captures: list[_FakeCapture] = []

            def capture_factory(region: Region) -> _FakeCapture:
                capture = _FakeCapture(region, root, frame, clicker)
                captures.append(capture)
                return capture

            automation = PageAutomation(
                config,
                config.page_profile("test"),
                EventBus(),
                answerer,
                clicker,
                None,
                capture_factory=capture_factory,
            )
            await automation.poll_once()
            self.assertEqual(answerer.calls, 1)
            self.assertEqual(clicker.points, [(321, 654)])
            self.assertEqual(automation.state, WorkflowState.REARMING)
            await automation.stop()
            self.assertTrue(all(capture.closed for capture in captures))

    async def test_waiting_on_answered_question_is_not_a_session_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path, frame = _trigger_image(root)
            config = _configuration(template_path, root / "config.yaml")
            clicker = _FakeClicker()
            answerer = _FakeAnswerer(
                ConsensusAnswer(
                    option="B", actionable=True, agreement_ratio=1.0
                )
            )
            bus = EventBus()
            bus.publish = AsyncMock()
            automation = PageAutomation(
                config,
                config.pages[0],
                bus,
                answerer,
                clicker,
                None,
                capture_factory=lambda region: _FakeCapture(
                    region, root, frame, clicker
                ),
            )
            try:
                await automation.poll_once()
                self.assertEqual(automation.outcome, "verified")
                # The next slide can legitimately take longer than rearm timeout.
                automation._progress.rearm_deadline = 0  # pylint: disable=protected-access
                await automation.poll_once()
                await automation.poll_once()
                self.assertEqual(clicker.points, [(321, 654)])
                self.assertEqual(answerer.calls, 1)
                self.assertEqual(automation.state, WorkflowState.REARMING)
                self.assertFalse(
                    any(
                        call.args[0].type == EventType.ERROR
                        for call in bus.publish.call_args_list
                    )
                )
            finally:
                await automation.stop()

    async def test_rejected_consensus_never_clicks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path, frame = _trigger_image(root)
            config = _configuration(template_path, root / "config.yaml")
            clicker = _FakeClicker()
            answerer = _FakeAnswerer(
                ConsensusAnswer(
                    option=None,
                    actionable=False,
                    agreement_ratio=0.5,
                    reason="model vote is tied",
                )
            )

            def capture_factory(region: Region) -> _FakeCapture:
                return _FakeCapture(region, root, frame, clicker)

            automation = PageAutomation(
                config,
                config.page_profile("test"),
                EventBus(),
                answerer,
                clicker,
                None,
                capture_factory=capture_factory,
            )
            await automation.poll_once()
            self.assertEqual(answerer.calls, 1)
            self.assertFalse(clicker.points)
            self.assertEqual(automation.state, WorkflowState.REARMING)

    async def test_detect_only_calls_neither_model_nor_mouse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path, frame = _trigger_image(root)
            config = _configuration(template_path, root / "config.yaml")
            clicker = _FakeClicker()
            answerer = _FakeAnswerer(
                ConsensusAnswer(
                    option="B",
                    actionable=True,
                    agreement_ratio=1.0,
                )
            )

            def capture_factory(region: Region) -> _FakeCapture:
                return _FakeCapture(region, root, frame, clicker)

            automation = PageAutomation(
                config,
                config.page_profile("test"),
                EventBus(),
                answerer,
                clicker,
                None,
                detect_only=True,
                capture_factory=capture_factory,
            )
            await automation.poll_once()
            self.assertEqual(answerer.calls, 0)
            self.assertFalse(clicker.points)
            self.assertEqual(automation.state, WorkflowState.REARMING)

    async def test_partial_capture_construction_is_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path, frame = _trigger_image(root)
            config = _configuration(template_path, root / "config.yaml")
            clicker = _FakeClicker()
            answerer = _FakeAnswerer(
                ConsensusAnswer(
                    option=None,
                    actionable=False,
                    agreement_ratio=0.0,
                )
            )
            captures: list[_FakeCapture] = []

            def capture_factory(region: Region) -> _FakeCapture:
                if len(captures) == 2:
                    raise RuntimeError("capture unavailable")
                capture = _FakeCapture(region, root, frame, clicker)
                captures.append(capture)
                return capture

            with self.assertRaises(RuntimeError):
                PageAutomation(
                    config,
                    config.page_profile("test"),
                    EventBus(),
                    answerer,
                    clicker,
                    None,
                    capture_factory=capture_factory,
                )
            self.assertEqual(len(captures), 2)
            self.assertTrue(all(capture.closed for capture in captures))

    async def test_answer_is_rejected_if_question_changed_during_models(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_path, frame = _trigger_image(root)
            config = _configuration(template_path, root / "config.yaml")
            clicker = _FakeClicker()
            answerer = _FakeAnswerer(
                ConsensusAnswer(
                    option="B",
                    actionable=True,
                    agreement_ratio=1.0,
                    votes={"B": 2},
                )
            )

            def capture_factory(region: Region) -> _FakeCapture:
                return _FakeCapture(
                    region,
                    root,
                    frame,
                    clicker,
                    question_changes=True,
                )

            automation = PageAutomation(
                config,
                config.page_profile("test"),
                EventBus(),
                answerer,
                clicker,
                None,
                capture_factory=capture_factory,
            )
            await automation.poll_once()
            self.assertEqual(answerer.calls, 1)
            self.assertFalse(clicker.points)
            self.assertEqual(automation.state, WorkflowState.REARMING)


def _trigger_image(directory: Path) -> tuple[Path, np.ndarray]:
    random = np.random.default_rng(7)
    template = random.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    frame = random.integers(0, 256, (20, 20, 3), dtype=np.uint8)
    frame[8:14, 9:15] = template
    path = directory / "trigger.png"
    if not cv2.imwrite(str(path), template):
        raise OSError(f"could not write test template: {path}")
    return path, frame


class WorkflowScene(unittest.IsolatedAsyncioTestCase):
    """Exercise changing pages without network, OCR, or a real mouse."""

    async def asyncSetUp(self) -> None:
        from unittest.mock import AsyncMock

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        template_path, trigger = _trigger_image(self.root)
        self.config = _configuration(template_path, self.root / "config.yaml")
        self.scene = {
            size: np.zeros((size, size, 3), dtype=np.uint8)
            for size in (20, 30, 40, 50, 60)
        }
        self.scene[20] = trigger
        self.clicker = _FakeClicker()
        self.result = ConsensusAnswer("B", True, 1.0, votes={"B": 1})
        self.answerer = _FakeAnswerer(self.result)
        self.answer_request = AsyncMock(return_value=self.result)
        self.answerer.answer = self.answer_request
        self.bus = EventBus()
        self.events = AsyncMock()
        self.bus.publish = self.events

    def _automation(self, detect_only: bool = False) -> PageAutomation:
        scene = self.scene
        root, clicker = self.root, self.clicker

        class SceneCapture(_FakeCapture):

            def grab_frame(self) -> np.ndarray:
                width, height = self._region[2:]
                if width == height and width in scene:
                    return scene[width].copy()
                return np.zeros((height, width, 3), dtype=np.uint8)

        def capture(region: Region) -> _FakeCapture:
            return SceneCapture(region, root, scene[20], clicker)

        automation = PageAutomation(
            self.config,
            self.config.pages[0],
            self.bus,
            self.answerer,
            self.clicker,
            None,
            detect_only=detect_only,
            capture_factory=capture,
        )
        self.addAsyncCleanup(automation.stop)
        return automation

    def _successful_click(self, x: int, y: int) -> bool:
        self.clicker.points.append((x, y))
        self.scene[50][:] = 255
        return True


class WorkflowRegressionTest(WorkflowScene):
    """Regression tests for unchanged, stale, and uncertain question pages."""

    async def test_stale_answer_does_not_swallow_next_question(self) -> None:
        async def change_question(*args, **kwargs):
            del args, kwargs
            self.scene[30][:] = 255
            self.scene[60][:] = 255
            return self.result

        self.answer_request.side_effect = change_question
        self.clicker.click_point = self._successful_click
        automation = self._automation()
        await automation.poll_once()
        self.assertFalse(self.clicker.points)
        await automation.poll_once()
        self.assertEqual(automation.state, WorkflowState.WAITING)
        self.answer_request.side_effect = None
        await automation.poll_once()
        self.assertEqual(self.answer_request.await_count, 2)
        self.assertEqual(len(self.clicker.points), 1)

    async def test_immediate_next_question_can_rearm(self) -> None:
        def next_question(x, y):
            self._successful_click(x, y)
            self.scene[30][:] = 255
            self.scene[60][:] = 255
            return True

        self.clicker.click_point = next_question
        automation = self._automation()
        await automation.poll_once()
        await automation.poll_once()
        self.assertEqual(automation.state, WorkflowState.WAITING)

    async def test_completed_question_is_not_answered_twice(self) -> None:
        self.clicker.click_point = self._successful_click
        automation = self._automation()
        for _ in range(5):
            await automation.poll_once()
        self.assertEqual(self.answer_request.await_count, 1)
        self.assertEqual(len(self.clicker.points), 1)
        self.assertEqual(automation.state, WorkflowState.REARMING)

    async def test_page_changed_after_selection_is_not_submitted(self) -> None:
        from autoykt.core.config import ClickTargetConfig

        self.config.pages[0].submit_target = ClickTargetConfig(point=(2, 3))

        def change_after_click(x, y):
            self._successful_click(x, y)
            self.scene[30][:] = 255
            self.scene[60][:] = 255
            return True

        self.clicker.click_point = change_after_click
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.clicker.points, [(321, 654)])
        self.assertEqual(automation.state, WorkflowState.ERROR)
        self.assertTrue(automation.needs_attention)

    async def test_preexisting_success_marker_does_not_allow_clicks(
        self,
    ) -> None:
        from autoykt.core.config import ImageTemplateConfig

        template_path, _ = _trigger_image(self.root)
        self.config.pages[0].verification.success_templates = [
            ImageTemplateConfig(path=str(template_path), threshold=0.99)
        ]
        marker = read_image(template_path)
        self.scene[50][2:8, 2:8] = marker
        automation = self._automation()
        for _ in range(5):
            await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.outcome, "already_completed")
        self.assertEqual(automation.state, WorkflowState.REARMING)
        self.assertFalse(automation.needs_attention)
        self.assertFalse(
            any(
                call.args[0].payload.get("success")
                for call in self.events.call_args_list
            )
        )

    async def test_expired_question_skips_models_and_input(self) -> None:
        from autoykt.core.config import ImageTemplateConfig

        template_path, _ = _trigger_image(self.root)
        self.config.pages[0].verification.failure_templates = [
            ImageTemplateConfig(path=str(template_path), threshold=0.99)
        ]
        self.scene[50][2:8, 2:8] = read_image(template_path)
        automation = self._automation()
        for _ in range(5):
            await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.outcome, "unavailable")
        self.assertEqual(automation.state, WorkflowState.REARMING)
        self.assertFalse(automation.needs_attention)

    async def test_completed_question_skips_entry_and_preview_answering(
        self,
    ) -> None:
        from autoykt.core.config import ClickTargetConfig, ImageTemplateConfig

        template_path, _ = _trigger_image(self.root)
        self.config.pages[0].verification.success_templates = [
            ImageTemplateConfig(path=str(template_path), threshold=0.99)
        ]
        self.config.pages[0].entry_target = ClickTargetConfig(point=(2, 3))
        self.scene[50][2:8, 2:8] = read_image(template_path)
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                self.config.runtime.dry_run = dry_run
                automation = self._automation()
                await automation.poll_once()
                self.answer_request.assert_not_awaited()
                self.assertFalse(self.clicker.points)
                self.assertEqual(automation.outcome, "already_completed")

    async def test_completion_during_capture_skips_model_and_can_rearm(
        self,
    ) -> None:
        from autoykt.core.config import ImageTemplateConfig

        template_path, _ = _trigger_image(self.root)
        self.config.pages[0].verification.success_templates = [
            ImageTemplateConfig(path=str(template_path), threshold=0.99)
        ]
        marker = read_image(template_path)
        automation = self._automation()
        capture = automation._capture_stable_question

        async def completed_during_capture():
            frame = await capture()
            self.scene[50][2:8, 2:8] = marker
            return frame

        automation._capture_stable_question = completed_during_capture
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.outcome, "already_completed")
        self.assertEqual(automation.state, WorkflowState.REARMING)

        automation._capture_stable_question = capture
        self.scene[50][:] = 0
        self.scene[30][:] = 255
        self.scene[60][:] = 255

        def complete_new_question(x, y):
            self.clicker.points.append((x, y))
            self.scene[50][2:8, 2:8] = marker
            return True

        self.clicker.click_point = complete_new_question
        await automation.poll_once()
        self.assertEqual(automation.state, WorkflowState.WAITING)
        await automation.poll_once()
        self.answer_request.assert_awaited_once()
        self.assertEqual(self.clicker.points, [(321, 654)])
        self.assertEqual(automation.outcome, "verified")

    async def test_success_after_selection_skips_extra_submit_and_blocks_feedback(
        self,
    ) -> None:
        from autoykt.core.config import ClickTargetConfig, ImageTemplateConfig

        template_path, _ = _trigger_image(self.root)
        self.config.pages[0].verification.success_templates = [
            ImageTemplateConfig(path=str(template_path), threshold=0.99)
        ]
        self.config.pages[0].submit_target = ClickTargetConfig(point=(2, 3))
        marker = read_image(template_path)

        def show_success(x, y):
            self.clicker.points.append((x, y))
            self.scene[50][2:8, 2:8] = marker
            self.scene[60][:] = 255
            return True

        self.clicker.click_point = show_success
        automation = self._automation()
        await automation.poll_once()
        await automation.poll_once()
        self.assertEqual(self.clicker.points, [(321, 654)])
        self.assertEqual(automation.state, WorkflowState.REARMING)
        self.assertTrue(
            any(
                call.args[0].payload.get("success")
                for call in self.events.call_args_list
            )
        )
        self.scene[50][:] = 0
        await automation.poll_once()
        self.assertEqual(automation.state, WorkflowState.WAITING)

    async def test_dry_run_and_auto_apply_are_independent_guards(self) -> None:
        for dry_run, auto_apply in ((True, True), (False, False)):
            with self.subTest(dry_run=dry_run, auto_apply=auto_apply):
                self.config.runtime.dry_run = dry_run
                self.config.answering.auto_apply = auto_apply
                automation = self._automation()
                await automation.poll_once()
                self.assertFalse(self.clicker.points)

    async def test_unchanged_submission_reports_failure(self) -> None:
        automation = self._automation()
        await automation.poll_once()
        results = [call.args[0] for call in self.events.call_args_list]
        self.assertTrue(
            any(event.payload.get("success") is False for event in results)
        )
        self.assertFalse(
            any(event.payload.get("success") is True for event in results)
        )
        for _ in range(3):
            await automation.poll_once()
        self.assertEqual(len(self.clicker.points), 1)

    async def test_question_is_checked_again_after_option_location(
        self,
    ) -> None:
        automation = self._automation()
        original = automation._session.save_frame

        def change_after_location(label, frame):
            path = original(label, frame)
            self.scene[30][:] = 255
            return path

        automation._session.save_frame = change_after_location
        await automation.poll_once()
        self.assertFalse(self.clicker.points)

    async def test_dry_run_outputs_plan_without_clicking(self) -> None:
        import json

        self.config.runtime.dry_run = True
        automation = self._automation()
        await automation.poll_once()
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.completed_cycles, 1)
        self.assertEqual(automation.outcome, "preview_only")
        assert automation.report_path is not None
        report = json.loads(automation.report_path.read_text(encoding="utf-8"))
        plan = next(step for step in report["steps"] if step["step"] == "plan")
        self.assertEqual(plan["plan"]["clicks"][0]["point"], [321, 654])
        self.assertTrue(
            (automation.report_path.parent / "plan_preview.png").is_file()
        )

    async def test_entry_preview_does_not_answer_an_unopened_question(
        self,
    ) -> None:
        from autoykt.core.config import ClickTargetConfig

        self.config.runtime.dry_run = True
        self.config.pages[0].entry_target = ClickTargetConfig(point=(2, 3))
        automation = self._automation()
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.outcome, "entry_preview")
        self.assertEqual(automation.completed_cycles, 1)

    async def test_answer_layout_changed_after_preview_does_not_click(
        self,
    ) -> None:
        automation = self._automation()
        original = automation._session.preview

        def change_answers(plan, capture, **kwargs):
            path = original(plan, capture, **kwargs)
            self.scene[40][:] = 255
            return path

        automation._session.preview = change_answers
        await automation.poll_once()
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.outcome, "failed")

    async def test_failure_marker_after_selection_prevents_submit(self) -> None:
        from autoykt.core.config import ClickTargetConfig, ImageTemplateConfig

        template_path, _ = _trigger_image(self.root)
        self.config.pages[0].verification.failure_templates = [
            ImageTemplateConfig(path=str(template_path), threshold=0.99)
        ]
        self.config.pages[0].submit_target = ClickTargetConfig(point=(2, 3))
        marker = read_image(template_path)

        def fail_selection(x, y):
            self.clicker.points.append((x, y))
            self.scene[50][2:8, 2:8] = marker
            return True

        self.clicker.click_point = fail_selection
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(len(self.clicker.points), 1)
        self.assertEqual(automation.outcome, "failed")
        for _ in range(3):
            await automation.poll_once()
        self.assertEqual(len(self.clicker.points), 1)

    async def test_last_moment_change_after_log_write_prevents_click(
        self,
    ) -> None:
        automation = self._automation()
        original = automation._session.record

        def record_then_change(step, **details):
            original(step, **details)
            if step == "click_attempt":
                self.scene[30][:] = 255

        automation._session.record = record_then_change
        await automation.poll_once()
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.outcome, "failed")

    async def test_idle_window_pause_can_resume_without_answering_early(
        self,
    ) -> None:
        from unittest.mock import Mock
        from autoykt.monitor.windows import WindowUnavailable

        self.config.runtime.dry_run = True
        automation = self._automation()
        check = Mock(side_effect=WindowUnavailable("not foreground"))
        automation._session.check_page = check
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertEqual(automation.completed_cycles, 0)
        check.side_effect = None
        await automation.poll_once()
        self.answer_request.assert_awaited_once()
        self.assertEqual(automation.outcome, "preview_only")

    async def test_detect_only_does_not_load_answering_templates(self) -> None:
        from autoykt.core.config import ImageTemplateConfig
        from autoykt.monitor.image_utils import write_png

        path = self.root / "uncalibrated.png"
        write_png(path, np.zeros((8, 8, 3), dtype=np.uint8))
        self.config.pages[0].answer_style.option_templates = {"B": str(path)}
        self.config.pages[0].verification.success_templates = [
            ImageTemplateConfig(path=str(path))
        ]
        automation = self._automation(detect_only=True)
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)
