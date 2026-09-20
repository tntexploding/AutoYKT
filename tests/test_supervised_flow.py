"""Supervised classroom workflows, bounded recovery, and restart evidence."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from autoykt.agent.models import ConsensusAnswer, ModelAnswer
from autoykt.core.config import (
    ClickTargetConfig,
    ConfigError,
    ImageTemplateConfig,
    PageFlowConfig,
    PageGuardConfig,
)
from autoykt.core.state import WorkflowState
from autoykt.monitor.image_utils import write_png
from autoykt.monitor.page_tools import recover_current
from tests.test_profile_runtime import WorkflowScene


class SupervisedFlowTest(WorkflowScene):
    """Operate synthetic pages without models, network, OCR, or a desktop."""

    def _marker(self, name: str, seed: int) -> tuple[Path, np.ndarray]:
        frame = np.random.default_rng(seed).integers(
            0, 256, (8, 8, 3), dtype=np.uint8
        )
        path = self.root / (name + ".png")
        write_png(path, frame)
        return path, frame

    def _result_flow(self):
        success_path, success = self._marker("success", 201)
        close_path, close = self._marker("close", 202)
        next_path, next_marker = self._marker("next", 203)
        self.config.pages[0].verification.success_templates = [
            ImageTemplateConfig(path=str(success_path), threshold=0.99)
        ]
        self.scene[70] = np.zeros((70, 70, 3), dtype=np.uint8)
        self.config.pages[0].page_flow = PageFlowConfig.model_validate(
            {
                "after_submit": [
                    {
                        "name": name,
                        "when": {
                            "path": str(path),
                            "threshold": 0.99,
                            "region": [0, 0, 70, 70],
                        },
                        "target": {
                            "point": point,
                            "coordinate_space": "screen",
                        },
                        "stable_hits": 1,
                    }
                    for name, path, point in (
                        ("close", close_path, [101, 102]),
                        ("next", next_path, [201, 202]),
                    )
                ],
                "transition_timeout_seconds": 0.01,
            }
        )
        return success, close, next_marker

    async def test_two_questions_close_result_next_and_return_to_waiting(self):
        success, close, next_marker = self._result_flow()
        question_index = 0

        def interact(x, y):
            nonlocal question_index
            self.clicker.points.append((x, y))
            if (x, y) == (321, 654):
                self.scene[50][2:10, 2:10] = success
                self.scene[70][2:10, 2:10] = close
            elif (x, y) == (101, 102):
                self.scene[70][2:10, 2:10] = next_marker
            else:
                self.scene[50][:] = 0
                self.scene[70][:] = 0
                question_index += 1
                self.scene[30][:] = question_index * 80
                self.scene[60][:] = question_index * 80
            return True

        self.clicker.click_point = interact
        automation = self._automation()
        for _ in range(8):
            await automation.poll_once()
        self.assertEqual(self.answer_request.await_count, 2)
        self.assertEqual(
            self.clicker.points, [(321, 654), (101, 102), (201, 202)] * 2
        )
        self.assertEqual(automation.state, WorkflowState.WAITING)
        self.assertFalse(automation._session.checkpoint.path.exists())
        assert automation.report_path is not None
        report = json.loads(automation.report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "verified")
        self.assertTrue(
            (
                automation.report_path.parent / "advance_next_preview.png"
            ).is_file()
        )

    async def test_unchanged_result_button_is_not_clicked_twice(self):
        success, close, _ = self._result_flow()

        def interact(x, y):
            self.clicker.points.append((x, y))
            self.scene[50][2:10, 2:10] = success
            self.scene[70][2:10, 2:10] = close
            return True

        self.clicker.click_point = interact
        automation = self._automation()
        await automation.poll_once()
        await automation.poll_once()
        await asyncio.sleep(0.02)
        for _ in range(3):
            await automation.poll_once()
        self.assertEqual(self.clicker.points, [(321, 654), (101, 102)])
        self.assertTrue(automation.needs_attention)

    async def test_result_control_is_rechecked_after_preview(self):
        success, close, _ = self._result_flow()

        def interact(x, y):
            self.clicker.points.append((x, y))
            self.scene[50][2:10, 2:10] = success
            self.scene[70][2:10, 2:10] = close
            return True

        self.clicker.click_point = interact
        automation = self._automation()
        await automation.poll_once()
        preview = automation._session.preview

        def move_control(*args, **kwargs):
            result = preview(*args, **kwargs)
            self.scene[70][:] = 0
            return result

        automation._session.preview = move_control
        await automation.poll_once()
        self.assertEqual(self.clicker.points, [(321, 654)])
        self.assertTrue(automation.needs_attention)

    async def test_navigation_requires_explicit_success_evidence(self):
        self._result_flow()
        self.config.pages[0].verification.success_templates = []
        with self.assertRaisesRegex(ConfigError, "success templates"):
            self._automation()

    def _question_notice(self):
        success_path, self.notice_success = self._marker("notice-success", 213)
        self.config.pages[0].verification.success_templates = [
            ImageTemplateConfig(path=str(success_path), threshold=0.99)
        ]
        path, marker = self._marker("question-notice", 212)
        self.scene[70] = np.zeros((70, 70, 3), dtype=np.uint8)
        self.scene[70][2:10, 2:10] = marker
        self.config.pages[0].page_flow = PageFlowConfig.model_validate(
            {
                "before_question": [
                    {
                        "name": "new_question",
                        "when": {
                            "path": str(path),
                            "region": [0, 0, 70, 70],
                            "threshold": 0.99,
                        },
                        "target": {
                            "point": [101, 102],
                            "coordinate_space": "screen",
                        },
                        "stable_hits": 1,
                    }
                ],
                "transition_timeout_seconds": 0.5,
            }
        )
        return marker

    async def test_question_notice_enters_from_waiting_and_rearming(self):
        marker = self._question_notice()

        def click(x, y):
            if (x, y) == (101, 102):
                self.clicker.points.append((x, y))
                self.scene[70][:] = 0
                self.scene[50][:] = 0
                self.scene[30][:] += 60
                self.scene[60][:] += 60
                return True
            self.clicker.points.append((x, y))
            self.scene[50][2:10, 2:10] = self.notice_success
            return True

        self.clicker.click_point = click
        automation = self._automation()
        for _ in range(2):
            await automation.poll_once()
            self.assertEqual(automation.state, WorkflowState.WAITING)
            await automation.poll_once()
            self.assertEqual(automation.outcome, "verified")
            self.scene[70][2:10, 2:10] = marker
        self.assertEqual(self.answer_request.await_count, 2)
        self.assertEqual(self.clicker.points, [(101, 102), (321, 654)] * 2)

    async def test_unchanged_notice_is_not_repeated_or_answered(self):
        self._question_notice()
        self.config.pages[0].page_flow.transition_timeout_seconds = 0.5
        automation = self._automation()
        for _ in range(3):
            await automation.poll_once()
        self.assertEqual(self.clicker.points, [(101, 102)])
        self.assertTrue(automation.needs_attention)
        self.answer_request.assert_not_awaited()

    async def test_focus_autoadvance_reobserves_instead_of_clicking_notice(
        self,
    ):
        self._question_notice()

        def answer_click(x, y):
            self.clicker.points.append((x, y))
            self.scene[50][2:10, 2:10] = self.notice_success
            return True

        self.clicker.click_point = answer_click
        automation = self._automation()
        window = Mock()

        def focus(*_args):
            self.scene[70][:] = 0
            return True

        window.prepare_input.side_effect = focus
        automation._session.window = window
        await automation.poll_once()
        self.assertEqual(self.clicker.points, [])
        self.answer_request.assert_not_awaited()
        automation._session.window = None
        await automation.poll_once()
        self.answer_request.assert_awaited_once()
        self.assertEqual(self.clicker.points, [(321, 654)])

    async def test_notice_detection_starts_the_shared_question_budget(self):
        self._question_notice()
        self.config.pages[0].question_budget.total_seconds = 10

        def click(x, y):
            self.clicker.points.append((x, y))
            self.scene[70][:] = 0
            return True

        self.clicker.click_point = click
        automation = self._automation()
        await automation.poll_once()
        self.assertIsNotNone(automation._notice_deadline)
        automation._notice_deadline = time.monotonic() - 1
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertEqual(self.clicker.points, [(101, 102)])
        self.assertEqual(automation.outcome, "failed")

    async def test_transient_unclicked_notice_does_not_expire_next_question(
        self,
    ):
        self._question_notice()
        self.config.pages[0].page_flow.before_question[0].stable_hits = 2
        automation = self._automation()
        await automation.poll_once()
        self.assertIsNotNone(automation._notice_deadline)
        self.scene[70][:] = 0
        self.scene[20][:] = 0
        await automation.poll_once()
        self.assertIsNone(automation._notice_deadline)
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)

    async def test_entry_without_a_question_does_not_poison_later_question(
        self,
    ):
        self._question_notice()
        automation = self._automation()
        window = Mock()
        window.prepare_input.side_effect = (
            lambda *_: self.scene[70].fill(0) or True
        )
        automation._session.window = window
        await automation.poll_once()
        automation._session.window = None
        trigger = self.scene[20].copy()
        self.scene[20][:] = 0
        automation._notice_deadline = time.monotonic() - 1
        await automation.poll_once()
        self.assertIsNone(automation._notice_deadline)
        self.scene[20][:] = trigger

        # Supply the configured completion marker after the later answer.
        def answer_click(x, y):
            self.clicker.points.append((x, y))
            self.scene[50][2:10, 2:10] = self.notice_success
            return True

        self.clicker.click_point = answer_click
        await automation.poll_once()
        self.answer_request.assert_awaited_once()
        self.assertEqual(automation.outcome, "verified")

    async def test_focus_without_autoadvance_keeps_original_notice_deadline(
        self,
    ):
        self._question_notice()
        automation = self._automation()
        window = Mock()
        window.prepare_input.return_value = True
        automation._session.window = window
        await automation.poll_once()
        original = automation._notice_deadline
        await automation.poll_once()
        self.assertEqual(automation._notice_deadline, original)
        self.assertFalse(self.clicker.points)

    async def test_dynamic_notice_clicks_body_and_rechecks_movement(self):
        self._question_notice()
        settings = self.config.pages[0].page_flow.model_dump()
        step = settings["before_question"][0]
        step["target"] = None
        step["click_match"] = True
        self.config.pages[0].page_flow = PageFlowConfig.model_validate(settings)
        automation = self._automation()
        preview = automation._session.preview

        def move_notice(*args, **kwargs):
            result = preview(*args, **kwargs)
            marker = self.scene[70][2:10, 2:10].copy()
            self.scene[70][:] = 0
            self.scene[70][20:28, 30:38] = marker
            return result

        automation._session.preview = move_notice
        await automation.poll_once()
        self.assertEqual(self.clicker.points, [])
        self.assertTrue(automation.needs_attention)

    async def test_dynamic_notice_uses_observed_center(self):
        self._question_notice()
        settings = self.config.pages[0].page_flow.model_dump()
        step = settings["before_question"][0]
        step["target"] = None
        step["click_match"] = True
        self.config.pages[0].page_flow = PageFlowConfig.model_validate(settings)

        def enter(x, y):
            self.clicker.points.append((x, y))
            self.scene[70][:] = 0
            return True

        self.clicker.click_point = enter
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.clicker.points, [(1006, 2006)])
        self.assertEqual(automation.outcome, "question_opened")
        self.assertEqual(automation.state, WorkflowState.WAITING)

    async def test_preview_notice_never_activates_or_clicks(self):
        self._question_notice()
        self.config.runtime.dry_run = True
        automation = self._automation()
        window = Mock()
        automation._session.window = window
        await automation.poll_once()
        window.prepare_input.assert_not_called()
        self.answer_request.assert_not_awaited()
        self.assertEqual(self.clicker.points, [])

    def _blocking_state(self, state):
        path, marker = self._marker(state, 205)
        self.scene[70] = np.zeros((70, 70, 3), dtype=np.uint8)
        self.scene[70][2:10, 2:10] = marker
        self.config.pages[0].page_flow = PageFlowConfig.model_validate(
            {
                state: [
                    {
                        "path": str(path),
                        "region": [0, 0, 70, 70],
                        "threshold": 0.99,
                    }
                ],
                "transition_timeout_seconds": 0.01,
            }
        )

    async def test_loading_waits_and_resumes_when_clear(self):
        self._blocking_state("loading")
        self.clicker.click_point = self._successful_click
        automation = self._automation()
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)
        self.scene[70][:] = 0
        await automation.poll_once()
        self.answer_request.assert_awaited_once()
        self.assertEqual(automation.outcome, "verified")

    async def test_loading_timeout_requests_manual_takeover(self):
        self._blocking_state("loading")
        automation = self._automation()
        await automation.poll_once()
        await asyncio.sleep(0.02)
        await automation.poll_once()
        self.assertTrue(automation.needs_attention)
        self.answer_request.assert_not_awaited()

    async def test_manual_state_holds_even_if_it_disappears(self):
        self._blocking_state("manual")
        automation = self._automation()
        await automation.poll_once()
        self.scene[70][:] = 0
        await automation.poll_once()
        self.assertTrue(automation.needs_attention)
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)

    async def test_end_of_class_stops_after_two_observations(self):
        self._blocking_state("finished")
        automation = self._automation()
        await automation.poll_once()
        self.assertNotEqual(automation.state, WorkflowState.STOPPED)
        await automation.poll_once()
        self.assertEqual(automation.state, WorkflowState.STOPPED)
        self.assertEqual(automation.outcome, "class_finished")
        self.assertEqual(automation.snapshot()["monitoring"]["pause_count"], 0)
        self.answer_request.assert_not_awaited()

    async def test_transient_models_retry_once_before_selection(self):
        failure = ConsensusAnswer(
            None,
            False,
            0,
            responses=(
                ModelAnswer("fake", "vision", error="timeout", retryable=True),
            ),
        )
        self.config.pages[0].recovery.delay_seconds = 0
        self.answer_request.side_effect = [failure, self.result]
        self.clicker.click_point = self._successful_click
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.answer_request.await_count, 2)
        self.assertEqual(self.clicker.points, [(321, 654)])
        self.assertEqual(automation.outcome, "verified")

    async def test_transient_retries_are_bounded(self):
        self.answer_request.return_value = ConsensusAnswer(
            None,
            False,
            0,
            responses=(
                ModelAnswer("fake", "vision", error="timeout", retryable=True),
            ),
        )
        self.config.pages[0].recovery.delay_seconds = 0
        automation = self._automation()
        await automation.poll_once()
        for _ in range(3):
            await automation.poll_once()
        self.assertEqual(self.answer_request.await_count, 2)
        self.assertFalse(self.clicker.points)

    async def test_invalid_model_output_does_not_retry(self):
        self.answer_request.return_value = ConsensusAnswer(
            None,
            False,
            0,
            responses=(ModelAnswer("fake", "vision", error="invalid output"),),
        )
        automation = self._automation()
        await automation.poll_once()
        self.answer_request.assert_awaited_once()
        self.assertFalse(self.clicker.points)

    async def test_changed_question_is_not_retried_and_can_rearm(self):
        async def change(*args, **kwargs):
            self.scene[30][:] = 255
            self.scene[60][:] = 255
            return ConsensusAnswer(
                None,
                False,
                0,
                responses=(
                    ModelAnswer(
                        "fake", "vision", error="timeout", retryable=True
                    ),
                ),
            )

        self.config.pages[0].recovery.delay_seconds = 0
        self.answer_request.side_effect = change
        automation = self._automation()
        await automation.poll_once()
        self.answer_request.assert_awaited_once()
        self.assertFalse(self.clicker.points)
        await automation.poll_once()
        self.assertEqual(automation.state, WorkflowState.WAITING)

    async def test_slow_context_falls_back_to_screenshot(self):
        self.config.pages[0].question_budget.context_seconds = 0.005
        self.clicker.click_point = self._successful_click
        automation = self._automation()

        async def slow_context(_):
            await asyncio.sleep(1)
            return "late text"

        automation._recognize_question = AsyncMock(side_effect=slow_context)
        await automation.poll_once()
        self.assertEqual(automation.outcome, "verified")
        self.assertEqual(
            self.answer_request.call_args.kwargs["question_text"], ""
        )
        self.assertEqual(
            self.answer_request.call_args.kwargs["knowledge_context"], ""
        )

    async def test_entire_cycle_budget_includes_question_settling(self):
        self.config.pages[0].question_budget.total_seconds = 0.02
        self.config.pages[0].question_ready.delay_seconds = 1
        automation = self._automation()
        started = time.monotonic()
        await automation.poll_once()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(automation.outcome, "failed")
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)

    async def test_model_budget_cancels_late_answer_before_input(self):
        from autoykt.agent.answer_agent import AnswerCoordinator

        self.config.pages[0].question_budget.answer_seconds = 0.02
        cancelled = asyncio.Event()
        coordinator = AnswerCoordinator(self.config.answering)
        self.addAsyncCleanup(coordinator.close)

        async def delayed(provider, model, *args, **kwargs):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.set()
            return ModelAnswer(provider.name, model, option="B")

        coordinator._query_model = AsyncMock(side_effect=delayed)
        automation = self._automation()
        automation._answerer = coordinator
        await automation.poll_once()
        self.assertTrue(cancelled.is_set())
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.outcome, "answer_rejected")

    async def test_uncertain_submission_holds_even_after_page_changes(self):
        automation = self._automation()
        await automation.poll_once()
        self.scene[30][:] = 255
        self.scene[60][:] = 255
        for _ in range(4):
            await automation.poll_once()
        self.assertEqual(self.clicker.points, [(321, 654)])
        self.assertTrue(automation.needs_attention)
        self.assertEqual(self.answer_request.await_count, 1)

    async def test_restart_restores_verified_question_without_repeating(self):
        self.clicker.click_point = self._successful_click
        first = self._automation()
        await first.poll_once()
        await first.stop()
        second = self._automation()
        self.assertEqual(second.state, WorkflowState.REARMING)
        await second.poll_once()
        self.assertEqual(second.state, WorkflowState.REARMING)
        self.assertEqual(self.answer_request.await_count, 1)
        self.assertEqual(self.clicker.points, [(321, 654)])
        self.scene[30][:] = 255
        self.scene[60][:] = 255
        self.scene[50][:] = 0
        await second.poll_once()
        self.assertEqual(second.state, WorkflowState.WAITING)

    async def test_restart_refuses_unresolved_input(self):
        first = self._automation()
        await first.poll_once()
        await first.stop()
        with self.assertRaisesRegex(
            ConfigError, "previous input needs manual review"
        ):
            self._automation()
        self.assertEqual(self.clicker.points, [(321, 654)])

    async def test_manual_recovery_skips_current_page_without_clicks(self):
        first = self._automation()
        await first.poll_once()
        await first.stop()
        count = len(self.clicker.points)
        factory = first._session._factory
        assert factory is not None
        with patch(
            "autoykt.monitor.page_tools.ScreenCapture",
            side_effect=lambda roi, **_: factory(roi),
        ):
            report = recover_current(self.config, "test")
        self.assertTrue(report.is_file())
        self.assertEqual(len(self.clicker.points), count)
        second = self._automation()
        await second.poll_once()
        self.assertEqual(second.state, WorkflowState.REARMING)
        self.assertEqual(self.answer_request.await_count, 1)
        self.scene[60][:] = 255
        await second.poll_once()
        self.assertEqual(second.state, WorkflowState.WAITING)

    async def test_dry_run_preserves_unresolved_live_checkpoint(self):
        first = self._automation()
        await first.poll_once()
        await first.stop()
        path = first._session.checkpoint.path
        original = path.read_bytes()
        self.config.runtime.dry_run = True
        preview = self._automation()
        await preview.poll_once()
        self.assertEqual(preview.outcome, "preview_only")
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.clicker.points, [(321, 654)])

    async def test_checkpoint_write_failure_prevents_mouse_input(self):
        automation = self._automation()
        with patch.object(
            automation._session.checkpoint,
            "write",
            side_effect=OSError("disk full"),
        ):
            await automation.poll_once()
        self.assertFalse(self.clicker.points)
        self.assertEqual(automation.outcome, "failed")

    async def test_restart_does_not_repeat_a_result_action(self):
        success, close, _ = self._result_flow()

        def interact(x, y):
            self.clicker.points.append((x, y))
            self.scene[50][2:10, 2:10] = success
            self.scene[70][2:10, 2:10] = close
            return True

        self.clicker.click_point = interact
        first = self._automation()
        await first.poll_once()
        await first.poll_once()
        await first.stop()
        second = self._automation()
        await second.poll_once()
        self.assertTrue(second.needs_attention)
        self.assertEqual(self.clicker.points, [(321, 654), (101, 102)])

    async def test_blocking_state_appearing_during_answer_prevents_input(self):
        self._blocking_state("manual")
        marker = self.scene[70].copy()
        self.scene[70][:] = 0

        async def respond(*args, **kwargs):
            self.scene[70][:] = marker
            return self.result

        self.answer_request.side_effect = respond
        automation = self._automation()
        await automation.poll_once()
        await automation.poll_once()
        self.assertFalse(self.clicker.points)
        self.assertTrue(automation.needs_attention)

    async def test_deadline_is_checked_after_writing_input_intent(self):
        from autoykt.monitor.operations import PlannedClick

        automation = self._automation()
        session = automation._session
        session.start("live")
        session.set_deadline(1)
        with patch(
            "autoykt.monitor.operations.time.monotonic", side_effect=[0, 2]
        ):
            clicked = session.click(
                PlannedClick("option_B", (321, 654), "screen"), self.clicker
            )
        self.assertFalse(clicked)
        self.assertFalse(self.clicker.points)

    async def test_failure_to_write_report_still_revokes_input(self):
        automation = self._automation()
        with patch.object(
            automation._session, "finish", side_effect=OSError("disk full")
        ):
            await automation._handle_failure(RuntimeError("capture failed"))
        self.assertTrue(automation._session._cancelled.is_set())
        self.assertTrue(automation.needs_attention)
        self.assertEqual(automation.state, WorkflowState.ERROR)

    async def test_timed_coordinator_can_submit_timely_consensus(self):
        from autoykt.agent.answer_agent import AnswerCoordinator

        self.config.answering.providers[0].models = ["fast", "slow"]
        self.config.pages[0].question_budget.answer_seconds = 0.05
        self.clicker.click_point = self._successful_click
        coordinator = AnswerCoordinator(self.config.answering)
        self.addAsyncCleanup(coordinator.close)
        cancelled = asyncio.Event()

        async def model(provider, name, *args, **kwargs):
            if name == "slow":
                try:
                    await asyncio.sleep(1)
                finally:
                    cancelled.set()
            return ModelAnswer(
                provider.name, name, option="B", reported_confidence=0.9
            )

        coordinator._query_model = AsyncMock(side_effect=model)
        automation = self._automation()
        automation._answerer = coordinator
        await automation.poll_once()
        self.assertTrue(cancelled.is_set())
        self.assertEqual(automation.outcome, "verified")
        self.assertEqual(self.clicker.points, [(321, 654)])

    async def test_result_preview_uses_same_target_without_any_input(self):
        from autoykt.monitor.page_tools import preview_step

        _, close, _ = self._result_flow()
        self.scene[70][2:10, 2:10] = close
        automation = self._automation()
        factory = automation._session._factory
        assert factory is not None
        with patch(
            "autoykt.monitor.page_tools.ScreenCapture",
            side_effect=lambda roi, **_: factory(roi),
        ):
            report_path = preview_step(self.config, "test", "close")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        plan = next(step for step in report["steps"] if step["step"] == "plan")
        self.assertEqual(plan["plan"]["clicks"][0]["point"], [101, 102])
        self.assertEqual(report["status"], "preview_only")
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)
        self.assertFalse(automation._session.checkpoint.path.exists())
