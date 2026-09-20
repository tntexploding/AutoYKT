"""One page profile's explicit and fail-closed automation workflow."""

from __future__ import annotations

import asyncio
from collections import Counter
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np


from autoykt.agent.models import AnswerProvider, ConsensusAnswer
from autoykt.core.config import (
    AppConfig,
    PageProfileConfig,
    Region,
)
from autoykt.core.event_bus import Event, EventBus, EventType
from autoykt.core.state import (
    InvalidStateTransition,
    WorkflowState,
    WorkflowStateMachine,
)
from autoykt.knowledge.store import (
    KnowledgeStore,
    format_knowledge_context,
)
from autoykt.monitor.clicker import MouseController
from autoykt.monitor.cycle import (
    CycleProgress,
    WorkflowError,
    verify_submission,
)
from autoykt.monitor.detector import (
    ImageTemplateMatcher,
    MultiTemplateDetector,
    TemplateMatch,
)
from autoykt.monitor.image_utils import frame_change_ratio
from autoykt.monitor.health import MonitoringHealth
from autoykt.monitor.selection import (
    apply_checked_selection,
    selected_options,
    selection_change_ratio,
)
from autoykt.monitor.ocr_engine import (
    BaseOcrEngine,
    create_ocr_engine,
    format_question_text,
)
from autoykt.monitor.screen_capture import ScreenCapture
from autoykt.monitor.operations import (
    ActionPlan,
    CaptureDevice,
    ProfileSession,
    CaptureFactory,
    open_capture_devices,
    close_capture_devices,
)
from autoykt.monitor.windows import WindowUnavailable
from autoykt.monitor.page_flow import PageBlocked, PageFlow


# Keep the state transitions together for review of the complete workflow.
# pylint: disable=too-many-lines

logger = logging.getLogger("autoykt")


class PageAutomation:
    """Run a single page profile from trigger through verified submission."""

    def __init__(
        self,
        app_config: AppConfig,
        profile: PageProfileConfig,
        event_bus: EventBus,
        answerer: AnswerProvider,
        clicker: MouseController,
        knowledge_store: KnowledgeStore | None,
        *,
        detect_only: bool = False,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self._app_config = app_config
        self._profile = profile
        self._bus = event_bus
        self._answerer = answerer
        self._clicker = clicker
        self._knowledge_store = knowledge_store
        self._detect_only = detect_only
        self._interactions_enabled = (
            not detect_only
            and not app_config.runtime.dry_run
            and app_config.answering.auto_apply
        )
        self._session = ProfileSession(
            app_config, profile, load_options=not detect_only
        )
        if self._interactions_enabled:
            self._session.check_live_configuration()
        self._progress = CycleProgress(baseline=self._session.restore())
        screenshot_dir = app_config.resolve_path(
            app_config.storage.screenshot_dir
        )

        if capture_factory is None:

            def capture(region: Region) -> CaptureDevice:
                return ScreenCapture(
                    roi=region,
                    screenshot_dir=screenshot_dir,
                    monitor_index=profile.monitor_index,
                    window_guard=self._session.window,
                )

            capture_factory = capture

        self._session.attach_capture_factory(capture_factory)
        self._entry_flow = PageFlow(
            app_config,
            profile.page_flow,
            self._session.capture,
            enabled=not detect_only,
            before_question=True,
        )
        self._detector = MultiTemplateDetector(
            [
                {
                    "name": trigger.name,
                    "path": str(app_config.resolve_path(trigger.path)),
                    "threshold": trigger.threshold,
                    "consecutive_hits": trigger.consecutive_hits,
                    "action": trigger.action,
                    "question_type": trigger.question_type,
                }
                for trigger in profile.triggers
            ]
        )
        self._success_matcher = ImageTemplateMatcher(
            [
                (
                    f"success_{index + 1}",
                    str(app_config.resolve_path(template.path)),
                    template.threshold,
                )
                for index, template in enumerate(
                    profile.verification.success_templates
                )
                if not detect_only
            ]
        )
        self._failure_matcher = ImageTemplateMatcher(
            [
                (
                    f"failure_{index + 1}",
                    str(app_config.resolve_path(template.path)),
                    template.threshold,
                )
                for index, template in enumerate(
                    profile.verification.failure_templates
                )
                if not detect_only
            ]
        )
        captures = open_capture_devices(
            capture_factory,
            (
                profile.regions.detection,
                profile.regions.question,
                profile.regions.answer_region,
                profile.regions.verification_region,
                profile.regions.rearm_region,
            ),
        )
        (
            self._detection_capture,
            self._question_capture,
            self._answer_capture,
            self._verification_capture,
            self._rearm_capture,
        ) = captures
        self._state = WorkflowStateMachine(
            profile.id,
            WorkflowState.REARMING
            if self._progress.baseline is not None
            else WorkflowState.WAITING,
        )
        self._progress.rearm_deadline = (
            time.monotonic() + profile.rearm.timeout_seconds
        )
        self._ocr: BaseOcrEngine | None = None
        self._ocr_unavailable = False
        self._outcome_counts: Counter[str] = Counter()
        self._notice_deadline: float | None = None
        self._health = MonitoringHealth()

    async def prepare(self) -> None:
        """Load local OCR before monitoring, outside every question deadline."""
        if (
            self._detect_only
            or not self._app_config.knowledge.question_ocr_enabled
            or self._ocr is not None
            or self._ocr_unavailable
        ):
            return
        logger.info(
            "Loading question OCR before monitoring '%s'", self.profile_id
        )
        try:
            self._ocr = await asyncio.to_thread(create_ocr_engine)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Optional OCR has several independently installed dependencies.
            self._ocr_unavailable = True
            if not any(
                provider.input_mode == "vision"
                for provider in self._app_config.answering.providers
            ):
                raise WorkflowError(
                    "local OCR could not be initialized"
                ) from error
            logger.warning("Question OCR unavailable; using vision models")
            return
        logger.info("Question OCR ready for '%s'", self.profile_id)

    @property
    def profile_id(self) -> str:
        """Return the page profile identifier."""
        return self._profile.id

    @property
    def state(self) -> WorkflowState:
        """Return the current automation state."""
        return self._state.state

    @property
    def completed_cycles(self) -> int:
        """Return terminal attempts for the scheduler's single-question mode."""
        return self._session.completed_cycles

    @property
    def outcome(self) -> str:
        """Return the latest attempt outcome for single-question exit codes."""
        return self._session.outcome

    @property
    def report_path(self) -> Path | None:
        """Return the latest private evidence report."""
        return self._session.report_path

    @property
    def needs_attention(self) -> bool:
        """Whether automation is held for a supervised manual takeover."""
        return self._progress.manual_required

    def snapshot(self) -> dict[str, Any]:
        """Return bounded status data without retaining question images."""
        return {
            "profile_id": self.profile_id,
            "state": self.state.value,
            "completed_cycles": self.completed_cycles,
            "outcome_counts": dict(self._outcome_counts),
            "last_outcome": self.outcome,
            "needs_attention": self.needs_attention,
            "pause_reason": self._session.pause_reason,
            "report": str(self.report_path) if self.report_path else None,
            "monitoring": self._health.snapshot(),
        }

    def _complete_cycle(self) -> None:
        self._session.completed_cycles += 1
        self._outcome_counts[self.outcome] += 1

    async def poll_once(self) -> None:
        """Pause until the calibrated window and page are available again."""
        if self.state == WorkflowState.STOPPED or self.needs_attention:
            return
        try:
            self._session.check_page()
            self._health.observed()
            if self.state in {WorkflowState.WAITING, WorkflowState.REARMING}:
                if await self._open_question_notice():
                    self._session.pause_reason = ""
                    self._health.resume()
                    return
            await self._poll_ready_page()
        except PageBlocked as error:
            self._session.pause_reason = f"page state: {error.state}"
            if error.state != "finished":
                self._health.pause(self._session.pause_reason)
            else:
                self._health.resume()
            await self._handle_page_state(error)
            return
        except WindowUnavailable as error:
            self._health.pause(str(error))
            self._detector.reset()
            if self._session.pause_reason != str(error):
                self._session.pause_reason = str(error)
                logger.warning(
                    "Profile '%s' paused: %s", self.profile_id, error
                )
                await self._report_error("page_guard", str(error))
            return
        self._session.pause_reason = ""
        self._health.resume()
        self._progress.blocked_since = 0.0
        self._progress.finished_hits = 0

    async def _open_question_notice(self) -> bool:
        """Enter a visible new-question notice without relying on page focus."""
        step, visible = self._entry_flow.next_step()
        if visible and self._progress.notice_entered:
            self._notice_deadline = None
            self._progress.notice_entered = False
        if not visible and not self._progress.notice_entered:
            # An unclicked notice can disappear before debounce finishes.
            # It must not leave a deadline that expires the next question.
            self._notice_deadline = None
        if visible and self._notice_deadline is None:
            self._notice_deadline = (
                time.monotonic() + self._profile.question_budget.total_seconds
            )
        if step is None or not self._interactions_enabled:
            return visible
        if self._session.window and self._session.window.prepare_input(
            self._clicker.click_point
        ):
            # Focus can open the latest slide on its own. Reobserve first.
            self._progress.notice_entered = not self._entry_flow.matches(
                step.when
            )
            self._detector.reset()
            return True
        previous = self._session.checkpoint.read()
        if previous and previous["status"] == "input_pending":
            await self._hold_for_review(
                "unconfirmed input precedes new question"
            )
            return True
        # A recognized new-question notice can leave a verified or reviewed
        # prior page. Never transfer ownership of an unconfirmed input.
        self._session.release_question()
        self._session.start("question_entry")
        self._session.record("entry_from", previous_checkpoint=previous)
        deadline = min(
            time.monotonic()
            + self._profile.page_flow.transition_timeout_seconds,
            self._notice_deadline or float("inf"),
        )
        self._session.set_deadline(deadline)
        try:
            self._session.save_frame(
                "notice_before", self._session.capture(step.when.region)
            )
            action = self._session.step_target("entry_" + step.name, step)
            self._session.preview(
                ActionPlan("", (action,)),
                self._rearm_capture,
                label="entry_" + step.name + "_preview",
            )
            self._entry_flow.attempted(step)
            applied = await asyncio.to_thread(
                self._session.click,
                action,
                self._clicker,
                required_state=step.when,
                required_step=step if step.click_match else None,
            )
            if not applied:
                raise WorkflowError("new-question entry was not applied")
            while self._entry_flow.matches(step.when):
                if time.monotonic() >= deadline:
                    raise WorkflowError("new-question notice did not disappear")
                await asyncio.sleep(0.1)
            self._session.finish("question_opened")
            self._progress.notice_entered = True
            self._session.release_question()
            self._session.set_deadline(float("inf"))
            self._entry_flow.reset()
            self._progress.baseline = None
            self._detector.reset()
            if self.state == WorkflowState.REARMING:
                await self._transition(WorkflowState.WAITING)
        except Exception as error:  # pylint: disable=broad-exception-caught
            await self._hold_for_review(str(error))
        return True

    async def _handle_page_state(self, error: PageBlocked) -> None:
        self._detector.reset()
        if error.state == "finished":
            self._progress.finished_hits += 1
            if self._progress.finished_hits >= 2:
                if self._session.flow.verified:
                    self._session.release_question()
                self._session.finish("class_finished")
                await self._transition(WorkflowState.STOPPED)
                logger.info("Profile '%s': class finished", self.profile_id)
            return
        self._progress.finished_hits = 0
        if error.state == "manual":
            await self._hold_for_review("page requires manual intervention")
            return
        if not self._progress.blocked_since:
            self._progress.blocked_since = time.monotonic()
        timeout = self._profile.page_flow.transition_timeout_seconds
        if time.monotonic() - self._progress.blocked_since >= timeout:
            await self._hold_for_review("page loading timed out")

    async def _report_error(self, source: str, message: str) -> None:
        await self._bus.publish(
            Event(
                type=EventType.ERROR,
                profile_id=self.profile_id,
                payload={
                    "source": source,
                    "error": message,
                    "report_path": str(self.report_path or ""),
                },
            )
        )

    async def _hold_for_review(self, message: str) -> None:
        self._session.cancel()
        self._progress.manual_required = True
        try:
            if self.report_path is None:
                self._session.start("page_hold")
                self._session.cancel()
            self._session.finish("manual_required", error=message)
            self._session.save_frame(
                "manual_review", self._rearm_capture.grab_frame()
            )
        except (OSError, RuntimeError) as evidence_error:
            logger.error(
                "Could not capture manual-review evidence: %s", evidence_error
            )
        if self.state != WorkflowState.ERROR:
            await self._transition(WorkflowState.ERROR)
        logger.error(
            "Profile '%s' needs manual review: %s", self.profile_id, message
        )
        await self._report_error("manual_review", message)

    async def _poll_ready_page(self) -> None:
        """Poll or advance this page profile exactly once."""
        if self._state.state == WorkflowState.ERROR:
            await self._begin_rearm()
            return
        if self._state.state == WorkflowState.REARMING:
            try:
                await self._poll_rearm()
            except Exception as error:  # pylint: disable=broad-exception-caught
                await self._hold_for_review(str(error))
            return
        if self._state.state != WorkflowState.WAITING:
            return
        frame = self._detection_capture.grab_frame()
        observation = self._detector.observe(frame)
        trigger = observation.triggered
        self._health.detected(trigger.name if trigger else None)
        if observation.ambiguous:
            raise WindowUnavailable(
                "conflicting single/multiple question templates"
            )
        await self._expire_pending_entry(bool(observation.present))
        if trigger is None:
            return
        self._progress.trigger_name = trigger.name
        if trigger.action == "notify" or self._detect_only:
            try:
                self._session.start("capture_only")
                await self._capture_without_answer(trigger)
                self._session.finish("captured")
            except Exception as error:  # pylint: disable=broad-exception-caught
                # Capture failures must also leave this trigger fail-closed.
                await self._handle_failure(error)
            finally:
                self._complete_cycle()
            return
        if self._interactions_enabled and self._session.window:
            if self._session.window.prepare_input(self._clicker.click_point):
                # Focusing may advance a classroom. Discard the old detection.
                self._detector.reset()
                return
        await self._bus.publish(
            Event(
                type=EventType.TEMPLATE_DETECTED,
                profile_id=self.profile_id,
                payload={
                    "template_name": trigger.name,
                    "action": trigger.action,
                    "question_type": trigger.question_type,
                    "confidence": trigger.confidence,
                },
            )
        )
        await self._run_answer_cycle(trigger)

    async def _expire_pending_entry(self, question_visible: bool) -> None:
        """Retire an entry that never produced a question within its budget."""
        if (
            not question_visible
            and self._progress.notice_entered
            and self._notice_deadline is not None
            and time.monotonic() >= self._notice_deadline
        ):
            self._notice_deadline = None
            self._progress.notice_entered = False
            await self._report_error(
                "question_entry",
                "no question appeared before the entry deadline",
            )

    async def _capture_without_answer(self, trigger: TemplateMatch) -> None:
        question_frame = self._question_capture.grab_frame()
        self._remember_question(question_frame)
        screenshot = self._question_capture.save_screenshot(
            question_frame,
            prefix=f"{self.profile_id}_{trigger.name}",
        )
        event_type = (
            EventType.TEMPLATE_DETECTED
            if trigger.action == "notify"
            else EventType.QUESTION_CAPTURED
        )
        await self._bus.publish(
            Event(
                type=event_type,
                profile_id=self.profile_id,
                payload={
                    "template_name": trigger.name,
                    "action": trigger.action,
                    "question_type": trigger.question_type,
                    "confidence": trigger.confidence,
                    "screenshot_path": str(screenshot),
                },
            )
        )
        self._session.record("question", screenshot_path=str(screenshot))
        await self._begin_rearm()

    async def _run_answer_cycle(self, trigger: TemplateMatch) -> None:
        try:
            self._session.start(
                "live" if self._interactions_enabled else "preview"
            )
            budget = self._profile.question_budget.total_seconds
            self._progress.deadline = min(
                time.monotonic() + budget,
                self._notice_deadline
                if self._notice_deadline is not None
                else float("inf"),
            )
            self._notice_deadline = None
            self._progress.notice_entered = False
            self._session.set_deadline(self._progress.deadline)
            await asyncio.wait_for(
                self._execute_answer_cycle(trigger),
                timeout=max(0, self._progress.deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            self._session.cancel()
            await self._handle_failure(
                WorkflowError("question time budget exhausted")
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            await self._handle_failure(error)
        except asyncio.CancelledError:
            self._session.cancel()
            self._session.finish("interrupted")
            raise
        finally:
            self._complete_cycle()
            logger.info("Operation report: %s", self.report_path)

    # The completion checks intentionally stay beside the preparation steps.
    # pylint: disable-next=too-many-statements
    async def _execute_answer_cycle(self, trigger: TemplateMatch) -> None:
        self._progress.multiple = trigger.question_type == "multiple"
        await self._transition(WorkflowState.PREPARING)
        if await self._skip_completed_question():
            return
        if not await self._prepare_entry():
            await self._begin_rearm()
            return
        if self._profile.question_ready.delay_seconds:
            await asyncio.sleep(self._profile.question_ready.delay_seconds)
        await self._transition(WorkflowState.CAPTURING)
        question_frame = await self._capture_stable_question()
        self._remember_question(question_frame)
        if await self._skip_completed_question():
            return
        self._progress.allowed_options = self._profile.answer_style.option_keys
        if self._profile.answer_style.button_colors is not None:
            visible = self._session.inspect_options(
                self._answer_capture.grab_frame()
            )
            selected_options(visible)
            self._progress.allowed_options = frozenset(visible)
        elif self._progress.multiple:
            raise WorkflowError(
                "multiple choice requires observed selection states"
            )
        screenshot = self._question_capture.save_screenshot(
            question_frame,
            prefix=f"{self.profile_id}_question",
        )
        self._session.record("question", screenshot_path=str(screenshot))
        question_text, knowledge_context = await self._bounded_context(
            question_frame
        )
        await self._bus.publish(
            Event(
                type=EventType.QUESTION_CAPTURED,
                profile_id=self.profile_id,
                payload={
                    "template_name": trigger.name,
                    "confidence": trigger.confidence,
                    "question": question_text,
                    "screenshot_path": str(screenshot),
                },
            )
        )
        await self._transition(WorkflowState.ANSWERING)
        consensus = await self._answer_with_retries(
            screenshot,
            question_frame,
            question_text,
            knowledge_context,
        )
        if consensus.actionable:
            consensus = self._reject_if_question_changed(
                consensus, question_frame
            )
        self._session.record(
            "answer",
            option=consensus.option,
            actionable=consensus.actionable,
            reason=consensus.reason,
            consensus=consensus.event_payload(),
            question_text=question_text,
        )
        await self._publish_answer(consensus)
        if not consensus.actionable or consensus.option is None:
            self._session.finish("answer_rejected", reason=consensus.reason)
            await self._begin_rearm()
            return
        self._session.check_page()
        answer_frame = self._answer_capture.grab_frame()
        if self._profile.answer_style.button_colors is not None:
            plan = self._session.plan(
                consensus.option,
                None,
                self._answer_capture,
                frame=answer_frame,
                multiple=self._progress.multiple,
            )
        else:
            point, annotated = self._locate_answer(
                consensus.option, answer_frame
            )
            self._session.save_frame("located_option", annotated)
            plan = self._session.plan(
                consensus.option, point, self._answer_capture
            )
        self._ensure_question_current(question_frame)
        if not self._interactions_enabled:
            self._session.finish("preview_only")
            await self._begin_rearm()
            return
        if time.monotonic() + self._input_reserve() >= self._progress.deadline:
            raise WorkflowError(
                "not enough time remains to select and verify submission"
            )
        await self._transition(WorkflowState.APPLYING)
        await self._apply_and_verify(plan, question_frame, answer_frame)
        self._session.finish("verified")
        self._session.mark_verified()
        await self._begin_rearm()

    async def _skip_completed_question(self) -> bool:
        """Skip questions that are already completed or unavailable."""
        frame = self._verification_capture.grab_frame()
        success = self._success_matcher.best(frame)
        failure = self._failure_matcher.best(frame)
        if success is not None and failure is not None:
            raise WorkflowError("conflicting success and failure states")
        match = success or failure
        if match is None:
            return False
        outcome = "already_completed" if success else "unavailable"
        evidence_kind = "success" if success else "failure"
        self._session.save_frame(outcome, frame)
        self._session.record(
            outcome,
            evidence=(
                f"{evidence_kind}_template:{match.name}:"
                f"{match.confidence:.3f}"
            ),
        )
        self._session.finish(outcome)
        await self._begin_rearm()
        return True

    def _input_reserve(self) -> float:
        reserve = self._profile.verification.timeout_seconds + 1.0
        style = self._profile.answer_style
        if style.button_colors is not None:
            count = (
                len(self._progress.allowed_options)
                if self._progress.multiple
                else 1
            )
            reserve += count * style.selection_timeout_seconds
        return reserve

    async def _bounded_context(self, frame: np.ndarray) -> tuple[str, str]:
        async def collect() -> tuple[str, str]:
            question_text = await self._recognize_question(frame)
            return question_text, await self._knowledge_context(question_text)

        available = (
            self._progress.deadline - time.monotonic() - self._input_reserve()
        )
        if available <= 0:
            raise WorkflowError("question budget exhausted before answering")
        try:
            return await asyncio.wait_for(
                collect(),
                min(self._profile.question_budget.context_seconds, available),
            )
        except asyncio.TimeoutError:
            if not any(
                provider.input_mode == "vision"
                for provider in self._app_config.answering.providers
            ):
                self._session.record("context_timeout", fallback="unavailable")
                raise WorkflowError(
                    "question OCR timed out; text-only answering is unavailable"
                ) from None
            self._session.record("context_timeout", fallback="vision_only")
            logger.warning(
                "Profile '%s': context timed out; using the screenshot",
                self.profile_id,
            )
            return "", ""

    async def _answer_with_retries(
        self,
        screenshot: Path,
        frame: np.ndarray,
        question_text: str,
        context: str,
    ) -> ConsensusAnswer:
        deadline = min(
            self._progress.deadline - self._input_reserve() - 1.0,
            time.monotonic() + self._profile.question_budget.answer_seconds,
        )
        settings = self._profile.recovery
        for attempt in range(settings.maximum_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkflowError("model answer budget exhausted")
            self._session.check_page()
            self._ensure_question_current(frame)
            # The coordinator owns the cutoff and retains timely votes.
            # The outer cycle still enforces the total question budget.
            result = await self._answerer.answer(
                screenshot,
                self._progress.allowed_options,
                question_text=question_text,
                knowledge_context=context,
                timeout_seconds=remaining,
                multiple=self._progress.multiple,
            )
            transient = bool(result.responses) and all(
                response.error is not None and response.retryable
                for response in result.responses
            )
            if (
                not transient
                or attempt + 1 >= settings.maximum_attempts
                or deadline - time.monotonic() <= settings.delay_seconds
            ):
                return result
            self._session.record("answer_retry", attempt=attempt + 2)
            await asyncio.sleep(settings.delay_seconds)
        raise AssertionError("retry loop must return an answer")

    async def _prepare_entry(self) -> bool:
        target = self._profile.entry_target
        if target is None:
            return True
        entry = self._session.target("entry", target, self._detection_capture)
        self._session.preview(ActionPlan("", (entry,)), self._detection_capture)
        if not self._interactions_enabled:
            self._session.finish("entry_preview")
            return False
        if not await asyncio.to_thread(
            self._session.click, entry, self._clicker
        ):
            raise WorkflowError("clicking entry failed")
        return True

    def _remember_question(self, question_frame: np.ndarray) -> None:
        regions = self._profile.regions
        baseline = (
            question_frame
            if regions.rearm_region == regions.question
            else self._rearm_capture.grab_frame()
        )
        # Remember the answered question, even if a new question appears
        # during a model request or immediately after submission.
        self._progress.baseline = baseline.copy()
        self._progress.rearm_options = None
        self._session.save_frame("rearm_before", baseline)

    async def _capture_stable_question(self) -> np.ndarray:
        settings = self._profile.question_ready
        previous = self._question_capture.grab_frame()
        stable_frames = 1
        deadline = time.monotonic() + settings.timeout_seconds
        while stable_frames < settings.stable_frames:
            if time.monotonic() >= deadline:
                raise WorkflowError("question region did not become stable")
            await asyncio.sleep(0.1)
            current = self._question_capture.grab_frame()
            ratio = frame_change_ratio(previous, current)
            stable_frames = (
                stable_frames + 1
                if ratio <= settings.maximum_change_ratio
                else 1
            )
            previous = current
        return previous

    async def _recognize_question(self, frame: np.ndarray) -> str:
        settings = self._app_config.knowledge
        if not settings.question_ocr_enabled or self._ocr_unavailable:
            return ""
        if self._ocr is None:
            try:
                self._ocr = await asyncio.to_thread(create_ocr_engine)
            except Exception as error:  # pylint: disable=broad-exception-caught
                # Optional OCR may be unavailable for many backend reasons.
                logger.warning("Question OCR is unavailable: %s", error)
                self._ocr_unavailable = True
                return ""
        try:
            result = await asyncio.to_thread(self._ocr.recognize, frame)
            return format_question_text(
                result, self._session.inspect_options(frame)
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            # OCR failure must not block vision models.
            logger.warning("Question OCR failed: %s", error)
            return ""

    async def _knowledge_context(self, question_text: str) -> str:
        settings = self._app_config.knowledge
        if (
            self._knowledge_store is None
            or not settings.enabled
            or not question_text
        ):
            return ""
        matches = await asyncio.to_thread(
            self._knowledge_store.search,
            self._profile.course_id,
            question_text,
            settings.maximum_results,
            settings.minimum_score,
        )
        return format_knowledge_context(
            matches,
            settings.maximum_context_characters,
        )

    async def _publish_answer(self, answer: ConsensusAnswer) -> None:
        event_type = (
            EventType.ANSWER_READY
            if answer.actionable
            else EventType.ANSWER_REJECTED
        )
        await self._bus.publish(
            Event(
                type=event_type,
                profile_id=self.profile_id,
                payload=answer.event_payload(),
            )
        )

    def _reject_if_question_changed(
        self,
        answer: ConsensusAnswer,
        original_frame: np.ndarray,
    ) -> ConsensusAnswer:
        current_frame = self._question_capture.grab_frame()
        change_ratio = frame_change_ratio(original_frame, current_frame)
        maximum = self._profile.question_ready.maximum_change_ratio
        if change_ratio <= maximum:
            return answer
        return ConsensusAnswer(
            option=None,
            actionable=False,
            agreement_ratio=answer.agreement_ratio,
            votes=dict(answer.votes),
            responses=answer.responses,
            reason=(
                "question changed while models were answering "
                f"({change_ratio:.4f} > {maximum:.4f})"
            ),
        )

    async def _apply_and_verify(
        self,
        plan: ActionPlan,
        question_frame: np.ndarray,
        answer_frame: np.ndarray,
    ) -> None:
        option = plan.option
        before_interaction = self._verification_capture.grab_frame()
        if self._success_matcher.best(before_interaction) is not None:
            raise WorkflowError("success state was present before interaction")
        if self._failure_matcher.best(before_interaction) is not None:
            raise WorkflowError("failure state was present before interaction")
        self._session.save_frame("before_selection", before_interaction)
        self._ensure_question_current(question_frame)
        current_answers = self._answer_capture.grab_frame()
        if (
            frame_change_ratio(answer_frame, current_answers)
            > self._profile.question_ready.maximum_change_ratio
        ):
            raise WorkflowError("answer layout changed after planning")
        reference = None
        if self._profile.answer_style.button_colors is not None:
            reference = await apply_checked_selection(
                self._session,
                self._answer_capture,
                self._clicker,
                plan,
                question_frame,
                answer_frame,
                multiple=self._progress.multiple,
            )
        else:
            if not await asyncio.to_thread(
                self._session.click,
                plan.clicks[0],
                self._clicker,
                question_frame=question_frame,
                answer_frame=answer_frame,
            ):
                raise WorkflowError(f"clicking option {option} failed")
            await asyncio.sleep(0.3)
        self._session.save_frame(
            "selection_result", self._answer_capture.grab_frame()
        )

        verification_baseline = before_interaction
        if self._profile.submit_target is not None:
            after_selection = self._verification_capture.grab_frame()
            self._session.save_frame("after_selection", after_selection)
            failure = self._failure_matcher.best(after_selection)
            if failure is not None:
                self._session.record(
                    "verification",
                    success=False,
                    evidence=f"failure_template:{failure.name}",
                )
                raise WorkflowError("page reported failure after selection")
            if self._success_matcher.best(after_selection) is None:
                verification_baseline = after_selection
                await self._transition(WorkflowState.SUBMITTING)
                if reference is None:
                    self._ensure_question_current(question_frame)
                if not await asyncio.to_thread(
                    self._session.click,
                    plan.clicks[-1],
                    self._clicker,
                    question_frame=question_frame,
                    selection_reference=reference,
                ):
                    raise WorkflowError("clicking submit failed")
        await self._transition(WorkflowState.VERIFYING)
        verified, final_frame, evidence = await verify_submission(
            self._session,
            self._verification_capture,
            self._success_matcher,
            self._failure_matcher,
            verification_baseline,
        )
        final_path = self._verification_capture.save_screenshot(
            final_frame,
            prefix=f"{self.profile_id}_verified",
        )
        await self._bus.publish(
            Event(
                type=EventType.INTERACTION_COMPLETED,
                profile_id=self.profile_id,
                payload={
                    "answer": option,
                    "verified": verified,
                    "verification_evidence": evidence,
                    "located_screenshot_path": str(
                        self.report_path.parent / "plan_preview.png"
                    )
                    if self.report_path
                    else "",
                    "result_screenshot_path": str(final_path),
                },
            )
        )
        await self._bus.publish(
            Event(
                type=EventType.SUBMISSION_VERIFIED,
                profile_id=self.profile_id,
                payload={
                    "answer": option,
                    "success": verified,
                    "evidence": evidence,
                    "screenshot_path": str(final_path),
                },
            )
        )
        self._session.record(
            "verification",
            success=verified,
            evidence=evidence,
            screenshot_path=str(final_path),
        )
        if not verified:
            raise WorkflowError("submission could not be verified")

    def _ensure_question_current(self, original_frame: np.ndarray) -> None:
        current = self._question_capture.grab_frame()
        maximum = self._profile.question_ready.maximum_change_ratio
        if frame_change_ratio(original_frame, current) > maximum:
            raise WorkflowError("question changed before mouse interaction")

    def _locate_answer(
        self,
        option: str,
        frame: np.ndarray,
    ) -> tuple[tuple[int, int], np.ndarray]:
        return self._session.locate_answer(option, frame, self._answer_capture)

    async def _begin_rearm(self) -> None:
        baseline = self._progress.baseline
        if baseline is None:
            baseline = self._rearm_capture.grab_frame()
        if self._state.state != WorkflowState.REARMING:
            await self._transition(WorkflowState.REARMING)
        self._progress.baseline = baseline.copy()
        self._progress.hits = 0
        self._progress.rearm_deadline = (
            time.monotonic() + self._profile.rearm.timeout_seconds
        )
        self._progress.timeout_reported = False

    async def _poll_rearm(self) -> None:
        if self._session.flow.verified and await self._advance_result():
            return
        frame = self._rearm_capture.grab_frame()
        observation = self._detector.observe(
            self._detection_capture.grab_frame()
        )
        self._health.detected(None)
        changed = self._rearm_content_changed(frame)
        trigger_present = any(
            self._progress.trigger_name is None
            or match.name == self._progress.trigger_name
            for match in observation.present
        )
        ready = changed or not trigger_present
        if (
            self._profile.verification.success_templates
            or self._profile.verification.failure_templates
        ):
            # Feedback on this question is not a new question.
            feedback = self._verification_capture.grab_frame()
            ready = (
                ready
                and self._success_matcher.best(feedback) is None
                and self._failure_matcher.best(feedback) is None
            )
        self._progress.hits = self._progress.hits + 1 if ready else 0
        if self._progress.hits >= self._profile.rearm.stable_hits:
            self._detector.reset()
            self._progress.baseline = None
            self._session.release_question()
            await self._transition(WorkflowState.WAITING)
            return
        if (
            time.monotonic() >= self._progress.rearm_deadline
            and not self._progress.timeout_reported
        ):
            self._progress.timeout_reported = True
            logger.info(
                "Profile '%s': waiting for the next question without "
                "repeating input",
                self.profile_id,
            )

    def _rearm_content_changed(self, frame: np.ndarray) -> bool:
        """Ignore selection colors when detecting a new question."""
        baseline = self._progress.baseline
        if baseline is None:
            return True
        if self._profile.answer_style.button_colors is None:
            ratio = frame_change_ratio(baseline, frame)
        else:
            if self._progress.rearm_options is None:
                self._progress.rearm_options = self._session.inspect_options(
                    baseline
                )
            region = self._profile.regions.rearm_region
            ratio = selection_change_ratio(
                baseline, frame, region, region, self._progress.rearm_options
            )
        return ratio >= self._profile.rearm.minimum_change_ratio

    async def _advance_result(self) -> bool:
        step, visible = self._session.flow.next_step()
        if step is None:
            return visible
        action = self._session.step_target("advance_" + step.name, step)
        self._session.preview(
            ActionPlan("", (action,)),
            self._rearm_capture,
            label="advance_" + step.name + "_preview",
        )
        if not self._interactions_enabled:
            return True
        self._session.flow.attempted(step)
        applied = await asyncio.to_thread(
            self._session.click,
            action,
            self._clicker,
            required_state=step.when,
            required_step=step if step.click_match else None,
        )
        if not applied:
            raise WorkflowError("result navigation could not be confirmed")
        self._session.mark_verified()
        self._session.save_frame(
            "advance_" + step.name + "_result", self._rearm_capture.grab_frame()
        )
        return True

    async def _handle_failure(
        self,
        error: Exception,
    ) -> None:
        self._session.cancel()
        try:
            self._session.finish("failed", error=str(error))
        except OSError as log_error:
            self._progress.manual_required = True
            logger.error("Could not persist failure evidence: %s", log_error)
        logger.exception("Profile '%s' failed: %s", self.profile_id, error)
        if self._state.state not in {
            WorkflowState.ERROR,
            WorkflowState.STOPPED,
        }:
            try:
                await self._transition(WorkflowState.ERROR)
            except InvalidStateTransition:
                pass
        await self._bus.publish(
            Event(
                type=EventType.ERROR,
                profile_id=self.profile_id,
                payload={"source": "workflow", "error": str(error)},
            )
        )
        if (
            self._session.answer_input_attempted
            or self._progress.manual_required
        ):
            self._progress.manual_required = True
            self._session.cancel()
            await self._report_error(
                "manual_review",
                "operation could not be safely completed; "
                "inspect the page before restarting",
            )
            return
        try:
            # Preserve the original baseline so a new question is not swallowed.
            await self._begin_rearm()
        except Exception as rearm_error:  # pylint: disable=broad-exception-caught
            # Stay ERROR because retrying this question may click twice.
            logger.error(
                "Profile '%s' could not enter rearm safely: %s",
                self.profile_id,
                rearm_error,
            )

    async def _transition(self, new_state: WorkflowState) -> None:
        previous = self._state.transition(new_state)
        if previous == new_state:
            return
        await self._bus.publish(
            Event(
                type=EventType.STATE_CHANGED,
                profile_id=self.profile_id,
                payload={
                    "previous": previous.value,
                    "current": new_state.value,
                },
            )
        )

    async def stop(self) -> None:
        """Mark this profile stopped and release screen handles."""
        self._session.cancel()
        self._health.resume()
        if self._state.state != WorkflowState.STOPPED:
            await self._transition(WorkflowState.STOPPED)
        self.close_capture_devices()

    def close_capture_devices(self) -> None:
        """Release capture handles without changing workflow state."""
        close_capture_devices(
            (
                self._detection_capture,
                self._question_capture,
                self._answer_capture,
                self._verification_capture,
                self._rearm_capture,
            )
        )
