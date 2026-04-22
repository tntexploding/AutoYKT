"""Main screen monitoring loop - polls screen, detects questions, triggers OCR."""

import asyncio
import logging
from pathlib import Path

import cv2
import numpy as np
import pyautogui

from autoykt.core.event_bus import EventBus, Event, EventType
from autoykt.core.config import AppConfig
from autoykt.monitor.screen_capture import ScreenCapture
from autoykt.monitor.detector import QuestionDetector, OptionTemplateDetector
from autoykt.monitor.clicker import Clicker

logger = logging.getLogger("auto_answer")


class ScreenWatcher:
    """Orchestrates the screen monitoring pipeline:

    poll screen → detect question → OCR → publish event → wait for answer →
    click option → confirm → publish result
    """

    def __init__(self, config: AppConfig, event_bus: EventBus, detect_only: bool = False) -> None:
        self._config = config
        self._bus = event_bus
        self._detect_only = detect_only

        # Always needed: capture + detector
        feature_roi = config.monitor.feature_roi or config.monitor.roi
        task_roi = config.monitor.task_roi or config.monitor.question_roi or feature_roi
        entry_roi = config.monitor.entry_roi or task_roi

        self._feature_capture = ScreenCapture(
            roi=feature_roi,
            screenshot_dir="storage/screenshots",
            monitor_index=config.monitor.monitor_index,
        )
        self._task_capture = ScreenCapture(
            roi=task_roi,
            screenshot_dir="storage/screenshots",
            monitor_index=config.monitor.monitor_index,
        )
        self._entry_capture = ScreenCapture(
            roi=entry_roi,
            screenshot_dir="storage/screenshots",
            monitor_index=config.monitor.monitor_index,
        )
        finish_task_roi = config.monitor.finish_task_roi
        if finish_task_roi and len(finish_task_roi) == 4 and all(value > 0 for value in finish_task_roi):
            self._finish_capture = ScreenCapture(
                roi=finish_task_roi,
                screenshot_dir="storage/screenshots",
                monitor_index=config.monitor.monitor_index,
            )
        else:
            self._finish_capture = None
        self._detector = QuestionDetector(
            template_path=config.detector.question_feature_template_path,
            threshold=config.monitor.match_threshold,
            debounce_frames=config.monitor.debounce_frames,
        )
        self._option_detector = OptionTemplateDetector(
            template_paths=config.detector.option_templates,
            threshold=config.detector.option_match_threshold,
        )

        self._running = False
        self._pending_answer: asyncio.Future[str] | None = None
        self._detect_only_paused = False  # pause after first detection until template disappears
        self._post_answer_paused = False  # pause after a completed answer until question feature disappears
        self._post_answer_change_hits = 0
        self._post_answer_baseline: np.ndarray | None = None
        self._auto_click = config.agent.auto_click

        if detect_only or not self._auto_click:
            logger.info("ScreenWatcher running in raw-answer mode (no OCR/Agent/Clicker loop).")
            return

        # Full mode: init clicker and subscribe to answers (OCR replaced by vision LLM in agent)
        self._clicker = Clicker(
            options_positions=config.clicker.options_positions,
            confirm_delay=config.clicker.confirm_delay,
            screen_capture=self._task_capture,
        )

        # Subscribe to ANSWER_READY so we can click the answer
        self._bus.subscribe(EventType.ANSWER_READY, self._on_answer_ready)      #屏幕监控器类初始订阅answerready状态

    async def start(self) -> None:                                              #启动函数
        """Start the screen polling loop."""
        self._running = True                                                    #标志位开启
        interval = self._config.monitor.poll_interval
        logger.info(f"ScreenWatcher started. Polling every {interval}s.")       #每i秒扫一次屏幕

        while self._running:                                                    #主循环
            try:
                await self._poll_once()                                         #扫一次屏幕
            except Exception as e:
                logger.exception(f"Error in poll loop: {e}")
                await self._bus.publish(Event(
                    type=EventType.ERROR,
                    payload={"source": "screen_watcher", "error": str(e)},      #问题定义为ERROR
                ))
            await asyncio.sleep(interval)                                       #等i秒

    def stop(self) -> None:                                                     #关闭函数
        """Stop the polling loop."""
        self._running = False                                                   #关闭标志位
        self._feature_capture.close()                                           #关闭截屏
        self._task_capture.close()
        self._entry_capture.close()
        if self._finish_capture is not None:
            self._finish_capture.close()
        logger.info("ScreenWatcher stopped.")                                   #发布log信息

    async def _poll_once(self) -> None:                                         #扫一次屏幕
        """Single poll iteration: grab frame → detect → (OCR → publish) or lite publish."""
        frame = self._feature_capture.grab_frame()                              #获取到指定区域截图
        detected, confidence, location = self._detector.detect(frame)

        if self._post_answer_paused:
            if self._post_answer_baseline is None:
                self._post_answer_baseline = frame
                return

            change_ratio = self._frame_change_ratio(self._post_answer_baseline, frame)
            if change_ratio >= self._config.monitor.post_answer_resume_change_ratio:
                self._post_answer_change_hits += 1
                logger.info(
                    f"[auto-click] Feature ROI change ratio={change_ratio:.3f} "
                    f"hit {self._post_answer_change_hits}/"
                    f"{self._config.monitor.post_answer_resume_change_hits}"
                )
                if self._post_answer_change_hits >= self._config.monitor.post_answer_resume_change_hits:
                    logger.info("[auto-click] Feature ROI changed enough, resuming detection.")
                    self._post_answer_paused = False
                    self._post_answer_change_hits = 0
                    self._post_answer_baseline = None
                    self._detector.reset()
            else:
                self._post_answer_change_hits = 0
            return

        if not detected:
            if self._detect_only and self._detect_only_paused:
                logger.info("[detect-only] Question disappeared, resuming detection.")
                self._detect_only_paused = False
            return

        # Detect-only mode: publish event once, then pause until template disappears
        if self._detect_only:
            if self._detect_only_paused:
                # Still detecting the same question, skip
                return

            question_frame = self._task_capture.grab_frame()
            screenshot_path = self._task_capture.save_screenshot(question_frame, prefix="question")
            logger.info(f"[detect-only] Question detected! confidence={confidence:.3f}")
            await self._bus.publish(Event(
                type=EventType.QUESTION_DETECTED,
                payload={
                    "question": "[检测模式] 题目区域已匹配",
                    "options": {},
                    "raw_text": "",
                    "screenshot_path": str(screenshot_path),
                    "confidence": confidence,
                },
            ))
            self._detect_only_paused = True
            return

        logger.info("Question frame detected, sending to vision LLM...")

        if self._auto_click:
            self._click_entry_center()
            await asyncio.sleep(0.3)

        # Save the question screenshot
        question_frame = self._task_capture.grab_frame()
        screenshot_path = self._task_capture.save_screenshot(question_frame, prefix="question")
        detected_path = self._feature_capture.save_screenshot(frame, prefix="detected")

        # Create a future to wait for the answer
        loop = asyncio.get_running_loop()
        self._pending_answer = loop.create_future()

        # Publish QUESTION_DETECTED with screenshot path (agent will use vision LLM)
        await self._bus.publish(Event(
            type=EventType.QUESTION_DETECTED,
            payload={
                "question": "[vision mode]",
                "options": {},
                "raw_text": "",
                "screenshot_path": str(screenshot_path),
                "detected_screenshot_path": str(detected_path),
                "confidence": confidence,
            },
        ))

        if not self._auto_click:
            return

        # Wait for answer with timeout
        try:
            answer = await asyncio.wait_for(                                    #等待answer，最多超时+5秒
                self._pending_answer,
                timeout=self._config.agent.timeout + 5,
            )
            self._post_answer_paused = True
            self._post_answer_change_hits = 0
            self._post_answer_baseline = self._feature_capture.grab_frame()
            await self._execute_answer(answer)                                  #等待点击答案完成
        except asyncio.TimeoutError:                                            #如果出了问题
            logger.error("Timed out waiting for answer.")
            self._detector.reset()
        finally:
            self._pending_answer = None                                         #重置

    async def _on_answer_ready(self, event: Event) -> None:
        """Handle ANSWER_READY event: resolve the pending future."""
        answer = event.payload.get("answer_option") or event.payload.get("answer", "")
        if self._pending_answer and not self._pending_answer.done():            #当前有答案需求且没有回传答案
            self._pending_answer.set_result(answer)                             #回传答案给answer，future阻塞周期结束

    def _click_entry_center(self) -> None:
        region = self._entry_capture.monitor_region
        cx = int(region["left"] + region["width"] / 2)
        cy = int(region["top"] + region["height"] / 2)
        logger.info(f"Clicking entry center at ({cx}, {cy})")
        pyautogui.click(cx, cy)

    def _click_finish_center(self) -> bool:
        if self._finish_capture is None:
            return False
        region = self._finish_capture.monitor_region
        cx = int(region["left"] + region["width"] / 2)
        cy = int(region["top"] + region["height"] / 2)
        logger.info(f"Clicking finish-submit center at ({cx}, {cy})")
        pyautogui.click(cx, cy)
        return True

    def _annotate_option_matches(
        self,
        frame: np.ndarray,
        matches: dict[str, dict[str, float | tuple[int, int]]],
        answer: str,
    ) -> np.ndarray:
        annotated = frame.copy()
        for key, info in matches.items():
            top_left = info.get("top_left")
            size = info.get("size")
            score = info.get("score")
            if not isinstance(top_left, tuple) or not isinstance(size, tuple):
                continue
            color = (0, 255, 0) if key == answer else (255, 255, 0)
            x1, y1 = int(top_left[0]), int(top_left[1])
            w, h = int(size[0]), int(size[1])
            cv2.rectangle(annotated, (x1, y1), (x1 + w, y1 + h), color, 2)
            cv2.putText(
                annotated,
                f"{key}:{float(score):.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return annotated

    async def _execute_answer(self, answer: str) -> None:                       #点击屏幕答案
        """Click the answer option and confirm result."""
        logger.info(f"Executing answer: {answer}")

        task_frame = self._task_capture.grab_frame()
        matches = self._option_detector.detect(task_frame)
        located = matches.get(answer)

        located_path = None
        annotated_frame = self._annotate_option_matches(task_frame, matches, answer)
        located_path = self._task_capture.save_screenshot(annotated_frame, prefix="located")

        success = False
        confirm_frame = None
        finish_clicked = False
        post_submit_task_path = None
        if located and isinstance(located.get("center"), tuple):
            center = located["center"]
            region = self._task_capture.monitor_region
            abs_x = int(region["left"] + center[0])
            abs_y = int(region["top"] + center[1])
            before = self._task_capture.grab_frame()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._clicker.click_point, abs_x, abs_y)
            await asyncio.sleep(self._clicker._confirm_delay)
            confirm_frame = self._task_capture.grab_frame()
            success = self._compare_frames(before, confirm_frame)
        else:
            logger.warning(f"No template match for option '{answer}', fallback to configured click positions.")
            before = self._task_capture.grab_frame()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._clicker.click_option, answer)
            await asyncio.sleep(self._clicker._confirm_delay)
            confirm_frame = self._task_capture.grab_frame()
            success = self._compare_frames(before, confirm_frame)

        # Finish step: click "submit answer" button in finish_task ROI if configured,
        # then capture task ROI again as final state.
        if self._finish_capture is not None:
            finish_clicked = self._click_finish_center()
            await asyncio.sleep(self._config.monitor.finish_click_delay)
            final_task_frame = self._task_capture.grab_frame()
            post_submit_task_path = self._task_capture.save_screenshot(
                final_task_frame,
                prefix="task_after_submit",
            )

        # Save confirmation screenshot
        confirm_path = None                                                     #存储确认截图
        if confirm_frame is not None:
            confirm_path = self._task_capture.save_screenshot(confirm_frame, prefix="confirm")

        # Publish CLICK_DONE event
        await self._bus.publish(Event(                                          #发布完成事件
            type=EventType.CLICK_DONE,
            payload={
                "answer": answer,
                "success": success,
                "finish_clicked": finish_clicked,
                "located_option_screenshot_path": str(located_path) if located_path else None,
                "confirm_screenshot_path": str(confirm_path) if confirm_path else None,
                "post_submit_task_screenshot_path": str(post_submit_task_path) if post_submit_task_path else None,
            },
        ))

        # Reset detector for next question
        self._detector.reset()                                                  #重置，等待下一个循环
        self._post_answer_paused = True
        self._post_answer_change_hits = 0
        self._post_answer_baseline = self._feature_capture.grab_frame()
        logger.info(f"Answer cycle complete. success={success}")

    @staticmethod
    def _compare_frames(before: np.ndarray, after: np.ndarray) -> bool:
        """Compare two frames and return whether a meaningful screen change occurred."""
        diff = cv2.absdiff(before, after)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed_pixels = np.count_nonzero(gray_diff > 30)
        total_pixels = gray_diff.size
        if total_pixels == 0:
            return False
        change_ratio = changed_pixels / total_pixels
        return bool(change_ratio > 0.01)

    @staticmethod
    def _frame_change_ratio(before: np.ndarray, after: np.ndarray) -> float:
        """Compute changed-pixel ratio between two frames."""
        diff = cv2.absdiff(before, after)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed_pixels = np.count_nonzero(gray_diff > 30)
        total_pixels = gray_diff.size
        if total_pixels == 0:
            return 0.0
        return changed_pixels / total_pixels