"""Main screen monitoring loop - polls screen, detects questions, triggers OCR."""

import asyncio
import logging
from pathlib import Path

import numpy as np

from autoykt.core.event_bus import EventBus, Event, EventType
from autoykt.core.config import AppConfig
from autoykt.monitor.screen_capture import ScreenCapture
from autoykt.monitor.detector import QuestionDetector
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
        self._capture = ScreenCapture(
            roi=config.monitor.roi,
            screenshot_dir="storage/screenshots",
            monitor_index=config.monitor.monitor_index,
        )
        self._detector = QuestionDetector(
            template_path=config.monitor.template_path,
            threshold=config.monitor.match_threshold,
            debounce_frames=config.monitor.debounce_frames,
        )

        self._running = False
        self._pending_answer: asyncio.Future[str] | None = None
        self._detect_only_paused = False  # pause after first detection until template disappears

        if detect_only:
            logger.info("ScreenWatcher running in detect-only mode (no OCR/Agent/Clicker).")
            return

        # Full mode: init clicker and subscribe to answers (OCR replaced by vision LLM in agent)
        self._clicker = Clicker(
            options_positions=config.clicker.options_positions,
            confirm_delay=config.clicker.confirm_delay,
            screen_capture=self._capture,
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
        self._capture.close()                                                   #关闭截屏
        logger.info("ScreenWatcher stopped.")                                   #发布log信息

    async def _poll_once(self) -> None:                                         #扫一次屏幕
        """Single poll iteration: grab frame → detect → (OCR → publish) or lite publish."""
        frame = self._capture.grab_frame()                                      #获取到指定区域截图
        detected, confidence, location = self._detector.detect(frame)

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

            screenshot_path = self._capture.save_screenshot(frame, prefix="question")
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

        # Save the question screenshot
        screenshot_path = self._capture.save_screenshot(frame, prefix="question")

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
                "confidence": confidence,
            },
        ))

        # Wait for answer with timeout
        try:
            answer = await asyncio.wait_for(                                    #等待answer，最多超时+5秒
                self._pending_answer,
                timeout=self._config.agent.timeout + 5,
            )
            await self._execute_answer(answer)                                  #等待点击答案完成
        except asyncio.TimeoutError:                                            #如果出了问题
            logger.error("Timed out waiting for answer.")
            self._detector.reset()
        finally:
            self._pending_answer = None                                         #重置

    async def _on_answer_ready(self, event: Event) -> None:
        """Handle ANSWER_READY event: resolve the pending future."""
        answer = event.payload.get("answer", "")                                #payload中获取answer字段
        if self._pending_answer and not self._pending_answer.done():            #当前有答案需求且没有回传答案
            self._pending_answer.set_result(answer)                             #回传答案给answer，future阻塞周期结束

    async def _execute_answer(self, answer: str) -> None:                       #点击屏幕答案
        """Click the answer option and confirm result."""
        logger.info(f"Executing answer: {answer}")

        loop = asyncio.get_running_loop()                                       #点击进程置入进程池
        success, confirm_frame = await loop.run_in_executor(
            None, self._clicker.click_and_confirm, answer
        )                                                                       #等待点击确认

        # Save confirmation screenshot
        confirm_path = None                                                     #存储确认截图
        if confirm_frame is not None:
            confirm_path = self._capture.save_screenshot(confirm_frame, prefix="confirm")

        # Publish CLICK_DONE event
        await self._bus.publish(Event(                                          #发布完成事件
            type=EventType.CLICK_DONE,
            payload={
                "answer": answer,
                "success": success,
                "confirm_screenshot_path": str(confirm_path) if confirm_path else None,
            },
        ))

        # Reset detector for next question
        self._detector.reset()                                                  #重置，等待下一个循环
        logger.info(f"Answer cycle complete. success={success}")