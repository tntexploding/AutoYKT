"""Abstract base class for notification backends."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from autoykt.core.event_bus import EventBus, Event, EventType

logger = logging.getLogger("auto_answer")


class BaseNotifier(ABC):
    """Base class for all notification backends (Telegram, QQ, etc).

    Subscribes to three events and pushes messages accordingly:
    1. QUESTION_DETECTED → question text + screenshot
    2. ANSWER_READY      → answer + source + confidence
    3. CLICK_DONE        → result screenshot + success status
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._bus.subscribe(EventType.QUESTION_DETECTED, self._on_question)
        self._bus.subscribe(EventType.ANSWER_READY, self._on_answer)
        self._bus.subscribe(EventType.CLICK_DONE, self._on_click_done)

    async def _on_question(self, event: Event) -> None:
        question = event.payload.get("question", "")
        options = event.payload.get("options", {})
        screenshot_path = event.payload.get("screenshot_path")

        opts_text = "\n".join(f"  {k}. {v}" for k, v in options.items())
        text = f"📝 题目检测到\n\n{question}\n\n{opts_text}"

        try:
            await self.send_text(text)
            if screenshot_path and Path(screenshot_path).exists():
                await self.send_image(screenshot_path, caption="题目截图")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to send question: {e}")

    async def _on_answer(self, event: Event) -> None:
        raw_responses = event.payload.get("raw_responses")
        answer = event.payload.get("answer", "?")
        answer_option = event.payload.get("answer_option", "")
        source = event.payload.get("source", "unknown")
        confidence = event.payload.get("confidence", 0.0)

        if raw_responses:
            sections = []
            for item in raw_responses:
                model = item.get("model", "unknown")
                raw_response = item.get("raw_response", "")
                sections.append(f"[{model}]\n{raw_response}")

            text = (
                f"🧠 模型原始回答\n\n"
                f"来源: {source}\n"
                f"置信度: {confidence:.0%}\n\n"
                + "\n\n".join(sections)
            )
            if answer_option:
                text += f"\n\n🎯 提取选项: {answer_option}"
        else:
            text = (
                f"✅ 答案生成\n\n"
                f"答案: {answer}\n"
                f"来源: {source}\n"
                f"置信度: {confidence:.0%}"
            )

        try:
            await self.send_text(text)
        except Exception as e:
            logger.error(f"[{self.name}] Failed to send answer: {e}")

    async def _on_click_done(self, event: Event) -> None:
        success = event.payload.get("success", False)
        answer = event.payload.get("answer", "?")
        finish_clicked = event.payload.get("finish_clicked", False)
        located_path = event.payload.get("located_option_screenshot_path")
        confirm_path = event.payload.get("confirm_screenshot_path")
        post_submit_task_path = event.payload.get("post_submit_task_screenshot_path")

        status = "✅ 成功" if success else "⚠️ 提交后变化不明显"
        finish_status = "✅ 已点击" if finish_clicked else "⏭️ 未配置或未点击"
        text = (
            f"📋 答题完成\n\n"
            f"选项: {answer}\n"
            f"状态: {status}\n"
            f"提交按钮: {finish_status}"
        )

        try:
            await self.send_text(text)
            if located_path and Path(located_path).exists():
                await self.send_image(located_path, caption="选项识别与点击定位截图")
            if confirm_path and Path(confirm_path).exists():
                await self.send_image(confirm_path, caption="答题结果截图")
            if post_submit_task_path and Path(post_submit_task_path).exists():
                await self.send_image(post_submit_task_path, caption="点击提交后 task 区域截图")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to send result: {e}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this notifier backend."""
        ...

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send a text message."""
        ...

    @abstractmethod
    async def send_image(self, image_path: str, caption: str = "") -> None:
        """Send an image with optional caption."""
        ...

    async def close(self) -> None:
        """Release resources. Override in subclasses if needed."""
        pass