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
        answer = event.payload.get("answer", "?")
        source = event.payload.get("source", "unknown")
        confidence = event.payload.get("confidence", 0.0)

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
        confirm_path = event.payload.get("confirm_screenshot_path")

        status = "✅ 成功" if success else "❌ 可能失败"
        text = f"📋 答题完成\n\n选项: {answer}\n状态: {status}"

        try:
            await self.send_text(text)
            if confirm_path and Path(confirm_path).exists():
                await self.send_image(confirm_path, caption="答题结果截图")
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