"""Event formatting shared by remote notification backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path

from autoykt.core.event_bus import Event, EventBus, EventType


logger = logging.getLogger("autoykt")
_MAX_RESPONSE_CHARACTERS = 4000


class BaseNotifier(ABC):
    """Convert observer events into concise text and image notifications."""

    def __init__(self, event_bus: EventBus) -> None:
        event_bus.subscribe(EventType.TEMPLATE_DETECTED, self._on_template)
        event_bus.subscribe(EventType.QUESTION_CAPTURED, self._on_question)
        event_bus.subscribe(EventType.ANSWER_READY, self._on_answer)
        event_bus.subscribe(EventType.ANSWER_REJECTED, self._on_answer)
        event_bus.subscribe(EventType.SUBMISSION_VERIFIED, self._on_submission)
        event_bus.subscribe(EventType.ERROR, self._on_error)

    async def _on_template(self, event: Event) -> None:
        if event.payload.get("action") != "notify":
            return
        screenshot = event.payload.get("screenshot_path")
        if not screenshot:
            return
        await self._safe_text(
            f"检测到页面样式 [{_profile(event)}]\n"
            f"模板：{event.payload.get('template_name', 'unknown')}\n"
            f"匹配度：{float(event.payload.get('confidence', 0.0)):.1%}"
        )
        await self._safe_image(str(screenshot), "页面截图")

    async def _on_question(self, event: Event) -> None:
        question = str(event.payload.get("question") or "（未识别文本）")
        await self._safe_text(f"检测到题目 [{_profile(event)}]\n{question}")
        screenshot = event.payload.get("screenshot_path")
        if screenshot:
            await self._safe_image(str(screenshot), "题目截图")

    async def _on_answer(self, event: Event) -> None:
        answer = event.payload.get("answer") or "无可执行答案"
        votes = event.payload.get("votes") or {}
        reason = event.payload.get("reason") or ""
        sections = [
            f"答题结果 [{_profile(event)}]",
            f"答案：{answer}",
            f"投票：{votes}",
            f"结论：{reason}",
        ]
        for response in event.payload.get("responses") or []:
            if not isinstance(response, dict):
                continue
            identity = f"{response.get('provider')}/{response.get('model')}"
            raw = str(
                response.get("raw_response") or response.get("error") or ""
            )
            sections.append(f"[{identity}]\n{raw}")
        text = "\n\n".join(sections)[:_MAX_RESPONSE_CHARACTERS]
        await self._safe_text(text)

    async def _on_submission(self, event: Event) -> None:
        success = bool(event.payload.get("success"))
        status = "已验证成功" if success else "验证失败，已停止本题流程"
        await self._safe_text(
            f"提交结果 [{_profile(event)}]\n"
            f"答案：{event.payload.get('answer', '?')}\n"
            f"状态：{status}\n"
            f"证据：{event.payload.get('evidence', '')}"
        )
        screenshot = event.payload.get("screenshot_path")
        if screenshot:
            await self._safe_image(str(screenshot), "提交后页面")

    async def _on_error(self, event: Event) -> None:
        await self._safe_text(
            f"AutoYKT 错误 [{_profile(event)}]\n"
            f"来源：{event.payload.get('source', 'observer')}\n"
            f"信息：{event.payload.get('error', 'unknown error')}"
        )

    async def _safe_text(self, text: str) -> None:
        try:
            await self.send_text(text)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Notifications are an isolation boundary around remote backends.
            logger.error(
                "[%s] text notification failed: %s",
                self.name,
                type(error).__name__,
            )

    async def _safe_image(self, image_path: str, caption: str) -> None:
        if not Path(image_path).is_file():
            return
        try:
            await self.send_image(image_path, caption)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # Notifications are an isolation boundary around remote backends.
            logger.error(
                "[%s] image notification failed: %s",
                self.name,
                type(error).__name__,
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend's user-facing name."""

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send one text message."""

    @abstractmethod
    async def send_image(self, image_path: str, caption: str = "") -> None:
        """Send one local image."""

    async def close(self) -> None:
        """Release optional backend resources."""


def _profile(event: Event) -> str:
    return event.profile_id or "system"
