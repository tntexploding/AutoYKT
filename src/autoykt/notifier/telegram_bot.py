"""Telegram Bot API notification backend."""

from __future__ import annotations

from pathlib import Path

from telegram import Bot

from autoykt.core.event_bus import EventBus
from autoykt.notifier.base import BaseNotifier


class TelegramNotifier(BaseNotifier):
    """Send workflow notifications to one Telegram chat."""

    def __init__(self, event_bus: EventBus, token: str, chat_id: str) -> None:
        super().__init__(event_bus)
        self._bot = Bot(token=token)
        self._chat_id = chat_id
        self._initialized = False

    @property
    def name(self) -> str:
        return "Telegram"

    async def send_text(self, text: str) -> None:
        await self._ensure_initialized()
        await self._bot.send_message(chat_id=self._chat_id, text=text)

    async def send_image(self, image_path: str, caption: str = "") -> None:
        await self._ensure_initialized()
        with Path(image_path).open("rb") as image_file:
            await self._bot.send_photo(
                chat_id=self._chat_id,
                photo=image_file,
                caption=caption or None,
            )

    async def close(self) -> None:
        await self._bot.shutdown()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._bot.initialize()
            self._initialized = True
