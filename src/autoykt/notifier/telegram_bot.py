"""Telegram notification backend using python-telegram-bot."""

import logging

from telegram import Bot

from autoykt.core.event_bus import EventBus
from autoykt.notifier.base import BaseNotifier

logger = logging.getLogger("auto_answer")


class TelegramNotifier(BaseNotifier):
    """Push notifications to a Telegram chat via Bot API."""

    def __init__(self, event_bus: EventBus, token: str, chat_id: str) -> None:
        super().__init__(event_bus)
        self._bot = Bot(token=token)
        self._chat_id = chat_id
        logger.info(f"TelegramNotifier initialized, chat_id={chat_id}")

    @property
    def name(self) -> str:
        return "Telegram"

    async def send_text(self, text: str) -> None:
        await self._bot.send_message(
            chat_id=self._chat_id,
            text=text,
            parse_mode=None,
        )
        logger.debug(f"[Telegram] Sent text: {text[:60]}...")

    async def send_image(self, image_path: str, caption: str = "") -> None:
        with open(image_path, "rb") as f:
            await self._bot.send_photo(
                chat_id=self._chat_id,
                photo=f,
                caption=caption or None,
            )
        logger.debug(f"[Telegram] Sent image: {image_path}")