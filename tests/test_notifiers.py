"""Tests for notification client lifecycle without network access."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from autoykt.core.event_bus import EventBus
from autoykt.notifier.telegram_bot import TelegramNotifier


class TelegramNotifierTest(unittest.IsolatedAsyncioTestCase):
    """Verify that python-telegram-bot is initialized lazily and once."""

    async def test_initializes_before_send_and_shuts_down(self) -> None:
        bot = MagicMock()
        bot.initialize = AsyncMock()
        bot.send_message = AsyncMock()
        bot.shutdown = AsyncMock()
        with patch("autoykt.notifier.telegram_bot.Bot", return_value=bot):
            notifier = TelegramNotifier(EventBus(), "token", "chat")
            await notifier.send_text("first")
            await notifier.send_text("second")
            await notifier.close()

        bot.initialize.assert_awaited_once_with()
        self.assertEqual(bot.send_message.await_count, 2)
        bot.shutdown.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
