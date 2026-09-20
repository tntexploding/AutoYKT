"""CLI lifecycle regressions without external connections."""

import asyncio
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock, patch

from autoykt.cli import _run_automation
from autoykt.core.config import load_config
from autoykt.core.event_bus import Event, EventBus, EventType


class ShutdownRegressionTest(unittest.IsolatedAsyncioTestCase):
    """Observers finish queued work before their clients are closed."""

    async def test_stop_before_dispatch_start_does_not_hang(self) -> None:
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe(EventType.ERROR, handler)
        await bus.publish(Event(EventType.ERROR))
        bus.stop()
        await asyncio.wait_for(bus.start(), 1)
        handler.assert_awaited_once()

    async def test_notifications_drain_before_client_close(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config.example.yaml")
        order = []
        notifier = Mock()
        notifier.close = AsyncMock(side_effect=lambda: order.append("close"))
        bus_holder = []

        def notifiers(_, bus):
            bus_holder.append(bus)

            async def handle(_):
                await asyncio.sleep(0)
                order.append("send")

            bus.subscribe(EventType.ANSWER_READY, handle)
            return [notifier]

        watcher = Mock()
        watcher.close = AsyncMock()

        async def start():
            await bus_holder[0].publish(Event(EventType.ANSWER_READY))
            raise RuntimeError("end fake run")

        watcher.start = start
        with (
            patch("autoykt.cli.setup_logger"),
            patch("autoykt.cli.create_notifiers", side_effect=notifiers),
            patch("autoykt.cli.ScreenWatcher", return_value=watcher),
        ):
            with self.assertRaisesRegex(RuntimeError, "end fake run"):
                await asyncio.wait_for(_run_automation(config, False), 2)
        self.assertEqual(order, ["send", "close"])

    async def test_notifiers_close_when_watcher_construction_fails(
        self,
    ) -> None:
        config = load_config(Path(__file__).parents[1] / "config.example.yaml")
        notifier = Mock(close=AsyncMock())
        with (
            patch("autoykt.cli.setup_logger"),
            patch("autoykt.cli.create_notifiers", return_value=[notifier]),
            patch(
                "autoykt.cli.ScreenWatcher",
                side_effect=RuntimeError("missing screen"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing screen"):
                await _run_automation(config, False)
        notifier.close.assert_awaited_once()
