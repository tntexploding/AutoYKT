"""Tests for non-business observer event delivery."""

import asyncio
import unittest

from autoykt.core.event_bus import Event, EventBus, EventType


class EventBusTest(unittest.IsolatedAsyncioTestCase):

    async def test_delivers_event_and_stops(self) -> None:
        bus = EventBus()
        delivered = asyncio.Event()

        async def handler(event: Event) -> None:
            self.assertEqual(event.profile_id, "page")
            delivered.set()

        bus.subscribe(EventType.ANSWER_READY, handler)
        task = asyncio.create_task(bus.start())
        await bus.publish(Event(type=EventType.ANSWER_READY, profile_id="page"))
        await asyncio.wait_for(delivered.wait(), timeout=1)
        bus.stop()
        await asyncio.wait_for(task, timeout=1)

    async def test_stop_drains_events_queued_before_sentinel(self) -> None:
        bus = EventBus()
        delivered: list[int] = []

        async def handler(event: Event) -> None:
            delivered.append(int(event.payload["sequence"]))

        bus.subscribe(EventType.ANSWER_READY, handler)
        task = asyncio.create_task(bus.start())
        await asyncio.sleep(0)
        await bus.publish(
            Event(type=EventType.ANSWER_READY, payload={"sequence": 1})
        )
        await bus.publish(
            Event(type=EventType.ANSWER_READY, payload={"sequence": 2})
        )
        bus.stop()
        await asyncio.wait_for(task, timeout=1)
        self.assertEqual(delivered, [1, 2])

    async def test_stop_drains_error_created_during_dispatch(self) -> None:
        bus = EventBus()
        error_delivered = asyncio.Event()

        async def failing_handler(_: Event) -> None:
            raise RuntimeError("observer failed")

        async def error_handler(_: Event) -> None:
            error_delivered.set()

        bus.subscribe(EventType.ANSWER_READY, failing_handler)
        bus.subscribe(EventType.ERROR, error_handler)
        task = asyncio.create_task(bus.start())
        await asyncio.sleep(0)
        await bus.publish(Event(type=EventType.ANSWER_READY))
        bus.stop()
        await asyncio.wait_for(task, timeout=1)
        self.assertTrue(error_delivered.is_set())


if __name__ == "__main__":
    unittest.main()
