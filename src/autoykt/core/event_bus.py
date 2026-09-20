"""Asynchronous events used for observation, logging, and notifications."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Observable events emitted by the automation workflow."""

    STATE_CHANGED = "state_changed"
    TEMPLATE_DETECTED = "template_detected"
    QUESTION_CAPTURED = "question_captured"
    ANSWER_READY = "answer_ready"
    ANSWER_REJECTED = "answer_rejected"
    INTERACTION_COMPLETED = "interaction_completed"
    SUBMISSION_VERIFIED = "submission_verified"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    ERROR = "error"

    # Compatibility name used by prototype integrations.
    QUESTION_DETECTED = "question_captured"
    CLICK_DONE = "interaction_completed"


@dataclass(frozen=True)
class Event:
    """One immutable workflow event."""

    type: EventType
    profile_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


Subscriber = Callable[[Event], Awaitable[None]]


class EventBus:
    """Queue events without coupling workflow progress to observers."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Subscriber]] = {}
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()
        self._running = False
        self._stopping = False

    def subscribe(self, event_type: EventType, handler: Subscriber) -> None:
        """Register an async handler for one event type."""
        self._subscribers.setdefault(event_type, []).append(handler)

    async def publish(self, event: Event) -> None:
        """Place an event on the observer queue."""
        await self._queue.put(event)

    async def start(self) -> None:
        """Dispatch queued events until ``stop`` is called."""
        if self._running:
            raise RuntimeError("event bus is already running")
        self._running = True
        try:
            while True:
                event = await self._queue.get()
                if event is None:
                    if self._queue.empty():
                        break
                    self._queue.put_nowait(None)
                    continue
                await self._dispatch(event)
        finally:
            self._running = False
            self._stopping = False

    async def _dispatch(self, event: Event) -> None:
        handlers = tuple(self._subscribers.get(event.type, ()))
        if not handlers:
            return
        results = await asyncio.gather(
            *(handler(event) for handler in handlers),
            return_exceptions=True,
        )
        if event.type == EventType.ERROR:
            return
        for result in results:
            if isinstance(result, BaseException):
                await self._queue.put(
                    Event(
                        type=EventType.ERROR,
                        profile_id=event.profile_id,
                        payload={
                            "source_event": event.type.value,
                            "error": str(result),
                        },
                    )
                )

    def stop(self) -> None:
        """Wake and stop the dispatch loop."""
        if not self._stopping:
            self._stopping = True
            self._queue.put_nowait(None)
