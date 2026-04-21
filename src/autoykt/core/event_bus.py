"""Lightweight async event bus using asyncio.Queue."""

import asyncio
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from datetime import datetime


class EventType(Enum):
    """All event types flowing through the system."""
    QUESTION_DETECTED = auto()   # 题目被检测到
    ANSWER_READY = auto()        # 答案已生成
    CLICK_DONE = auto()          # 点击完成，答题结束
    ERROR = auto()               # 错误事件


@dataclass                                                                      #生成数据类样板代码
class Event:
    """A single event with type, payload, and metadata."""
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)                       #field＆defaultfactory
    timestamp: datetime = field(default_factory=datetime.now)                   #对每一个新对象，给出新的字典和记录时间


# Subscriber type: async callable that takes an Event
Subscriber = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Async pub/sub event bus. Modules subscribe to event types."""

    def __init__(self) -> None:                                                 #初始化
        self._subscribers: dict[EventType, list[Subscriber]] = {}               #字典，四种不同事件的subscriber列表
        self._queue: asyncio.Queue[Event] = asyncio.Queue()                     #事件队列
        self._running = False

    def subscribe(self, event_type: EventType, handler: Subscriber) -> None:    #注册订阅
        """Register a handler for a specific event type."""
        if event_type not in self._subscribers:                                 #如果此前没有订阅某种event，则新增该event的订阅列表
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)                           #向订阅列表加入subscriber

    async def publish(self, event: Event) -> None:                              #向队列发布事件
        """Publish an event to the bus."""
        await self._queue.put(event)

    async def start(self) -> None:                                              #启动自动运行
        """Start the event dispatch loop."""
        self._running = True
        while self._running:                                                    #自动运行标志
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)  #从事件队列取一个事件，没有就1s后再来
            except asyncio.TimeoutError:
                continue

            handlers = self._subscribers.get(event.type, [])                    #获取subscriber列表
            for handler in handlers:
                try:
                    await handler(event)                                        #顺序执行handler
                except Exception as e:                                          #错误拦截，升级成ERROR
                    # Publish error event but avoid infinite loop
                    if event.type != EventType.ERROR:
                        err_event = Event(
                            type=EventType.ERROR,
                            payload={"source_event": event.type.name, "error": str(e)},
                        )
                        await self._queue.put(err_event)                        #将拦截到的错误置入队列，不等待handler

    def stop(self) -> None:                                                     #关闭标志位
        """Stop the dispatch loop."""
        self._running = False