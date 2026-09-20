"""Bounded classroom sessions with private evidence and stop control."""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Protocol
from uuid import uuid4

from autoykt.core.config import AppConfig, ConfigError
from autoykt.core.event_bus import Event, EventBus, EventType


logger = logging.getLogger("autoykt")
HEARTBEAT_SECONDS = 1.0


class SessionWatcher(Protocol):
    """Monitoring operations needed by a timed session."""

    exit_code: int
    stop_reason: str

    async def prepare(self) -> None:
        """Warm local engines."""
        raise NotImplementedError

    async def start(self) -> None:
        """Poll until stopped."""
        raise NotImplementedError

    def stop(self) -> None:
        """Finish the current bounded cycle."""
        raise NotImplementedError

    def snapshot(self) -> list[dict[str, Any]]:
        """Return current profile states and cumulative outcomes."""
        raise NotImplementedError


def session_directory(config: AppConfig) -> Path:
    """Locate session artifacts alongside the configured private run data."""
    return config.resolve_path(config.storage.data_dir) / "sessions"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            # Windows readers may briefly prevent atomic replacement.
            sleep(0.02 * (2**attempt))


def read_session_status(config: AppConfig) -> dict[str, Any]:
    """Read the latest status, explicitly exposing an expired heartbeat."""
    path = session_directory(config) / "latest.json"
    if not path.is_file():
        raise ConfigError(
            "no classroom session has been started for this config"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    heartbeat = datetime.fromisoformat(data["heartbeat_at"])
    age = max(0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds())
    data["heartbeat_age_seconds"] = round(age, 1)
    data["heartbeat_stale"] = (
        data["status"] in {"preparing", "monitoring", "finishing"} and age > 15
    )
    return data


def request_session_stop(config: AppConfig) -> Path:
    """Address a stop request to this exact session, never to the next one."""
    data = read_session_status(config)
    identifier = data["id"]
    if (
        not isinstance(identifier, str)
        or len(identifier) != 32
        or any(char not in "0123456789abcdef" for char in identifier)
    ):
        raise ConfigError("invalid session identifier")
    if data["status"] not in {"preparing", "monitoring", "finishing"}:
        raise ConfigError("the latest classroom session has already ended")
    root = session_directory(config).resolve()
    directory = (root / identifier).resolve()
    directory.relative_to(root)
    path = directory / "stop.request"
    path.write_text("stop\n", encoding="utf-8")
    return path


class ClassSession:
    """Time the monitoring period, leaving each question's own budget intact."""

    def __init__(
        self, config: AppConfig, event_bus: EventBus, minutes: float
    ) -> None:
        if not math.isfinite(minutes) or minutes <= 0:
            raise ConfigError("session minutes must be positive and finite")
        self._config = config
        self._duration = minutes * 60
        self._root = session_directory(config)
        identifier = uuid4().hex
        self.directory = self._root / identifier
        self.directory.mkdir(parents=True, exist_ok=False)
        self.report_path = self.directory / "report.json"
        self._started: float | None = None
        self._ended: float | None = None
        self._created = monotonic()
        self._data: dict[str, Any] = {
            "id": identifier,
            "pid": os.getpid(),
            "config": str(config.source_path),
            "mode": "dry_run" if config.runtime.dry_run else "live",
            "status": "preparing",
            "duration_seconds": self._duration,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "monitoring_started_at": None,
            "planned_end_at": None,
            "elapsed_seconds": 0.0,
            "reason": "",
            "exit_code": None,
            "models": [
                {
                    "provider": provider.name,
                    "models": provider.models,
                    "input_mode": provider.input_mode,
                    "thinking": provider.thinking,
                }
                for provider in config.answering.providers
            ],
            "profiles": [],
            "outcome_counts": {},
            "error_events": 0,
            "last_error": None,
            "report": str(self.report_path),
        }
        self.refresh()
        for event_type in EventType:
            event_bus.subscribe(event_type, self.record_event)

    async def record_event(self, event: Event) -> None:
        """Append evidence as it arrives without retaining a growing list."""
        payload = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.type.value,
            "profile_id": event.profile_id,
            "payload": event.payload,
        }
        with (self.directory / "events.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if event.type == EventType.ERROR:
            self._data["error_events"] += 1
            self._data["last_error"] = payload

    def refresh(self, watcher: SessionWatcher | None = None) -> None:
        """Atomically publish a heartbeat and cumulative outcome counts."""
        if watcher is not None:
            profiles = watcher.snapshot()
            self._data["profiles"] = profiles
            counts: Counter[str] = Counter()
            for profile in profiles:
                counts.update(profile["outcome_counts"])
            self._data["outcome_counts"] = dict(counts)
        if self._started is not None:
            end = self._ended if self._ended is not None else monotonic()
            self._data["elapsed_seconds"] = round(end - self._started, 3)
        if self._data["status"] == "ended":
            self._data["assessment"] = self._assessment()
        self._data["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(self.report_path, self._data)
        _write_json(self._root / "latest.json", self._data)

    def finish(
        self, reason: str, code: int, watcher: SessionWatcher | None
    ) -> None:
        """Persist the terminal reason, including startup failures."""
        self._ended = monotonic()
        self._data.update(
            status="ended",
            reason=reason,
            exit_code=code,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        self.refresh(watcher)

    def _assessment(self) -> str:
        """Include observer errors delivered while shutdown drains events."""
        counts = self._data["outcome_counts"]
        return (
            "review_required"
            if self._data["exit_code"]
            or any(
                counts.get(key, 0)
                for key in (
                    "failed",
                    "answer_rejected",
                    "interrupted",
                    "unavailable",
                )
            )
            or self._data["error_events"]
            or any(
                profile.get("monitoring", {}).get("pause_count", 0)
                for profile in self._data["profiles"]
            )
            else "verified_cycles_observed"
            if counts.get("verified", 0)
            else "no_verified_questions"
        )

    async def run(self, watcher: SessionWatcher) -> int:
        """Warm OCR, monitor for the duration, then finish the cycle."""
        task = asyncio.create_task(watcher.prepare())
        try:
            if await self._wait_prepared(task, watcher):
                self.finish("stop_requested", 130, watcher)
                return 130
            now = datetime.now(timezone.utc)
            self._started = monotonic()
            self._data.update(
                status="monitoring",
                monitoring_started_at=now.isoformat(),
                planned_end_at=(
                    now + timedelta(seconds=self._duration)
                ).isoformat(),
                preparation_seconds=round(self._started - self._created, 3),
            )
            self.refresh(watcher)
            logger.info(
                "Class session ready: %.1f minutes; status: %s",
                self._duration / 60,
                self._root / "latest.json",
            )
            task = asyncio.create_task(watcher.start())
            reason = await self._monitor(task, watcher)
            code = 130 if reason == "stop_requested" else watcher.exit_code
            self.finish(reason, code, watcher)
            return code
        except asyncio.CancelledError:
            await self._cancel(task)
            self.finish("interrupted", 130, watcher)
            raise
        except Exception:
            await self._cancel(task)
            self.finish("error", 1, watcher)
            raise

    async def _wait_prepared(
        self, task: asyncio.Task, watcher: SessionWatcher
    ) -> bool:
        while not task.done():
            if self._stop_requested():
                await self._cancel(task)
                return True
            self.refresh(watcher)
            await asyncio.wait({task}, timeout=HEARTBEAT_SECONDS)
        await task
        return self._stop_requested()

    async def _monitor(
        self, task: asyncio.Task, watcher: SessionWatcher
    ) -> str:
        assert self._started is not None
        deadline = self._started + self._duration
        finishing_since = None
        grace = (
            max(
                page.question_budget.total_seconds
                for page in self._config.active_page_profiles()
            )
            + 5
        )
        next_log = self._started + 60
        while not task.done():
            now = monotonic()
            if self._stop_requested():
                await self._cancel(task)
                return "stop_requested"
            if now >= deadline and finishing_since is None:
                watcher.stop()
                finishing_since = now
                self._data["status"] = "finishing"
                logger.info(
                    "Session duration reached; finishing the current cycle"
                )
            if finishing_since is not None and now - finishing_since >= grace:
                await self._cancel(task)
                watcher.exit_code = 1
                return "shutdown_timeout"
            self.refresh(watcher)
            if now >= next_log:
                logger.info("Class session: %s", self._data["profiles"])
                next_log = now + 60
            timeout = HEARTBEAT_SECONDS
            if finishing_since is None:
                timeout = min(timeout, max(0.001, deadline - now))
            await asyncio.wait({task}, timeout=timeout)
        await task
        if any(profile["needs_attention"] for profile in watcher.snapshot()):
            watcher.exit_code = 1
            return "manual_required"
        if finishing_since is not None:
            return (
                "duration_reached"
                if watcher.exit_code == 0
                else "manual_required"
            )
        return watcher.stop_reason

    def _stop_requested(self) -> bool:
        return (self.directory / "stop.request").exists()

    @staticmethod
    async def _cancel(task: asyncio.Task) -> None:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
