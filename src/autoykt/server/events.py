"""Journal events and persist prioritized webhook deliveries across restarts."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import sqlite3
import time
from uuid import uuid4

import aiohttp

from autoykt.server.config import ServerConfig


logger = logging.getLogger("autoykt.server")
URGENT = frozenset(
    {
        "question_detected",
        "question_ready",
        "question_image",
        "question_image_unavailable",
        "question_unavailable",
        "login_required",
        "login_qr",
        "login_succeeded",
        "notification_test",
        "answer_failed",
        "answer_ready",
        "submission_result",
        "submission_verified",
        "submission_unresolved",
        "lesson_disconnected",
        "classroom_entry_result",
    }
)


class EventWriter:
    """Deliver urgent reminders before routine telemetry."""

    def __init__(self, config: ServerConfig):
        self.config = config
        config.storage.mkdir(parents=True, exist_ok=True)
        self.path = config.storage / "events.jsonl"
        self.worker: asyncio.Task | None = None
        self.changed = asyncio.Event()
        self.last_delivery: float | None = None
        self.last_error: str | None = None
        self.destination = hashlib.sha256(
            (config.webhook.url or "").encode()
        ).hexdigest()
        self.db: sqlite3.Connection | None = None
        if config.webhook.url:
            self.db = sqlite3.connect(config.storage / "webhook.sqlite3")
            self.db.row_factory = sqlite3.Row
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS deliveries (
                id TEXT PRIMARY KEY, destination TEXT, event TEXT,
                priority INTEGER, created REAL, expires REAL,
                due REAL, attempts INTEGER DEFAULT 0
            )"""
            )
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS receipts "
                "(id TEXT PRIMARY KEY, delivered REAL, expires REAL)"
            )
            self.db.commit()

    async def start(self):
        """Resume unexpired deliveries for the configured destination."""
        if self.db is not None and self.worker is None:
            self.worker = asyncio.create_task(self._deliver())

    async def emit(
        self,
        kind: str,
        *,
        payload: dict | None = None,
        dedup_key: str | None = None,
        **identity,
    ):
        """Journal and enqueue the event before returning to the classroom."""
        event_id = (
            hashlib.sha256(
                (
                    f"{self.config.platform.base_url}|{self.destination}|"
                    f"{kind}|{dedup_key}"
                ).encode()
            ).hexdigest()
            if dedup_key
            else uuid4().hex
        )
        event = {
            "schema_version": 1,
            "event_id": event_id,
            "type": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domain": self.config.platform.base_url,
            **identity,
            "payload": payload or {},
        }
        line = json.dumps(event, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        logger.info(
            "%s lesson=%s question=%s event_id=%s",
            kind,
            identity.get("lesson_id", ""),
            identity.get("question_id", ""),
            event_id,
        )
        selected = self.config.webhook.events
        if self.db is not None and (not selected or kind in selected):
            self._enqueue(event, line)
            self.changed.set()
        return event

    def _enqueue(self, event, line):
        assert self.db is not None
        now = time.time()
        priority = 0 if event["type"] in URGENT else 2
        expires = now + self.config.webhook.retention_seconds
        payload = event["payload"]
        if payload.get("expires_at"):
            expires = min(expires, float(payload["expires_at"]))
        if payload.get("deadline") is not None:
            expires = min(expires, float(payload["deadline"]) + 30)
        with self.db:
            self.db.execute("DELETE FROM deliveries WHERE expires <= ?", (now,))
            if self.db.execute(
                "SELECT 1 FROM deliveries WHERE id=? UNION ALL "
                "SELECT 1 FROM receipts WHERE id=? AND expires>?",
                (event["event_id"], event["event_id"], now),
            ).fetchone():
                return
            count = self.db.execute(
                "SELECT COUNT(*) FROM deliveries WHERE destination=?",
                (self.destination,),
            ).fetchone()[0]
            if count >= self.config.webhook.queue_size:
                victim = self.db.execute(
                    "SELECT id, priority FROM deliveries WHERE "
                    "destination=? ORDER BY priority DESC, created LIMIT 1",
                    (self.destination,),
                ).fetchone()
                if victim["priority"] < priority:
                    logger.warning(
                        "outbox full; routine event retained only in JSONL"
                    )
                    return
                self.db.execute(
                    "DELETE FROM deliveries WHERE id=?", (victim["id"],)
                )
                logger.warning(
                    "outbox full; oldest low-priority delivery evicted"
                )
            self.db.execute(
                "INSERT OR IGNORE INTO deliveries VALUES (?,?,?,?,?,?,?,0)",
                (
                    event["event_id"],
                    self.destination,
                    line,
                    priority,
                    now,
                    expires,
                    now,
                ),
            )

    def _next(self):
        assert self.db is not None
        now = time.time()
        with self.db:
            self.db.execute("DELETE FROM deliveries WHERE expires <= ?", (now,))
            self.db.execute("DELETE FROM receipts WHERE expires <= ?", (now,))
            row = self.db.execute(
                "SELECT * FROM deliveries WHERE destination=? AND due<=? "
                "ORDER BY priority, created LIMIT 1",
                (self.destination, now),
            ).fetchone()
            if row:
                self.db.execute(
                    "UPDATE deliveries SET due=? WHERE id=?",
                    (
                        now + self.config.webhook.timeout_seconds + 2,
                        row["id"],
                    ),
                )
            return row

    async def _deliver(self):
        settings = self.config.webhook
        assert settings.url is not None
        assert self.db is not None
        token = os.environ.get(settings.token_env, "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.timeout_seconds)
        ) as session:
            while True:
                self.changed.clear()
                row = self._next()
                if row is None:
                    try:
                        await asyncio.wait_for(self.changed.wait(), 1)
                    except asyncio.TimeoutError:
                        pass
                    continue
                try:
                    async with session.post(
                        settings.url,
                        json=json.loads(row["event"]),
                        headers={**headers, "Idempotency-Key": row["id"]},
                        allow_redirects=False,
                    ) as response:
                        self.last_error = (
                            None
                            if 200 <= response.status < 300
                            else f"HTTP {response.status}"
                        )
                except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                    self.last_error = type(error).__name__
                with self.db:
                    if self.last_error is None:
                        self.last_delivery = time.time()
                        self.db.execute(
                            "INSERT OR REPLACE INTO receipts VALUES (?,?,?)",
                            (row["id"], self.last_delivery, row["expires"]),
                        )
                        self.db.execute(
                            "DELETE FROM deliveries WHERE id=?", (row["id"],)
                        )
                    elif (
                        row["priority"] > 0
                        and row["attempts"] + 1 >= settings.attempts
                    ):
                        self.db.execute(
                            "DELETE FROM deliveries WHERE id=?", (row["id"],)
                        )
                        logger.warning(
                            "routine webhook retries exhausted: %s", row["id"]
                        )
                    else:
                        delay = min(2 ** min(row["attempts"], 5), 30)
                        self.db.execute(
                            "UPDATE deliveries SET attempts=attempts+1, due=? "
                            "WHERE id=?",
                            (time.time() + delay, row["id"]),
                        )
                        logger.warning(
                            "webhook delivery delayed: %s", self.last_error
                        )

    def delivery_status(self) -> dict:
        """Expose delivery progress independently of the process heartbeat."""
        count = 0
        if self.db is not None:
            count = self.db.execute(
                "SELECT COUNT(*) FROM deliveries WHERE "
                "destination=? AND expires>?",
                (self.destination, time.time()),
            ).fetchone()[0]
        return {
            "configured": bool(self.config.webhook.url),
            "pending": count,
            "worker_alive": self.worker is not None and not self.worker.done(),
            "last_delivery": self.last_delivery,
            "last_error": self.last_error,
        }

    async def flush(self, timeout: float = 5):
        """Wait for current unexpired deliveries to leave the outbox."""

        async def wait():
            while self.delivery_status()["pending"]:
                await asyncio.sleep(0.02)

        await asyncio.wait_for(wait(), timeout)

    async def wait_delivered(self, event_id: str, timeout: float):
        """Require a 2xx receipt, including one from another sender process."""
        db = self.db
        assert db is not None

        async def wait():
            while not db.execute(
                "SELECT 1 FROM receipts WHERE id=?", (event_id,)
            ).fetchone():
                await asyncio.sleep(0.02)

        await asyncio.wait_for(wait(), timeout)

    async def close(self):
        """Drain briefly and retain remaining deliveries for the next start."""
        if self.worker:
            try:
                await self.flush()
            except asyncio.TimeoutError:
                pass
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
            self.worker = None
        if self.db is not None:
            self.db.close()
            self.db = None
