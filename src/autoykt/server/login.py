"""Phone-assisted QR login for a continuously running headless service."""

from __future__ import annotations

import asyncio
import base64
from functools import partial
import time
from uuid import uuid4

from autoykt.server.client import RECOVERY_ERRORS
from autoykt.server.solver import AssetStore


async def publish_ticket(
    config, client, events, url, *, attempt_id, expires_at
):
    """Send both a QR image URL and, when available, its original bytes."""
    assets = AssetStore(config.storage / "images", client)
    try:
        image = await asyncio.wait_for(assets.image(url), 5)
    except asyncio.TimeoutError:
        image = None
    payload = {
        "image_url": url,
        "attempt_id": attempt_id,
        "expires_at": expires_at,
    }
    if image:
        data, mime = image
        extension = {
            "image/jpeg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
            "image/gif": "gif",
        }[mime]
        target = config.storage / f"login-qr.{extension}"
        target.write_bytes(data)
        payload.update(
            {
                "image_base64": base64.b64encode(data).decode("ascii"),
                "image_mime": mime,
                "local_file": str(target),
            }
        )
    await events.emit("login_qr", payload=payload)


class LoginRecovery:
    """Keep a single QR attempt active across all classroom listeners."""

    def __init__(self, config, client, events, on_success):
        self.config = config
        self.client = client
        self.events = events
        self.on_success = on_success
        self.task: asyncio.Task | None = None

    @property
    def active(self) -> bool:
        """Whether the service is currently waiting for a phone scan."""
        return self.task is not None and not self.task.done()

    def request(self):
        """Coalesce simultaneous expired-session reports into one QR flow."""
        if (
            not self.client.supports_browser_login
            or not self.config.platform.auto_login
            or self.active
        ):
            return
        self.task = asyncio.create_task(self._recover())

    async def _recover(self):
        while True:
            attempt = uuid4().hex
            try:
                await self.client.login(
                    partial(
                        publish_ticket,
                        self.config,
                        self.client,
                        self.events,
                        attempt_id=attempt,
                        expires_at=time.time() + 120,
                    )
                )
                await self.client.request("GET", "/api/v3/user/basic-info")
                await self.on_success()
                await self.events.emit(
                    "login_succeeded", payload={"attempt_id": attempt}
                )
                return
            except RECOVERY_ERRORS as error:
                await self.events.emit(
                    "login_retry",
                    payload={
                        "reason": type(error).__name__,
                        "attempt_id": attempt,
                    },
                )
                await asyncio.sleep(self.config.platform.login_retry_seconds)

    async def close(self):
        """Cancel an outstanding scan when shutting down."""
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
