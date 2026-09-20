"""QQ private-message notifications through the OneBot HTTP API."""

from __future__ import annotations

import base64
from pathlib import Path

import aiohttp

from autoykt.core.event_bus import EventBus
from autoykt.notifier.base import BaseNotifier


class QQNotifier(BaseNotifier):
    """Send events to one QQ account through a local OneBot service."""

    def __init__(
        self,
        event_bus: EventBus,
        onebot_url: str,
        target_qq: str,
        access_token: str = "",
    ) -> None:
        if not target_qq.isdigit():
            raise ValueError("QQ target environment value must be numeric")
        super().__init__(event_bus)
        self._url = onebot_url.rstrip("/")
        self._target = int(target_qq)
        self._access_token = access_token
        self._session: aiohttp.ClientSession | None = None

    @property
    def name(self) -> str:
        return "QQ"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
            )
        return self._session

    async def send_text(self, text: str) -> None:
        await self._send_message([{"type": "text", "data": {"text": text}}])

    async def send_image(self, image_path: str, caption: str = "") -> None:
        image_data = base64.b64encode(Path(image_path).read_bytes()).decode(
            "ascii"
        )
        message: list[dict[str, object]] = []
        if caption:
            message.append({"type": "text", "data": {"text": caption + "\n"}})
        message.append(
            {
                "type": "image",
                "data": {"file": f"base64://{image_data}"},
            }
        )
        await self._send_message(message)

    async def _send_message(self, message: list[dict[str, object]]) -> None:
        session = await self._get_session()
        payload = {"user_id": self._target, "message": message}
        async with session.post(
            f"{self._url}/send_private_msg", json=payload
        ) as response:
            response.raise_for_status()
            result = await response.json()
            if result.get("retcode") != 0:
                raise RuntimeError("OneBot rejected message")

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
