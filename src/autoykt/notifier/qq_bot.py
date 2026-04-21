"""QQ notification backend using OneBot (Lagrange) HTTP API."""

import base64
import logging
from pathlib import Path

import aiohttp

from autoykt.core.event_bus import EventBus
from autoykt.notifier.base import BaseNotifier

logger = logging.getLogger("auto_answer")


class QQNotifier(BaseNotifier):
    """Push notifications to a QQ account via OneBot HTTP API."""

    def __init__(self, event_bus: EventBus, onebot_url: str, target_qq: str, access_token: str = "") -> None:
        super().__init__(event_bus)
        self._url = onebot_url.rstrip("/")
        self._target = int(target_qq)
        self._access_token = access_token
        self._session: aiohttp.ClientSession | None = None
        logger.info(f"QQNotifier initialized, target={target_qq}, url={onebot_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self._access_token:
                headers["Authorization"] = f"Bearer {self._access_token}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    @property
    def name(self) -> str:
        return "QQ"

    async def send_text(self, text: str) -> None:
        session = await self._get_session()
        payload = {
            "user_id": self._target,
            "message": [{"type": "text", "data": {"text": text}}],
        }
        async with session.post(f"{self._url}/send_private_msg", json=payload) as resp:
            result = await resp.json()
            if result.get("retcode") != 0:
                logger.warning(f"[QQ] send_text failed: {result}")
            else:
                logger.debug(f"[QQ] Sent text: {text[:60]}...")

    async def send_image(self, image_path: str, caption: str = "") -> None:
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("utf-8")

        message = []
        if caption:
            message.append({"type": "text", "data": {"text": caption + "\n"}})
        message.append({"type": "image", "data": {"file": f"base64://{b64}"}})

        session = await self._get_session()
        payload = {
            "user_id": self._target,
            "message": message,
        }
        async with session.post(f"{self._url}/send_private_msg", json=payload) as resp:
            result = await resp.json()
            if result.get("retcode") != 0:
                logger.warning(f"[QQ] send_image failed: {result}")
            else:
                logger.debug(f"[QQ] Sent image: {image_path}")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()