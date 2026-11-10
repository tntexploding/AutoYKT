"""Async adaptation of thuhollow2/yuketang's v3 classroom protocol.

Source: b5da7837dc187c9b7a960623e4d196727637aa3d/yuketang.py.
Copyright (c) 2025 thuhollow2, MIT; see third_party/thuhollow2-LICENSE.txt.
"""

from __future__ import annotations

import asyncio
from http.cookies import SimpleCookie
import json
import os
import time
from uuid import uuid4
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from autoykt.server.config import ServerConfig


class ProtocolError(RuntimeError):
    """An explicit platform response rejected the request."""

    def __init__(self, message, *, http_status=None):
        self.http_status = http_status
        super().__init__(message)


class LoginRequired(ProtocolError):
    """The platform requires a new login."""

    def __init__(self, message: str, code=None, *, http_status=None):
        self.code = code
        super().__init__(message, http_status=http_status)


class RejectedSubmission(ProtocolError):
    """The platform explicitly returned a business rejection."""

    def __init__(self, message, *, code=None, server_msg="", http_status=None):
        self.code, self.server_msg = code, server_msg
        super().__init__(message, http_status=http_status)


class ClassroomEntryRejected(ProtocolError):
    """A classroom entry business response, including dynamic-code refusal."""

    def __init__(self, code):
        self.code = code
        super().__init__(f"classroom entry rejected: code={code}")


PLATFORM_ERRORS = (
    ProtocolError,
    aiohttp.ClientError,
    asyncio.TimeoutError,
    ValueError,
)
RECOVERY_ERRORS = PLATFORM_ERRORS + (KeyError, TypeError)

class YuketangClient:
    """Use a private cookie and per-lesson rotated Bearer tokens."""

    entry_mode = "web"
    supports_browser_login = True
    block_on_entry_rejection = False

    USER_AGENT = (
        "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/141.0.0.0 Mobile Safari/537.36"
    )
    CATALOG_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, config: ServerConfig):
        self.config = config
        self.origin = config.platform.base_url
        self._loaded_session_path = self.session_path
        self.state: dict[str, Any] = {
            "domain": self.origin,
            "cookie": "",
            "tokens": {},
        }
        if self.session_path.exists():
            self.reload_session()
        self.http: aiohttp.ClientSession | None = None

    @property
    def session_path(self):
        """Locate the private Web session file."""
        return self.config.resolve(self.config.platform.session_file)

    def select_entry(self, mode):
        """Refresh the current entry adapter's session."""
        if mode != self.entry_mode:
            raise ValueError("此入口配置只支持 Web 会话。")
        self.reload_session()

    def reload_session(self):
        """Allow a separate QR-login command to renew a running service."""
        if self._loaded_session_path != self.session_path:
            self.state = {"domain": self.origin, "cookie": "", "tokens": {}}
            self._loaded_session_path = self.session_path
        if self.session_path.exists():
            state = json.loads(self.session_path.read_text(encoding="utf-8"))
            if state.get("domain") != self.origin:
                raise ValueError(
                    "session belongs to a different platform origin"
                )
            self.state = {"tokens": {}, **state}

    async def __aenter__(self):
        self.http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.config.platform.request_timeout
            ),
            cookie_jar=aiohttp.DummyCookieJar(),
        )
        return self

    async def __aexit__(self, *_):
        if self.http:
            await self.http.close()

    def save_session(self):
        """Persist rotated credentials atomically in private storage."""
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.session_path.with_name(
            f".{self.session_path.name}.{uuid4().hex}.tmp"
        )
        temporary.write_text(json.dumps(self.state), encoding="utf-8")
        if os.name != "nt":
            temporary.chmod(0o600)
        temporary.replace(self.session_path)

    def headers(self, lesson: str = "") -> dict[str, str]:
        """Build request headers without exposing them to observers."""
        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json",
            "xtbz": "ykt",
            "x-client": "h5",
            "Referer": self.origin
            + (
                f"/lesson/fullscreen/v3/{lesson}?source=5"
                if lesson
                else "/web?next=/v2/web/index&type=3"
            ),
            "Cookie": self.state.get("cookie", ""),
        }
        token = self.state.get("tokens", {}).get(lesson)
        if token:
            headers["Authorization"] = "Bearer " + token.removeprefix("Bearer ")
        return headers

    async def request(
        self,
        method: str,
        path: str,
        *,
        lesson: str = "",
        headers: dict | None = None,
        allow_missing_code: bool = False,
        **kwargs,
    ):
        """Check HTTP and business status, and capture Set-Auth rotation."""
        assert self.http is not None
        request_cookie = self.state.get("cookie", "")
        async with self.http.request(
            method,
            self.origin + path,
            headers=self.headers(lesson) if headers is None else headers,
            **kwargs,
        ) as response:
            if response.status in {401, 403}:
                raise LoginRequired(
                    "platform session expired or access denied",
                    response.status,
                    http_status=response.status,
                )
            if response.status >= 400:
                raise ProtocolError(
                    f"platform HTTP {response.status} at {path.split('?')[0]}",
                    http_status=response.status,
                )
            token = response.headers.get("Set-Auth")
            if (token and lesson) or response.cookies:
                self.reload_session()
                if self.state.get("cookie", "") == request_cookie:
                    if token and lesson:
                        self.state.setdefault("tokens", {})[lesson] = token
                    if response.cookies:
                        cookies = SimpleCookie()
                        cookies.load(request_cookie)
                        for key, value in response.cookies.items():
                            cookies[key] = value.value
                        self.state["cookie"] = "; ".join(
                            f"{key}={value.value}"
                            for key, value in cookies.items()
                        )
                    self.save_session()
            data = await response.json(content_type=None)
            if not isinstance(data, dict):
                raise ProtocolError("platform response is not an object")
            if data.get("code") != 0 and not (
                allow_missing_code and "code" not in data
            ):
                code = data.get("code")
                message = str(data.get("msg") or "request rejected")[:160]
                if str(code) in {"401", "50000"} or any(
                    word in message.upper()
                    for word in (
                        "NOT_LOGIN",
                        "UNLOGIN",
                        "SESSION_EXPIRED",
                        "TOKEN_EXPIRED",
                    )
                ):
                    raise LoginRequired(
                        "platform login expired", code, http_status=response.status
                    )
                if path == "/api/v3/lesson/checkin" and code is not None:
                    raise ClassroomEntryRejected(code)
                if "code" in data:
                    raise RejectedSubmission(
                        f"platform code={code}: {message}",
                        code=code,
                        server_msg=message,
                        http_status=response.status,
                    )
                raise ProtocolError(
                    f"platform code={data.get('code')}: {message}"
                )
            return data

    async def probe(self) -> dict:
        """Read the session and classroom list without entering a class."""
        started = time.monotonic()
        results = []
        for name, path in (
            ("session", "/api/v3/user/basic-info"),
            ("lessons", self.lessons_path),
        ):
            try:
                data = await self.request("GET", path)
                item = {"endpoint": name, "ok": True, "code": data.get("code", 0)}
                if name == "lessons":
                    item["active_lessons"] = self._lessons(data)
                results.append(item)
            except PLATFORM_ERRORS as error:
                results.append(
                    {"endpoint": name, "ok": False, "error": type(error).__name__}
                )
        return {
            "ok": all(item["ok"] for item in results),
            "checks": results,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    async def active_lessons(self) -> list[dict]:
        """Read ongoing classrooms without enrolling in unrelated courses."""
        data = await self.request("GET", self.lessons_path)
        return [
            lesson
            for lesson in self._lessons(data)
            if str(lesson.get("lessonId"))
            not in self.config.platform.ignored_lesson_ids
        ]

    @staticmethod
    def _lessons(data):
        lessons = (data.get("data") or {}).get("onLessonClassrooms", [])
        return [
            {
                "lessonId": lesson.get("lessonId")
                or lesson.get("id")
                or lesson.get("lesson_id"),
                "classroomId": lesson.get("classroomId")
                or lesson.get("classroom_id")
                or lesson.get("courseId")
                or lesson.get("course_id"),
                "classroomName": lesson.get("classroomName")
                or lesson.get("name")
                or lesson.get("courseName"),
            }
            for lesson in lessons
        ]

    @property
    def lessons_path(self) -> str:
        """Use the Web classroom discovery endpoint."""
        return "/api/v3/classroom/on-lesson-upcoming-exam"

    async def account_id(self, response: dict | None = None) -> str:
        """Resolve an account ID from the Web profile."""
        if response is None:
            if self.state.get("user_id"):
                return str(self.state["user_id"])
            response = await self.request("GET", "/api/v3/user/basic-info")
        profile = response.get("data") or {}
        if not isinstance(profile, dict):
            raise ProtocolError("account user profile was not an object")
        user_id = str(profile.get("id") or profile.get("user_id") or "")
        if not user_id:
            raise ProtocolError("account profile has no user ID")
        if self.state.get("user_id") != user_id:
            self.state["user_id"] = user_id
            self.save_session()
        return user_id

    async def checkin(self, lesson: str) -> dict:
        """Obtain a classroom connection token through standard Web entry."""
        data = await self.request(
            "POST",
            "/api/v3/lesson/checkin",
            lesson=lesson,
            json={"source": 5, "lessonId": str(lesson)},
        )
        identity = data.get("data") or {}
        if not identity.get("lessonToken") or not identity.get("identityId"):
            raise ProtocolError("checkin lacks lessonToken/identityId")
        return identity

    async def presentation_catalog(self, lesson: str) -> list[dict]:
        """Read the lesson page's catalog with the account Cookie."""
        data = await self.request(
            "GET",
            "/v/lesson/lesson_info_v2",
            params={"lesson_id": lesson},
            headers={
                "Accept": "application/json",
                "Cookie": self.state.get("cookie", ""),
                "Referer": f"{self.origin}/lesson/fullscreen/v3/{lesson}",
                "User-Agent": self.CATALOG_USER_AGENT,
            },
            allow_missing_code=True,
        )
        detail = data.get("data")
        presentations = (
            detail.get("presentationList") if isinstance(detail, dict) else None
        )
        if not isinstance(presentations, list):
            raise ProtocolError("lesson info has no presentationList array")
        result = {}
        for item in presentations:
            if not isinstance(item, dict):
                raise ProtocolError("lesson catalog contains an invalid deck")
            key = str(item.get("id") or item.get("presentationID") or "")
            if not key:
                raise ProtocolError("lesson catalog contains a deck without an ID")
            slides = item.get("slides")
            result[key] = {
                "id": key,
                "title": item.get("Title") or item.get("title") or "",
                "slide_count": len(slides) if isinstance(slides, list) else 0,
            }
        return list(result.values())

    async def presentation(self, lesson: str, presentation: str) -> dict:
        """Fetch original slides and question metadata."""
        data = await self.request(
            "GET",
            "/api/v3/lesson/presentation/fetch",
            lesson=lesson,
            params={"presentation_id": presentation},
        )
        result = data.get("data") or {}
        if not isinstance(result.get("slides"), list):
            raise ProtocolError("presentation has no slides array")
        return result

    async def submit(self, lesson: str, question: str, kind: int, answer: Any):
        """Send once; the caller journals ambiguous transport outcomes."""
        if kind == 5:
            answer = {
                "content": answer["content"],
                "pics": [{"pic": "", "thumb": ""}],
            }
        return await self.request(
            "POST",
            "/api/v3/lesson/problem/answer",
            lesson=lesson,
            json={
                "problemId": question,
                "problemType": kind,
                "dt": int(time.time() * 1000),
                "result": answer,
            },
        )

    def classroom_hello(self, lesson: str, identity: dict) -> dict:
        """Authenticate the Web classroom connection."""
        return {
            "op": "hello",
            "role": "student",
            "lessonid": lesson,
            "userid": str(identity["identityId"]),
            "auth": identity["lessonToken"],
        }

    async def websocket(self, *, lesson: str = ""):
        """Open the Web gateway with a heartbeat and private cookie."""
        assert self.http is not None
        scheme = "wss" if self.origin.startswith("https:") else "ws"
        url = scheme + "://" + urlsplit(self.origin).netloc + "/wsapp/"
        headers = self.headers()
        headers["Origin"] = self.origin
        return await self.http.ws_connect(url, headers=headers, heartbeat=5)

    async def login(self, on_ticket):
        """Obtain a QR ticket and exchange the scan result for a cookie."""
        assert self.http is not None
        async with await self.websocket() as ws:
            await ws.send_json(
                {
                    "op": "requestlogin",
                    "role": "web",
                    "version": 1.4,
                    "type": "qrcode",
                    "from": "web",
                }
            )
            await asyncio.wait_for(self._login_messages(ws, on_ticket), 120)
            return

    async def _login_messages(self, ws, on_ticket):
        assert self.http is not None
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(message.data)
            if data.get("ticket"):
                await on_ticket(str(data["ticket"]))
            if data.get("UserID") and data.get("Auth"):
                async with self.http.post(
                    self.origin + "/pc/web_login",
                    headers=self.headers(),
                    json={"UserID": data["UserID"], "Auth": data["Auth"]},
                ) as response:
                    response.raise_for_status()
                    body = await response.json(content_type=None)
                    if body.get("code", 0) != 0 or not response.cookies:
                        raise LoginRequired(
                            "QR exchange returned no session cookie"
                        )
                    self.state["cookie"] = "; ".join(
                        f"{key}={value.value}"
                        for key, value in response.cookies.items()
                    )
                    self.state["tokens"] = {}
                    self.state.pop("user_id", None)
                    self.save_session()
                    return
        raise LoginRequired("QR login ended before completion")
