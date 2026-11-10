"""Headless service checks against real local HTTP/WebSocket boundaries."""

import asyncio
import base64
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from aiohttp import web

from autoykt.server.cache import ClassroomCache
from autoykt.server.client import YuketangClient
from autoykt.server.cli import build_parser, execute
from autoykt.server.config import ServerConfig
from autoykt.server.control import ControlServer
from autoykt.server.events import EventWriter
from autoykt.server.models import Question, SIGNED_SLIDE_CDN, validate_answer
from autoykt.server.courseware import export_courseware
from autoykt.server.service import ClassroomService, LessonState


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jSucAAAAASUVORK5CYII="
)

CAPTURED_EMPTY_LESSON = json.loads(
    (
        Path(__file__).parent
        / "fixtures/server/changjiang_empty_lesson_20261104.json"
    ).read_text(encoding="utf-8")
)


def presentation(origin=""):
    """Use protocol field names from the reference v3 implementation."""
    return {
        "title": "Course",
        "slides": [
            {
                "id": "previous",
                "index": 1,
                "cover": origin + "/image",
                "shapes": [{"Text": "课程定义：正确示例使用选项 B"}],
            },
            {
                "id": "slide-2",
                "index": 2,
                "cover": origin + "/image",
                "problem": {
                    "problemId": "problem-7",
                    "problemType": 1,
                    "body": "根据此前课件选择正确示例",
                    "result": None,
                    "options": [
                        {"key": "A", "value": "甲"},
                        {"key": "B", "value": "乙"},
                    ],
                },
            },
            {
                "id": "future",
                "index": 3,
                "shapes": [{"Text": "未来页面，不应提前成为上下文"}],
            },
        ],
    }


def submission_status(cache, lesson, question):
    """Assert that a persisted journal entry exists before checking its state."""
    record = cache.submission(lesson, question)
    assert record is not None
    return record["status"]


class CacheTest(unittest.TestCase):
    """Validate SQLite persistence, invalidation and cache interchange."""

    def test_server_defaults_to_notifications_without_model_credentials(self):
        config = ServerConfig()
        self.assertEqual(config.answering.mode, "notify")
        self.assertFalse(config.answering.providers)
        self.assertTrue(config.platform.auto_login)

    def test_cache_scope_manual_precedence_and_content_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            cache = ClassroomCache(path, "https://www.yuketang.cn")
            cache.save_presentation("course", "deck", presentation())
            question = cache.questions("course", "deck")[0]
            cache.import_answers(
                "course",
                "deck",
                {"answers": {"2": ["A"]}, "manual_answers": {"2": ["B"]}},
            )
            cache.save_answer(question, ["A"], "ai:model")
            self.assertEqual(cache.answer(question), (["B"], "manual"))
            exported = cache.export_answers("course", "deck")
            self.assertEqual(exported["manual_answers"]["2"], ["B"])
            self.assertEqual(
                [s["id"] for s in cache.previous_slides(question, 3)],
                ["previous"],
            )
            cache.mark_submission("lesson", question, ["B"], "pending")
            cache.close()
            cache = ClassroomCache(path, "https://www.yuketang.cn")
            self.assertEqual(
                submission_status(cache, "lesson", question.question_id),
                "pending",
            )
            self.assertIsNone(
                cache.answer(replace(question, course_id="other"))
            )
            self.assertIsNone(
                cache.answer(replace(question, options={"A": "乙", "B": "甲"}))
            )
            changed_deck = presentation()
            changed_deck["slides"][1]["problem"]["body"] = "题目已改变"
            cache.save_presentation("course", "deck", changed_deck)
            with self.assertRaisesRegex(ValueError, "changed"):
                cache.import_answers("course", "deck", exported)
            cache.close()

    def test_payload_types_are_checked_without_confidence_voting(self):
        question = Question.from_slide("c", "p", presentation()["slides"][1])
        self.assertEqual(validate_answer(question, ["B"]), ["B"])
        for answer in (["A", "B"], ["Z"]):
            with self.assertRaises(ValueError):
                validate_answer(question, answer)
        self.assertEqual(
            validate_answer(replace(question, kind=4, blanks=2), ["1", "2"]),
            ["1", "2"],
        )
        self.assertEqual(
            validate_answer(replace(question, kind=5), "回答"),
            {"content": "回答", "pics": []},
        )

    def test_old_cache_keeps_the_answer_when_question_version_returns(self):
        question = Question.from_slide("c", "p", presentation()["slides"][1])
        changed = replace(question, text="更改后的题面")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classroom.sqlite3"
            domain = "https://www.yuketang.cn"
            with sqlite3.connect(path) as old:
                old.execute(
                    "CREATE TABLE answers (domain TEXT, course TEXT, "
                    "presentation TEXT, question TEXT, signature TEXT, "
                    "answer TEXT, source TEXT, updated REAL, "
                    "PRIMARY KEY(domain, course, presentation, question))"
                )
                old.execute(
                    "INSERT INTO answers VALUES (?,?,?,?,?,?,?,?)",
                    (
                        domain, "c", "p", question.question_id, question.signature,
                        '["B"]', "manual", 1,
                    ),
                )
            old.close()
            cache = ClassroomCache(path, domain)
            try:
                self.assertIsNone(cache.answer(changed))
                cache.save_answer(changed, ["A"], "ai:model")
                self.assertEqual(cache.answer(changed), (["A"], "ai:model"))
                self.assertEqual(cache.answer(question), (["B"], "manual"))
            finally:
                cache.close()


class ProtocolServiceFixture:
    """Loopback transport and model fixture shared with optional adapters."""

    config_type = ServerConfig
    client_type = YuketangClient
    service_type = ClassroomService
    control_type = ControlServer
    lesson_paths = ("/api/v3/classroom/on-lesson-upcoming-exam",)

    def extra_platform_settings(self):
        return {}

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.posts, self.events, self.prompts = [], [], []
        self.model_started, self.release_model = (
            asyncio.Event(),
            asyncio.Event(),
        )
        self.detected, self.accepted = asyncio.Event(), asyncio.Event()
        self.ready, self.qr_delivered = asyncio.Event(), asyncio.Event()
        self.image_delivered, self.release_image = (
            asyncio.Event(),
            asyncio.Event(),
        )
        self.block_image = False
        self.classroom_sockets = []
        self.allow_scan, self.probe_failed = asyncio.Event(), asyncio.Event()
        self.fetch_started, self.disconnected = asyncio.Event(), asyncio.Event()
        self.revalidation_started, self.release_revalidation = (
            asyncio.Event(),
            asyncio.Event(),
        )
        self.entry_reported, self.heartbeat_received = (
            asyncio.Event(),
            asyncio.Event(),
        )
        self.login_required = asyncio.Event()
        self.entry_code = 0
        self.hello_auth_failed = False
        self.user_reads = 0
        self.rotate_profile_cookie = False
        self.captured_lesson = False
        self.connected = asyncio.Event()
        self.no_lessons = self.profile_missing_id = False
        self.web_profile_expired = False
        self.web_profile_unavailable = False
        self.catalog = []
        self.catalog_reads = 0
        self.catalog_rejected = False
        self.decks = {}
        self.fetch_rejected = False
        self.fetch_reads = 0
        self.block_revalidation = False
        self.bad_model_answer = False
        self.require_auth = self.delay_scan = False
        self.bad_handshake = self.reject_checkin = self.block_probes = False
        self.qr_requests = 0
        self.block_model = self.reject_submit = self.disconnect_submit = False
        self.change_on_model = False
        self.webhook_failures = self.checkins = 0
        self.persist_submitted = False
        self.current: dict = {}
        app = web.Application()
        app.router.add_get("/api/v3/user/basic-info", self.user)
        for path in self.lesson_paths:
            app.router.add_get(path, self.lessons)
        app.router.add_post("/api/v3/lesson/checkin", self.checkin)
        app.router.add_get("/api/v3/lesson/presentation/fetch", self.fetch)
        app.router.add_get(
            "/v/lesson/lesson_info_v2", self.presentation_catalog
        )
        app.router.add_post("/api/v3/lesson/problem/answer", self.submit)
        app.router.add_get("/wsapp/", self.websocket)
        app.router.add_get("/image", self.image)
        app.router.add_post("/v1/chat/completions", self.model)
        app.router.add_post("/events", self.webhook)
        app.router.add_post("/pc/web_login", self.exchange_login)
        app.router.add_post("/control/command", self.control_command)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.origin = f"http://127.0.0.1:{port}"
        self.current = presentation(self.origin)
        self.config = self.config_type.model_validate(
            {
                "data_dir": self.temp.name,
                "platform": {
                    "base_url": self.origin,
                    "session_file": str(Path(self.temp.name) / "session.json"),
                    **self.extra_platform_settings(),
                    "poll_seconds": 0.1,
                    "reconnect_seconds": 0.1,
                    "metadata_retry_seconds": 0.03,
                    "metadata_wait_seconds": 1,
                },
                "answering": {
                    "mode": "auto",
                    "precompute": False,
                    "timeout_seconds": 2,
                    "providers": [
                        {
                            "name": "local",
                            "base_url": self.origin + "/v1",
                            "api_key": "local-model-key",
                            "models": ["test"],
                            "input_mode": "vision",
                        }
                    ],
                },
                "webhook": {"url": self.origin + "/events", "attempts": 2},
            }
        )
        self.writer = EventWriter(self.config)
        await self.writer.start()
        self.client = self.client_type(self.config)
        self.client.state["cookie"] = "sessionid=local-cookie"
        self.client.save_session()
        await self.client.__aenter__()
        self.service = self.service_type(self.config, self.client, self.writer)
        self.controller = self.control_type(self.service)
        self.controller.token = "fixture-control"
        self.addAsyncCleanup(self.cleanup_service)

    async def cleanup_service(self):
        self.release_model.set()
        self.release_image.set()
        self.release_revalidation.set()
        await self.service.close()
        await self.writer.close()
        await self.client.__aexit__()
        await self.runner.cleanup()

    async def user(self, _):
        self.user_reads += 1
        if self.web_profile_unavailable:
            return web.json_response({"code": 900}, status=503)
        if self.web_profile_expired and "renewed-cookie" not in _.headers.get(
            "Cookie", ""
        ):
            return web.json_response({"code": 50000})
        if self.captured_lesson:
            return web.json_response(CAPTURED_EMPTY_LESSON["account_response"])
        profile = {"name": "test"}
        if not self.profile_missing_id:
            profile["id"] = "student"
        response = web.json_response({"code": 0, "data": profile})
        if self.rotate_profile_cookie:
            response.set_cookie("csrftoken", "sample-csrf")
        return response

    async def lessons(self, _):
        if (
            self.require_auth
            and _.headers.get("Cookie") != "sessionid=renewed-cookie"
        ):
            return web.json_response({"code": 401}, status=401)
        if self.no_lessons:
            return web.json_response(
                {"code": 0, "data": {"onLessonClassrooms": []}}
            )
        return web.json_response(
            {
                "code": 0,
                "data": {
                    "onLessonClassrooms": [
                        {
                            "lessonId": "lesson-1",
                            "classroomId": "course-1",
                            "classroomName": "Test",
                        }
                    ]
                },
            }
        )

    async def checkin(self, request):
        self.checkins += 1
        payload = await request.json()
        self.assertEqual(payload, {"source": 5, "lessonId": "lesson-1"})
        if self.entry_code:
            return web.json_response({"code": self.entry_code})
        if (
            self.require_auth
            and request.headers.get("Cookie") != "sessionid=renewed-cookie"
        ):
            return web.json_response({"code": 401}, status=401)
        if self.reject_checkin:
            return web.json_response({"code": 900, "msg": "checkin rejected"})
        if self.captured_lesson:
            return web.json_response(
                CAPTURED_EMPTY_LESSON["checkin_response"],
                headers={"set-auth": CAPTURED_EMPTY_LESSON["checkin_set_auth"]},
            )
        identity = {"lessonToken": "ws-auth", "identityId": "student"}
        return web.json_response(
            {
                "code": 0,
                "data": identity,
            },
            headers={"Set-Auth": "lesson-token"},
        )

    async def fetch(self, request):
        self.fetch_started.set()
        self.fetch_reads += 1
        if self.block_revalidation and self.fetch_reads >= 2:
            self.revalidation_started.set()
            await self.release_revalidation.wait()
        if request.headers.get("Authorization"):
            self.assertEqual(
                request.headers["Authorization"], "Bearer lesson-token"
            )
        if self.fetch_rejected:
            return web.json_response(
                {"code": 900, "msg": "presentation not visible"}
            )
        return web.json_response(
            {
                "code": 0,
                "data": self.decks.get(
                    request.query.get("presentation_id"), self.current
                ),
            }
        )

    async def presentation_catalog(self, request):
        self.catalog_reads += 1
        self.assertEqual(request.query["lesson_id"], "lesson-1")
        self.assertEqual(request.headers["Cookie"], self.client.state["cookie"])
        self.assertEqual(
            request.headers["Referer"], self.origin + "/lesson/fullscreen/v3/lesson-1"
        )
        self.assertNotIn("Authorization", request.headers)
        self.assertNotIn("x-client", request.headers)
        if self.catalog_rejected:
            return web.json_response({"code": 900}, status=500)
        # The catalog can return a data envelope without a business code.
        return web.json_response({"data": {"presentationList": self.catalog}})

    async def submit(self, request):
        self.posts.append(await request.json())
        if self.reject_submit:
            return web.json_response({"code": 900, "msg": "closed"})
        if self.disconnect_submit:
            request.transport.close()
            return web.Response()
        self.accepted.set()
        if self.persist_submitted:
            self.current["slides"][1]["problem"]["result"] = self.posts[-1][
                "result"
            ]
        return web.json_response({"code": 0, "msg": "OK"})

    async def image(self, _):
        if self.block_image:
            await self.release_image.wait()
        return web.Response(body=PNG, content_type="image/png")

    async def websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        hello = await ws.receive_json()
        if hello.get("op") == "requestlogin":
            self.qr_requests += 1
            self.assertEqual(hello["version"], 1.4)
            await ws.send_json({"ticket": self.origin + "/image"})
            if self.delay_scan:
                await self.allow_scan.wait()
            await ws.send_json({"UserID": "student", "Auth": "qr-auth"})
            await ws.close()
            return ws
        self.assertEqual(hello["auth"], "ws-auth")
        self.assertEqual(request.headers["Origin"], self.origin)
        if self.hello_auth_failed:
            await ws.send_json({"op": "hello", "status": False})
            await ws.close()
            return ws
        if self.bad_handshake:
            await ws.send_json({"op": "error", "msg": "rejected"})
            await ws.close()
            return ws
        self.classroom_sockets.append(ws)
        if self.captured_lesson:
            await ws.send_json(CAPTURED_EMPTY_LESSON["received_hello"])
            await ws.send_json(CAPTURED_EMPTY_LESSON["received_notification"])
        else:
            await ws.send_json(
                {"op": "hello", "presentation": "deck-1", "timeline": []}
            )
            await ws.send_json(
                {"op": "unlockproblem", "problem": self.unlock()}
            )
        async for message in ws:
            self.assertEqual(
                json.loads(message.data),
                {"op": "heartbeat", "lessonid": "lesson-1"},
            )
            self.heartbeat_received.set()
        return ws

    async def exchange_login(self, request):
        self.assertEqual(
            await request.json(), {"UserID": "student", "Auth": "qr-auth"}
        )
        response = web.json_response({"code": 0})
        response.set_cookie("sessionid", "renewed-cookie", httponly=True)
        return response

    def unlock(self):
        return {
            "prob": "problem-7",
            "sid": "slide-2",
            "pres": "deck-1",
            "dt": int(time.time() * 1000),
            "limit": 60,
        }

    async def model(self, request):
        body = await request.json()
        self.prompts.append(body)
        self.model_started.set()
        if self.block_model:
            await self.release_model.wait()
        if self.change_on_model:
            self.current["slides"][1]["problem"]["body"] = "新的题目"
        return web.json_response(
            {
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"answer":["Z"]}'
                            if self.bad_model_answer
                            else '{"answer":["B"],"confidence":0.01}',
                        },
                    }
                ],
            }
        )

    async def webhook(self, request):
        event = await request.json()
        self.assertEqual(event["event_id"], request.headers["Idempotency-Key"])
        if event["type"] == "probe_result" and self.block_probes:
            self.probe_failed.set()
            return web.Response(status=503)
        if self.webhook_failures:
            self.webhook_failures -= 1
            return web.Response(status=503)
        self.events.append(event)
        if event["type"] == "question_detected":
            self.detected.set()
        if event["type"] == "question_ready":
            self.ready.set()
        if event["type"] == "question_image":
            self.image_delivered.set()
        if event["type"] == "login_qr":
            self.qr_delivered.set()
        if event["type"] == "lesson_disconnected":
            self.disconnected.set()
        if event["type"] == "lesson_connected":
            self.connected.set()
        if event["type"] == "classroom_entry_result":
            self.entry_reported.set()
        if event["type"] == "login_required":
            self.login_required.set()
        return web.Response(status=204)

    async def control_command(self, request):
        return await self.controller.command(request)

    async def command(
        self, action, *, request_id=None, token="fixture-control"
    ):
        async with self.client.http.post(
            self.origin + "/control/command",
            json={
                "request_id": request_id or f"test-{time.time_ns()}",
                "action": action,
            },
            headers={"Authorization": "Bearer " + token},
        ) as response:
            return response.status, await response.json()

    async def run_job(self):
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        await self.service.handle_message(
            state, {"op": "unlockproblem", "problem": self.unlock()}
        )
        await asyncio.gather(*tuple(self.service.jobs.values()))
        await self.writer.flush()
        return state


class ProtocolServiceTest(
    ProtocolServiceFixture, unittest.IsolatedAsyncioTestCase
):
    """Exercise the public Web answer service end to end."""

    async def test_standalone_cli_reads_back_an_answer_without_integrations(self):
        directory = Path(self.temp.name)
        config = self.config.model_copy(deep=True)
        config.data_dir = str(directory / "standalone")
        config.webhook.url = None
        config.webhook.inline_question_images = False
        config.control.enabled = False
        config.platform.poll_seconds = 5
        config.answering.providers[0].api_key_env = "AUTOYKT_LOOPBACK_MODEL_KEY"
        path = directory / "standalone.yaml"
        path.write_text(config.model_dump_json(), encoding="utf-8")
        self.persist_submitted = True
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-m",
            "autoykt.server",
            "--config",
            str(path),
            "run",
            cwd=directory,
            env={**os.environ, "AUTOYKT_LOOPBACK_MODEL_KEY": "fixture-key"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def await_readback():
            events = config.storage / "events.jsonl"
            while process.returncode is None:
                if events.exists():
                    for line in events.read_text(encoding="utf-8").splitlines():
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if event["type"] == "submission_verified":
                            return event["payload"]["matched"]
                await asyncio.sleep(0.05)
            return False

        try:
            self.assertTrue(await asyncio.wait_for(await_readback(), 10))
            self.assertEqual([item["result"] for item in self.posts], [["B"]])
            self.assertFalse(self.events)
            self.assertEqual(self.checkins, 1)
        finally:
            if process.returncode is None:
                process.terminate()
            await process.communicate()

    async def test_captured_classroom_identity_and_empty_timeline(self):
        self.config.answering.mode = "notify"
        self.config.diagnostics.capture_until = datetime.now(
            timezone.utc
        ) + timedelta(minutes=5)
        self.captured_lesson = True
        running = asyncio.create_task(self.service.run())
        try:
            await asyncio.wait_for(self.connected.wait(), 3)
            state = self.service.states["lesson-1"]
            self.assertEqual(state.connection, "connected")
            self.assertEqual(state.presentation_id, "")
            self.assertFalse(self.detected.is_set())
            self.assertFalse(self.fetch_started.is_set())
            self.assertFalse(self.posts)
            self.assertFalse(self.prompts)
            trace = (self.config.storage / "protocol-shapes.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(json.loads(trace.splitlines()[0])["op"], "hello")
            self.assertNotIn("ws-auth", trace)
        finally:
            self.service.stop.set()
            await running

    async def test_web_hello_matches_the_captured_browser_frame(self):
        capture = CAPTURED_EMPTY_LESSON
        self.assertEqual(
            self.client.classroom_hello(
                capture["lesson_id"], capture["checkin_response"]["data"]
            ),
            capture["sent_hello"],
        )

    async def test_hello_auth_failure_never_reports_connected(self):
        self.config.platform.auto_login = False
        self.hello_auth_failed = True
        running = asyncio.create_task(self.service.run())
        try:
            await asyncio.wait_for(self.login_required.wait(), 3)
            self.assertFalse(
                any(e["type"] == "lesson_connected" for e in self.events)
            )
            self.assertFalse(self.detected.is_set())
            self.assertEqual(
                self.service.states["lesson-1"].connection, "login_required"
            )
        finally:
            self.service.stop.set()
            await running

    async def test_slow_image_download_does_not_delay_question_details(self):
        self.config.answering.mode = "notify"
        self.block_image = True
        await self.run_job()
        self.assertTrue(self.detected.is_set())
        self.assertTrue(self.ready.is_set())
        self.assertFalse(self.image_delivered.is_set())
        self.release_image.set()
        await asyncio.wait_for(self.image_delivered.wait(), 3)
        event = next(e for e in self.events if e["type"] == "question_image")
        self.assertEqual(
            base64.b64decode(event["payload"]["image_base64"]), PNG
        )
        self.assertEqual(event["payload"]["image_mime"], "image/png")
        self.assertFalse(self.posts)
        self.assertFalse(self.prompts)

    async def test_unreadable_image_preserves_alert_and_official_lesson_link(
        self,
    ):
        self.config.answering.mode = "notify"
        self.current["slides"][1]["cover"] = self.origin + "/missing"
        await self.run_job()
        await asyncio.gather(*tuple(self.service.background))
        await self.writer.flush()
        event = next(
            e for e in self.events if e["type"] == "question_image_unavailable"
        )
        self.assertTrue(self.ready.is_set())
        self.assertTrue(event["payload"]["lesson_url"].endswith("/lesson-1"))
        self.assertFalse(self.posts)

    async def test_notify_command_requires_a_real_http_acknowledgement(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = await execute(
                build_parser().parse_args(["notify-test"]), self.config
            )
        report = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertTrue(report["webhook_accepted"])
        self.assertEqual(report["phone_receipt"], "confirm_on_phone")
        self.assertEqual(
            sum(e["event_id"] == report["event_id"] for e in self.events), 1
        )
        self.assertEqual(self.checkins, 0)
        self.assertFalse(self.prompts)

    async def test_filtered_test_event_is_not_reported_as_delivered(self):
        self.config.webhook.events = ["question_detected"]
        with self.assertRaisesRegex(ValueError, "excludes notification_test"):
            await execute(
                build_parser().parse_args(["notify-test"]), self.config
            )

    async def test_expired_event_is_not_reported_as_acknowledged(self):
        event = await self.writer.emit(
            "login_qr", payload={"expires_at": time.time() - 1}
        )
        await self.writer.flush()
        with self.assertRaises(asyncio.TimeoutError):
            await self.writer.wait_delivered(event["event_id"], timeout=0.1)

    async def test_phone_login_after_connection_restarts_existing_listener(
        self,
    ):
        self.config.answering.mode = "notify"
        running = asyncio.create_task(self.service.run())
        try:
            await asyncio.wait_for(self.ready.wait(), 3)
            self.require_auth = self.delay_scan = True
            await self.classroom_sockets[0].close()
            await asyncio.wait_for(self.qr_delivered.wait(), 3)
            self.allow_scan.set()

            async def reconnected():
                while (
                    sum(e["type"] == "lesson_connected" for e in self.events)
                    < 2
                    or self.service.states["lesson-1"].connection != "connected"
                ):
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(reconnected(), 3)
            self.assertEqual(self.qr_requests, 1)
            self.assertFalse(self.posts)
            self.assertFalse(self.prompts)
        finally:
            self.allow_scan.set()
            self.service.stop.set()
            await running

    async def test_incomplete_question_is_forwarded_after_metadata_recovers(
        self,
    ):
        self.config.answering.mode = "notify"
        original = self.current
        self.current = {"slides": [], "datacompleted": False}
        pending = asyncio.create_task(self.run_job())
        await asyncio.wait_for(self.detected.wait(), 3)
        await asyncio.wait_for(self.fetch_started.wait(), 3)
        self.assertFalse(self.ready.is_set())
        self.current = original
        await asyncio.wait_for(pending, 3)
        messages = [e for e in self.events if e["type"] == "question_ready"]
        self.assertEqual(
            messages[0]["payload"]["options"], {"A": "甲", "B": "乙"}
        )
        self.assertTrue(messages[0]["payload"]["image_url"].endswith("/image"))
        await self.run_job()
        self.assertEqual(
            sum(e["type"] == "question_detected" for e in self.events), 1
        )
        self.assertEqual(
            sum(e["type"] == "question_ready" for e in self.events), 1
        )
        self.assertFalse(self.posts)
        self.assertFalse(self.prompts)

    async def test_phone_scan_renews_expired_login_and_resumes_listening(self):
        self.config.answering.mode = "notify"
        self.require_auth = self.delay_scan = True
        running = asyncio.create_task(self.service.run())
        try:
            await asyncio.wait_for(self.qr_delivered.wait(), 3)
            self.service.login.request()
            self.service.login.request()
            qr = next(e for e in self.events if e["type"] == "login_qr")
            self.assertEqual(
                base64.b64decode(qr["payload"]["image_base64"]), PNG
            )
            self.assertGreater(qr["payload"]["expires_at"], time.time())
            self.assertEqual(self.qr_requests, 1)
            self.assertFalse(self.detected.is_set())
            self.allow_scan.set()
            await asyncio.wait_for(self.ready.wait(), 3)
            self.assertEqual(
                self.client.state["cookie"], "sessionid=renewed-cookie"
            )
            self.assertEqual(
                self.service.states["lesson-1"].connection, "connected"
            )
            self.assertFalse(self.posts)
            self.assertFalse(self.prompts)
        finally:
            self.allow_scan.set()
            self.service.stop.set()
            await running

    async def test_rejected_hello_does_not_report_a_healthy_classroom(self):
        self.config.answering.mode = "notify"
        self.bad_handshake = True
        running = asyncio.create_task(self.service.run())
        try:
            await asyncio.wait_for(self.disconnected.wait(), 3)
            status = json.loads(
                (self.config.storage / "status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["status"], "degraded")
            self.assertFalse(
                any(e["type"] == "lesson_connected" for e in self.events)
            )
        finally:
            self.service.stop.set()
            await running

    async def test_checkin_failure_is_degraded_despite_successful_discovery(
        self,
    ):
        self.reject_checkin = True
        running = asyncio.create_task(self.service.run())
        try:
            await asyncio.wait_for(self.entry_reported.wait(), 3)
            status = json.loads(
                (self.config.storage / "status.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(status["status"], "degraded")
            self.assertFalse(self.detected.is_set())
        finally:
            self.service.stop.set()
            await running

    async def test_durable_outbox_survives_a_sender_restart(self):
        await self.writer.close()
        writer = EventWriter(self.config)
        event = await writer.emit("question_detected", question_id="restart")
        await writer.close()
        self.writer = EventWriter(self.config)
        self.service.events = self.writer
        await self.writer.start()
        await self.writer.flush()
        self.assertEqual(
            sum(e["event_id"] == event["event_id"] for e in self.events), 1
        )

    async def test_question_delivery_overtakes_a_delayed_probe_retry(self):
        self.block_probes = True
        await self.writer.emit("probe_result", payload={"ok": True})
        await asyncio.wait_for(self.probe_failed.wait(), 3)
        await self.writer.emit("question_detected", question_id="urgent")
        await asyncio.wait_for(self.detected.wait(), 0.8)
        self.assertFalse(any(e["type"] == "probe_result" for e in self.events))
        self.block_probes = False
        await self.writer.flush()

    async def test_expired_qr_is_not_replayed_to_the_phone(self):
        await self.writer.emit(
            "login_qr",
            payload={
                "image_url": "https://example.invalid/expired",
                "expires_at": time.time() - 1,
            },
        )
        await self.writer.flush()
        await asyncio.sleep(0.05)
        self.assertFalse(self.qr_delivered.is_set())

    async def test_two_senders_claim_a_delivery_only_once(self):
        other = EventWriter(self.config)
        await other.start()
        try:
            event = await self.writer.emit(
                "question_detected", question_id="claim"
            )
            other.changed.set()
            await self.writer.flush()
            self.assertEqual(
                sum(e["event_id"] == event["event_id"] for e in self.events), 1
            )
        finally:
            await other.close()

    async def test_short_payload_uses_reference_shape(self):
        await self.client.submit(
            "lesson-1", "short", 5, {"content": "answer", "pics": []}
        )
        self.assertEqual(
            self.posts[0]["result"],
            {"content": "answer", "pics": [{"pic": "", "thumb": ""}]},
        )

    async def test_probe_is_read_only_and_emits_no_secrets(self):
        result = await self.client.probe()
        await self.writer.emit("probe_result", payload=result)
        await self.writer.flush()
        self.assertTrue(result["ok"])
        self.assertEqual(self.checkins, 0)
        self.assertFalse(self.posts)
        self.assertFalse(self.prompts)
        self.assertNotIn(
            "local-cookie", self.writer.path.read_text(encoding="utf-8")
        )

    async def test_completed_lesson_can_be_excluded_without_excluding_course(
        self,
    ):
        self.config.platform.ignored_lesson_ids = ["lesson-1"]
        self.assertEqual(await self.client.active_lessons(), [])
        self.config.platform.ignored_lesson_ids = ["previous-lesson"]
        self.assertEqual(
            (await self.client.active_lessons())[0]["lessonId"], "lesson-1"
        )
        self.assertFalse(self.config.platform.classroom_ids)

    async def test_full_discovery_websocket_ai_post_and_reminder_before_ai(
        self,
    ):
        self.block_model = True
        run = asyncio.create_task(self.service.run())
        try:
            await asyncio.wait_for(self.detected.wait(), 3)
            await asyncio.wait_for(self.model_started.wait(), 3)
            self.assertFalse(self.posts)
            self.release_model.set()
            await asyncio.wait_for(self.accepted.wait(), 3)
            await asyncio.sleep(0.05)
            self.assertEqual(self.posts[0]["problemId"], "problem-7")
            self.assertEqual(self.posts[0]["result"], ["B"])
            prompt = self.prompts[0]["messages"][0]["content"]
            self.assertIn("课程定义", prompt[0]["text"])
            self.assertNotIn("未来页面", prompt[0]["text"])
            self.assertEqual(len(prompt), 3)
            self.assertEqual(
                submission_status(self.service.cache, "lesson-1", "problem-7"),
                "accepted",
            )
            self.assertEqual(
                self.client.state["tokens"]["lesson-1"], "lesson-token"
            )
        finally:
            self.service.stop.set()
            await run

    async def test_changed_question_is_not_submitted(self):
        self.change_on_model = True
        await self.run_job()
        self.assertFalse(self.posts)
        self.assertTrue(any(e["type"] == "answer_failed" for e in self.events))

    async def test_unattended_submission_reads_back_result_without_second_post(
        self,
    ):
        self.config.answering.auto_from = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
        self.config.answering.auto_until = datetime.now(
            timezone.utc
        ) + timedelta(minutes=1)
        self.persist_submitted = True
        fetch = self.client.presentation

        async def signed_presentation(lesson, presentation_id):
            data = await fetch(lesson, presentation_id)
            data["slides"][1]["cover"] = (
                SIGNED_SLIDE_CDN
                + f"quiz.png?e={self.fetch_reads}&token=fixture"
            )
            return data

        with (
            patch.object(
                self.client, "presentation", side_effect=signed_presentation
            ),
            patch.object(
                self.service.assets, "image", return_value=(PNG, "image/png")
            ),
        ):
            await self.run_job()
            await asyncio.gather(*tuple(self.service.background))
        await self.writer.flush()
        reports = [e for e in self.events if e["type"] == "submission_verified"]
        self.assertTrue(reports[-1]["payload"]["matched"])
        self.assertEqual(len(self.posts), 1)

    async def test_unattended_rejection_retains_stage_and_platform_reason(self):
        self.reject_submit = True
        await self.run_job()
        report = [e for e in self.events if e["type"] == "answer_failed"][-1][
            "payload"
        ]
        self.assertEqual(report["stage"], "submission")
        self.assertEqual(report["code"], 900)
        self.assertEqual(report["server_msg"], "closed")
        self.assertEqual(len(self.posts), 1)

    async def test_unattended_answer_never_posts_outside_automatic_window(self):
        self.config.answering.auto_until = datetime.now(
            timezone.utc
        ) - timedelta(seconds=1)
        await self.run_job()
        self.assertFalse(self.posts)
        self.assertFalse(
            [e for e in self.events if e["type"] == "answer_ready"][-1][
                "payload"
            ]["auto_submit"]
        )
        self.assertTrue(
            any(
                e["type"] == "submission_skipped"
                and e["payload"]["reason"] == "outside_automatic_window"
                for e in self.events
            )
        )

    async def test_cache_hit_avoids_ai_and_duplicate_submission(self):
        self.service.cache.save_presentation("course-1", "deck-1", self.current)
        question = self.service.cache.questions("course-1", "deck-1")[0]
        self.service.cache.save_answer(question, ["A"], "manual")
        await self.run_job()
        await self.run_job()
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0]["result"], ["A"])
        self.assertFalse(self.prompts)

    async def test_explicit_rejection_is_not_reported_as_success(self):
        self.reject_submit = True
        await self.run_job()
        self.assertEqual(
            submission_status(self.service.cache, "lesson-1", "problem-7"),
            "rejected",
        )
        self.assertFalse(
            any(e["type"] == "submission_result" for e in self.events)
        )

    async def test_uncertain_post_is_journaled_and_not_repeated(self):
        self.disconnect_submit = True
        await self.run_job()
        self.assertEqual(
            submission_status(self.service.cache, "lesson-1", "problem-7"),
            "uncertain",
        )
        await self.run_job()
        self.assertEqual(len(self.posts), 1)

    async def test_notify_mode_never_calls_ai_or_submits(self):
        self.config.answering.mode = "notify"
        await self.run_job()
        self.assertTrue(self.detected.is_set())
        self.assertFalse(self.posts)
        self.assertFalse(self.prompts)

    async def test_webhook_retries_keep_event_identity(self):
        self.webhook_failures = 1
        event = await self.writer.emit("probe_result", payload={"ok": True})
        await asyncio.wait_for(self.writer.flush(), 3)
        self.assertEqual(self.events[-1]["event_id"], event["event_id"])

    async def test_qr_login_persists_a_session_and_running_client_can_reload(
        self,
    ):
        tickets = []

        async def receive_ticket(ticket):
            tickets.append(ticket)

        async with YuketangClient(self.config) as login_client:
            await login_client.login(receive_ticket)
        self.assertEqual(tickets, [self.origin + "/image"])
        self.client.reload_session()
        self.assertEqual(
            self.client.state["cookie"], "sessionid=renewed-cookie"
        )

    async def test_precomputed_answer_is_used_without_another_model_call(self):
        self.config.answering.precompute = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        self.service.states[state.lesson_id] = state
        self.service.cache.save_presentation("course-1", "deck-1", self.current)
        self.service.start_precompute()
        assert self.service.warmer is not None
        await self.service.warmer
        self.assertEqual(len(self.prompts), 1)
        await self.run_job()
        self.assertEqual(len(self.prompts), 1)
        self.assertEqual(self.posts[0]["result"], ["B"])

    async def test_catalog_prepares_every_deck_and_waits_for_publication(self):
        self.config.answering.precompute = True
        second = presentation(self.origin)
        second["slides"][1]["problem"]["problemId"] = "problem-8"
        self.decks = {"deck-1": self.current, "deck-2": second}
        self.catalog = [
            {"id": "deck-1", "Title": "One"},
            {"presentationID": "deck-2", "title": "Two"},
        ]
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        self.service.states[state.lesson_id] = state
        await self.service.collect_courseware(state)
        assert self.service.warmer is not None
        await asyncio.wait_for(self.service.warmer, 4)
        self.assertEqual(
            self.service.cache.presentation_ids("course-1"),
            ["deck-1", "deck-2"],
        )
        self.assertEqual(
            len(
                self.service.cache.presentation("course-1", "deck-2")["slides"]
            ),
            3,
        )
        self.assertEqual(len(self.prompts), 2)
        self.assertFalse(self.posts)
        question = self.service.cache.questions("course-1", "deck-2")[0]
        with self.assertRaisesRegex(ValueError, "not released"):
            await self.service.commit(state, question, ["B"], "manual")
        await self.service.handle_message(
            state,
            {
                "op": "problemon",
                "problem": {
                    **self.unlock(),
                    "prob": "problem-8",
                    "pres": "deck-2",
                },
            },
        )
        await asyncio.gather(*tuple(self.service.jobs.values()))
        self.assertEqual(len(self.prompts), 2)
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0]["problemId"], "problem-8")
        exported = export_courseware(
            self.service.cache,
            self.service.assets,
            "course-1",
            Path(self.temp.name) / "export",
        )
        self.assertEqual(len(exported["presentations"]), 2)
        self.assertEqual(
            exported["presentations"][0]["images"][0]["file"], "slide-0001.png"
        )

    async def test_unavailable_catalog_and_hidden_questions_keep_live_capture(
        self,
    ):
        self.config.answering.precompute = True
        self.catalog_rejected = True
        self.current = {"slides": [{"id": "future", "hasProblem": True}]}
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        self.service.states[state.lesson_id] = state
        await self.service.collect_courseware(state)
        if self.service.warmer:
            await self.service.warmer
        self.assertFalse(self.prompts)
        self.current = presentation(self.origin)
        await self.service.handle_message(
            state, {"op": "unlockproblem", "problem": self.unlock()}
        )
        await asyncio.gather(*tuple(self.service.jobs.values()))
        await self.writer.flush()
        self.assertEqual(self.posts[0]["result"], ["B"])
        fallback = [
            event
            for event in self.events
            if event["type"] == "courseware_unavailable"
        ]
        self.assertEqual(fallback[0]["payload"]["fallback"], "live_capture")

    async def test_catalog_http_failure_backs_off_and_keeps_prepared_deck(self):
        self.config.answering.precompute = True
        self.catalog_rejected = True
        self.client.state["tokens"]["lesson-1"] = "lesson-token"
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        self.service.states[state.lesson_id] = state
        for _ in range(3):
            await self.service.collect_courseware(state)
        self.assertIsNotNone(self.service.warmer)
        await self.service.warmer
        self.assertEqual(self.catalog_reads, 1)
        self.assertEqual(self.fetch_reads, 1)
        await self.service.collect_courseware(state, force=True)
        if self.service.warmer:
            await self.service.warmer
        await self.writer.flush()
        self.assertEqual(self.catalog_reads, 2)
        self.assertEqual(self.fetch_reads, 2)
        self.assertEqual(len(self.prompts), 1)
        failures = [
            e for e in self.events if e["type"] == "courseware_unavailable"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["payload"]["http_status"], 500)
        self.assertEqual(
            sum(e["type"] == "courseware_updated" for e in self.events), 1
        )
        self.assertFalse(self.posts)

    async def test_released_inline_question_survives_deck_api_refusal(self):
        self.fetch_rejected = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        await self.service.handle_message(
            state,
            {
                "op": "problemon",
                "problem": {
                    **self.unlock(),
                    "problemType": 1,
                    "body": "实时题干",
                    "options": [
                        {"key": "A", "value": "甲"},
                        {"key": "B", "value": "乙"},
                    ],
                },
            },
        )
        await asyncio.gather(*tuple(self.service.jobs.values()))
        self.assertEqual(self.posts[0]["result"], ["B"])
        self.assertEqual(len(self.prompts), 1)

    async def test_teacher_closing_question_cancels_ai_and_blocks_post(self):
        self.block_model = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        await self.service.handle_message(
            state, {"op": "unlockproblem", "problem": self.unlock()}
        )
        await asyncio.wait_for(self.model_started.wait(), 3)
        tasks = tuple(self.service.jobs.values())
        await self.service.handle_message(
            state,
            {
                "op": "problemfinished",
                "prob": "problem-7",
                "problem": self.unlock(),
            },
        )
        self.release_model.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.assertFalse(self.posts)
        question = self.service.cache.questions("course-1", "deck-1")[0]
        with self.assertRaisesRegex(ValueError, "closed"):
            await self.service.commit(state, question, ["B"], "manual")

    async def test_later_deck_is_prepared_when_first_model_call_is_running(
        self,
    ):
        self.block_model = True
        self.config.answering.precompute = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        self.service.states[state.lesson_id] = state
        self.service.cache.save_presentation("course-1", "deck-1", self.current)
        self.service.start_precompute()
        await asyncio.wait_for(self.model_started.wait(), 3)
        second = presentation(self.origin)
        second["slides"][1]["problem"]["problemId"] = "problem-8"
        self.decks["deck-2"] = second
        await self.service.fetch(state, "deck-2")
        self.release_model.set()
        assert self.service.warmer is not None
        await asyncio.wait_for(self.service.warmer, 4)
        question = self.service.cache.questions("course-1", "deck-2")[0]
        self.assertEqual(self.service.cache.answer(question)[0], ["B"])
        self.assertEqual(len(self.prompts), 2)
        self.assertFalse(self.posts)

    async def test_control_auth_idempotency_and_persistent_mode(self):
        status, _ = await self.command("mode_assist", token="wrong")
        self.assertEqual(status, 401)
        self.assertEqual(self.config.answering.mode, "auto")
        status, first = await self.command(
            "mode_assist", request_id="owner-command-1"
        )
        self.assertEqual(status, 200)
        self.assertTrue(first["ok"])
        generation = self.service.runtime.generation
        _, repeated = await self.command(
            "mode_auto", request_id="owner-command-1"
        )
        self.assertEqual(repeated, first)
        self.assertEqual(self.config.answering.mode, "assist")
        self.assertEqual(self.service.runtime.generation, generation)
        await self.service.close()
        self.config.answering.mode = "notify"
        self.service = ClassroomService(self.config, self.client, self.writer)
        self.controller = ControlServer(self.service)
        self.controller.token = "fixture-control"
        self.assertEqual(self.config.answering.mode, "assist")

    async def test_switching_to_half_while_ai_runs_prevents_auto_post(self):
        self.block_model = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        await self.service.handle_message(
            state, {"op": "unlockproblem", "problem": self.unlock()}
        )
        await asyncio.wait_for(self.model_started.wait(), 3)
        _, response = await self.command("mode_assist")
        self.assertTrue(response["ok"])
        self.release_model.set()
        await asyncio.gather(*tuple(self.service.jobs.values()))
        await self.writer.flush()
        self.assertFalse(self.posts)
        answers = [
            event for event in self.events if event["type"] == "answer_ready"
        ]
        self.assertFalse(answers[-1]["payload"]["auto_submit"])

    async def test_selecting_auto_applies_only_to_newly_released_questions(
        self,
    ):
        self.config.answering.mode = "assist"
        self.block_model = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        await self.service.handle_message(
            state, {"op": "unlockproblem", "problem": self.unlock()}
        )
        await asyncio.wait_for(self.model_started.wait(), 3)
        await self.command("mode_auto")
        self.release_model.set()
        await asyncio.gather(*tuple(self.service.jobs.values()))
        self.assertFalse(self.posts)
        self.current["slides"][1]["problem"]["problemId"] = "problem-8"
        await self.service.handle_message(
            state,
            {
                "op": "unlockproblem",
                "problem": {**self.unlock(), "prob": "problem-8"},
            },
        )
        await asyncio.gather(*tuple(self.service.jobs.values()))
        self.assertEqual(len(self.posts), 1)
        self.assertEqual(self.posts[0]["problemId"], "problem-8")

    async def test_pause_during_final_revalidation_blocks_the_post(self):
        self.block_revalidation = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        await self.service.handle_message(
            state, {"op": "unlockproblem", "problem": self.unlock()}
        )
        await asyncio.wait_for(self.revalidation_started.wait(), 3)
        _, response = await self.command("pause")
        self.assertTrue(response["ok"])
        self.release_revalidation.set()
        await asyncio.gather(*tuple(self.service.jobs.values()))
        await self.writer.flush()
        self.assertFalse(self.posts)
        self.assertTrue(
            any(event["type"] == "submission_skipped" for event in self.events)
        )

    async def test_extraction_and_preparation_report_real_counts_and_partial_pages(
        self,
    ):
        self.config.answering.mode = "assist"
        self.config.answering.precompute = True
        self.catalog = [{"id": "deck-1"}]
        state = LessonState(
            "lesson-1", "course-1", name="测试课堂", presentation_id="deck-1"
        )
        self.service.states[state.lesson_id] = state
        await self.service.collect_courseware(state)
        if self.service.warmer:
            await asyncio.wait_for(self.service.warmer, 4)
        await self.writer.flush()
        summaries = [
            event["payload"]
            for event in self.events
            if event["type"] == "courseware_summary"
        ]
        self.assertEqual(summaries[-1]["status"], "all_visible")
        self.assertEqual(summaries[-1]["slides"], 3)
        progress = [
            event["payload"]
            for event in self.events
            if event["type"] == "preparation_progress"
        ][-1]
        self.assertEqual(progress["prepared"], 1)
        self.assertEqual(progress["questions"], 1)
        self.assertEqual(progress["remaining"], 0)
        self.assertFalse(self.posts)
        self.catalog[0]["slides"] = [{}] * 10
        await self.service.collect_courseware(state)
        await self.writer.flush()
        summaries = [
            event["payload"]
            for event in self.events
            if event["type"] == "courseware_summary"
        ]
        self.assertEqual(summaries[-1]["status"], "partial")
        _, response = await self.command("status")
        self.assertIn("备答 1", response["message"])

    async def test_preparation_failure_is_counted_in_progress(self):
        self.config.answering.mode = "assist"
        self.config.answering.precompute = True
        self.bad_model_answer = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        self.service.states[state.lesson_id] = state
        self.service.cache.save_presentation("course-1", "deck-1", self.current)
        self.service.start_precompute()
        assert self.service.warmer is not None
        await self.service.warmer
        await self.writer.flush()
        progress = [
            event["payload"]
            for event in self.events
            if event["type"] == "preparation_progress"
        ][-1]
        self.assertEqual(progress["failed"], 1)
        self.assertEqual(progress["prepared"], 0)
        self.assertFalse(self.posts)

    async def test_missing_ai_can_notify_or_submit_an_explicit_fallback(self):
        self.config.answering.providers = []
        await self.run_job()
        self.assertFalse(self.posts)
        self.config.answering.fallback = "first_option"
        await self.run_job()
        self.assertEqual(self.posts[0]["result"], ["A"])
        results = [e for e in self.events if e["type"] == "submission_result"]
        self.assertEqual(
            results[-1]["payload"]["source"], "fallback:first_option"
        )

    async def test_actual_model_timeout_reports_failure_without_submission(
        self,
    ):
        self.block_model = True
        self.config.answering.timeout_seconds = 0.5
        await self.run_job()
        self.assertFalse(self.posts)
        failures = [
            event for event in self.events if event["type"] == "answer_failed"
        ]
        self.assertEqual(failures[-1]["payload"]["reason"], "TimeoutError")

    async def test_new_question_preempts_background_model_request(self):
        self.block_model = True
        self.config.answering.precompute = True
        self.service.solver.limit = asyncio.Semaphore(1)
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        self.service.states[state.lesson_id] = state
        self.service.cache.save_presentation("course-1", "deck-1", self.current)
        self.service.start_precompute()
        await asyncio.wait_for(self.model_started.wait(), 3)
        warmer = self.service.warmer
        await self.service.handle_message(
            state, {"op": "unlockproblem", "problem": self.unlock()}
        )
        await asyncio.wait_for(self.detected.wait(), 3)
        self.release_model.set()
        await asyncio.wait_for(
            asyncio.gather(*tuple(self.service.jobs.values())), 3
        )
        assert warmer is not None
        self.assertTrue(warmer.cancelled())
        self.assertEqual(len(self.posts), 1)

    async def test_lesson_end_stops_background_model_requests(self):
        self.block_model = True
        self.config.answering.precompute = True
        state = LessonState("lesson-1", "course-1", presentation_id="deck-1")
        self.service.states[state.lesson_id] = state
        self.service.cache.save_presentation("course-1", "deck-1", self.current)
        self.service.start_precompute()
        await asyncio.wait_for(self.model_started.wait(), 3)
        warmer = self.service.warmer
        assert warmer is not None
        await self.service.handle_message(state, {"op": "lessonfinished"})
        await asyncio.gather(warmer, return_exceptions=True)
        self.assertTrue(warmer.cancelled())
        self.assertFalse(self.posts)

    async def test_unresolved_submission_survives_service_restart(self):
        self.disconnect_submit = True
        await self.run_job()
        await self.service.close()
        self.service = ClassroomService(self.config, self.client, self.writer)
        await self.run_job()
        self.assertEqual(len(self.posts), 1)
        self.current["slides"][1]["problem"]["result"] = ["B"]
        await self.run_job()
        self.assertEqual(
            submission_status(self.service.cache, "lesson-1", "problem-7"),
            "accepted",
        )
