"""Classroom discovery, realtime reminders and deadline-aware answering."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import time
from typing import TypedDict

import aiohttp

from autoykt.knowledge.store import KnowledgeStore
from autoykt.server.cache import ClassroomCache
from autoykt.server.client import (
    PLATFORM_ERRORS,
    RECOVERY_ERRORS,
    ClassroomEntryRejected,
    LoginRequired,
    ProtocolError,
    RejectedSubmission,
    YuketangClient,
)
from autoykt.server.config import ServerConfig
from autoykt.server.control import RuntimeSettings, SubmissionPaused
from autoykt.server.events import EventWriter
from autoykt.server.login import LoginRecovery
from autoykt.server.models import (
    Question,
    first_field,
    identifier,
    slide_text,
    validate_answer,
)
from autoykt.server.protocol_trace import ProtocolTrace
from autoykt.server.solver import (
    AssetStore,
    FirstAnswerSolver,
    SolverUnavailable,
)


logger = logging.getLogger("autoykt.server")
CLASSROOM_HEARTBEAT_SECONDS = 20
CATALOG_FAILURE_RETRY_SECONDS = 300


class MetadataUnavailable(RuntimeError):
    """The question did not become readable within its available time."""


class QuestionIdentity(TypedDict):
    """Envelope identity passed to event delivery."""

    lesson_id: str
    course_id: str
    question_id: str


@dataclass
class LessonState:
    """Protocol identity of a connected classroom."""

    lesson_id: str
    course_id: str
    name: str = ""
    presentation_id: str = ""
    ended: bool = False
    connection: str = "discovered"
    last_message_at: float | None = None
    entry_code: int | str | None = None
    presentations: set[str] = field(default_factory=set)
    courseware: dict[str, dict] = field(default_factory=dict)
    expected_pages: dict[str, int] = field(default_factory=dict)
    catalog_ok: bool = False
    catalog_retry_after: float = 0
    catalog_error: dict | None = None
    extraction: str = "waiting"
    entry_blocked: bool = False


def question_deadline(data: dict) -> float | None:
    """Translate the platform's epoch milliseconds and seconds limit."""
    limit = float(data.get("limit", -1))
    if limit < 0:
        return None
    started = float(data.get("dt") or time.time())
    if started > 100_000_000_000:
        started /= 1000
    return started + limit


class ClassroomService:
    """Run independent lessons without a browser, display or physical mouse."""

    def __init__(
        self, config: ServerConfig, client: YuketangClient, events: EventWriter
    ):
        self.config = config
        self.client = client
        self.events = events
        self.cache = ClassroomCache(
            config.storage / "classroom.sqlite3", client.origin
        )
        self.runtime = RuntimeSettings(config)
        self.knowledge = KnowledgeStore(config.storage / "knowledge.sqlite3")
        self.assets = AssetStore(config.storage / "images", client)
        self.protocol_trace = ProtocolTrace(config)
        self.solver = FirstAnswerSolver(config.answering)
        self.states: dict[str, LessonState] = {}
        self.discovery_blocked = False
        self.listeners: dict[str, asyncio.Task] = {}
        self.jobs: dict[tuple[str, str], asyncio.Task] = {}
        self.fetches: dict[tuple[str, str], asyncio.Task] = {}
        self.catalogs: dict[str, asyncio.Task] = {}
        self.courseware_jobs: dict[tuple[str, str], asyncio.Task] = {}
        self.catalog_checked: dict[str, float] = {}
        self.courseware_revision = 0
        self.background: set[asyncio.Task] = set()
        self.preparation_failures: set[tuple[str, str, str, str]] = set()
        self.last_preparation_report: dict[str, float] = {}
        self.preparation_reports: dict[str, tuple] = {}
        self.extraction_reports: dict[str, tuple] = {}
        self.warmer: asyncio.Task | None = None
        self.stop = asyncio.Event()
        self.login = LoginRecovery(
            config, client, events, self._login_succeeded
        )
        self.notified: set[tuple[str, str]] = set()
        self.question_details: set[tuple[str, str]] = set()
        self.deadlines: dict[tuple[str, str], float | None] = {}
        self.publications: dict[tuple[str, str], dict] = {}
        self.closed_questions: set[tuple[str, str]] = set()

    async def run(self):
        """Discover active lessons and keep a local heartbeat/status file."""
        await self.events.emit(
            "service_started",
            payload={
                "mode": self.config.answering.mode,
                "entry_mode": self.client.entry_mode,
            },
        )
        login_reported = False
        while not self.stop.is_set():
            if self.discovery_blocked:
                self._write_status("entry_refused")
                await self._pause()
                continue
            try:
                self.client.reload_session()
                if not self.client.state.get("cookie"):
                    raise LoginRequired("no session cookie")
                lessons = await self.client.active_lessons()
                await self.events.emit(
                    "probe_result",
                    payload={
                        "ok": True,
                        "active_lessons": [
                            {
                                k: item.get(k)
                                for k in (
                                    "lessonId",
                                    "classroomId",
                                    "classroomName",
                                )
                            }
                            for item in lessons
                        ],
                    },
                )
                login_reported = False
                allowed = self.config.platform.classroom_ids
                for item in lessons:
                    lesson_id = identifier(item.get("lessonId"))
                    course_id = identifier(item.get("classroomId"))
                    if not lesson_id or (allowed and course_id not in allowed):
                        continue
                    course_id = course_id or f"lesson:{lesson_id}"
                    state = self.states.setdefault(
                        lesson_id,
                        LessonState(
                            lesson_id,
                            course_id,
                            str(item.get("classroomName") or ""),
                        ),
                    )
                    if state.ended or state.entry_blocked:
                        continue
                    task = self.listeners.get(lesson_id)
                    if task is None or task.done():
                        self.listeners[lesson_id] = asyncio.create_task(
                            self.listen(state)
                        )
                    if state.connection == "connected":
                        self.schedule_courseware(state)
                self._write_status("running")
            except LoginRequired as error:
                if await self._reject_discovery_login(error):
                    self._write_status("entry_refused")
                    await self._pause()
                    continue
                if not login_reported:
                    await self.events.emit("login_required")
                    login_reported = True
                self.login.request()
                self._write_status("login_required")
            except PLATFORM_ERRORS as error:
                await self.events.emit(
                    "probe_result",
                    payload={"ok": False, "error": type(error).__name__},
                )
                self._write_status("reconnecting")
            await self._pause()

    async def _pause(self):
        stop = asyncio.create_task(self.stop.wait())
        waiting = {stop}
        if self.login.active:
            assert self.login.task is not None
            waiting.add(self.login.task)
        try:
            await asyncio.wait(
                waiting,
                timeout=self.config.platform.poll_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            stop.cancel()
            await asyncio.gather(stop, return_exceptions=True)

    async def _login_succeeded(self):
        for task in self.listeners.values():
            task.cancel()
        await asyncio.gather(*self.listeners.values(), return_exceptions=True)
        self.listeners.clear()

    def _lesson_url(self, state: LessonState) -> str:
        return f"{self.client.origin}/lesson/fullscreen/v3/{state.lesson_id}"

    def lesson_progress(self, state):
        presentations = set(state.presentations)
        if state.presentation_id:
            presentations.add(state.presentation_id)
        totals = {
            "presentations": len(presentations),
            "slides": 0,
            "questions": 0,
            "prepared": 0,
            "failed": 0,
            "answered": 0,
            "remaining": 0,
            "unpublished_visible": 0,
        }
        for presentation in presentations:
            document = self.cache.presentation(state.course_id, presentation)
            totals["slides"] += len(document.get("slides") or [])
            for question in self.cache.questions(state.course_id, presentation):
                totals["questions"] += 1
                if (
                    state.lesson_id,
                    question.question_id,
                ) not in self.publications and (
                    state.lesson_id,
                    question.slide_id,
                ) not in self.publications:
                    totals["unpublished_visible"] += 1
                key = (
                    question.course_id,
                    question.presentation_id,
                    question.question_id,
                    question.signature,
                )
                if question.result:
                    totals["answered"] += 1
                elif self.cache.answer(question):
                    totals["prepared"] += 1
                elif key in self.preparation_failures:
                    totals["failed"] += 1
                else:
                    totals["remaining"] += 1
        return totals

    def snapshot(self, status="running"):
        active = [state for state in self.states.values() if not state.ended]
        webhook = self.events.delivery_status()
        if self.login.active:
            status = "login_required"
        elif status == "running" and (
            any(state.connection != "connected" for state in active)
            or (
                webhook["configured"]
                and (not webhook["worker_alive"] or webhook["last_error"])
            )
        ):
            status = "degraded"
        return {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "active_lessons": [
                s.lesson_id for s in self.states.values() if not s.ended
            ],
            "pending_answers": len(self.jobs),
            "mode": self.config.answering.mode,
            "entry_mode": self.client.entry_mode,
            "login_kind": self.client.state.get("login_kind", "web_login"),
            "submission_paused": self.runtime.paused,
            "classrooms": [
                {
                    "lesson_id": state.lesson_id,
                    "name": state.name,
                    "course_id": state.course_id,
                    "connection": state.connection,
                    "last_message_at": state.last_message_at,
                    "entry_code": state.entry_code,
                    "courseware": state.courseware,
                    "extraction": state.extraction,
                    "progress": self.lesson_progress(state),
                    "entry_blocked": state.entry_blocked,
                }
                for state in active
            ],
            "webhook": webhook,
        }

    def _write_status(self, status: str):
        path = self.config.storage / "status.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.snapshot(status)), encoding="utf-8"
        )
        temporary.replace(path)

    async def listen(self, state: LessonState):
        """Reconnect the classroom stream after transport loss."""
        while not self.stop.is_set() and not state.ended:
            state.connection = "connecting"
            try:
                identity = await self.client.checkin(state.lesson_id)
                await self._entry_result(state, 0)
                await self._classroom_stream(state, identity)
            except ClassroomEntryRejected as error:
                state.connection = "entry_refused"
                await self._entry_result(state, error.code)
                if self.client.block_on_entry_rejection:
                    state.entry_blocked = True
                    return
            except LoginRequired as error:
                if await self._reject_lesson_login(state, error):
                    return
                state.connection = "login_required"
                await self.events.emit(
                    "login_required", lesson_id=state.lesson_id
                )

                self.login.request()
                if self.login.active:
                    assert self.login.task is not None
                    await asyncio.shield(self.login.task)
            except RECOVERY_ERRORS as error:
                state.connection = "disconnected"
                await self.events.emit(
                    "lesson_disconnected",
                    lesson_id=state.lesson_id,
                    payload={"reason": type(error).__name__},
                )
            finally:
                if not state.ended and state.connection == "connected":
                    state.connection = "disconnected"
                    if not self.stop.is_set() and not self.login.active:
                        await self.events.emit(
                            "lesson_disconnected",
                            lesson_id=state.lesson_id,
                            payload={"reason": "socket_closed"},
                        )
                self._write_status("running")
            if not state.ended:
                await asyncio.sleep(
                    self.config.platform.poll_seconds
                    if state.connection == "entry_refused"
                    else self.config.platform.reconnect_seconds
                )

    async def restore_entry(self, mode="web"):
        previous = self.client.entry_mode
        if previous != mode and (
            self.jobs
            or any(
                not state.ended and state.connection == "connected"
                for state in self.states.values()
            )
        ):
            raise ValueError("请在当前课堂结束后切换课堂入口。")
        self.client.select_entry(mode)
        self.discovery_blocked = False
        self.runtime.save(entry_mode=mode)
        reconnect = [
            state
            for state in self.states.values()
            if not state.ended and state.connection != "connected"
        ]
        for state in reconnect:
            state.entry_blocked = True
            task = self.listeners.get(state.lesson_id)
            if task:
                task.cancel()
        await asyncio.gather(
            *(
                self.listeners[state.lesson_id]
                for state in reconnect
                if state.lesson_id in self.listeners
            ),
            return_exceptions=True,
        )
        for state in reconnect:
            state.entry_blocked = False
            self.listeners[state.lesson_id] = asyncio.create_task(
                self.listen(state)
            )

    async def _entry_result(self, state, code):
        if state.entry_code == code:
            return
        state.entry_code = code
        await self.events.emit(
            "classroom_entry_result",
            lesson_id=state.lesson_id,
            course_id=state.course_id,
            payload={
                "name": state.name,
                "entry_mode": self.client.entry_mode,
                "code": code,
                "accepted": code == 0,
                "attendance_verified": False,
                "reason": self._entry_reason(code),
            },
        )

    async def _reject_discovery_login(self, error):
        """Allow a separate entry adapter to classify a terminal rejection."""
        return False

    async def _reject_lesson_login(self, state, error):
        """Allow a separate entry adapter to classify a classroom rejection."""
        return False

    def _entry_reason(self, code):
        if code == 0:
            return "accepted"
        if str(code) == "50070":
            return "dynamic_code_refused"
        return "entry_rejected"

    @staticmethod
    def _check_classroom_auth(data):
        if data.get("op") == "hello" and (
            data.get("message") == "authorize failed"
            or data.get("status") is False
            or str(data.get("code")) in {"401", "50000"}
        ):
            raise LoginRequired("classroom hello authorization failed")

    async def _classroom_stream(self, state, identity):
        async with await self.client.websocket(lesson=state.lesson_id) as ws:
            await ws.send_json(
                self.client.classroom_hello(state.lesson_id, identity)
            )
            first = await ws.receive(
                timeout=self.config.platform.handshake_timeout
            )
            if first.type != aiohttp.WSMsgType.TEXT:
                raise ProtocolError("classroom handshake was not text")
            initial = json.loads(first.data)
            self.protocol_trace.record(state.lesson_id, initial)
            if not isinstance(initial, dict):
                raise ProtocolError("classroom handshake was not an object")
            self._check_classroom_auth(initial)
            if initial.get("op") not in {"hello", "fetchtimeline"}:
                raise ProtocolError("classroom hello was not acknowledged")
            state.connection = "connected"
            state.last_message_at = time.time()
            self._write_status("running")
            await self.events.emit(
                "lesson_connected",
                lesson_id=state.lesson_id,
                course_id=state.course_id,
                payload={
                    "name": state.name,
                    "entry_mode": self.client.entry_mode,
                    "attendance_verified": False,
                },
            )
            heartbeat = asyncio.create_task(
                self._classroom_heartbeat(ws, state.lesson_id)
            )
            try:
                await self.handle_message(state, initial)
                self.schedule_courseware(state)
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        state.last_message_at = time.time()
                        try:
                            data = json.loads(message.data)
                            self.protocol_trace.record(state.lesson_id, data)
                            if isinstance(data, dict):
                                self._check_classroom_auth(data)
                                await self.handle_message(state, data)
                        except (ValueError, TypeError, KeyError):
                            await self.events.emit(
                                "protocol_error",
                                lesson_id=state.lesson_id,
                                payload={
                                    "error": "unsupported websocket message"
                                },
                            )
                        if state.ended:
                            break
                    elif message.type in {
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    }:
                        break
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _classroom_heartbeat(self, ws, lesson):
        try:
            while not self.stop.is_set():
                await asyncio.sleep(CLASSROOM_HEARTBEAT_SECONDS)
                await ws.send_json({"op": "heartbeat", "lessonid": lesson})
        except (aiohttp.ClientError, ConnectionError, RuntimeError):
            await ws.close()

    # Keep platform event dispatch in one place.
    # pylint: disable-next=too-many-branches
    async def handle_message(self, state: LessonState, data: dict):
        """Emit a reminder before any metadata fetch, OCR or model call."""
        operation = data.get("op")
        if operation == "lessonfinished":
            state.ended = True
            state.connection = "ended"
            if self.warmer and all(s.ended for s in self.states.values()):
                self.warmer.cancel()
            if state.lesson_id in self.catalogs:
                self.catalogs[state.lesson_id].cancel()
            await self.events.emit("lesson_finished", lesson_id=state.lesson_id)
            for (lesson, _), task in tuple(self.jobs.items()):
                if lesson == state.lesson_id:
                    task.cancel()
            return
        if operation in {"problemfinished", "closeproblem", "problemoff"}:
            problem = data.get("problem")
            problem = {**data, **problem} if isinstance(problem, dict) else data
            keys = {
                identifier(
                    first_field(problem, "prob", "problemId", "problemid", "id")
                ),
                identifier(problem.get("sid")),
            } - {""}
            for key, released in tuple(self.publications.items()):
                aliases = {key[1], identifier(released.get("sid"))} - {""}
                if key[0] == state.lesson_id and (not keys or keys & aliases):
                    self.closed_questions.add(key)
                    task = self.jobs.get(key)
                    if task:
                        task.cancel()
            return
        presentation = identifier(
            first_field(data, "presentation", "presentationid", "pres")
        )
        if presentation:
            state.presentations.add(presentation)
            if operation != "presentationcreated" or not state.presentation_id:
                state.presentation_id = presentation
        if operation in {"hello", "fetchtimeline"}:
            for piece in data.get("timeline") or []:
                if not isinstance(piece, dict):
                    continue
                past = identifier(
                    first_field(piece, "pres", "presentation", "presId")
                )
                if past:
                    state.presentations.add(past)
                    if not presentation:
                        state.presentation_id = past
                if piece.get("type") == "problem" and not piece.get("isClosed"):
                    await self.handle_message(
                        state, {"op": "unlockproblem", "problem": piece}
                    )
        if operation in {"unlockproblem", "problemon", "extendtime"}:
            problem = data.get("problem")
            if not isinstance(problem, dict):
                problem = {"prob": problem} if problem is not None else {}
            problem = {**data, **problem}
            problem.pop("problem", None)
            problem["prob"] = first_field(
                problem, "prob", "problemId", "problemid", "id"
            )
            problem.setdefault("pres", state.presentation_id)
            key = identifier(problem.get("prob") or problem.get("sid"))
            if not key:
                raise ValueError("unlockproblem lacks a question identifier")
            deadline = question_deadline(problem)
            job_key = (state.lesson_id, key)
            if operation == "extendtime" and job_key in self.deadlines:
                previous = self.deadlines[job_key]
                if previous is not None:
                    extension = float(problem.get("extend") or 0)
                    if deadline is None or deadline <= previous:
                        deadline = previous + max(0, extension)
            if deadline is not None and deadline <= time.time():
                return
            self.publications[job_key] = problem
            self.closed_questions.discard(job_key)
            self.deadlines[job_key] = deadline
            if job_key in self.jobs:
                return
            if (
                self.config.answering.mode == "notify"
                and job_key in self.question_details
            ):
                return
            attempt = self.cache.submission(
                state.lesson_id, identifier(problem.get("prob"))
            )
            if attempt and attempt["status"] == "accepted":
                return
            if job_key not in self.notified:
                await self.events.emit(
                    "question_detected",
                    lesson_id=state.lesson_id,
                    course_id=state.course_id,
                    question_id=key,
                    dedup_key=f"{state.lesson_id}:{key}",
                    payload={
                        "presentation_id": identifier(problem.get("pres")),
                        "deadline": deadline,
                        "name": state.name,
                        "lesson_url": self._lesson_url(state),
                    },
                )
                self.notified.add(job_key)
            if self.warmer:
                self.warmer.cancel()
            task = asyncio.create_task(
                self.answer(
                    state,
                    problem,
                    deadline,
                    auto_allowed=self.config.answering.mode == "auto"
                    and not self.runtime.paused,
                    auto_generation=self.runtime.generation,
                )
            )
            self.jobs[job_key] = task
            task.add_done_callback(
                lambda done, k=job_key: self._job_done(k, done)
            )
        elif state.presentation_id and operation in {
            "hello",
            "fetchtimeline",
            "showpresentation",
            "presentationstart",
            "presentationupdated",
            "presentationcreated",
            "slidenav",
            "slide",
        }:
            target = presentation or state.presentation_id
            if (
                operation not in {"presentationupdated", "presentationcreated"}
                and not self._courseware_needed(state, target)
            ):
                return
            task = asyncio.create_task(
                self._background_fetch(state, target)
            )
            self.background.add(task)
            task.add_done_callback(self.background.discard)

    def schedule_courseware(self, state: LessonState, *, force=False):
        """Refresh all visible decks independently of the realtime stream."""
        if (
            (not self.config.courseware.enabled and not force)
            or state.ended
            or self.stop.is_set()
        ):
            return
        current = self.catalogs.get(state.lesson_id)
        if current and not current.done():
            return
        now = time.monotonic()
        if not force and (
            now - self.catalog_checked.get(state.lesson_id, float("-inf"))
            < self.config.courseware.refresh_seconds
        ):
            return
        self.catalog_checked[state.lesson_id] = now
        self.catalogs[state.lesson_id] = asyncio.create_task(
            self.collect_courseware(state, force=force)
        )

    def _courseware_needed(self, state, presentation):
        loaded = state.courseware.get(presentation, {})
        return (
            not loaded.get("ok")
            or not loaded.get("complete")
            or loaded["slides"] < state.expected_pages.get(presentation, 0)
            or bool(loaded.get("images_unavailable"))
        )

    async def collect_courseware(
        self, state: LessonState, *, prepare=True, force=False
    ):
        """Follow the official catalog, retaining live capture on failure."""
        if force or time.monotonic() >= state.catalog_retry_after:
            try:
                catalog = await self.client.presentation_catalog(state.lesson_id)
                state.catalog_ok = True
                state.catalog_retry_after = 0
                state.catalog_error = None
                state.presentations.update(item["id"] for item in catalog)
                state.expected_pages.update(
                    {item["id"]: item["slide_count"] for item in catalog}
                )
                if not state.presentation_id and catalog:
                    state.presentation_id = catalog[0]["id"]
            except RECOVERY_ERRORS as error:
                state.catalog_ok = False
                state.catalog_retry_after = time.monotonic() + max(
                    CATALOG_FAILURE_RETRY_SECONDS,
                    self.config.courseware.refresh_seconds,
                )
                detail = {
                    "scope": "catalog",
                    "reason": type(error).__name__,
                    "http_status": getattr(error, "http_status", None),
                    "code": getattr(error, "code", None),
                    "fallback": "live_capture",
                }
                if detail != state.catalog_error:
                    state.catalog_error = detail
                    await self.events.emit(
                        "courseware_unavailable",
                        lesson_id=state.lesson_id,
                        payload=detail,
                    )
        if state.presentation_id:
            state.presentations.add(state.presentation_id)
        for presentation in sorted(state.presentations):
            if self.stop.is_set() or state.ended:
                break
            if force or self._courseware_needed(state, presentation):
                await self._background_fetch(state, presentation, prepare=prepare)
        if not state.ended and not self.stop.is_set():
            await self.report_extraction(state)
            if (
                self.warmer
                and self.warmer.done()
                and not self.lesson_progress(state)["remaining"]
            ):
                await self.report_preparation("completed", force=True)
        return state.courseware

    def _job_done(self, key, task):
        self.jobs.pop(key, None)
        if not task.cancelled() and task.exception():
            logger.error(
                "answer task failed: %s", type(task.exception()).__name__
            )
        if not self.jobs and not self.stop.is_set():
            self.start_precompute()

    async def wait_for_question(self, state, data) -> Question:
        """Retry incomplete presentations within the question deadline."""
        limit = time.monotonic() + self.config.platform.metadata_wait_seconds
        key = (state.lesson_id, identifier(data.get("prob") or data.get("sid")))
        last_error = "metadata_not_ready"
        while not self.stop.is_set() and not state.ended:
            remaining = limit - time.monotonic()
            deadline = self.deadlines.get(key)
            if deadline is not None:
                remaining = min(remaining, deadline - time.time())
            if remaining <= 0:
                break
            presentation = identifier(data.get("pres") or state.presentation_id)
            try:
                if presentation:
                    await asyncio.wait_for(
                        self.fetch(state, presentation), remaining
                    )
                    question = self.cache.find_question(
                        state.course_id,
                        presentation,
                        identifier(data.get("prob")),
                        identifier(data.get("sid")),
                    )
                    if question and (
                        question.text.strip() or question.image_url
                    ):
                        return question
            except LoginRequired:
                self.login.request()
                last_error = "LoginRequired"
            except RECOVERY_ERRORS as error:
                last_error = type(error).__name__
            try:
                released = self.publications.get(key, data)
                question = Question.from_event(
                    state.course_id, presentation, released
                )
                if question.text.strip() or question.image_url:
                    return question
            except (ValueError, TypeError, KeyError):
                pass
            await asyncio.sleep(
                min(remaining, self.config.platform.metadata_retry_seconds)
            )
        raise MetadataUnavailable(last_error)

    async def fetch(self, state: LessonState, presentation: str) -> dict:
        """Share fetches while isolating cancellation of a cache warmer."""
        key = (state.lesson_id, presentation)
        task = self.fetches.get(key)
        if task is None or task.done():
            task = asyncio.create_task(self._fetch_deck(state, presentation))
            self.fetches[key] = task
        return await asyncio.shield(task)

    def _question_versions(self, course, presentation):
        return {
            (question.question_id, question.signature, bool(question.result))
            for question in self.cache.questions(course, presentation)
        }

    async def _fetch_deck(self, state, presentation):
        data = await self.client.presentation(state.lesson_id, presentation)
        previous = self._question_versions(state.course_id, presentation)
        self.cache.save_presentation(state.course_id, presentation, data)
        if previous != self._question_versions(state.course_id, presentation):
            self.courseware_revision += 1
        state.presentations.add(presentation)
        if (
            data.get("datacompleted") is False
            or len(data.get("slides") or []) < state.expected_pages.get(presentation, 0)
        ) and presentation in state.courseware:
            state.courseware[presentation]["complete"] = False
        return data

    async def _background_fetch(self, state, presentation, *, prepare=True):
        key = (state.lesson_id, presentation)
        task = self.courseware_jobs.get(key)
        if task is None or task.done():
            task = asyncio.create_task(
                self._load_courseware(state, presentation, prepare=prepare)
            )
            self.courseware_jobs[key] = task
        await asyncio.shield(task)

    async def _load_courseware(self, state, presentation, *, prepare=True):
        previous = state.courseware.get(presentation)
        try:
            data = await self.fetch(state, presentation)
            questions = self.cache.questions(state.course_id, presentation)
            slides = self.cache.presentation(
                state.course_id, presentation
            ).get("slides", [])
            status = {
                "ok": True,
                "complete": data.get("datacompleted") is not False,
                "slides": len(slides),
                "questions": len(questions),
                "images_saved": 0,
                "images_unavailable": 0,
                "images_expected": sum(
                    bool(slide.get("cover_url"))
                    for slide in slides
                    if isinstance(slide, dict)
                ),
            }
            state.courseware[presentation] = status
            if prepare:
                self.start_precompute()
            if self.config.courseware.download_images:
                for slide in slides:
                    if self.stop.is_set() or state.ended:
                        break
                    if not isinstance(slide, dict) or not slide.get(
                        "cover_url"
                    ):
                        continue
                    while (
                        self.jobs and not self.stop.is_set() and not state.ended
                    ):
                        await asyncio.sleep(0.1)
                    if self.stop.is_set() or state.ended:
                        break
                    try:
                        saved = await self.assets.image(slide["cover_url"])
                    except OSError:
                        saved = None
                    status[
                        "images_saved" if saved else "images_unavailable"
                    ] += 1
            if status != previous:
                await self.events.emit(
                    "courseware_updated",
                    lesson_id=state.lesson_id,
                    course_id=state.course_id,
                    payload={"presentation_id": presentation, **status},
                )
        except RECOVERY_ERRORS as error:
            failed = {
                **state.courseware.get(presentation, {}),
                "ok": False,
                "reason": type(error).__name__,
            }
            state.courseware[presentation] = failed
            if failed != previous:
                await self.events.emit(
                    "courseware_unavailable",
                    lesson_id=state.lesson_id,
                    payload={
                        "presentation_id": presentation,
                        "reason": failed["reason"],
                        "fallback": "live_capture",
                    },
                )

    async def report_extraction(self, state):
        progress = self.lesson_progress(state)
        successful = [
            value for value in state.courseware.values() if value.get("ok")
        ]
        failed = sum(
            not state.courseware.get(key, {}).get("ok")
            for key in state.presentations
        )
        saved = sum(value.get("images_saved", 0) for value in successful)
        expected = sum(value.get("images_expected", 0) for value in successful)
        incomplete = any(
            not value.get("complete")
            or value.get("slides", 0) < state.expected_pages.get(key, 0)
            for key, value in state.courseware.items()
        )
        complete = (
            progress["slides"] > 0
            and state.catalog_ok
            and not failed
            and not incomplete
            and (not self.config.courseware.download_images or saved == expected)
        )
        state.extraction = (
            "all_visible"
            if complete
            else "partial"
            if progress["slides"]
            else "waiting"
            if state.catalog_ok and not failed
            else "unavailable"
        )
        payload = {
            **progress,
            "name": state.name,
            "status": state.extraction,
            "failed_presentations": failed,
            "images_saved": saved,
            "images_expected": expected,
            "unreleased_visibility": "unknown",
            "fallback": "live_capture",
        }
        fingerprint = tuple(
            sorted((key, str(value)) for key, value in payload.items())
        )
        if self.extraction_reports.get(state.lesson_id) != fingerprint:
            self.extraction_reports[state.lesson_id] = fingerprint
            await self.events.emit(
                "courseware_summary",
                lesson_id=state.lesson_id,
                course_id=state.course_id,
                payload=payload,
            )

    async def report_preparation(self, phase, *, force=False):
        for state in self.states.values():
            if state.ended:
                continue
            progress = self.lesson_progress(state)
            fingerprint = tuple(progress.values()) + (phase,)
            old = self.preparation_reports.get(state.lesson_id)
            now = time.monotonic()
            completed = progress["prepared"] + progress["failed"]
            if (
                not force
                and old is not None
                and completed % 5
                and now - self.last_preparation_report.get(state.lesson_id, 0)
                < 30
            ):
                continue
            if old == fingerprint:
                continue
            self.last_preparation_report[state.lesson_id] = now
            self.preparation_reports[state.lesson_id] = fingerprint
            await self.events.emit(
                "preparation_progress",
                lesson_id=state.lesson_id,
                course_id=state.course_id,
                payload={**progress, "name": state.name, "phase": phase},
            )

    def start_precompute(self):
        """Precompute in idle time; realtime questions preempt this worker."""
        policy = self.config.answering
        if (
            self.jobs
            or not policy.precompute
            or policy.mode == "notify"
            or not policy.providers
        ):
            return False
        if self.warmer and not self.warmer.done():
            return True
        if not any(
            not state.ended and self.lesson_progress(state)["remaining"]
            for state in self.states.values()
        ):
            return False
        self.warmer = asyncio.create_task(self._prepare_reported())
        self.warmer.add_done_callback(
            lambda task: self.start_precompute()
            if task.cancelled() and not self.stop.is_set()
            else None
        )
        return True

    async def _prepare_reported(self):
        await self.report_preparation("started", force=True)
        try:
            await self.precompute()
        except asyncio.CancelledError:
            await self.report_preparation("paused", force=True)
            raise
        else:
            phase = (
                "waiting_for_content"
                if any(not task.done() for task in self.catalogs.values())
                else "completed"
            )
            await self.report_preparation(phase, force=True)

    async def precompute(self):
        """Analyze question slides once and retain answers across restarts."""
        while not self.stop.is_set() and not self.jobs:
            revision = self.courseware_revision
            for state in tuple(self.states.values()):
                if state.ended:
                    continue
                presentations = state.presentations | (
                    {state.presentation_id} if state.presentation_id else set()
                )
                for presentation in sorted(presentations):
                    for question in self.cache.questions(
                        state.course_id, presentation
                    ):
                        if state.ended or self.jobs or self.stop.is_set():
                            break
                        key = (
                            question.course_id,
                            question.presentation_id,
                            question.question_id,
                            question.signature,
                        )
                        if (
                            question.result
                            or key in self.preparation_failures
                            or self.cache.answer(question)
                        ):
                            continue
                        try:
                            answer, source = await asyncio.wait_for(
                                self.compute(question),
                                self.config.answering.timeout_seconds,
                            )
                            latest = self.cache.find_question(
                                question.course_id,
                                question.presentation_id,
                                question.question_id,
                                "",
                            )
                            if (
                                latest is None
                                or latest.result
                                or latest.signature != question.signature
                            ):
                                continue
                            self.cache.save_answer(latest, answer, source)
                            await self.events.emit(
                                "answer_cached",
                                course_id=question.course_id,
                                question_id=question.question_id,
                                payload={
                                    "presentation_id": question.presentation_id,
                                    "source": source,
                                },
                            )
                        except (
                            SolverUnavailable,
                            ValueError,
                            asyncio.TimeoutError,
                        ):
                            self.preparation_failures.add(key)
                        await self.report_preparation("progress")
            # A second deck may finish downloading while the first AI call runs.
            if revision == self.courseware_revision:
                return

    async def compute(self, question: Question):
        """Combine earlier slide images/text and course-isolated knowledge."""
        policy = self.config.answering
        prior_material = self.cache.previous_slides(question, 100000)
        previous = (
            prior_material[-policy.previous_slides :]
            if policy.previous_slides
            else []
        )
        course_scope = self.client.origin + "|" + question.course_id
        for slide in prior_material:
            text = slide_text(slide)
            if text.strip():
                self.knowledge.add_document(
                    course_scope,
                    f"{question.presentation_id}/{slide.get('id')}",
                    f"课件 {question.presentation_id} 第 {slide.get('index')} 页",
                    text,
                )
        context = "\n".join(
            f"此前第 {s.get('index')} 页：{slide_text(s)}" for s in previous
        )
        for match in self.knowledge.search(
            course_scope, question.describe(), limit=4
        ):
            context += f"\n历史资料 {match.title}：{match.content}"
        for prior in self.cache.questions(
            question.course_id, question.presentation_id
        ):
            if prior.index < question.index:
                cached = self.cache.answer(prior)
                if cached:
                    context += (
                        f"\n此前题目（缓存答案未保证正确）：{prior.describe()}"
                        f"\n{cached[0]}"
                    )
        urls = [question.image_url] + [
            str(s.get("cover_url") or s.get("cover") or "") for s in previous
        ]
        downloaded = [None] * len(urls)
        if any(p.input_mode == "vision" for p in policy.providers):
            downloaded = await asyncio.gather(
                *(self.assets.image(url) for url in urls)
            )
        images = [item for item in downloaded if item]
        return await self.solver.solve(
            question,
            images,
            context[: policy.context_characters],
            target_image=bool(downloaded[0]),
        )

    # Preserve the deadline and single-write state in one coroutine.
    # pylint: disable-next=too-many-branches,too-many-statements
    async def answer(
        self,
        state: LessonState,
        data: dict,
        deadline: float | None,
        *,
        auto_allowed=False,
        auto_generation=None,
    ):
        """Use cached/model answers and optional fallback within the budget."""
        policy = self.config.answering
        identity: QuestionIdentity = {
            "lesson_id": state.lesson_id,
            "course_id": state.course_id,
            "question_id": identifier(data.get("prob") or data.get("sid")),
        }
        question = None
        stage = "metadata"
        operation_started = time.monotonic()
        try:
            budget = (
                self.config.platform.metadata_wait_seconds
                if policy.mode == "notify"
                else policy.timeout_seconds
            )
            if deadline is not None:
                budget = min(
                    budget,
                    deadline
                    - time.time()
                    - (
                        policy.submit_reserve_seconds
                        if policy.mode == "auto"
                        else 0
                    ),
                )
            if budget <= 0:
                raise asyncio.TimeoutError
            started = time.monotonic()
            question = await asyncio.wait_for(
                self.wait_for_question(state, data), budget
            )
            presentation = question.presentation_id
            identity["question_id"] = question.question_id
            await self.events.emit(
                "question_ready",
                **identity,
                dedup_key=(
                    f"{state.lesson_id}:{question.question_id}:"
                    f"{question.signature}"
                ),
                payload={
                    "text": question.describe(),
                    "image_url": question.image_url,
                    "kind": question.kind,
                    "presentation_id": presentation,
                    "index": question.index,
                    "options": question.options,
                    "name": state.name,
                    "deadline": self.deadlines.get(
                        (
                            state.lesson_id,
                            identifier(data.get("prob") or data.get("sid")),
                        )
                    ),
                    "lesson_url": self._lesson_url(state),
                },
            )
            if (
                question.image_url
                and self.config.webhook.inline_question_images
            ):
                task = asyncio.create_task(
                    self._forward_question_image(state, question, deadline)
                )
                self.background.add(task)
                task.add_done_callback(self.background.discard)
            self.question_details.add(
                (
                    state.lesson_id,
                    identifier(data.get("prob") or data.get("sid")),
                )
            )
            if question.result:
                self.cache.mark_submission(
                    state.lesson_id, question, question.result, "accepted"
                )
                await self.events.emit("already_submitted", **identity)
                return
            if policy.mode == "notify":
                return
            prior = self.cache.submission(state.lesson_id, question.question_id)
            if prior and prior["status"] in {
                "accepted",
                "pending",
                "uncertain",
            }:
                await self.events.emit(
                    "submission_unresolved"
                    if prior["status"] != "accepted"
                    else "already_submitted",
                    **identity,
                )
                return
            cached = self.cache.answer(question)
            if cached:
                answer, source = cached
            else:
                try:
                    stage = "ai"
                    answer, source = await asyncio.wait_for(
                        self.compute(question),
                        max(0.001, budget - (time.monotonic() - started)),
                    )
                    self.cache.save_answer(question, answer, source)
                    answer, source = self.cache.answer(question) or (
                        answer,
                        source,
                    )
                except (SolverUnavailable, asyncio.TimeoutError):
                    if (
                        policy.fallback != "first_option"
                        or question.kind not in {1, 2, 3, 6}
                        or not question.options
                    ):
                        raise
                    answer = validate_answer(
                        question, [next(iter(question.options))]
                    )
                    source = "fallback:first_option"
            await self.events.emit(
                "answer_ready",
                **identity,
                payload={
                    "answer": answer,
                    "source": source,
                    "mode": policy.mode,
                    "auto_submit": auto_allowed
                    and policy.mode == "auto"
                    and not self.runtime.paused
                    and auto_generation == self.runtime.generation
                    and policy.automatic_window_open,
                    "name": state.name,
                    "deadline": self.deadlines.get(
                        (
                            state.lesson_id,
                            identifier(data.get("prob") or data.get("sid")),
                        )
                    ),
                    "lesson_url": self._lesson_url(state),
                },
            )
            if (
                auto_allowed
                and policy.mode == "auto"
                and not state.ended
                and not self.runtime.paused
                and auto_generation == self.runtime.generation
            ):
                remaining = self.config.platform.request_timeout * 2
                if deadline is not None:
                    remaining = min(remaining, deadline - time.time())
                if remaining <= 0:
                    raise asyncio.TimeoutError
                stage = "submission"
                await asyncio.wait_for(
                    self.commit(
                        state,
                        question,
                        answer,
                        source,
                        auto_generation=auto_generation,
                    ),
                    remaining,
                )
        except asyncio.CancelledError:
            await self._record_interrupted(state, question)
            raise
        except SubmissionPaused:
            await self.events.emit(
                "submission_skipped",
                **identity,
                payload={
                    "reason": "outside_automatic_window"
                    if not policy.automatic_window_open
                    else "owner_changed_mode_or_paused",
                    "lesson_url": self._lesson_url(state),
                },
            )
        except (
            ProtocolError,
            aiohttp.ClientError,
            SolverUnavailable,
            MetadataUnavailable,
            asyncio.TimeoutError,
            ValueError,
        ) as error:
            await self._record_interrupted(state, question)
            await self.events.emit(
                "question_unavailable" if question is None else "answer_failed",
                **identity,
                payload={
                    "reason": type(error).__name__,
                    "stage": stage,
                    "code": getattr(error, "code", None),
                    "server_msg": getattr(error, "server_msg", None),
                    "elapsed_ms": round(
                        (time.monotonic() - operation_started) * 1000
                    ),
                    "lesson_url": self._lesson_url(state),
                },
            )

    async def _forward_question_image(self, state, question, deadline):
        """Supplement earlier reminders without delaying them for a download."""
        image = None
        try:
            image = await asyncio.wait_for(
                self.assets.image(question.image_url), 3
            )
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            pass
        payload = {
            "image_url": question.image_url,
            "deadline": deadline,
            "lesson_url": self._lesson_url(state),
        }
        kind = "question_image_unavailable"
        if (
            image
            and len(image[0]) <= self.config.webhook.inline_image_max_bytes
        ):
            kind = "question_image"
            payload.update(
                image_base64=base64.b64encode(image[0]).decode("ascii"),
                image_mime=image[1],
            )
        else:
            payload["reason"] = (
                "image_too_large" if image else "download_failed"
            )
        await self.events.emit(
            kind,
            lesson_id=state.lesson_id,
            course_id=state.course_id,
            question_id=question.question_id,
            dedup_key=(
                f"{state.lesson_id}:{question.question_id}:{question.signature}"
            ),
            payload=payload,
        )

    async def _record_interrupted(self, state, question):
        if question:
            record = self.cache.submission(
                state.lesson_id, question.question_id
            )
            if record and record["status"] == "pending":
                self.cache.mark_submission(
                    state.lesson_id,
                    question,
                    json.loads(record["answer"]),
                    "uncertain",
                )
                await self.events.emit(
                    "submission_unresolved",
                    lesson_id=state.lesson_id,
                    question_id=question.question_id,
                )

    async def commit(
        self, state, question, answer, source, *, auto_generation=None
    ):
        """Re-read content and persist intent before the single POST."""
        self._check_automatic(auto_generation)
        released = self._check_publication(state, question)
        try:
            await self.fetch(state, question.presentation_id)
        except LoginRequired:
            raise
        except RECOVERY_ERRORS:
            # Inline released metadata remains usable when full decks are hidden.
            current = Question.from_event(
                state.course_id, question.presentation_id, released
            )
        else:
            current = self.cache.find_question(
                question.course_id,
                question.presentation_id,
                question.question_id,
                "",
            )
            if current is None or not (
                current.text.strip() or current.image_url
            ):
                current = Question.from_event(
                    state.course_id, question.presentation_id, released
                )
        if (
            current is None
            or current.signature != question.signature
            or state.ended
        ):
            raise ValueError("question changed before submission")
        if current.result:
            self.cache.mark_submission(
                state.lesson_id, question, current.result, "accepted"
            )
            await self.events.emit(
                "already_submitted",
                lesson_id=state.lesson_id,
                question_id=question.question_id,
            )
            return
        self._check_publication(state, question)
        self._check_automatic(auto_generation)
        self.cache.mark_submission(state.lesson_id, question, answer, "pending")
        try:
            await self.client.submit(
                state.lesson_id, question.question_id, question.kind, answer
            )
        except RejectedSubmission:
            self.cache.mark_submission(
                state.lesson_id, question, answer, "rejected"
            )
            raise
        self.cache.mark_submission(
            state.lesson_id, question, answer, "accepted"
        )
        await self.events.emit(
            "submission_result",
            lesson_id=state.lesson_id,
            course_id=question.course_id,
            question_id=question.question_id,
            payload={"accepted": True, "answer": answer, "source": source},
        )
        task = asyncio.create_task(
            self.verify_submission(state, question, answer)
        )
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    async def verify_submission(self, state, question, answer):
        """Read back accepted answers without another answer POST."""
        reason = "result_not_visible"
        for delay in (0, 2):
            if delay:
                await asyncio.sleep(delay)
            try:
                data = await self.client.presentation(
                    state.lesson_id, question.presentation_id
                )
                for slide in data.get("slides", []):
                    try:
                        current = Question.from_slide(
                            question.course_id, question.presentation_id, slide
                        )
                    except (ValueError, TypeError, KeyError):
                        continue
                    if (
                        current.question_id != question.question_id
                        or not current.result
                    ):
                        continue
                    matched = (
                        current.signature == question.signature
                        and validate_answer(current, current.result)
                        == validate_answer(question, answer)
                    )
                    await self.events.emit(
                        "submission_verified",
                        lesson_id=state.lesson_id,
                        question_id=question.question_id,
                        payload={"matched": matched, "result_present": True},
                    )
                    return
            except RECOVERY_ERRORS as error:
                reason = type(error).__name__
        await self.events.emit(
            "submission_verified",
            lesson_id=state.lesson_id,
            question_id=question.question_id,
            payload={
                "matched": None,
                "result_present": False,
                "reason": reason,
            },
        )

    def _check_automatic(self, generation):
        if (
            self.config.answering.mode != "auto"
            or not self.config.answering.automatic_window_open
            or self.runtime.paused
            or (
                generation is not None and generation != self.runtime.generation
            )
        ):
            raise SubmissionPaused("automatic submission disabled by owner")

    def _check_publication(self, state, question):
        for question_id in (question.question_id, question.slide_id):
            key = (state.lesson_id, question_id)
            if key in self.publications:
                deadline = self.deadlines.get(key)
                if (
                    state.ended
                    or key in self.closed_questions
                    or (deadline is not None and deadline <= time.time())
                ):
                    raise ValueError("question is closed or expired")
                return self.publications[key]
        raise ValueError("teacher has not released this question")

    async def close(self):
        """Cancel work, retain ambiguous writes and release resources."""
        self.stop.set()
        await self.login.close()
        tasks = list(self.listeners.values()) + list(self.jobs.values())
        tasks += list(self.fetches.values()) + list(self.background)
        tasks += list(self.catalogs.values())
        tasks += list(self.courseware_jobs.values())
        if self.warmer:
            tasks.append(self.warmer)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._write_status("stopped")
        await self.solver.close()
        self.runtime.close()
        self.knowledge.close()
        self.cache.close()
