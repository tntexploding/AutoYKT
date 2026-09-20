"""Timed-session lifecycle checks without a browser or external models."""

import asyncio
from contextlib import nullcontext, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from autoykt.cli import _build_parser, _session_command
from autoykt.core.config import ConfigError, load_config
from autoykt.core.event_bus import Event, EventBus, EventType
from autoykt.core.run_guard import keep_awake, run_lease
from autoykt.monitor.class_session import (
    ClassSession,
    _write_json,
    read_session_status,
    request_session_stop,
)


class _Clock:

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Watcher:

    def __init__(self, clock):
        self.clock = clock
        self.exit_code = 0
        self.stop_reason = "stopped"
        self.running = False
        self.polls = 0
        self.counts = {}

    async def prepare(self):
        self.clock.now += 300

    async def start(self):
        self.running = True
        while self.running:
            self.polls += 1
            self.clock.now += 60
            if self.polls % 10 == 0:
                self.counts["verified"] = self.counts.get("verified", 0) + 1
            await asyncio.sleep(0.002)

    def stop(self):
        self.running = False

    def snapshot(self):
        return [
            {
                "profile_id": "example_yuketang",
                "state": "waiting",
                "outcome_counts": dict(self.counts),
                "needs_attention": False,
            }
        ]


class ClassSessionTest(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = load_config("config.example.yaml")
        self.config.pages[0].enabled = True
        self.config.storage.data_dir = str(self.root)

    def test_status_write_retries_transient_windows_reader_lock(self):
        path = self.root / "status.json"
        path.write_text('{"old": true}', encoding="utf-8")
        original = Path.replace
        failures = 0

        def replace_file(source, target):
            nonlocal failures
            if failures < 2:
                failures += 1
                self.assertEqual(json.loads(path.read_text()), {"old": True})
                raise PermissionError("reader holds the status file")
            return original(source, target)

        with (
            patch.object(Path, "replace", replace_file),
            patch("autoykt.monitor.class_session.sleep") as wait,
        ):
            _write_json(path, {"new": True})
        self.assertEqual(wait.call_count, 2)
        self.assertEqual(json.loads(path.read_text()), {"new": True})

    async def test_two_hours_excludes_warmup_and_records_cumulative_outcomes(
        self,
    ):
        clock = _Clock()
        watcher = _Watcher(clock)
        with (
            patch("autoykt.monitor.class_session.monotonic", clock),
            patch("autoykt.monitor.class_session.HEARTBEAT_SECONDS", 0.001),
        ):
            session = ClassSession(self.config, EventBus(), 120)
            code = await session.run(watcher)
        data = read_session_status(self.config)
        self.assertEqual(code, 0)
        self.assertEqual(data["reason"], "duration_reached")
        self.assertEqual(data["elapsed_seconds"], 7200)
        self.assertEqual(data["preparation_seconds"], 300)
        self.assertEqual(watcher.polls, 120)
        self.assertEqual(data["outcome_counts"], {"verified": 12})
        self.assertFalse(data["heartbeat_stale"])
        self.assertEqual(
            json.loads(session.report_path.read_text())["id"], data["id"]
        )

    def test_verified_question_does_not_hide_missed_question_or_monitoring_pause(
        self,
    ):
        for counts, pauses, expected in (
            ({"verified": 1}, 0, "verified_cycles_observed"),
            ({"verified": 1, "unavailable": 1}, 0, "review_required"),
            ({"verified": 1}, 1, "review_required"),
            ({}, 0, "no_verified_questions"),
        ):
            with self.subTest(counts=counts, pauses=pauses):
                session = ClassSession(self.config, EventBus(), 120)
                watcher = Mock()
                watcher.snapshot.return_value = [
                    {
                        "outcome_counts": counts,
                        "monitoring": {"pause_count": pauses},
                    }
                ]
                session.finish("duration_reached", 0, watcher)
                self.assertEqual(
                    read_session_status(self.config)["assessment"], expected
                )

    async def test_errors_drained_after_finish_update_final_assessment(self):
        session = ClassSession(self.config, EventBus(), 120)
        watcher = Mock()
        watcher.snapshot.return_value = [{"outcome_counts": {"verified": 1}}]
        session.finish("duration_reached", 0, watcher)
        self.assertEqual(
            read_session_status(self.config)["assessment"],
            "verified_cycles_observed",
        )
        await session.record_event(
            Event(EventType.ERROR, payload={"error": "late observer error"})
        )
        session.refresh()
        data = read_session_status(self.config)
        self.assertEqual(data["assessment"], "review_required")
        self.assertEqual(data["error_events"], 1)
        self.assertEqual(data["outcome_counts"], {"verified": 1})

    async def test_stop_cancels_a_pending_answer_and_does_not_affect_next_session(
        self,
    ):
        ready = asyncio.Event()
        watcher = _Watcher(_Clock())

        async def answer_forever():
            ready.set()
            await asyncio.Event().wait()

        watcher.start = AsyncMock(side_effect=answer_forever)
        session = ClassSession(self.config, EventBus(), 120)
        with patch("autoykt.monitor.class_session.HEARTBEAT_SECONDS", 0.001):
            run = asyncio.create_task(session.run(watcher))
            await asyncio.wait_for(ready.wait(), 1)
            path = request_session_stop(self.config)
            self.assertEqual(await asyncio.wait_for(run, 1), 130)
        self.assertEqual(
            read_session_status(self.config)["reason"], "stop_requested"
        )
        self.assertTrue(path.exists())
        second = ClassSession(self.config, EventBus(), 1)
        self.assertNotEqual(second.directory, session.directory)
        self.assertFalse((second.directory / "stop.request").exists())

    async def test_stop_during_ocr_preparation(self):
        watcher = _Watcher(_Clock())
        watcher.prepare = AsyncMock(side_effect=asyncio.Event().wait)
        watcher.start = AsyncMock()
        session = ClassSession(self.config, EventBus(), 120)
        request_session_stop(self.config)
        self.assertEqual(await session.run(watcher), 130)
        watcher.start.assert_not_awaited()
        self.assertIsNone(
            read_session_status(self.config)["monitoring_started_at"]
        )

    async def test_preparation_error_is_persisted(self):
        watcher = _Watcher(_Clock())
        watcher.prepare = AsyncMock(side_effect=RuntimeError("OCR unavailable"))
        session = ClassSession(self.config, EventBus(), 120)
        with self.assertRaisesRegex(RuntimeError, "OCR unavailable"):
            await session.run(watcher)
        data = read_session_status(self.config)
        self.assertEqual(data["reason"], "error")
        self.assertEqual(data["exit_code"], 1)

    async def test_expiry_finishes_inflight_cycle_but_manual_failure_is_not_success(
        self,
    ):
        clock = _Clock()
        watcher = _Watcher(clock)
        attention = False

        async def finish_after_stop():
            nonlocal attention
            watcher.running = True
            clock.now += 60
            while watcher.running:
                await asyncio.sleep(0.001)
            clock.now += 2
            attention = True

        watcher.start = finish_after_stop
        watcher.snapshot = lambda: [
            {"outcome_counts": {"failed": 1}, "needs_attention": attention}
        ]
        with (
            patch("autoykt.monitor.class_session.monotonic", clock),
            patch("autoykt.monitor.class_session.HEARTBEAT_SECONDS", 0.001),
        ):
            session = ClassSession(self.config, EventBus(), 1)
            code = await session.run(watcher)
        self.assertEqual(code, 1)
        data = read_session_status(self.config)
        self.assertEqual(data["elapsed_seconds"], 62)
        self.assertEqual(data["reason"], "manual_required")


class SessionEntryTest(unittest.TestCase):

    def test_default_entry_is_two_hours_and_does_not_rewrite_private_config(
        self,
    ):
        config = load_config("config.example.yaml")
        before = config.model_dump()
        args = _build_parser().parse_args(["session"])
        self.assertEqual(args.minutes, 120)
        run = AsyncMock(return_value=0)
        with (
            patch("autoykt.cli.live_configuration_issues", return_value=[]),
            patch("autoykt.cli._check_config", return_value=0),
            patch("autoykt.cli.run_lease", return_value=nullcontext()),
            patch("autoykt.cli.keep_awake", return_value=nullcontext()),
            patch("autoykt.cli._run_automation", run),
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(_session_command(config, args), 0)
        self.assertEqual(config.model_dump(), before)
        effective = run.call_args.args[0]
        self.assertFalse(effective.runtime.dry_run)
        self.assertTrue(effective.answering.auto_apply)
        self.assertEqual(run.call_args.kwargs["minutes"], 120)

    def test_incomplete_session_refuses_before_any_capture_or_models(self):
        config = load_config("config.example.yaml")
        args = _build_parser().parse_args(["session"])
        with (
            patch("autoykt.cli._run_automation") as run,
            redirect_stdout(StringIO()),
        ):
            with self.assertRaisesRegex(ConfigError, "setup is incomplete"):
                _session_command(config, args)
        run.assert_not_called()

    def test_selected_disabled_profile_is_reported_by_preflight(self):
        from autoykt.monitor.preflight import live_configuration_issues

        config = load_config("config.example.yaml")
        config.runtime.active_profiles = [config.pages[0].id]
        issues = "\n".join(live_configuration_issues(config))
        self.assertIn("profile is disabled", issues)
        self.assertIn("target_window", issues)

    def test_duplicate_run_is_rejected_and_ownership_released(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with run_lease(root):
                with self.assertRaisesRegex(ConfigError, "another AutoYKT"):
                    with run_lease(root):
                        self.fail("second run acquired ownership")
            with run_lease(root):
                pass

    def test_power_request_is_restored_on_error(self):
        api = Mock()
        api.SetThreadExecutionState.return_value = 0x80000000
        with (
            patch("autoykt.core.run_guard.sys.platform", "win32"),
            patch(
                "autoykt.core.run_guard.ctypes.WinDLL",
                return_value=api,
                create=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with keep_awake():
                    raise RuntimeError("cancelled")
        self.assertEqual(
            [
                call.args[0]
                for call in api.SetThreadExecutionState.call_args_list
            ],
            [0x80000003, 0x80000000],
        )
