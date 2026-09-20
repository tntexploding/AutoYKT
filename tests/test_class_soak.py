"""Accelerated two-hour schedule through production polling and image controls.

The clock for the session advances 30 seconds per poll. Question deadlines,
selection, and verification still use real time. No browser, mouse or API is used.
"""

import asyncio
import json
import unittest
from unittest.mock import patch

from autoykt.core.config import ClickTargetConfig
from autoykt.core.event_bus import EventBus
from autoykt.monitor.class_session import ClassSession, read_session_status
from autoykt.monitor.image_utils import write_png
from autoykt.monitor.profile_runtime import PageAutomation
from autoykt.monitor.rehearsal import _Capture, _Scene, _rehearsal_config
from autoykt.monitor.screen_watcher import ScreenWatcher
from autoykt.monitor.windows import WindowUnavailable
from tests.test_class_session import _Clock
from tests import test_image_calibration as calibration_fixture
from tests.test_image_calibration import _button, GRAY
from tests.test_rehearsal import _Answers


class ClassSoakTest(unittest.IsolatedAsyncioTestCase):

    async def test_two_hours_of_repeated_questions_pauses_and_answer_rejections(
        self,
    ):
        asyncio.get_running_loop().slow_callback_duration = 2
        fixture = calibration_fixture.ImageCalibrationTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = _rehearsal_config(
            fixture._calibrate(), "example_yuketang", fixture.root / "soak"
        )
        config.knowledge.question_ocr_enabled = False
        config.runtime.poll_interval_seconds = 0.01
        profile = config.pages[0]
        profile.triggers[0].question_type = "multiple"
        profile.triggers[0].consecutive_hits = 1
        profile.rearm.minimum_change_ratio = 0.02
        profile.question_ready.delay_seconds = 0
        profile.question_ready.stable_frames = 1
        profile.verification.poll_interval_seconds = 0.001
        second = fixture.frame.copy()
        second[80:480, :700] = 255
        for key, (x, y) in zip("ABC", [(70, 140), (350, 340), (360, 190)]):
            second[y : y + 60, x : x + 60] = _button(key, GRAY)
        second_path = fixture.root / "second.png"
        write_png(second_path, second)
        images = [fixture.image_path, second_path]
        answers = _Answers(
            [None if index in (9, 19, 29) else "A,C" for index in range(30)]
        )
        scene = _Scene(config, profile)
        scene.load(images[0])
        scene.waiting = True
        profile.submit_target = ClickTargetConfig(
            point=scene.submit_point, coordinate_space="monitor"
        )
        screenshots = fixture.root / "screenshots"
        screenshots.mkdir()
        bus = EventBus()
        runtime = PageAutomation(
            config,
            profile,
            bus,
            answers,
            scene,
            None,
            capture_factory=lambda region: _Capture(scene, region, screenshots),
        )
        clock = _Clock()
        original_poll = runtime.poll_once
        introduced = 0
        next_question = 120
        last_cycles = 0
        submissions = []
        observed_pause = False
        observed_recovery = False

        async def scheduled_poll():
            nonlocal introduced, next_question, last_cycles, observed_pause, observed_recovery
            if clock.now >= next_question and introduced < 30:
                scene.load(images[introduced % 2])
                introduced += 1
                next_question += 240
            if 1500 <= clock.now < 1620:
                with patch.object(
                    runtime._session,
                    "check_page",
                    side_effect=WindowUnavailable("simulated focus loss"),
                ):
                    await original_poll()
                observed_pause |= bool(runtime.snapshot()["pause_reason"])
            else:
                await original_poll()
                observed_recovery |= (
                    clock.now >= 1620 and not runtime.snapshot()["pause_reason"]
                )
            if runtime.completed_cycles != last_cycles:
                submissions.append(list(scene.submissions))
                last_cycles = runtime.completed_cycles
            clock.now += 30

        runtime.poll_once = scheduled_poll
        with (
            patch(
                "autoykt.monitor.screen_watcher.PageAutomation",
                return_value=runtime,
            ),
            patch("autoykt.monitor.screen_watcher.Clicker") as native_clicker,
            patch(
                "autoykt.monitor.profile_runtime.ScreenCapture"
            ) as native_capture,
            patch("autoykt.monitor.class_session.monotonic", clock),
            patch("autoykt.monitor.class_session.HEARTBEAT_SECONDS", 0.005),
        ):
            watcher = ScreenWatcher(config, bus)
            session = ClassSession(config, bus, 120)
            bus_task = asyncio.create_task(bus.start())
            try:
                self.assertEqual(await session.run(watcher), 0)
            finally:
                await watcher.close()
                bus.stop()
                await bus_task
                session.refresh()
            native_capture.assert_not_called()
            native_clicker.return_value.click_point.assert_not_called()
        data = read_session_status(config)
        self.assertEqual(data["reason"], "duration_reached")
        self.assertEqual(data["elapsed_seconds"], 7200)
        self.assertEqual(introduced, 30)
        self.assertEqual(len(answers.calls), 30)
        self.assertEqual(
            data["outcome_counts"], {"verified": 27, "answer_rejected": 3}
        )
        self.assertEqual(sum(len(items) for items in submissions), 27)
        self.assertTrue(all(len(items) <= 1 for items in submissions))
        self.assertTrue(observed_pause)
        self.assertTrue(observed_recovery)
        self.assertEqual(data["assessment"], "review_required")
        events = (
            (session.directory / "events.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertGreater(len(events), 100)
        self.assertTrue(
            all(isinstance(json.loads(event), dict) for event in events)
        )
