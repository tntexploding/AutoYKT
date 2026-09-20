"""Monitoring progress must describe real observations, not heartbeat ticks."""

import unittest
from unittest.mock import patch

from autoykt.core.config import TriggerTemplateConfig
from autoykt.monitor.health import MonitoringHealth
from autoykt.monitor.windows import WindowUnavailable
from tests.test_profile_runtime import WorkflowScene


class MonitoringHealthTest(unittest.TestCase):

    def test_pauses_accumulate_once_and_do_not_advance_page_checks(self):
        health = MonitoringHealth()
        with patch(
            "autoykt.monitor.health.monotonic", return_value=10
        ) as clock:
            health.observed()
            health.detected("single")
            health.pause("covered")
            clock.return_value = 13
            health.pause("covered")
            self.assertEqual(health.snapshot()["paused_seconds"], 3)
            self.assertEqual(health.snapshot()["successful_page_checks"], 1)
            self.assertEqual(health.snapshot()["detection_scans"], 1)
            health.resume()
            clock.return_value = 30
            self.assertEqual(health.snapshot()["paused_seconds"], 3)
            health.pause("resized")
            clock.return_value = 32
            health.resume()
        self.assertEqual(health.snapshot()["pause_count"], 2)
        self.assertEqual(health.snapshot()["paused_seconds"], 5)
        self.assertEqual(health.snapshot()["last_pause_reason"], "resized")


class MonitoringProgressTest(WorkflowScene):

    async def test_window_pause_stops_scan_count_and_recovers(self):
        self.scene[20][:] = 0
        automation = self._automation()
        await automation.poll_once()
        with patch.object(
            automation._session,
            "check_page",
            side_effect=WindowUnavailable("covered"),
        ):
            await automation.poll_once()
            await automation.poll_once()
        health = automation.snapshot()["monitoring"]
        self.assertEqual(health["successful_page_checks"], 1)
        self.assertEqual(health["detection_scans"], 1)
        self.assertEqual(health["pause_count"], 1)
        await automation.poll_once()
        self.assertEqual(
            automation.snapshot()["monitoring"]["detection_scans"], 2
        )
        self.assertEqual(automation.snapshot()["pause_reason"], "")

    async def test_conflicting_question_types_do_not_request_or_click(self):
        # Preview also invokes models, but must not do so for conflicting types.
        self.config.runtime.dry_run = True
        original = self.config.pages[0].triggers[0]
        self.config.pages[0].triggers.append(
            TriggerTemplateConfig(
                name="multiple",
                path=original.path,
                question_type="multiple",
                consecutive_hits=1,
                threshold=0.99,
            )
        )
        automation = self._automation()
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertEqual(self.clicker.points, [])
        self.assertIn("conflicting", automation.snapshot()["pause_reason"])
        self.assertFalse(automation.needs_attention)
