"""Scheduler regressions using mock profiles instead of desktop devices."""

from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock, patch

from autoykt.core.config import load_config
from autoykt.core.event_bus import EventBus
from autoykt.monitor.screen_watcher import ScreenWatcher


class ScreenWatcherRegressionTest(unittest.IsolatedAsyncioTestCase):
    """Detection-only startup and stopping do not trigger unrelated work."""

    async def test_detect_only_ignores_prompt_and_knowledge_initialization(
        self,
    ) -> None:
        config = load_config(Path(__file__).parents[1] / "config.example.yaml")
        config.pages[0].enabled = True
        config.answering.prompt_template = "not-created-yet.j2"
        config.knowledge.enabled = True
        profile = Mock(prepare=AsyncMock(), stop=AsyncMock())
        with (
            patch(
                "autoykt.monitor.screen_watcher.PageAutomation",
                return_value=profile,
            ),
            patch("autoykt.monitor.screen_watcher.KnowledgeStore") as store,
        ):
            watcher = ScreenWatcher(config, EventBus(), detect_only=True)
        try:
            store.assert_not_called()
        finally:
            await watcher.close()

    async def test_stop_does_not_poll_another_page(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config.example.yaml")
        config.pages[0].enabled = True
        config.runtime.poll_interval_seconds = 0.001
        profile = Mock(prepare=AsyncMock(), stop=AsyncMock())
        with patch(
            "autoykt.monitor.screen_watcher.PageAutomation",
            return_value=profile,
        ):
            watcher = ScreenWatcher(config, EventBus(), detect_only=True)
        second = Mock(
            poll_once=AsyncMock(), prepare=AsyncMock(), stop=AsyncMock()
        )
        profile.poll_once = AsyncMock(side_effect=watcher.stop)
        watcher._profiles.append(second)
        try:
            await watcher.start()
            second.poll_once.assert_not_awaited()
        finally:
            await watcher.close()


class SingleQuestionTest(unittest.IsolatedAsyncioTestCase):
    """A single-question run ends after its first terminal attempt."""

    async def test_once_stops_on_success_preview_and_failure(self):
        for outcome, code in (
            ("verified", 0),
            ("already_completed", 0),
            ("preview_only", 0),
            ("failed", 1),
        ):
            with self.subTest(outcome=outcome):
                config = load_config(
                    Path(__file__).parents[1] / "config.example.yaml"
                )
                config.pages[0].enabled = True
                profile = Mock(
                    prepare=AsyncMock(),
                    stop=AsyncMock(),
                    completed_cycles=0,
                    outcome=outcome,
                    report_path=Path("private/report.json"),
                )

                async def poll():
                    profile.completed_cycles += 1

                profile.poll_once = AsyncMock(side_effect=poll)
                with patch(
                    "autoykt.monitor.screen_watcher.PageAutomation",
                    return_value=profile,
                ):
                    watcher = ScreenWatcher(
                        config, EventBus(), detect_only=True, once=True
                    )
                try:
                    await watcher.start()
                    profile.poll_once.assert_awaited_once()
                    self.assertEqual(watcher.exit_code, code)
                finally:
                    await watcher.close()

    async def test_once_rejects_multiple_active_profiles(self):
        from autoykt.core.config import ConfigError

        config = load_config(Path(__file__).parents[1] / "config.example.yaml")
        config.pages[0].enabled = True
        other = config.pages[0].model_copy(update={"id": "other"})
        config.pages.append(other)
        with self.assertRaisesRegex(ConfigError, "exactly one"):
            ScreenWatcher(config, EventBus(), once=True)


class SupervisedExitTest(unittest.IsolatedAsyncioTestCase):
    """A finished class or manual takeover terminates continuous monitoring."""

    async def test_stops_when_every_profile_is_finished_or_needs_attention(
        self,
    ):
        from autoykt.core.state import WorkflowState

        for needs_attention, state, code in (
            (False, WorkflowState.STOPPED, 0),
            (True, WorkflowState.ERROR, 1),
        ):
            with self.subTest(state=state):
                config = load_config(
                    Path(__file__).parents[1] / "config.example.yaml"
                )
                config.pages[0].enabled = True
                profile = Mock(
                    prepare=AsyncMock(),
                    stop=AsyncMock(),
                    poll_once=AsyncMock(),
                    needs_attention=needs_attention,
                    state=state,
                )
                with patch(
                    "autoykt.monitor.screen_watcher.PageAutomation",
                    return_value=profile,
                ):
                    watcher = ScreenWatcher(
                        config, EventBus(), detect_only=True
                    )
                try:
                    await watcher.start()
                    profile.poll_once.assert_awaited_once()
                    self.assertEqual(watcher.exit_code, code)
                finally:
                    await watcher.close()
