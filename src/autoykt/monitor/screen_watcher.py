"""Top-level scheduler for configured page automation profiles."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from autoykt.agent.answer_agent import AnswerCoordinator
from autoykt.core.config import AppConfig, ConfigError
from autoykt.core.event_bus import Event, EventBus, EventType
from autoykt.core.state import WorkflowState
from autoykt.knowledge.store import KnowledgeStore
from autoykt.monitor.clicker import Clicker
from autoykt.monitor.profile_runtime import PageAutomation


logger = logging.getLogger("autoykt")


class ScreenWatcher:
    """Poll profiles serially because they share one physical mouse cursor."""

    def __init__(
        self,
        config: AppConfig,
        event_bus: EventBus,
        detect_only: bool = False,
        *,
        once: bool = False,
    ) -> None:
        self._config = config
        self._once = once
        self.exit_code = 0
        self._bus = event_bus
        profiles = config.active_page_profiles()
        if not profiles:
            raise ConfigError("no enabled page profiles are active")
        if once and len(profiles) != 1:
            raise ConfigError(
                "single-question mode requires exactly one active "
                "profile; use --profile"
            )
        prompt_path = (
            config.resolve_path(config.answering.prompt_template)
            if config.answering.prompt_template and not detect_only
            else None
        )
        self._answerer = AnswerCoordinator(
            config.answering,
            prompt_path=prompt_path,
        )
        self._knowledge_store: KnowledgeStore | None = None
        if config.knowledge.enabled and not detect_only:
            self._knowledge_store = KnowledgeStore(
                config.resolve_path(config.knowledge.database_path),
                chunk_size=config.knowledge.chunk_size_characters,
                chunk_overlap=config.knowledge.chunk_overlap_characters,
            )
        clicker = Clicker()
        self._profiles: list[PageAutomation] = []
        try:
            for profile in profiles:
                self._profiles.append(
                    PageAutomation(
                        app_config=config,
                        profile=profile,
                        event_bus=event_bus,
                        answerer=self._answerer,
                        clicker=clicker,
                        knowledge_store=self._knowledge_store,
                        detect_only=detect_only,
                    )
                )
        except BaseException:  # pylint: disable=broad-exception-caught
            for automation in self._profiles:
                automation.close_capture_devices()
            if self._knowledge_store is not None:
                self._knowledge_store.close()
            raise
        self._running = False
        self._closed = False
        self._prepared = False
        self._stop_requested = False
        self.stop_reason = "stopped"

    async def prepare(self) -> None:
        """Warm local engines once before starting the monitoring clock."""
        if not self._prepared:
            for profile in self._profiles:
                await profile.prepare()
            self._prepared = True

    def snapshot(self) -> list[dict[str, Any]]:
        """Return the latest bounded status for every monitored profile."""
        return [profile.snapshot() for profile in self._profiles]

    async def start(self) -> None:
        """Poll profiles until stopped or cancelled."""
        await self.prepare()
        if self._stop_requested:
            return
        self._running = True
        interval = self._config.runtime.poll_interval_seconds
        logger.info(
            "Monitoring profiles %s every %.2f seconds",
            [profile.profile_id for profile in self._profiles],
            interval,
        )
        while self._running:
            for profile in self._profiles:
                if not self._running:
                    break
                try:
                    await profile.poll_once()
                    if self._once and profile.completed_cycles:
                        self.exit_code = (
                            0
                            if profile.outcome
                            in {
                                "verified",
                                "already_completed",
                                "preview_only",
                                "entry_preview",
                                "captured",
                            }
                            else 1
                        )
                        print(
                            f"Outcome: {profile.outcome}; report: "
                            f"{profile.report_path}"
                        )
                        self.stop("once_complete")
                except Exception as error:  # pylint: disable=broad-exception-caught
                    # One profile must not hide observations from the others.
                    logger.exception(
                        "Polling profile '%s' failed: %s",
                        profile.profile_id,
                        error,
                    )
                    await self._bus.publish(
                        Event(
                            type=EventType.ERROR,
                            profile_id=profile.profile_id,
                            payload={
                                "source": "scheduler",
                                "error": str(error),
                            },
                        )
                    )
            if self._running and all(
                profile.needs_attention
                or profile.state == WorkflowState.STOPPED
                for profile in self._profiles
            ):
                self.exit_code = int(
                    any(profile.needs_attention for profile in self._profiles)
                )
                self.stop(
                    "manual_required" if self.exit_code else "class_finished"
                )
            if self._running:
                await asyncio.sleep(interval)

    def stop(self, reason: str = "stopped") -> None:
        """Request termination after the current bounded question cycle."""
        if self._running:
            self.stop_reason = reason
        self._stop_requested = True
        self._running = False

    async def close(self) -> None:
        """Release provider, database, and capture resources."""
        if self._closed:
            return
        self.stop()
        await asyncio.gather(
            *(profile.stop() for profile in self._profiles),
            return_exceptions=True,
        )
        await self._answerer.close()
        if self._knowledge_store is not None:
            self._knowledge_store.close()
        self._closed = True
