"""Continuously OCR changed lecture slides into a course knowledge store."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging

import numpy as np

from autoykt.knowledge.store import KnowledgeStore
from autoykt.monitor.ocr_engine import BaseOcrEngine
from autoykt.monitor.image_utils import frame_change_ratio
from autoykt.monitor.screen_capture import ScreenCapture


logger = logging.getLogger("autoykt")


class LiveKnowledgeRecorder:
    """Capture changed slides and persist useful OCR text until cancelled."""

    def __init__(
        self,
        store: KnowledgeStore,
        ocr_engine: BaseOcrEngine,
        course_id: str,
        monitor_index: int,
        region: tuple[int, int, int, int],
        interval_seconds: float = 2.0,
        minimum_change_ratio: float = 0.05,
        minimum_characters: int = 20,
    ) -> None:
        self._store = store
        self._ocr = ocr_engine
        self._course_id = course_id
        self._interval = interval_seconds
        self._minimum_change = minimum_change_ratio
        self._minimum_characters = minimum_characters
        self._capture = ScreenCapture(
            roi=region,
            monitor_index=monitor_index,
            screenshot_dir=None,
        )
        self._running = False

    async def run(self) -> None:
        """Record slide text until ``stop`` or task cancellation."""
        self._running = True
        baseline: np.ndarray | None = None
        while self._running:
            frame = self._capture.grab_frame()
            if baseline is None or frame_change_ratio(baseline, frame) >= (
                self._minimum_change
            ):
                if await self._record_frame(frame):
                    baseline = frame
            await asyncio.sleep(self._interval)

    async def _record_frame(self, frame: np.ndarray) -> bool:
        try:
            result = await asyncio.to_thread(self._ocr.recognize, frame)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # OCR engines expose varied native and model-runtime exceptions.
            logger.warning("Live OCR failed: %s", error)
            return False
        text = result.raw_text.strip()
        if len(text) < self._minimum_characters:
            return False
        captured_at = datetime.now(timezone.utc)
        source = "live://slide/" + captured_at.strftime("%Y%m%dT%H%M%S.%fZ")
        ingest_result = await asyncio.to_thread(
            self._store.add_document,
            self._course_id,
            source,
            "实时课程画面 " + captured_at.astimezone().strftime("%H:%M:%S"),
            text,
        )
        if ingest_result.added:
            logger.info(
                "Recorded live course material with %d chunk(s)",
                ingest_result.chunks,
            )
        return True

    def stop(self) -> None:
        """Stop recording and release the capture handle."""
        self._running = False
        self._capture.close()
