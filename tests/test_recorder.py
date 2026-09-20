"""Regressions for transient failures while recording course slides."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from autoykt.knowledge.recorder import LiveKnowledgeRecorder
from autoykt.knowledge.store import KnowledgeStore
from autoykt.monitor.ocr_engine import OcrResult


class RecorderRegressionTest(unittest.IsolatedAsyncioTestCase):
    """An OCR failure must not mark a slide as already recorded."""

    async def test_retries_the_same_slide_after_ocr_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with KnowledgeStore(Path(directory) / "knowledge.db") as store:
                engine = Mock()
                engine.recognize.side_effect = [
                    RuntimeError("transient"),
                    OcrResult("课程特有知识" * 10),
                ]
                with patch(
                    "autoykt.knowledge.recorder.ScreenCapture"
                ) as capture_type:
                    capture_type.return_value.grab_frame.return_value = (
                        np.zeros((20, 20, 3), dtype=np.uint8)
                    )
                    recorder = LiveKnowledgeRecorder(
                        store, engine, "course", 1, (0, 0, 20, 20)
                    )
                iterations = 0

                async def advance(_):
                    nonlocal iterations
                    iterations += 1
                    if iterations == 2:
                        recorder.stop()

                with patch(
                    "autoykt.knowledge.recorder.asyncio.sleep",
                    new=AsyncMock(side_effect=advance),
                ):
                    await recorder.run()
                self.assertEqual(engine.recognize.call_count, 2)
                self.assertEqual(store.document_count("course"), 1)
                self.assertFalse(store.search("another_course", "课程特有知识"))
