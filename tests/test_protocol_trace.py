"""Temporary live diagnostics must remain private, bounded and non-blocking."""

from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from autoykt.server.config import ServerConfig
from autoykt.server.protocol_trace import ProtocolTrace


class ProtocolTraceTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = ServerConfig(data_dir=self.temporary.name)
        self.now = datetime.now(timezone.utc)
        self.config.diagnostics.capture_until = self.now + timedelta(hours=1)
        self.trace = ProtocolTrace(self.config)

    def test_window_retains_shapes_without_credentials_or_question_content(
        self,
    ):
        self.config.diagnostics.capture_from = self.now + timedelta(minutes=5)
        message = {
            "op": "hello",
            "auth": "private-token",
            "timeline": [
                {
                    "type": "problem",
                    "sid": "private-id",
                    "body": "private-question",
                    "limit": 60,
                }
            ],
        }
        self.trace.record("private-lesson", message)
        self.assertFalse(self.trace.path.exists())
        self.config.diagnostics.capture_from = self.now - timedelta(minutes=1)
        self.trace.record("private-lesson", message)
        content = self.trace.path.read_text(encoding="utf-8")
        self.assertNotIn("private-", content)
        record = json.loads(content)
        self.assertEqual(record["op"], "hello")
        self.assertEqual(
            record["shape"]["timeline"]["items"][0]["limit"], "int"
        )
        if os.name != "nt":
            self.assertEqual(self.trace.path.stat().st_mode & 0o777, 0o600)
        self.config.diagnostics.capture_until = self.now - timedelta(seconds=1)
        self.trace.record("private-lesson", message)
        self.assertEqual(self.trace.path.read_text(encoding="utf-8"), content)
        with self.assertRaises(ValueError):
            type(self.config.diagnostics)(capture_until="2026-11-05T12:30:00")

    def test_full_or_unwritable_capture_does_not_interrupt_the_listener(self):
        self.config.diagnostics.max_bytes = 1024
        for _ in range(30):
            self.trace.record("lesson", {"op": "notification", "items": []})
        self.assertLessEqual(self.trace.path.stat().st_size, 1024)
        self.config.diagnostics.max_bytes = 5000
        with patch(
            "autoykt.server.protocol_trace.os.open", side_effect=PermissionError
        ):
            with self.assertLogs("autoykt.server", level="WARNING") as logs:
                self.trace.record("lesson", {"op": "notification"})
                self.trace.record("lesson", {"op": "notification"})
        self.assertEqual(len(logs.output), 1)


if __name__ == "__main__":
    unittest.main()
