"""Bounded, private message-shape capture for a scheduled classroom test."""

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import re


logger = logging.getLogger("autoykt.server")


def message_shape(value, depth=0):
    """Retain field names and types, without text, IDs, URLs or credentials."""
    if depth >= 8:
        return type(value).__name__
    if isinstance(value, dict):
        return {
            str(key)[:100]: message_shape(item, depth + 1)
            for key, item in list(value.items())[:60]
        }
    if isinstance(value, list):
        return {
            "length": len(value),
            "items": [message_shape(item, depth + 1) for item in value[:3]],
        }
    if isinstance(value, str):
        return {"type": "str", "length": len(value)}
    return type(value).__name__


class ProtocolTrace:
    """Observe incoming frames only; never affect classroom processing."""

    def __init__(self, config):
        self.policy = config.diagnostics
        self.path = config.storage / "protocol-shapes.jsonl"
        self.warned = False

    def record(self, lesson, data):
        """Write within the enabled window and the configured byte limit."""
        now = datetime.now(timezone.utc)
        policy = self.policy
        if (
            policy.capture_until is None
            or now >= policy.capture_until
            or (policy.capture_from is not None and now < policy.capture_from)
        ):
            return
        operation = data.get("op") if isinstance(data, dict) else None
        # Only the protocol operation is retained as a value. All other
        # string/numeric values become types, so auth and lesson tokens cannot
        # leak through a server response carrying an unexpected field.
        if not isinstance(operation, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_:-]{0,63}", operation
        ):
            operation = None
        line = (
            json.dumps(
                {
                    "timestamp": now.isoformat(),
                    "lesson_ref": hashlib.sha256(
                        str(lesson).encode()
                    ).hexdigest()[:12],
                    "op": operation,
                    "shape": message_shape(data),
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        try:
            size = self.path.stat().st_size if self.path.exists() else 0
            if size + len(line) > policy.max_bytes:
                return
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
            )
            with os.fdopen(descriptor, "ab") as stream:
                stream.write(line)
        except OSError:
            if not self.warned:
                logger.warning("protocol shape capture unavailable")
                self.warned = True
