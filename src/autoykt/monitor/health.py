"""Bounded evidence that page checks advance independently of heartbeats."""

from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any


@dataclass
class MonitoringHealth:
    """Count successful observations and measure recoverable pauses."""

    successful_checks: int = 0
    detections: int = 0
    last_page_checked_at: str | None = None
    last_detection_at: str | None = None
    last_trigger: str | None = None
    pause_count: int = 0
    last_pause_reason: str = ""
    _paused_since: float | None = None
    _paused_seconds: float = 0.0

    def observed(self) -> None:
        """Record an actual successful page check, not a timer tick."""
        self.successful_checks += 1
        self.last_page_checked_at = datetime.now(timezone.utc).isoformat()

    def detected(self, trigger: str | None) -> None:
        """Record a completed template scan, whether or not a rule matched."""
        self.detections += 1
        self.last_detection_at = datetime.now(timezone.utc).isoformat()
        if trigger is not None:
            self.last_trigger = trigger

    def pause(self, reason: str) -> None:
        """Count a pause once and retain its reason after recovery."""
        if self._paused_since is None:
            self._paused_since = monotonic()
            self.pause_count += 1
        self.last_pause_reason = reason

    def resume(self) -> None:
        """Accumulate pause time once when observations resume or stop."""
        if self._paused_since is not None:
            self._paused_seconds += max(0.0, monotonic() - self._paused_since)
            self._paused_since = None

    def snapshot(self) -> dict[str, Any]:
        """Return bounded operational evidence for session status."""
        active_pause = (
            max(0.0, monotonic() - self._paused_since)
            if self._paused_since is not None
            else 0.0
        )
        return {
            "successful_page_checks": self.successful_checks,
            "detection_scans": self.detections,
            "last_page_checked_at": self.last_page_checked_at,
            "last_detection_at": self.last_detection_at,
            "last_trigger": self.last_trigger,
            "pause_count": self.pause_count,
            "paused_seconds": round(self._paused_seconds + active_pause, 3),
            "last_pause_reason": self.last_pause_reason,
        }
