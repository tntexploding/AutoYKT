"""Private, atomic checkpoints that prevent replaying unresolved input."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoykt.core.config import ConfigError


class OperationCheckpoint:
    """Keep the current report until the page has safely left its question."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "pending.json"

    def read(self) -> dict[str, Any] | None:
        """Load a private checkpoint; corrupt state must never permit replay."""
        if not self.path.exists():
            return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("invalid checkpoint")
            report = (self.directory / value["report"]).resolve()
            report.relative_to(self.directory.resolve())
            if report.name != "report.json" or not report.is_file():
                raise ValueError("invalid report path")
            if value["status"] not in {"input_pending", "verified", "reviewed"}:
                raise ValueError("invalid checkpoint status")
            return value
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ConfigError(
                "operation checkpoint is unreadable; "
                "inspect private pending.json before restarting"
            ) from error

    def write(self, report: Path, status: str, action: str = "") -> None:
        """Persist intent before input; refuse another report's ownership."""
        relative = str(report.resolve().relative_to(self.directory.resolve()))
        previous = self.read()
        if previous and previous["report"] != relative:
            raise ConfigError(
                "another unresolved operation owns this page profile"
            )
        value = {"report": relative, "status": status, "action": action}
        self.directory.mkdir(parents=True, exist_ok=True)
        if previous is None:
            # Exclusive creation prevents two fresh runs from taking ownership.
            with self.path.open("x", encoding="utf-8") as stream:
                json.dump(value, stream)
        else:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value), encoding="utf-8")
            temporary.replace(self.path)

    def review(self, report: Path, expected: dict[str, Any] | None) -> None:
        """Replace a reviewed attempt with a snapshot of the page to skip."""
        if self.read() != expected:
            raise ConfigError(
                "operation changed during review; stop all runs first"
            )
        relative = str(report.resolve().relative_to(self.directory.resolve()))
        value = {
            "report": relative,
            "status": "reviewed",
            "action": "skip_current",
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value), encoding="utf-8")
        temporary.replace(self.path)

    def clear(self) -> None:
        """Release the checkpoint only once a fresh page has been observed."""
        self.path.unlink(missing_ok=True)
