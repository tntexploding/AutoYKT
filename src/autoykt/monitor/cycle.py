"""Answer-cycle progress and stable submission verification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING

import numpy as np

from autoykt.monitor.detector import ImageTemplateMatcher, OptionMatch
from autoykt.monitor.image_utils import frame_change_ratio

if TYPE_CHECKING:
    from autoykt.monitor.operations import CaptureDevice, ProfileSession


class WorkflowError(RuntimeError):
    """Raised when the current answer cycle cannot safely continue."""


@dataclass
class CycleProgress:
    """Mutable timing and rearm evidence for one page profile."""

    deadline: float = 0.0
    baseline: np.ndarray | None = None
    rearm_options: dict[str, OptionMatch] | None = None
    hits: int = 0
    rearm_deadline: float = 0.0
    timeout_reported: bool = False
    trigger_name: str | None = None
    manual_required: bool = False
    blocked_since: float = 0.0
    finished_hits: int = 0
    multiple: bool = False
    notice_entered: bool = False
    allowed_options: frozenset[str] = frozenset()


async def verify_submission(
    session: ProfileSession,
    capture: CaptureDevice,
    success_matcher: ImageTemplateMatcher,
    failure_matcher: ImageTemplateMatcher,
    baseline: np.ndarray,
) -> tuple[bool, np.ndarray, str]:
    """Require stable success evidence, stopping on explicit failure."""
    settings = session.profile.verification
    if settings.success_templates and success_matcher.best(baseline):
        return (
            False,
            baseline,
            "success state was present before interaction",
        )
    deadline = time.monotonic() + settings.timeout_seconds
    stable_hits = 0
    last_frame = baseline
    evidence = ""
    while time.monotonic() < deadline:
        await asyncio.sleep(settings.poll_interval_seconds)
        last_frame = capture.grab_frame()
        failure = failure_matcher.best(last_frame)
        if failure is not None:
            return (
                False,
                last_frame,
                f"failure_template:{failure.name}:{failure.confidence:.3f}",
            )
        if settings.success_templates:
            match = success_matcher.best(last_frame)
            passed = match is not None
            evidence = (
                f"success_template:{match.name}:{match.confidence:.3f}"
                if match is not None
                else "success template not present"
            )
        else:
            ratio = frame_change_ratio(baseline, last_frame)
            passed = ratio >= settings.minimum_change_ratio
            evidence = f"change_ratio:{ratio:.4f}"
        stable_hits = stable_hits + 1 if passed else 0
        if stable_hits >= settings.stable_hits:
            return True, last_frame, evidence
    return False, last_frame, "verification timeout: " + evidence
