"""Recognize blocking page states and advance a verified result once."""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from autoykt.core.config import (
    AppConfig,
    PageAdvanceConfig,
    PageFlowConfig,
    PageGuardConfig,
    Region,
)
from autoykt.monitor.detector import ImageTemplateMatcher, TemplateMatch
from autoykt.monitor.windows import WindowUnavailable


class PageBlocked(WindowUnavailable):
    """A configured loading, manual-intervention, or end-of-class state."""

    def __init__(self, state: str) -> None:
        super().__init__(f"page state: {state}")
        self.state = state


class PageFlow:
    """Hold all input on blocking states and never repeat a result action."""

    def __init__(
        self,
        config: AppConfig,
        settings: PageFlowConfig,
        capture: Callable[[Region], np.ndarray],
        *,
        enabled: bool = True,
        before_question: bool = False,
    ) -> None:
        self._config = config
        self._settings = settings if enabled else PageFlowConfig()
        self._steps = (
            self._settings.before_question
            if before_question
            else self._settings.after_submit
        )
        self._capture = capture
        self._matchers: dict[tuple, ImageTemplateMatcher] = {}
        self._used: set[str] = set()
        self._pending: tuple[str, float] | None = None
        self._candidate = ""
        self._hits = 0
        self.verified = False

    def matches(self, template: PageGuardConfig) -> bool:
        """Match one state using a fresh capture of its calibrated region."""
        return self.locate(template) is not None

    def locate(
        self, template: PageGuardConfig, *, unique: bool = False
    ) -> TemplateMatch | None:
        """Locate a state freshly, including configured display scales."""
        key = (
            template.path,
            template.threshold,
            tuple(template.scales),
            template.match_grayscale,
            unique,
        )
        matcher = self._matchers.get(key)
        if matcher is None:
            matcher = ImageTemplateMatcher(
                [
                    (
                        "state",
                        str(self._config.resolve_path(template.path)),
                        template.threshold,
                    )
                ],
                scales=template.scales,
                unique=unique,
                match_grayscale=template.match_grayscale,
            )
            self._matchers[key] = matcher
        return matcher.best(self._capture(template.region))

    def check_blocking(self) -> None:
        """Block input, including the final check in the input thread."""
        for state in ("manual", "finished", "loading"):
            templates: list[PageGuardConfig] = getattr(self._settings, state)
            if any(self.matches(template) for template in templates):
                raise PageBlocked(state)

    def reset(self, used: set[str] | None = None) -> None:
        """Start a new question, or restore verified navigation evidence."""
        self._used = set() if used is None else used.copy()
        self._pending = None
        self._candidate = ""
        self._hits = 0
        self.verified = False

    def next_step(self) -> tuple[PageAdvanceConfig | None, bool]:
        """Return an unused button and whether result UI is still visible."""
        matches = [
            step
            for step in self._steps
            if self.locate(step.when, unique=step.click_match) is not None
        ]
        if len(matches) > 1:
            raise WindowUnavailable(
                "multiple result actions match; manual review required"
            )
        if self._pending:
            name, deadline = self._pending
            if matches and matches[0].name == name:
                if time.monotonic() >= deadline:
                    raise WindowUnavailable(
                        "result action did not change the page; "
                        "not clicking again"
                    )
                return None, True
            self._pending = None
        if not matches:
            self._candidate, self._hits = "", 0
            return None, False
        step = matches[0]
        if step.name in self._used:
            raise WindowUnavailable(
                "completed result action reappeared; manual review required"
            )
        self._hits = self._hits + 1 if self._candidate == step.name else 1
        self._candidate = step.name
        return (step if self._hits >= step.stable_hits else None), True

    def attempted(self, step: PageAdvanceConfig) -> None:
        """Remember input attempts so a failed click cannot repeat."""
        self._used.add(step.name)
        self._pending = (
            step.name,
            time.monotonic() + self._settings.transition_timeout_seconds,
        )
        self._candidate, self._hits = "", 0
