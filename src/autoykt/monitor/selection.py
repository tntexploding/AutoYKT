"""State-aware selection and visual guards for single and multiple choice."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import time
from typing import TYPE_CHECKING

import numpy as np

from autoykt.core.config import Region
from autoykt.monitor.detector import OptionMatch
from autoykt.monitor.image_utils import frame_change_ratio

if TYPE_CHECKING:
    from autoykt.monitor.clicker import MouseController
    from autoykt.monitor.operations import (
        ActionPlan,
        CaptureDevice,
        ProfileSession,
    )


def selected_options(matches: Mapping[str, OptionMatch]) -> frozenset[str]:
    """Require a unique location and recognized state for every visible key."""
    if not matches:
        raise RuntimeError("no answer buttons were located")
    if any(
        match["ambiguous"] or match["selected"] is None
        for match in matches.values()
    ):
        raise RuntimeError(
            "answer button location or selection state is ambiguous"
        )
    return frozenset(key for key, match in matches.items() if match["selected"])


def selection_updates(
    answer: str, matches: Mapping[str, OptionMatch], *, multiple: bool
) -> tuple[str, ...]:
    """Plan only state changes; preserve options already selected correctly."""
    requested = answer.split(",")
    desired = frozenset(requested)
    if (
        not answer
        or len(desired) != len(requested)
        or not desired <= matches.keys()
    ):
        raise RuntimeError("answer contains unavailable or duplicate options")
    current = selected_options(matches)
    if not multiple:
        if len(desired) != 1 or len(current) > 1:
            raise RuntimeError(
                "single-choice selection contains multiple options"
            )
        return tuple(sorted(desired)) if current != desired else ()
    return (*sorted(current - desired), *sorted(desired - current))


def assert_same_layout(
    previous: Mapping[str, OptionMatch], current: Mapping[str, OptionMatch]
) -> None:
    """Stop if an option appears, disappears, moves, or becomes ambiguous."""
    selected_options(previous)
    selected_options(current)
    if previous.keys() != current.keys() or any(
        previous[key]["center"] != current[key]["center"]
        or previous[key]["size"] != current[key]["size"]
        for key in previous
    ):
        raise RuntimeError("answer button layout changed")


def selection_change_ratio(
    before: np.ndarray,
    after: np.ndarray,
    region: Region,
    answer_region: Region,
    matches: Mapping[str, OptionMatch],
) -> float:
    """Ignore only known button rectangles, preserving all question text."""
    if before.shape != after.shape:
        return 1.0
    first, second = before.copy(), after.copy()
    for match in matches.values():
        x = answer_region[0] + match["top_left"][0] - region[0]
        y = answer_region[1] + match["top_left"][1] - region[1]
        width, height = match["size"]
        left, top = max(0, x), max(0, y)
        right, bottom = min(first.shape[1], x + width), min(
            first.shape[0], y + height
        )
        if right > left and bottom > top:
            first[top:bottom, left:right] = 0
            second[top:bottom, left:right] = 0
    return frame_change_ratio(first, second)


async def apply_checked_selection(
    session: ProfileSession,
    capture: CaptureDevice,
    clicker: MouseController,
    plan: ActionPlan,
    question_frame: np.ndarray,
    answer_frame: np.ndarray,
    *,
    multiple: bool,
) -> dict[str, OptionMatch]:
    """Verify each toggle before another option or submission can be clicked."""
    reference = session.inspect_options(answer_frame)
    expected = selected_options(reference)
    required = selection_updates(plan.option, reference, multiple=multiple)
    actions = tuple(
        click for click in plan.clicks if click.label.startswith("option_")
    )
    if (
        tuple(click.label.removeprefix("option_") for click in actions)
        != required
    ):
        raise RuntimeError("selection plan no longer matches the current state")
    for index, action in enumerate(actions):
        key = action.label.removeprefix("option_")
        applied = await asyncio.to_thread(
            session.click,
            action,
            clicker,
            question_frame=question_frame,
            answer_frame=answer_frame,
            selection_reference=reference,
        )
        if not applied:
            raise RuntimeError(f"clicking option {key} failed")
        expected = expected ^ {key} if multiple else frozenset({key})
        reference, answer_frame = await _wait_for_selection(
            session,
            capture,
            reference,
            expected,
            question_frame,
        )
        session.save_frame(f"selection_{index + 1}_{key}", answer_frame)
        session.record(
            "selection_verified", option=key, selected=sorted(expected)
        )
    # Even a plan with no option clicks must recheck selection before submit.
    current = session.inspect_options(capture.grab_frame())
    assert_same_layout(reference, current)
    if selected_options(current) != frozenset(plan.option.split(",")):
        raise RuntimeError("selected options do not match the full answer")
    return current


async def _wait_for_selection(
    session: ProfileSession,
    capture: CaptureDevice,
    previous: Mapping[str, OptionMatch],
    expected: frozenset[str],
    question_frame: np.ndarray,
) -> tuple[dict[str, OptionMatch], np.ndarray]:
    settings = session.profile.answer_style
    deadline = time.monotonic() + settings.selection_timeout_seconds
    while time.monotonic() < deadline:
        await asyncio.sleep(0.1)
        session.check_page()
        frame = capture.grab_frame()
        current = session.inspect_options(frame)
        assert_same_layout(previous, current)
        profile = session.profile
        if (
            selection_change_ratio(
                question_frame,
                session.capture(profile.regions.question),
                profile.regions.question,
                profile.regions.answer_region,
                previous,
            )
            > profile.question_ready.maximum_change_ratio
        ):
            raise RuntimeError("question changed after selection")
        actual = selected_options(current)
        if actual == expected:
            return current, frame
        if actual != selected_options(previous):
            raise RuntimeError("unexpected option state after selection")
    raise RuntimeError(
        "option click did not reach the expected selection state"
    )
