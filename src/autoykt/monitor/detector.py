"""Robust OpenCV template matching for triggers and answer options."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import math
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np

from autoykt.core.config import (
    AppConfig,
    AnswerStyleConfig,
    ButtonColorsConfig,
    QuestionType,
)
from autoykt.monitor.image_utils import read_image


# OpenCV exports its native API dynamically.
# pylint: disable=no-member


logger = logging.getLogger("autoykt")


@dataclass(frozen=True)
class TemplateMatch:
    """The best match for one configured template."""

    name: str
    action: str
    confidence: float
    top_left: tuple[int, int]
    size: tuple[int, int]
    question_type: QuestionType = "single"

    @property
    def center(self) -> tuple[int, int]:
        """Return the match center in frame-relative coordinates."""
        return (
            self.top_left[0] + self.size[0] // 2,
            self.top_left[1] + self.size[1] // 2,
        )


@dataclass(frozen=True)
class DetectionObservation:
    """All currently visible templates and any newly fired trigger."""

    triggered: TemplateMatch | None
    present: tuple[TemplateMatch, ...]
    ambiguous: bool = False


@dataclass
class _RuleState:
    name: str
    action: str
    template: np.ndarray | None
    threshold: float
    required_hits: int
    question_type: QuestionType = "single"
    consecutive_hits: int = 0
    fired: bool = False


def _read_template(path: str | Path, name: str) -> np.ndarray | None:
    template_path = Path(path)
    if not template_path.is_file():
        logger.warning("Template '%s' does not exist: %s", name, template_path)
        return None
    try:
        template = read_image(template_path)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"invalid template '{name}': {template_path}"
        ) from error
    if not np.any(np.std(template, axis=(0, 1)) > 0):
        raise ValueError(f"template '{name}' has no spatial detail")
    return template


def _best_match(
    frame: np.ndarray,
    template: np.ndarray,
    unique_threshold: float | None = None,
) -> tuple[float, tuple[int, int]] | None:
    if frame.ndim not in (2, 3) or template.ndim != frame.ndim:
        return None
    frame_height, frame_width = frame.shape[:2]
    template_height, template_width = template.shape[:2]
    if template_width > frame_width or template_height > frame_height:
        return None
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, maximum, _, location = cv2.minMaxLoc(result)
    if not math.isfinite(maximum):
        return None
    if unique_threshold is not None:
        x, y = location
        result[
            max(0, y - template_height // 2) : y + template_height // 2 + 1,
            max(0, x - template_width // 2) : x + template_width // 2 + 1,
        ] = -1
        if np.any(result >= unique_threshold):
            return None
    return float(maximum), (int(location[0]), int(location[1]))


class MultiTemplateDetector:
    """Debounce multiple templates while firing once per appearance."""

    def __init__(self, rules: Sequence[Mapping[str, Any]]) -> None:
        self._states: list[_RuleState] = []
        names: set[str] = set()
        for index, rule in enumerate(rules):
            name = str(rule.get("name") or f"template_{index + 1}")
            if name in names:
                raise ValueError(f"duplicate template rule name: {name}")
            names.add(name)
            template_path = str(
                rule.get("path") or rule.get("template_path") or ""
            )
            required_hits = int(
                rule.get(
                    "consecutive_hits",
                    rule.get("debounce_frames", 3),
                )
            )
            if required_hits < 1:
                raise ValueError("consecutive_hits must be greater than zero")
            threshold = float(rule.get("threshold", 0.85))
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("template threshold must be between 0 and 1")
            old_action = str(rule.get("action", "answer"))
            action = {
                "question_detected": "answer",
                "notify_only": "notify",
            }.get(old_action, old_action)
            self._states.append(
                _RuleState(
                    name=name,
                    action=action,
                    template=_read_template(template_path, name),
                    threshold=threshold,
                    required_hits=required_hits,
                    question_type=rule.get("question_type", "single"),
                )
            )

    def observe(self, frame: np.ndarray) -> DetectionObservation:
        """Update every rule and return present and newly triggered matches."""
        present: list[TemplateMatch] = []
        newly_triggered: list[TemplateMatch] = []
        for state in self._states:
            if state.template is None:
                continue
            result = _best_match(frame, state.template)
            if result is None or result[0] < state.threshold:
                state.consecutive_hits = 0
                state.fired = False
                continue
            confidence, location = result
            height, width = state.template.shape[:2]
            match = TemplateMatch(
                name=state.name,
                action=state.action,
                question_type=state.question_type,
                confidence=confidence,
                top_left=location,
                size=(int(width), int(height)),
            )
            present.append(match)
            state.consecutive_hits += 1
            if (
                state.consecutive_hits >= state.required_hits
                and not state.fired
            ):
                state.fired = True
                newly_triggered.append(match)

        if (
            len(
                {
                    item.question_type
                    for item in present
                    if item.action == "answer"
                }
            )
            > 1
        ):
            # Different question types must not be chosen by relative score.
            # Reset their debounce so a later unambiguous frame can fire.
            for state in self._states:
                if state.action == "answer":
                    state.consecutive_hits = 0
                    state.fired = False
            return DetectionObservation(None, tuple(present), ambiguous=True)
        trigger = max(
            newly_triggered,
            key=lambda item: item.confidence,
            default=None,
        )
        if trigger is not None:
            logger.info(
                "Template '%s' triggered at %.3f (%s)",
                trigger.name,
                trigger.confidence,
                trigger.action,
            )
        return DetectionObservation(triggered=trigger, present=tuple(present))

    def detect(
        self, frame: np.ndarray
    ) -> tuple[str | None, float, tuple[int, int] | None, str | None]:
        """Compatibility wrapper returning the newly fired trigger."""
        match = self.observe(frame).triggered
        if match is None:
            return None, 0.0, None, None
        return match.name, match.confidence, match.top_left, match.action

    def reset(self, name: str | None = None) -> None:
        """Re-arm one template or every template."""
        for state in self._states:
            if name is None or state.name == name:
                state.consecutive_hits = 0
                state.fired = False


class QuestionDetector:
    """Compatibility adapter around a single question trigger."""

    def __init__(
        self,
        template_path: str,
        threshold: float = 0.90,
        debounce_frames: int = 3,
    ) -> None:
        self._threshold = threshold
        self._debounce_frames = debounce_frames
        self._detector = MultiTemplateDetector(
            [
                {
                    "name": "question",
                    "path": template_path,
                    "threshold": threshold,
                    "consecutive_hits": debounce_frames,
                    "action": "answer",
                }
            ]
        )

    def detect(
        self, frame: np.ndarray
    ) -> tuple[bool, float, tuple[int, int] | None]:
        """Return whether the single rule fired in this frame."""
        name, confidence, location, _ = self._detector.detect(frame)
        return name is not None, confidence, location

    def reset(self) -> None:
        """Re-arm the question rule."""
        self._detector.reset()

    def update_template(self, template_path: str) -> None:
        """Replace the question template and reset debounce state."""
        self._detector = MultiTemplateDetector(
            [
                {
                    "name": "question",
                    "path": template_path,
                    "threshold": self._threshold,
                    "consecutive_hits": self._debounce_frames,
                    "action": "answer",
                }
            ]
        )

    @property
    def threshold(self) -> float:
        """Return the configured threshold."""
        return self._threshold

    @property
    def debounce_frames(self) -> int:
        """Return the configured consecutive frame count."""
        return self._debounce_frames


class OptionMatch(TypedDict):
    """Location and observed selection state; None means unrecognized color."""

    ambiguous: bool
    center: tuple[int, int]
    score: float
    top_left: tuple[int, int]
    size: tuple[int, int]
    selected: bool | None


class OptionTemplateDetector:
    """Locate configured letter markers regardless of order or count."""

    @classmethod
    def from_config(
        cls, config: AppConfig, style: AnswerStyleConfig
    ) -> "OptionTemplateDetector":
        """Resolve private option assets consistently for image workflows."""
        return cls(
            {
                key: str(config.resolve_path(path))
                for key, path in style.option_templates.items()
            },
            style.option_match_threshold,
            match_grayscale=style.match_grayscale,
            button_colors=style.button_colors,
        )

    def __init__(
        self,
        template_paths: Mapping[str, str],
        threshold: float = 0.85,
        *,
        match_grayscale: bool = False,
        button_colors: ButtonColorsConfig | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("option threshold must be between 0 and 1")
        self._threshold = threshold
        self._match_grayscale = match_grayscale
        self._button_colors = button_colors
        self._templates: dict[str, np.ndarray] = {}
        for raw_key, path in template_paths.items():
            key = raw_key.strip().upper()
            template = _read_template(path, f"option {key}")
            if template is not None:
                if match_grayscale:
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    if not np.std(template):
                        raise ValueError(
                            f"option {key} has no grayscale spatial detail"
                        )
                self._templates[key] = template

    def detect(self, frame: np.ndarray) -> dict[str, OptionMatch]:
        """Return the best frame-relative location for each visible option."""
        matches: dict[str, OptionMatch] = {}
        if frame.ndim != 3 or frame.shape[2] != 3:
            return matches
        search_frame = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._match_grayscale
            else frame
        )
        for key, template in self._templates.items():
            result = _best_match(search_frame, template)
            if result is None or result[0] < self._threshold:
                continue
            confidence, location = result
            height, width = template.shape[:2]
            scores = cv2.matchTemplate(
                search_frame, template, cv2.TM_CCOEFF_NORMED
            )
            x, y = location
            # Ignore neighboring peaks for the same marker; another distant
            # peak means there is no unique location for this option.
            scores[
                max(0, y - height // 2) : y + height // 2 + 1,
                max(0, x - width // 2) : x + width // 2 + 1,
            ] = -1
            ambiguous = bool(np.any(scores >= self._threshold))
            matches[key] = {
                "ambiguous": ambiguous,
                "center": (
                    location[0] + width // 2,
                    location[1] + height // 2,
                ),
                "score": confidence,
                "top_left": location,
                "size": (int(width), int(height)),
                "selected": (
                    self._selected(frame[y : y + height, x : x + width])
                    if not ambiguous
                    else None
                ),
            }
        # Two labels may not claim the same letter button, even if each has
        # only one peak. Grayscale matching must not hide this ambiguity.
        items = list(matches.items())
        for index, (_, first) in enumerate(items):
            for _, second in items[index + 1 :]:
                if all(
                    abs(first["center"][axis] - second["center"][axis])
                    < min(first["size"][axis], second["size"][axis]) / 2
                    for axis in (0, 1)
                ):
                    first["ambiguous"] = second["ambiguous"] = True
                    first["selected"] = second["selected"] = None
        matches.update(self._unrecognized_buttons(frame, matches))
        return matches

    def _unrecognized_buttons(
        self,
        frame: np.ndarray,
        matches: dict[str, OptionMatch],
    ) -> dict[str, OptionMatch]:
        """Flag button-shaped color regions whose letters were not located."""
        colors = self._button_colors
        if colors is None or not self._templates:
            return {}
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for rgb in (colors.selected_rgb, colors.unselected_rgb):
            bgr = np.asarray(rgb[::-1], dtype=np.int16)
            color_mask = cv2.inRange(
                frame,
                np.clip(bgr - colors.tolerance, 0, 255).astype(np.uint8),
                np.clip(bgr + colors.tolerance, 0, 255).astype(np.uint8),
            )
            mask = cv2.bitwise_or(mask, color_mask)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        sizes = (
            [colors.button_size]
            if colors.button_size is not None
            else [
                (item.shape[1], item.shape[0])
                for item in self._templates.values()
            ]
        )
        unknown: dict[str, OptionMatch] = {}
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if (
                not any(
                    0.65 * expected_width <= width <= 1.35 * expected_width
                    and 0.65 * expected_height
                    <= height
                    <= 1.35 * expected_height
                    for expected_width, expected_height in sizes
                )
                or cv2.contourArea(contour) < 0.45 * width * height
            ):
                continue
            center = (x + width // 2, y + height // 2)
            recognized = [
                match
                for match in matches.values()
                if abs(match["center"][0] - center[0]) < width / 3
                and abs(match["center"][1] - center[1]) < height / 3
            ]
            if recognized:
                if colors.button_size is not None:
                    for match in recognized:
                        # Glyph crops locate either circles or squares. Mask
                        # the whole observed button when checking text changes.
                        match["top_left"] = (x, y)
                        match["size"] = (width, height)
                continue
            unknown[f"unrecognized_{len(unknown) + 1}"] = {
                "ambiguous": True,
                "center": center,
                "score": 0.0,
                "top_left": (x, y),
                "size": (width, height),
                "selected": None,
            }
        return unknown

    def _selected(self, patch: np.ndarray) -> bool | None:
        colors = self._button_colors
        if colors is None:
            return None
        # Tight button crops contain mostly background. The median excludes
        # white letter strokes and anti-aliased edges without assuming RGB.
        rgb = np.median(patch, axis=(0, 1))[::-1]
        for selected, expected in (
            (True, colors.selected_rgb),
            (False, colors.unselected_rgb),
        ):
            if np.max(np.abs(rgb - expected)) <= colors.tolerance:
                return selected
        return None


class ImageTemplateMatcher:
    """Check whether any non-debounced state template is present."""

    def __init__(
        self,
        templates: Sequence[tuple[str, str, float]],
        *,
        scales: Sequence[float] = (1.0,),
        unique: bool = False,
        match_grayscale: bool = False,
    ) -> None:
        self._unique = unique
        self._grayscale = match_grayscale
        self._templates: list[tuple[str, np.ndarray, float]] = []
        for name, path, threshold in templates:
            template = _read_template(path, name)
            if template is not None:
                if match_grayscale:
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    if not np.std(template):
                        raise ValueError(
                            "state template has no grayscale detail"
                        )
                for scale in scales:
                    resized = (
                        template
                        if scale == 1
                        else cv2.resize(
                            template,
                            (
                                max(1, round(template.shape[1] * scale)),
                                max(1, round(template.shape[0] * scale)),
                            ),
                        )
                    )
                    self._templates.append((name, resized, threshold))

    def best(self, frame: np.ndarray) -> TemplateMatch | None:
        """Return the highest-confidence configured state match."""
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matches: list[TemplateMatch] = []
        for name, template, threshold in self._templates:
            result = _best_match(
                frame, template, threshold if self._unique else None
            )
            if result is None or result[0] < threshold:
                continue
            confidence, location = result
            height, width = template.shape[:2]
            matches.append(
                TemplateMatch(
                    name=name,
                    action="state",
                    confidence=confidence,
                    top_left=location,
                    size=(int(width), int(height)),
                )
            )
        best = max(matches, key=lambda item: item.confidence, default=None)
        if (
            self._unique
            and best is not None
            and any(
                abs(match.center[0] - best.center[0]) > best.size[0] // 2
                or abs(match.center[1] - best.center[1]) > best.size[1] // 2
                for match in matches
            )
        ):
            return None
        return best
