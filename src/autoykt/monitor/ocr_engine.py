"""Local OCR abstraction used by questions and live course capture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Mapping
import logging
import re

import numpy as np

from autoykt.monitor.detector import OptionMatch


logger = logging.getLogger("autoykt")


@dataclass(frozen=True)
class OcrLine:
    """One OCR text line with bounds relative to its source frame."""

    text: str
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class OcrResult:
    """Text and best-effort multiple-choice structure from one frame."""

    raw_text: str
    question: str = ""
    options: dict[str, str] = field(default_factory=dict)
    lines: tuple[OcrLine, ...] = ()


class BaseOcrEngine(ABC):
    """Interface for local OCR implementations."""

    @abstractmethod
    def recognize(self, frame: np.ndarray) -> OcrResult:
        """Recognize text from one BGR frame."""


class RapidOcrEngine(BaseOcrEngine):
    """Recognize Chinese and Latin text with RapidOCR."""

    _OPTION_PATTERN = re.compile(r"^([A-Za-z0-9])\s*[.。、:：)）]\s*(.+)")

    def __init__(self) -> None:
        from rapidocr_onnxruntime import (  # pylint: disable=import-outside-toplevel
            RapidOCR,
        )

        self._engine = RapidOCR()

    def recognize(self, frame: np.ndarray) -> OcrResult:
        """Return OCR lines and a conservative question/options split."""
        result, _ = self._engine(frame)
        if not result:
            return OcrResult(raw_text="")
        lines = [
            str(item[1]).strip() for item in result if str(item[1]).strip()
        ]
        question, options = self._parse_question(lines)
        return OcrResult(
            raw_text="\n".join(lines),
            question=question,
            options=options,
            lines=tuple(
                _ocr_line(item) for item in result if str(item[1]).strip()
            ),
        )

    @classmethod
    def _parse_question(cls, lines: list[str]) -> tuple[str, dict[str, str]]:
        question_lines: list[str] = []
        options: dict[str, str] = {}
        found_option = False
        for line in lines:
            match = cls._OPTION_PATTERN.match(line)
            if match:
                options[match.group(1).upper()] = match.group(2).strip()
                found_option = True
            elif not found_option:
                question_lines.append(line)
        return " ".join(question_lines), options


def create_ocr_engine(engine_type: str = "rapidocr") -> BaseOcrEngine:
    """Create a supported local OCR engine."""
    if engine_type == "rapidocr":
        return RapidOcrEngine()
    raise ValueError(f"unknown OCR engine: {engine_type}")


def _ocr_line(item: list) -> OcrLine:
    points = np.asarray(item[0])
    left, top = points.min(axis=0)
    right, bottom = points.max(axis=0)
    return OcrLine(
        str(item[1]).strip(),
        (int(left), int(top), int(right - left), int(bottom - top)),
    )


def format_question_text(
    result: OcrResult,
    matches: Mapping[str, OptionMatch],
) -> str:
    """Pair nearby text with letter buttons without assuming row order."""
    if (
        not result.lines
        or not matches
        or any(match["ambiguous"] for match in matches.values())
    ):
        return result.raw_text.strip()
    options: dict[str, list[str]] = {key: [] for key in matches}
    question = []
    for line in result.lines:
        left, top, _, height = line.bounds
        middle = top + height / 2
        candidates = []
        for key, match in matches.items():
            x, _ = match["top_left"]
            width, button_height = match["size"]
            if (
                left >= x + width - 2
                and abs(middle - match["center"][1]) <= button_height
            ):
                distance = (
                    abs(middle - match["center"][1]) * 2 + left - x - width
                )
                candidates.append((distance, key))
        candidates.sort()
        if candidates and (
            len(candidates) == 1 or candidates[0][0] < candidates[1][0]
        ):
            options[candidates[0][1]].append(line.text)
        elif line.text.strip() not in matches:
            question.append(line.text)
    if any(not value for value in options.values()):
        return result.raw_text.strip()
    return "\n".join(
        [
            *question,
            *(
                f"{key}: {' '.join(lines)}"
                for key, lines in sorted(options.items())
            ),
        ]
    )
