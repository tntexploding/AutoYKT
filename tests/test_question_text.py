"""Preserve letter associations when OCR misses the button glyphs."""

import unittest

from autoykt.monitor.detector import OptionMatch
from autoykt.monitor.ocr_engine import OcrLine, OcrResult, format_question_text


def _match(x: int, y: int) -> OptionMatch:
    return {
        "ambiguous": False,
        "center": (x + 30, y + 30),
        "top_left": (x, y),
        "size": (60, 60),
        "score": 1.0,
        "selected": False,
    }


class QuestionTextTest(unittest.TestCase):

    def test_labels_variable_order_and_multiple_columns(self):
        matches = {
            "C": _match(20, 100),
            "A": _match(350, 100),
            "B": _match(20, 240),
        }
        result = OcrResult(
            "unlabeled",
            lines=(
                OcrLine("Question", (20, 20, 300, 30)),
                OcrLine("third", (100, 115, 190, 30)),
                OcrLine("first", (430, 115, 190, 30)),
                OcrLine("second", (100, 255, 190, 30)),
                OcrLine("continued", (100, 290, 190, 30)),
            ),
        )
        text = format_question_text(result, matches)
        self.assertEqual(
            text, "Question\nA: first\nB: second continued\nC: third"
        )

    def test_incomplete_geometry_does_not_invent_labels(self):
        result = OcrResult("raw", lines=(OcrLine("one", (100, 115, 100, 30)),))
        self.assertEqual(
            format_question_text(
                result,
                {"A": _match(20, 100), "B": _match(20, 240)},
            ),
            "raw",
        )
