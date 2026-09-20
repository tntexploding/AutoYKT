"""Tests for conservative answer parsing and consensus."""

import unittest

from autoykt.agent.consensus import build_consensus, parse_model_answer
from autoykt.agent.models import ModelAnswer


class AnswerParsingTest(unittest.TestCase):

    def test_parses_json_response(self) -> None:
        option, confidence = parse_model_answer(
            '{"answer":"B","confidence":0.75}',
            frozenset({"A", "B", "C", "D"}),
        )
        self.assertEqual(option, "B")
        self.assertEqual(confidence, 0.75)

    def test_rejects_unlabeled_reasoning_letter(self) -> None:
        option, _ = parse_model_answer(
            "A 看起来可能，但最终无法判断。",
            frozenset({"A", "B", "C", "D"}),
        )
        self.assertIsNone(option)

    def test_uses_last_labeled_answer(self) -> None:
        option, _ = parse_model_answer(
            "候选答案：A\n最终答案：C",
            frozenset({"A", "B", "C", "D"}),
        )
        self.assertEqual(option, "C")


class ConsensusTest(unittest.TestCase):

    @staticmethod
    def _answer(model: str, option: str) -> ModelAnswer:
        return ModelAnswer(provider="test", model=model, option=option)

    def test_accepts_clear_majority(self) -> None:
        result = build_consensus(
            [
                self._answer("one", "B"),
                self._answer("two", "B"),
                self._answer("three", "C"),
            ],
            minimum_responses=2,
            minimum_agreement=2,
        )
        self.assertTrue(result.actionable)
        self.assertEqual(result.option, "B")
        self.assertAlmostEqual(result.agreement_ratio, 2 / 3)

    def test_rejects_tie(self) -> None:
        result = build_consensus(
            [self._answer("one", "A"), self._answer("two", "B")],
            minimum_responses=2,
            minimum_agreement=1,
        )
        self.assertFalse(result.actionable)
        self.assertIsNone(result.option)
        self.assertIn("tied", result.reason)

    def test_rejects_too_few_valid_responses(self) -> None:
        result = build_consensus(
            [
                self._answer("one", "A"),
                ModelAnswer(provider="test", model="two", error="timeout"),
            ],
            minimum_responses=2,
            minimum_agreement=1,
        )
        self.assertFalse(result.actionable)
        self.assertIn("only 1", result.reason)


class ParsingRegressionTest(unittest.TestCase):
    """Ambiguous model prose must never become a mouse action."""

    def test_rejects_ambiguous_and_negative_answers(self) -> None:
        for text in (
            "答案：A或B",
            "答案：A, B",
            "option A is incorrect",
            "答案：A（错误）",
            "A\nB\n无法判断",
            '{"answer":"A", "option":"B"}',
            '{"answer":"A", "answer":"B"}',
            '{"answer":["A", "B"]}',
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    parse_model_answer(text, frozenset({"A", "B"})),
                    (None, None),
                )

    def test_accepts_complete_final_answers(self) -> None:
        for text in ("B", "答案：B。", "Final answer: B", "推理过程\n答案：B"):
            with self.subTest(text=text):
                self.assertEqual(
                    parse_model_answer(text, frozenset({"A", "B"}))[0], "B"
                )


if __name__ == "__main__":
    unittest.main()


class MultipleAnswerTest(unittest.TestCase):
    """Whole answer sets and confidence determine whether input is allowed."""

    def test_canonicalizes_complete_set(self):
        answer, confidence = parse_model_answer(
            '{"answer":["c","A"],"confidence":0.9}',
            frozenset("ABCDE"),
            multiple=True,
        )
        self.assertEqual((answer, confidence), ("A,C", 0.9))

    def test_rejects_ambiguous_or_unavailable_sets(self):
        for text in (
            '{"answer":["A","A"]}',
            '{"answer":[]}',
            '{"answer":["A","E"]}',
            '{"answer":["A",1]}',
            '{"answer":"AC"}',
            '{"answer":"A,C"}',
            '{"answer":["A","C"],"option":"A"}',
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    parse_model_answer(
                        text,
                        frozenset("ABCD"),
                        multiple=True,
                    ),
                    (None, None),
                )

    def test_votes_require_agreement_on_entire_set(self):
        responses = [
            ModelAnswer(
                "test", str(index), option=option, reported_confidence=0.9
            )
            for index, option in enumerate(("A,B", "A,C", "A,D"))
        ]
        result = build_consensus(responses, 2, 2, minimum_confidence=0.6)
        self.assertFalse(result.actionable)
        self.assertIsNone(result.option)
        self.assertEqual(result.votes, {"A,B": 1, "A,C": 1, "A,D": 1})

    def test_low_or_missing_confidence_does_not_vote(self):
        responses = [
            ModelAnswer(
                "test", str(index), option="A,C", reported_confidence=confidence
            )
            for index, confidence in enumerate((0.2, 0.25, None))
        ]
        result = build_consensus(responses, 2, 2, minimum_confidence=0.6)
        self.assertFalse(result.actionable)
        self.assertFalse(result.votes)
