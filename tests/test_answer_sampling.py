"""Bounded repeated sampling uses one text model without inventing votes."""

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from autoykt.agent.answer_agent import AnswerCoordinator
from autoykt.core.config import AnsweringConfig
from tests.test_answer_agent import _response


class AnswerSamplingTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.image = Path(directory.name) / "question.png"
        self.image.write_bytes(b"image stays local")
        config = AnsweringConfig.model_validate(
            {
                "providers": [
                    {
                        "name": "flash",
                        "models": ["deepseek-v4-flash"],
                        "input_mode": "text",
                    }
                ],
                "minimum_responses": 3,
                "minimum_agreement": 2,
                "maximum_rounds": 5,
                "maximum_parallel_requests": 3,
            }
        )
        self.coordinator = AnswerCoordinator(config)
        self.addAsyncCleanup(self.coordinator.close)
        self.request = AsyncMock()
        self.coordinator._client = Mock(
            return_value=SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=self.request)
                )
            )
        )

    async def _answer(self, **kwargs):
        return await self.coordinator.answer(
            self.image,
            frozenset("ABCD"),
            "Choose an option.\nA: first\nB: second\nC: third\nD: fourth",
            **kwargs,
        )

    async def test_initial_majority_stops_after_three_text_only_requests(self):
        self.request.side_effect = [_response(x) for x in ("C", "B", "C")]
        answer = await self._answer()
        self.assertTrue(answer.actionable)
        self.assertEqual(answer.option, "C")
        self.assertEqual(answer.votes, {"C": 2, "B": 1})
        self.assertAlmostEqual(answer.agreement_ratio, 2 / 3)
        self.assertEqual(self.request.await_count, 3)
        self.assertEqual(answer.event_payload()["rounds_used"], 3)
        for request in self.request.call_args_list:
            self.assertEqual(request.kwargs["model"], "deepseek-v4-flash")
            self.assertEqual(
                [
                    part["type"]
                    for part in request.kwargs["messages"][0]["content"]
                ],
                ["text"],
            )

    async def test_initial_tie_gets_two_more_independent_samples(self):
        self.request.side_effect = [
            _response(x) for x in ("A", "B", "C", "B", "B")
        ]
        answer = await self._answer()
        self.assertTrue(answer.actionable)
        self.assertEqual(answer.option, "B")
        self.assertEqual(answer.votes, {"A": 1, "B": 3, "C": 1})
        self.assertEqual(self.request.await_count, 5)
        self.assertEqual(
            [r.round_index for r in answer.responses], [1, 2, 3, 4, 5]
        )

    async def test_invalid_and_low_confidence_samples_do_not_vote(self):
        self.request.side_effect = [
            _response("A", finish_reason="length"),
            _response("C", confidence=0.2),
            _response("B"),
            _response("B"),
            _response("B"),
        ]
        answer = await self._answer()
        self.assertTrue(answer.actionable)
        self.assertEqual(answer.votes, {"B": 3})
        self.assertEqual(self.request.await_count, 5)

    async def test_exhausted_tie_stays_non_actionable(self):
        self.request.side_effect = [
            _response("A"),
            _response("B"),
            _response("C", confidence=0.2),
            _response("A"),
            _response("B"),
        ]
        answer = await self._answer()
        self.assertFalse(answer.actionable)
        self.assertEqual(answer.votes, {"A": 2, "B": 2})
        self.assertEqual(self.request.await_count, 5)

    async def test_multiple_choice_votes_are_complete_sets(self):
        self.request.side_effect = [
            _response(x) for x in (["A", "C"], ["A", "B"], ["C", "A"])
        ]
        answer = await self._answer(multiple=True)
        self.assertTrue(answer.actionable)
        self.assertEqual(answer.option, "A,C")
        self.assertEqual(answer.votes, {"A,C": 2, "A,B": 1})

    async def test_followup_samples_share_initial_deadline_and_are_cancelled(
        self,
    ):
        started = 0
        cancelled = 0

        async def respond(**_):
            nonlocal started, cancelled
            index = started
            started += 1
            if index < 3:
                return _response("ABC"[index])
            try:
                await asyncio.sleep(10)
            finally:
                cancelled += 1

        self.request.side_effect = respond
        answer = await self._answer(timeout_seconds=0.05)
        self.assertFalse(answer.actionable)
        self.assertEqual(answer.votes, {"A": 1, "B": 1, "C": 1})
        self.assertEqual(cancelled, 2)
        self.assertTrue(all(r.retryable for r in answer.responses[3:]))

    async def test_missing_option_text_never_calls_flash(self):
        answer = await self.coordinator.answer(self.image, frozenset("ABCD"))
        self.assertFalse(answer.actionable)
        self.request.assert_not_called()

    def test_response_threshold_cannot_exceed_total_samples(self):
        with self.assertRaises(ValueError):
            AnsweringConfig.model_validate(
                {
                    "providers": [
                        {"name": "flash", "models": ["deepseek-v4-flash"]}
                    ],
                    "maximum_rounds": 5,
                    "minimum_responses": 6,
                }
            )

    async def test_decisive_two_votes_do_not_wait_for_slow_third_sample(self):
        self.coordinator._config.minimum_responses = 2
        self.coordinator._config.initial_rounds = 3
        started = 0
        cancelled = asyncio.Event()

        async def respond(**_):
            nonlocal started
            started += 1
            if started == 3:
                try:
                    await asyncio.sleep(10)
                finally:
                    cancelled.set()
            return _response("C")

        self.request.side_effect = respond
        answer = await asyncio.wait_for(self._answer(), timeout=1)
        self.assertTrue(answer.actionable)
        self.assertEqual(answer.votes, {"C": 2})
        self.assertTrue(cancelled.is_set())
        self.assertEqual(self.request.await_count, 3)
