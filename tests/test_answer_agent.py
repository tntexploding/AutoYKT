"""Model-request regressions using in-process fake clients."""

import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

from autoykt.agent.answer_agent import AnswerCoordinator
from autoykt.core.config import AnsweringConfig


def _response(
    answer: str | list[str] = "B", finish_reason="stop", confidence=0.9
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=json.dumps(
                        {"answer": answer, "confidence": confidence}
                    )
                ),
            )
        ]
    )


class CoordinatorRegressionTest(unittest.IsolatedAsyncioTestCase):
    """Concurrency, failures, and response completion gate actionable votes."""

    async def asyncSetUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image = Path(self.directory.name) / "question.png"
        self.image.write_bytes(b"synthetic-question-image")
        self.config = AnsweringConfig.model_validate(
            {
                "providers": [
                    {"name": "fake", "models": ["one", "two", "three"]}
                ],
                "maximum_parallel_requests": 2,
                "minimum_responses": 2,
                "minimum_agreement": 2,
            }
        )
        self.coordinator = AnswerCoordinator(self.config)
        self.addAsyncCleanup(self.coordinator.close)
        self.request = AsyncMock(return_value=_response())
        self.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=self.request)
            )
        )
        self.coordinator._client = Mock(return_value=self.client)

    async def test_parallel_requests_are_bounded_and_receive_knowledge(
        self,
    ) -> None:
        active = maximum = 0

        async def respond(**_):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.001)
            active -= 1
            return _response()

        self.request.side_effect = respond
        result = await self.coordinator.answer(
            self.image,
            frozenset({"A", "B"}),
            "题目",
            "课程专有知识",
        )
        self.assertTrue(result.actionable)
        self.assertEqual(maximum, 2)
        self.assertEqual(self.request.await_count, 3)
        prompt = self.request.call_args.kwargs["messages"][0]["content"][0][
            "text"
        ]
        self.assertIn("课程专有知识", prompt)
        self.assertIn("题目", prompt)

    async def test_error_body_cannot_leak_credentials(self) -> None:
        self.request.side_effect = [
            RuntimeError("invalid API key private-test-value"),
            _response(),
            _response(),
        ]
        with self.assertLogs("autoykt", level="WARNING") as logs:
            result = await self.coordinator.answer(
                self.image, frozenset({"A", "B"})
            )
        self.assertTrue(result.actionable)
        self.assertNotIn("private-test-value", str(result.event_payload()))
        self.assertNotIn("private-test-value", str(logs.output))

    async def test_truncated_responses_do_not_vote(self) -> None:
        self.request.return_value = _response(finish_reason="length")
        result = await self.coordinator.answer(
            self.image, frozenset({"A", "B"})
        )
        self.assertFalse(result.actionable)
        self.assertFalse(result.votes)

    async def test_slow_model_is_cancelled_and_timely_votes_are_used(self):
        cancelled = asyncio.Event()

        async def respond(**request):
            if request["model"] == "three":
                try:
                    await asyncio.sleep(1)
                finally:
                    cancelled.set()
            return _response()

        self.request.side_effect = respond
        result = await self.coordinator.answer(
            self.image,
            frozenset({"A", "B"}),
            timeout_seconds=0.03,
        )
        self.assertTrue(result.actionable)
        self.assertEqual(result.votes, {"B": 2})
        self.assertTrue(cancelled.is_set())
        self.assertEqual(len(result.responses), 3)
        self.assertTrue(result.responses[2].retryable)

    async def test_deadline_includes_requests_waiting_for_concurrency_slot(
        self,
    ):
        self.coordinator._semaphore = asyncio.Semaphore(1)

        async def respond(**_):
            await asyncio.sleep(1)
            return _response()

        self.request.side_effect = respond
        result = await self.coordinator.answer(
            self.image,
            frozenset({"A", "B"}),
            timeout_seconds=0.02,
        )
        await asyncio.sleep(0.01)
        self.assertFalse(result.actionable)
        self.assertEqual(self.request.await_count, 1)
        self.assertEqual(len(result.responses), 3)
        self.assertTrue(all(response.error for response in result.responses))

    async def test_cancellation_drains_active_and_queued_model_tasks(self):
        started = asyncio.Event()
        cancelled = []

        async def respond(**request):
            started.set()
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.append(request["model"])
            return _response()

        self.request.side_effect = respond
        task = asyncio.create_task(
            self.coordinator.answer(self.image, frozenset({"A", "B"}))
        )
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(cancelled), 2)
        self.assertEqual(self.request.await_count, 2)

    async def test_multiple_choice_uses_array_and_full_set_consensus(self):
        self.request.return_value = _response(["C", "A"])
        result = await self.coordinator.answer(
            self.image,
            frozenset("ABCDE"),
            multiple=True,
        )
        self.assertTrue(result.actionable)
        self.assertEqual(result.options, ("A", "C"))
        prompt = self.request.call_args.kwargs["messages"][0]["content"][0][
            "text"
        ]
        self.assertIn("多选题", prompt)
        self.assertIn("非空数组", prompt)

    async def test_agreement_cannot_override_low_confidence(self):
        self.request.return_value = _response(confidence=0.25)
        result = await self.coordinator.answer(self.image, frozenset("AB"))
        self.assertFalse(result.actionable)
        self.assertFalse(result.votes)
        self.assertEqual(result.responses[0].reported_confidence, 0.25)

    async def test_text_provider_receives_labeled_text_without_image(self):
        self.config.providers[0].input_mode = "text"
        result = await self.coordinator.answer(
            self.image,
            frozenset("AB"),
            question_text="题目\nA: 苹果\nB: 梨子",
        )
        self.assertTrue(result.actionable)
        parts = self.request.call_args.kwargs["messages"][0]["content"]
        self.assertEqual([part["type"] for part in parts], ["text"])
        self.assertNotIn("data:image", str(parts))

    async def test_text_provider_does_not_guess_missing_option_mapping(self):
        self.config.providers[0].input_mode = "text"
        result = await self.coordinator.answer(
            self.image,
            frozenset("AB"),
            question_text="题目\n苹果\n梨子",
        )
        self.assertFalse(result.actionable)
        self.request.assert_not_awaited()
        self.assertIn("labeled", result.responses[0].error or "")

    async def test_vision_provider_still_receives_image_without_ocr(self):
        result = await self.coordinator.answer(self.image, frozenset("AB"))
        self.assertTrue(result.actionable)
        parts = self.request.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(
            [part["type"] for part in parts], ["text", "image_url"]
        )

    async def test_configured_reasoning_effort_is_forwarded(self):
        self.config.providers[0].reasoning_effort = "low"
        await self.coordinator.answer(self.image, frozenset("AB"))
        self.assertEqual(
            self.request.call_args.kwargs["reasoning_effort"], "low"
        )

    async def test_explicit_fast_mode_disables_thinking_without_changing_model(
        self,
    ):
        self.config.providers[0].thinking = "disabled"
        self.config.providers[0].reasoning_effort = "low"
        result = await self.coordinator.answer(self.image, frozenset("AB"))
        self.assertTrue(result.actionable)
        request = self.request.call_args.kwargs
        self.assertEqual(
            request["extra_body"], {"thinking": {"type": "disabled"}}
        )
        self.assertNotIn("reasoning_effort", request)
        self.assertIn(request["model"], ("one", "two", "three"))

    async def test_unconfigured_thinking_does_not_send_provider_specific_body(
        self,
    ):
        await self.coordinator.answer(self.image, frozenset("AB"))
        self.assertNotIn("extra_body", self.request.call_args.kwargs)
