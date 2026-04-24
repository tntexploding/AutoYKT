"""Answer agent: receives questions, queries cache or LLM, publishes answers."""

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Any

from jinja2 import Template
from openai import OpenAI

from autoykt.core.event_bus import EventBus, Event, EventType
from autoykt.core.config import AppConfig
from autoykt.agent.question_db import QuestionDB

logger = logging.getLogger("auto_answer")


class AnswerAgent:
    """Subscribes to QUESTION_DETECTED, produces ANSWER_READY.

    Flow: receive question → check cache → call one or more LLMs in parallel →
        store raw outputs in cache → publish answer event.
    """

    def __init__(self, config: AppConfig, event_bus: EventBus) -> None:
        self._config = config
        self._bus = event_bus

        # LLM client
        self._client = OpenAI(
            api_key=config.agent.api_key,
            base_url=config.agent.base_url or None,
        )
        self._models = self._resolve_models(config.agent.models, config.agent.model, config.agent.answer_count)
        self._timeout = config.agent.timeout
        self._min_response_count = config.agent.min_response_count
        self._auto_click = config.agent.auto_click
        self._question_db_enabled = config.agent.question_db_enabled

        # Load prompt template
        tpl_path = Path(config.agent.prompt_template)
        if tpl_path.exists():
            self._template = Template(tpl_path.read_text(encoding="utf-8"))
            logger.info(f"Prompt template loaded from {tpl_path}")
        else:
            # Fallback inline template
            self._template = Template(
                "请根据题目截图直接作答。\n\n"
                "题目：{{ question }}\n\n"
                "{% if options %}选项：\n{% for key, value in options.items() %}{{ key }}. {{ value }}\n{% endfor %}{% endif %}\n"
                "要求：\n"
                "1. 先给出简要判断依据\n"
                "2. 最后一行输出你认为最可能正确的答案\n"
                "3. 如果无法确定，也要给出最可能的选择\n\n"
                "答案："
            )
            logger.warning(f"Template not found at {tpl_path}, using fallback.")

        # Question cache
        self._db: QuestionDB | None = None
        if self._question_db_enabled:
            self._db = QuestionDB(
                db_path="storage/data/questions.db",
                similarity_threshold=config.agent.db_similarity_threshold,
            )
        else:
            logger.info("QuestionDB disabled by config; cache storage will be skipped.")

        # Subscribe to question events
        self._bus.subscribe(EventType.QUESTION_DETECTED, self._on_question)

    @staticmethod
    def _resolve_models(models: list[str], fallback_model: str, answer_count: int) -> list[str]:
        candidates = [model.strip() for model in models if model.strip()] or [fallback_model.strip()]
        if answer_count >= len(candidates):
            return candidates
        return candidates[:answer_count]

    async def _on_question(self, event: Event) -> None:
        """Handle a detected question: send screenshot to vision LLM."""
        screenshot_path = event.payload.get("screenshot_path", "")
        question_text = event.payload.get("question", "")
        options = event.payload.get("options", {})

        if not screenshot_path or not Path(screenshot_path).exists():
            logger.warning("No screenshot available, ignoring.")
            return

        logger.info(f"Agent received question with screenshot: {screenshot_path}; models={self._models}")

        # Step 1: call vision LLMs with screenshot in parallel
        try:
            responses = await self._query_vision_llms(screenshot_path)
        except Exception as e:
            logger.error(f"Vision LLM call failed: {e}")
            await self._bus.publish(Event(
                type=EventType.ERROR,
                payload={"source": "answer_agent", "error": str(e)},
            ))
            return

        if not responses:
            logger.error("No model responses received before timeout; skipping this question.")
            await self._bus.publish(Event(
                type=EventType.ERROR,
                payload={
                    "source": "answer_agent",
                    "error": "No model responses received before timeout",
                },
            ))
            return

        if len(responses) < self._min_response_count:
            logger.error(
                "Insufficient model responses before timeout: "
                f"got {len(responses)}, required {self._min_response_count}."
            )
            await self._bus.publish(Event(
                type=EventType.ERROR,
                payload={
                    "source": "answer_agent",
                    "error": (
                        "Insufficient model responses before timeout "
                        f"({len(responses)}/{self._min_response_count})"
                    ),
                },
            ))
            return

        raw_blob = "\n\n".join(
            f"[{item['model']}]\n{item['raw_response']}" for item in responses
        )
        answer_option = self._summarize_answer_option(responses)

        # Step 2: store raw outputs in cache for later lookup
        if self._db is not None:
            self._db.store(
                question_text=raw_blob,
                options="",
                answer=raw_blob,
                confidence=1.0,
                source=f"vision:{','.join(item['model'] for item in responses)}",
            )

        # Step 3: publish raw answers
        await self._publish_answer(
            raw_responses=responses,
            answer_option=answer_option,
            source=f"vision:{','.join(item['model'] for item in responses)}",
            confidence=1.0,
        )

    @staticmethod
    def _extract_option_letter(text: str) -> str | None:
        patterns = [
            r"答案\s*[：:]\s*([A-Da-d])",
            r"\b([A-Da-d])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).upper()
        return None

    def _summarize_answer_option(self, responses: list[dict[str, str]]) -> str:
        votes: dict[str, int] = {}
        for item in responses:
            raw = item.get("raw_response", "")
            opt = self._extract_option_letter(raw)
            if not opt:
                continue
            votes[opt] = votes.get(opt, 0) + 1

        if not votes:
            return ""

        best = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        logger.info(f"Summarized option vote: {votes}, selected={best}")
        return best

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read image file and return base64 encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def _query_single_model(self, model: str, screenshot_path: str) -> dict[str, str]:
        """Send screenshot to one vision LLM and return its raw response."""
        b64_image = self._encode_image(screenshot_path)
        prompt_text = self._template.render(
            question="请直接根据截图中的题目作答",
            options={},
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64_image}",
                        },
                    },
                ],
            }
        ]

        logger.debug("Sending screenshot to vision LLM...")
        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.1,
                timeout=self._timeout,
            ),
        )

        raw_answer = response.choices[0].message.content or ""
        logger.info(f"Vision LLM response from {model}:\n{raw_answer}")
        return {"model": model, "raw_response": raw_answer}

    async def _query_vision_llms(self, screenshot_path: str) -> list[dict[str, str]]:
        task_by_model: dict[asyncio.Task[dict[str, str]], str] = {}
        for model in self._models:
            task = asyncio.create_task(self._query_single_model(model, screenshot_path))
            task_by_model[task] = model

        responses: list[dict[str, str]] = []
        try:
            for done_task in asyncio.as_completed(task_by_model.keys(), timeout=self._timeout):
                try:
                    result = await done_task
                    responses.append(result)
                except Exception as e:
                    model = task_by_model.get(done_task, "unknown")
                    logger.warning(f"Vision LLM model failed ({model}): {e}")
        except asyncio.TimeoutError:
            logger.warning(
                f"Vision LLM partial timeout reached ({self._timeout}s). "
                f"Using {len(responses)} available response(s)."
            )
        finally:
            for task in task_by_model:
                if not task.done():
                    task.cancel()

        return responses

    async def _publish_answer(
        self,
        raw_responses: list[dict[str, str]],
        answer_option: str,
        source: str,
        confidence: float,
    ) -> None:
        """Publish ANSWER_READY event."""
        logger.info(f"Publishing {len(raw_responses)} raw answer(s) (source={source}, conf={confidence})")
        await self._bus.publish(Event(
            type=EventType.ANSWER_READY,
            payload={
                "raw_responses": raw_responses,
                "answer": answer_option,
                "answer_option": answer_option,
                "source": source,
                "confidence": confidence,
                "auto_click": self._auto_click,
            },
        ))

    def close(self) -> None:
        """Release resources."""
        if self._db is not None:
            self._db.close()