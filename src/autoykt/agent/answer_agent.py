"""Answer agent: receives questions, queries cache or LLM, publishes answers."""

import asyncio
import base64
import json
import logging
import re
from pathlib import Path

from jinja2 import Template
from openai import OpenAI

from autoykt.core.event_bus import EventBus, Event, EventType
from autoykt.core.config import AppConfig
from autoykt.agent.question_db import QuestionDB

logger = logging.getLogger("auto_answer")


class AnswerAgent:
    """Subscribes to QUESTION_DETECTED, produces ANSWER_READY.

    Flow: receive question → check cache → (miss) call LLM → parse answer →
          store in cache → publish answer event.
    """

    def __init__(self, config: AppConfig, event_bus: EventBus) -> None:
        self._config = config
        self._bus = event_bus

        # LLM client
        self._client = OpenAI(
            api_key=config.agent.api_key,
            base_url=config.agent.base_url or None,
        )
        self._model = config.agent.model
        self._timeout = config.agent.timeout

        # Load prompt template
        tpl_path = Path(config.agent.prompt_template)
        if tpl_path.exists():
            self._template = Template(tpl_path.read_text(encoding="utf-8"))
            logger.info(f"Prompt template loaded from {tpl_path}")
        else:
            # Fallback inline template
            self._template = Template(
                "请回答以下选择题，只返回选项字母。\n\n"
                "题目：{{ question }}\n\n"
                "{% for key, value in options.items() %}{{ key }}. {{ value }}\n{% endfor %}\n"
                "答案："
            )
            logger.warning(f"Template not found at {tpl_path}, using fallback.")

        # Question cache
        self._db = QuestionDB(
            db_path="storage/data/questions.db",
            similarity_threshold=config.agent.db_similarity_threshold,
        )

        # Subscribe to question events
        self._bus.subscribe(EventType.QUESTION_DETECTED, self._on_question)

    async def _on_question(self, event: Event) -> None:
        """Handle a detected question: send screenshot to vision LLM."""
        screenshot_path = event.payload.get("screenshot_path", "")
        question_text = event.payload.get("question", "")
        options = event.payload.get("options", {})

        if not screenshot_path or not Path(screenshot_path).exists():
            logger.warning("No screenshot available, ignoring.")
            return

        logger.info(f"Agent received question with screenshot: {screenshot_path}")

        # Step 1: call vision LLM with screenshot
        try:
            answer, raw_response = await self._query_vision_llm(screenshot_path)
        except Exception as e:
            logger.error(f"Vision LLM call failed: {e}")
            await self._bus.publish(Event(
                type=EventType.ERROR,
                payload={"source": "answer_agent", "error": str(e)},
            ))
            return

        # Step 2: store in cache (use raw_response as question_text for future reference)
        self._db.store(
            question_text=raw_response,
            options="",
            answer=answer,
            confidence=1.0,
            source=f"vision:{self._model}",
        )

        # Step 3: publish answer
        await self._publish_answer(
            answer=answer,
            source=f"vision:{self._model}",
            confidence=1.0,
        )

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read image file and return base64 encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def _query_vision_llm(self, screenshot_path: str) -> tuple[str, str]:
        """Send screenshot to vision LLM, return (parsed_answer, raw_response)."""
        b64_image = self._encode_image(screenshot_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "这是一道选择题的截图。请仔细阅读题目和所有选项，"
                            "给出正确答案的选项字母（A/B/C/D）。\n\n"
                            "要求：\n"
                            "1. 先简要分析题目（一两句话）\n"
                            "2. 最后一行只输出一个大写字母作为答案，格式为：答案：X"
                        ),
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
                model=self._model,
                messages=messages,
                max_tokens=200,
                temperature=0.1,
                timeout=self._timeout,
            ),
        )

        raw_answer = response.choices[0].message.content or ""
        logger.info(f"Vision LLM response:\n{raw_answer}")

        parsed = self._parse_answer(raw_answer)
        if parsed:
            return parsed, raw_answer

        # Fallback
        fallback = raw_answer.strip()[-1:].upper()
        if fallback in ("A", "B", "C", "D"):
            logger.warning(f"Used fallback parsing for: '{raw_answer}'")
            return fallback, raw_answer

        logger.error(f"Cannot parse answer from: '{raw_answer}'")
        raise ValueError(f"Unparseable vision LLM response: {raw_answer}")

    @staticmethod
    def _parse_answer(text: str) -> str:
        """Extract a single option letter (A/B/C/D) from LLM response.

        Handles formats like:
            "B"
            "B. len"
            "答案是B"
            "The answer is C."
        """
        text = text.strip()

        # Direct single letter
        if len(text) == 1 and text.upper() in "ABCD":
            return text.upper()

        # Look for patterns like "答案是X", "answer is X", just "X." etc.
        patterns = [
            r"^([A-Da-d])\b",           # starts with option letter
            r"答案[是为：:\s]*([A-Da-d])",  # 答案是X / 答案：X
            r"answer\s*(?:is)?\s*[:：]?\s*([A-Da-d])",  # answer is X
            r"\b([A-Da-d])\s*[.。)）]",  # X. or X)
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return ""

    async def _publish_answer(self, answer: str, source: str, confidence: float) -> None:
        """Publish ANSWER_READY event."""
        logger.info(f"Publishing answer: '{answer}' (source={source}, conf={confidence})")
        await self._bus.publish(Event(
            type=EventType.ANSWER_READY,
            payload={
                "answer": answer,
                "source": source,
                "confidence": confidence,
            },
        ))

    def close(self) -> None:
        """Release resources."""
        self._db.close()