"""OCR engine abstraction layer - supports RapidOCR (local) and Vision LLM (remote)."""

import base64
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("auto_answer")


class OcrResult:
    """Structured OCR output: question text + options."""

    def __init__(self, raw_text: str, question: str = "", options: dict[str, str] | None = None) -> None:
        self.raw_text = raw_text
        self.question = question
        self.options = options or {}

    def __repr__(self) -> str:
        opts = " | ".join(f"{k}: {v}" for k, v in self.options.items())
        return f"Q: {self.question}\nOptions: {opts}"


class BaseOcrEngine(ABC):
    """Abstract OCR engine interface."""

    @abstractmethod
    def recognize(self, frame: np.ndarray) -> OcrResult:
        """Run OCR on a BGR numpy frame and return structured result."""
        ...


class RapidOcrEngine(BaseOcrEngine):
    """Local OCR using RapidOCR - good Chinese + English support, no network needed."""

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR
        self._engine = RapidOCR()
        logger.info("RapidOCR engine initialized.")

    def recognize(self, frame: np.ndarray) -> OcrResult:
        result, _ = self._engine(frame)
        if not result:
            logger.warning("RapidOCR returned empty result.")
            return OcrResult(raw_text="")

        # result is list of [bbox, text, confidence]
        lines = [item[1] for item in result]
        raw_text = "\n".join(lines)
        question, options = self._parse_question(lines)

        logger.debug(f"OCR recognized {len(lines)} lines.")
        return OcrResult(raw_text=raw_text, question=question, options=options)

    # Matches lines like: A. xxx / A、xxx / A: xxx / A) xxx / A xxx
    _OPTION_RE = re.compile(r'^([A-Da-d])\s*[.。、:：)）]\s*(.+)')

    @staticmethod
    def _parse_question(lines: list[str]) -> tuple[str, dict[str, str]]:
        """Parse raw OCR lines into question text and options dict.

        Heuristic: lines matching 'X. text' pattern (where X is A-D) are options,
        everything before the first option is the question.
        """
        question_lines: list[str] = []
        options: dict[str, str] = {}
        found_option = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = RapidOcrEngine._OPTION_RE.match(stripped)
            if match:
                key = match.group(1).upper()
                value = match.group(2).strip()
                options[key] = value
                found_option = True
            elif not found_option:
                question_lines.append(stripped)

        question = " ".join(question_lines)
        return question, options


class VisionLlmOcrEngine(BaseOcrEngine):
    """Use a Vision LLM (e.g. GPT-4o) to read the screenshot directly.

    More accurate for complex layouts but requires network and costs tokens.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str = "", base_url: str = "") -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        logger.info(f"Vision LLM OCR engine initialized with model={model}")

    def recognize(self, frame: np.ndarray) -> OcrResult:
        # Encode frame to base64 PNG
        _, buffer = cv2.imencode(".png", frame)
        b64_image = base64.b64encode(buffer).decode("utf-8")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "这是一道题目的截图。请识别并输出：\n"
                                "1. 题目内容（一行）\n"
                                "2. 每个选项，格式为 A: xxx\\nB: xxx\\nC: xxx\\nD: xxx\n"
                                "只输出题目和选项，不要其他内容。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        raw_text = response.choices[0].message.content or ""
        question, options = self._parse_vision_response(raw_text)
        return OcrResult(raw_text=raw_text, question=question, options=options)

    @staticmethod
    def _parse_vision_response(text: str) -> tuple[str, dict[str, str]]:
        """Parse the LLM's structured text response into question + options."""
        lines = text.strip().split("\n")
        question_parts: list[str] = []
        options: dict[str, str] = {}

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Match "A: xxx" or "A. xxx"
            if len(stripped) >= 3 and stripped[0] in "ABCD" and stripped[1] in ":：.、":
                key = stripped[0]
                value = stripped[2:].strip()
                options[key] = value
            else:
                question_parts.append(stripped)

        question = " ".join(question_parts)
        return question, options


def create_ocr_engine(engine_type: str = "rapidocr", **kwargs) -> BaseOcrEngine:
    """Factory function to create the appropriate OCR engine."""
    if engine_type == "rapidocr":
        return RapidOcrEngine()
    elif engine_type == "vision_llm":
        return VisionLlmOcrEngine(
            model=kwargs.get("model", "gpt-4o"),
            api_key=kwargs.get("api_key", ""),
            base_url=kwargs.get("base_url", ""),
        )
    else:
        raise ValueError(f"Unknown OCR engine type: {engine_type}")