"""Value objects returned by model providers and answer consensus."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class ModelAnswer:
    """One model request outcome."""

    provider: str
    model: str
    raw_response: str = ""
    option: str | None = None
    reported_confidence: float | None = None
    latency_seconds: float = 0.0
    error: str | None = None
    retryable: bool = False
    round_index: int = 1

    @property
    def succeeded(self) -> bool:
        """Whether the response contained a valid option."""
        return self.error is None and self.option is not None


@dataclass(frozen=True)
class ConsensusAnswer:
    """Validated result of combining all model responses."""

    option: str | None
    actionable: bool
    agreement_ratio: float
    votes: dict[str, int] = field(default_factory=dict)
    responses: tuple[ModelAnswer, ...] = ()
    reason: str = ""

    @property
    def options(self) -> tuple[str, ...]:
        """Return the canonical option set in deterministic order."""
        return tuple(self.option.split(",")) if self.option else ()

    def event_payload(self) -> dict[str, object]:
        """Return a serializable, secret-free event representation."""
        return {
            "answer": self.option,
            "actionable": self.actionable,
            "agreement_ratio": self.agreement_ratio,
            "votes": dict(self.votes),
            "reason": self.reason,
            "rounds_used": max(
                (response.round_index for response in self.responses), default=0
            ),
            "responses": [
                {
                    "provider": response.provider,
                    "model": response.model,
                    "round": response.round_index,
                    "raw_response": response.raw_response,
                    "answer": response.option,
                    "reported_confidence": response.reported_confidence,
                    "latency_seconds": response.latency_seconds,
                    "error": response.error,
                }
                for response in self.responses
            ],
        }


class AnswerProvider(Protocol):
    """Answering surface consumed by a page automation."""

    async def answer(
        self,
        image_path: Path,
        allowed_options: frozenset[str],
        question_text: str = "",
        knowledge_context: str = "",
        *,
        timeout_seconds: float | None = None,
        multiple: bool = False,
    ) -> ConsensusAnswer:
        """Return consensus by the cutoff; cancel unfinished requests."""
        raise NotImplementedError
