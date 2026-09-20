"""Strict answer parsing and deterministic multi-model consensus."""

from __future__ import annotations

from collections import Counter
import json
import re

from autoykt.agent.models import ConsensusAnswer, ModelAnswer


_LABELED_ANSWER = re.compile(
    r"(?:(?:最终答案|答案|选项|final answer|answer|option)\s*[：:]\s*)?"
    r"([A-Za-z0-9_-]+)\s*[。.!！]?",
    flags=re.IGNORECASE,
)


def parse_model_answer(
    text: str,
    allowed_options: frozenset[str],
    *,
    multiple: bool = False,
) -> tuple[str | None, float | None]:
    """Parse one option or a complete multiple-choice set without guessing."""
    normalized_options = {
        option.strip().upper() for option in allowed_options if option.strip()
    }
    stripped = text.strip()
    if not stripped:
        return None, None

    json_text = stripped
    if json_text.startswith("```"):
        json_text = re.sub(r"^```(?:json)?\s*", "", json_text)
        json_text = re.sub(r"\s*```$", "", json_text)
    try:
        value = json.loads(json_text, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError:
        value = None
    except ValueError:
        value = {}
    if isinstance(value, dict):
        answers = [value[key] for key in ("answer", "option") if key in value]
        candidates = {
            _canonical_answer(item, normalized_options, multiple)
            for item in answers
        }
        if len(candidates) != 1 or None in candidates:
            return None, None
        return candidates.pop(), _parse_confidence(value.get("confidence"))

    # Only a complete final line may drive a click. Reasoning such as
    # "option A is incorrect" or "answer: A or B" is not an answer.
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    match = _LABELED_ANSWER.fullmatch(lines[-1])
    if match is not None and match.group(1).upper() in normalized_options:
        return match.group(1).upper(), None
    return None, None


def _canonical_answer(
    value: object, allowed_options: set[str], multiple: bool
) -> str | None:
    if isinstance(value, str):
        candidate = value.strip().upper()
        return candidate if candidate in allowed_options else None
    if not multiple or not isinstance(value, list) or not value:
        return None
    if any(not isinstance(item, str) for item in value):
        return None
    options = [item.strip().upper() for item in value]
    if len(set(options)) != len(options) or not set(options) <= allowed_options:
        return None
    return ",".join(sorted(options))


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _parse_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    confidence = float(value)
    if 0.0 <= confidence <= 1.0:
        return confidence
    return None


def build_consensus(
    responses: list[ModelAnswer],
    minimum_responses: int,
    minimum_agreement: int,
    *,
    minimum_confidence: float | None = None,
) -> ConsensusAnswer:
    """Require enough valid responses, an unambiguous winner, and agreement."""
    valid = [
        response
        for response in responses
        if response.succeeded
        and (
            minimum_confidence is None
            or (
                response.reported_confidence is not None
                and response.reported_confidence >= minimum_confidence
            )
        )
    ]
    votes = Counter(
        response.option for response in valid if response.option is not None
    )
    if len(valid) < minimum_responses:
        return ConsensusAnswer(
            option=None,
            actionable=False,
            agreement_ratio=0.0,
            votes=dict(votes),
            responses=tuple(responses),
            reason=(
                f"only {len(valid)} valid responses; "
                f"{minimum_responses} required"
            ),
        )
    ranked = votes.most_common()
    if not ranked:
        return ConsensusAnswer(
            option=None,
            actionable=False,
            agreement_ratio=0.0,
            responses=tuple(responses),
            reason="no model returned a configured option",
        )
    winner, count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == count:
        return ConsensusAnswer(
            option=None,
            actionable=False,
            agreement_ratio=count / len(valid),
            votes=dict(votes),
            responses=tuple(responses),
            reason="model vote is tied",
        )
    if count < minimum_agreement:
        return ConsensusAnswer(
            option=None,
            actionable=False,
            agreement_ratio=count / len(valid),
            votes=dict(votes),
            responses=tuple(responses),
            reason=(
                f"winning option has {count} votes; "
                f"{minimum_agreement} required"
            ),
        )
    return ConsensusAnswer(
        option=winner,
        actionable=True,
        agreement_ratio=count / len(valid),
        votes=dict(votes),
        responses=tuple(responses),
        reason="consensus reached",
    )
