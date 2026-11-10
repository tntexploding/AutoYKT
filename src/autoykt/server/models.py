"""Normalize presentation objects and validate platform answer payloads."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from urllib.parse import urlsplit


SLIDE_CDN = "https://qn-scd1.yuketang.cn/"
SIGNED_SLIDE_CDN = "https://changjiang-private-qn.yuketang.cn/"


def identifier(value: Any) -> str:
    """Normalize numeric/string identifiers without conflating missing IDs."""
    return "" if value is None else str(value)


def first_field(data: dict, *keys: str, default=None):
    """Keep meaningful zero values while accepting the sample's aliases."""
    return next(
        (data[key] for key in keys if data.get(key) not in (None, "")), default
    )


def problem_data(slide: dict) -> dict:
    """Read embedded metadata without confusing ordinary slides with quizzes."""
    for key in ("problem", "problemData", "Problem"):
        if isinstance(slide.get(key), dict) and slide[key]:
            return slide[key]
    if slide.get("hasProblem") or slide.get("prob"):
        return slide
    return {}


def image_url(value: Any) -> str:
    """Resolve relative slide covers using the CDN from the supplied sample."""
    value = str(value or "")
    if not value or value.startswith(("https://", "http://")):
        return value
    if value.startswith("//"):
        return "https:" + value
    return SLIDE_CDN + value.lstrip("/")


def slide_image_identity(url: str) -> str:
    """Keep the page and transformations, excluding observed access fields."""
    if not url.startswith(SIGNED_SLIDE_CDN):
        return url
    parts = urlsplit(url)
    fields = parts.query.split("&")
    names = {field.partition("=")[0] for field in fields}
    if not {"e", "token"} <= names:
        return url
    query = "&".join(
        field for field in fields if not field.startswith(("e=", "token="))
    )
    return parts._replace(query=query).geturl()


def normalize_presentation(data: dict) -> dict:
    """Preserve every returned field and page, adding usable covers/indexes."""
    slides = []
    for index, original in enumerate(data.get("slides") or [], 1):
        if not isinstance(original, dict):
            slides.append(original)
            continue
        slide = dict(original)
        if slide.get("index") is None:
            slide["index"] = index
        slide["cover_url"] = image_url(
            slide.get("cover_url") or slide.get("cover")
        )
        slides.append(slide)
    return {**data, "slides": slides}


def slide_text(slide: dict) -> str:
    """Use available structured text; an image-only slide may be empty."""
    problem = problem_data(slide)
    parts = [
        str(first_field(problem, "title", "Title", default="")),
        str(
            first_field(
                problem,
                "body",
                "Body",
                "content",
                "Content",
                "stem",
                "Stem",
                "desc",
                default="",
            )
        ),
    ]
    for shape in slide.get("shapes") or []:
        if isinstance(shape, dict) and shape.get("Text"):
            parts.append(str(shape["Text"]))
    return "\n".join(dict.fromkeys(part for part in parts if part.strip()))


def question_kind(problem: dict) -> int:
    """Normalize explicit numeric and named types observed in the reference."""
    value = first_field(
        problem,
        "problemType",
        "problemtype",
        "problem_type",
        "type",
        default=1,
    )
    names = {
        "single": 1,
        "multi": 2,
        "multiple": 2,
        "vote": 3,
        "polling": 3,
        "fill": 4,
        "fillblank": 4,
        "blank": 4,
        "subjective": 5,
        "shortanswer": 5,
        "essay": 5,
        "judgment": 6,
    }
    name = str(value).lower().strip()
    return names[name] if name in names else int(value)


def question_options(problem: dict) -> dict[str, str]:
    """Prefer actual options: real v3 bullets can contain only layout data."""
    raw = first_field(
        problem, "options", "Options", "bullets", "Bullets", default=[]
    )
    if isinstance(raw, dict):
        return {str(key): str(value or "") for key, value in raw.items()}
    result = {}
    for option in raw:
        if isinstance(option, str):
            result[option] = ""
        elif isinstance(option, dict):
            key = first_field(option, "key", "label")
            if key is not None:
                result[str(key)] = str(
                    first_field(option, "value", "content", default="")
                )
    return result


@dataclass(frozen=True)
class Question:
    """A question bound to its course, presentation and current content."""

    course_id: str
    presentation_id: str
    slide_id: str
    question_id: str
    index: int
    kind: int
    text: str
    options: dict[str, str]
    image_url: str
    blanks: int
    polling_count: int
    result: Any = None

    @classmethod
    def from_slide(cls, course_id: str, presentation_id: str, slide: dict):
        """Parse the v3 presentation schema used by the reference client."""
        problem = problem_data(slide)
        if problem is slide:
            question_id = identifier(
                first_field(slide, "prob", "problemId", "problemid")
            )
        else:
            question_id = identifier(
                first_field(
                    problem,
                    "problemId",
                    "prob",
                    "problemid",
                    "id",
                    default=first_field(
                        slide, "prob", "problemId", "problemid"
                    ),
                )
            )
        if not question_id:
            raise ValueError("slide has no problemId")
        return cls(
            course_id=course_id,
            presentation_id=presentation_id,
            slide_id=identifier(first_field(slide, "id", "slide_id", "sid")),
            question_id=question_id,
            index=int(slide.get("index", 0)),
            kind=question_kind(problem),
            text=slide_text(slide),
            options=question_options(problem),
            image_url=image_url(slide.get("cover_url") or slide.get("cover")),
            blanks=len(
                first_field(problem, "blanks", "Blanks", default=[]) or []
            ),
            polling_count=int(
                first_field(
                    problem,
                    "pollingCount",
                    "polling_count",
                    "pollingLimit",
                    "polling_limit",
                    default=1,
                )
            ),
            result=first_field(problem, "result", "user_answer", "my_answer"),
        )

    @classmethod
    def from_event(cls, course_id: str, presentation_id: str, event: dict):
        """Use a released inline question if the deck omits its metadata."""
        problem = {**event, **problem_data(event)}
        question_id = first_field(
            problem, "prob", "problemId", "problemid", "id"
        )
        problem["problemId"] = question_id
        # A WS timeline entry's type='problem' is an event type, not a quiz kind.
        if problem.get("type") in {"problem", "embedded_problem"}:
            problem.pop("type")
        return cls.from_slide(
            course_id,
            presentation_id,
            {
                "id": first_field(event, "sid", "slide_id"),
                "index": int(event.get("si") or 0) + 1,
                "cover": first_field(event, "cover_url", "cover", "image_url"),
                "problem": problem,
            },
        )

    @property
    def signature(self) -> str:
        """Invalidate answers when text, options or the image change."""
        return self._signature(slide_image_identity(self.image_url))

    @property
    def legacy_signature(self) -> str:
        return self._signature(self.image_url)

    def matches_signature(self, signature: str) -> bool:
        return signature == self.signature or signature == self.legacy_signature

    def _signature(self, image: str) -> str:
        content = [
            self.kind,
            self.text,
            self.options,
            image,
            self.blanks,
            self.polling_count,
        ]
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def describe(self) -> str:
        """Provide labeled options to text and multimodal models."""
        return "\n".join(
            [self.text]
            + [f"{key}. {value}" for key, value in self.options.items()]
        )


# The six platform question kinds have different payload contracts.
# pylint: disable-next=too-many-branches
def validate_answer(question: Question, answer: Any) -> Any:
    """Validate shape and option membership without judging correctness."""
    if question.kind in {1, 2, 3, 6}:
        if isinstance(answer, str):
            answer = [answer]
        if not isinstance(answer, list) or not answer:
            raise ValueError("choice answer must be a nonempty list")
        if any(not isinstance(item, str) for item in answer):
            raise ValueError("option keys must be strings")
        answer = list(dict.fromkeys(answer))
        if any(item not in question.options for item in answer):
            raise ValueError("answer contains an unknown option")
        if question.kind in {1, 6} and len(answer) != 1:
            raise ValueError("single-choice/judgment requires one option")
        if (
            question.kind == 3
            and question.polling_count >= 0
            and len(answer) > question.polling_count
        ):
            raise ValueError("too many voting options")
        return sorted(answer)
    if question.kind == 4:
        if (
            not isinstance(answer, list)
            or len(answer) != question.blanks
            or not answer
            or any(not isinstance(item, str) for item in answer)
        ):
            raise ValueError("fill-in answer must match the number of blanks")
        return answer
    if question.kind == 5:
        if isinstance(answer, str) and answer.strip():
            return {"content": answer, "pics": []}
        if isinstance(answer, dict) and isinstance(answer.get("content"), str):
            if answer["content"].strip():
                return {"content": answer["content"], "pics": []}
        raise ValueError("short answer requires nonempty content")
    raise ValueError(f"unsupported question type: {question.kind}")
