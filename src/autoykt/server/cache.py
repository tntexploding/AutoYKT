"""Persistent presentation/answer cache and submission journal.

The JSON interchange keeps the answers/manual_answers layout documented by
infstellar/RainClassroomAssistant, while binding entries to fetched slide IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import time
from typing import Any

from autoykt.server.models import (
    Question,
    SIGNED_SLIDE_CDN,
    identifier,
    normalize_presentation,
    validate_answer,
)


class ClassroomCache:
    """Keep course and environment boundaries in every lookup."""

    def __init__(self, path: Path, domain: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.domain = domain
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS presentations (
                domain TEXT, course TEXT, presentation TEXT, document TEXT,
                updated REAL,
                PRIMARY KEY(domain, course, presentation)
            );
            CREATE TABLE IF NOT EXISTS answers (
                domain TEXT, course TEXT, presentation TEXT, question TEXT,
                signature TEXT, answer TEXT, source TEXT, updated REAL,
                PRIMARY KEY(domain, course, presentation, question, signature)
            );
            CREATE TABLE IF NOT EXISTS submissions (
                domain TEXT, lesson TEXT, question TEXT, signature TEXT,
                answer TEXT, status TEXT, updated REAL,
                PRIMARY KEY(domain, lesson, question)
            );
            """
        )
        self._upgrade_answer_keys()
        self._upgrade_signed_answers()

    def _upgrade_answer_keys(self):
        """Keep saved answers when upgrading a one-version cache."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            signature = next(
                row for row in self.db.execute("PRAGMA table_info(answers)")
                if row["name"] == "signature"
            )
            if signature["pk"]:
                return
            self.db.execute("ALTER TABLE answers RENAME TO old_answers")
            self.db.execute(
                """
                CREATE TABLE answers (
                    domain TEXT, course TEXT, presentation TEXT, question TEXT,
                    signature TEXT, answer TEXT, source TEXT, updated REAL,
                    PRIMARY KEY(domain, course, presentation, question, signature)
                )
                """
            )
            self.db.execute("INSERT INTO answers SELECT * FROM old_answers")
            self.db.execute("DROP TABLE old_answers")

    def _upgrade_signed_answers(self):
        """Preserve old answers that match the last saved question exactly."""
        decks = self.db.execute(
            "SELECT p.course, p.presentation FROM presentations p "
            "WHERE p.domain=? AND p.document LIKE ? AND EXISTS ("
            "SELECT 1 FROM answers a WHERE a.domain=p.domain "
            "AND a.course=p.course AND a.presentation=p.presentation)",
            (self.domain, f"%{SIGNED_SLIDE_CDN}%"),
        ).fetchall()
        with self.db:
            for course, presentation in decks:
                for question in self.questions(course, presentation):
                    old, new = question.legacy_signature, question.signature
                    if old != new:
                        self.db.execute(
                            "UPDATE OR IGNORE answers SET signature=? WHERE domain=? "
                            "AND course=? AND presentation=? AND question=? "
                            "AND signature=?",
                            (
                                new,
                                self.domain,
                                course,
                                presentation,
                                question.question_id,
                                old,
                            ),
                        )

    def close(self) -> None:
        """Close the persistent connection."""
        self.db.close()

    def save_presentation(self, course: str, presentation: str, data: dict):
        """Replace metadata without erasing valid answers or manual edits."""
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO presentations VALUES (?, ?, ?, ?, ?)",
                (
                    self.domain,
                    course,
                    presentation,
                    json.dumps(normalize_presentation(data)),
                    time.time(),
                ),
            )

    def presentation(self, course: str, presentation: str) -> dict:
        """Return one downloaded presentation, or an empty object."""
        row = self.db.execute(
            "SELECT document FROM presentations WHERE "
            "domain=? AND course=? AND presentation=?",
            (self.domain, course, presentation),
        ).fetchone()
        return json.loads(row[0]) if row else {}

    def questions(self, course: str, presentation: str) -> list[Question]:
        """Parse supported question objects from a cached presentation."""
        questions = []
        for slide in self.presentation(course, presentation).get("slides", []):
            if not isinstance(slide, dict):
                continue
            try:
                questions.append(
                    Question.from_slide(course, presentation, slide)
                )
            except (ValueError, TypeError, KeyError):
                # Retain the original page; incomplete metadata can arrive later.
                continue
        return questions

    def presentation_ids(self, course: str) -> list[str]:
        """List every cached deck in this course and platform environment."""
        return [
            row[0]
            for row in self.db.execute(
                "SELECT presentation FROM presentations WHERE domain=? AND course=? "
                "ORDER BY presentation",
                (self.domain, course),
            )
        ]

    def find_question(
        self, course: str, presentation: str, question_id: str, slide_id: str
    ) -> Question | None:
        """Prefer problemId; sid is a fallback only when prob is absent."""
        for question in self.questions(course, presentation):
            if question_id and question.question_id == question_id:
                return question
            if not question_id and slide_id and question.slide_id == slide_id:
                return question
        return None

    def previous_slides(self, question: Question, limit: int) -> list[dict]:
        """Select earlier pages; never send a later page as prior context."""
        if not limit:
            return []
        slides = self.presentation(
            question.course_id, question.presentation_id
        ).get("slides", [])
        earlier = [
            s
            for s in slides
            if isinstance(s, dict) and int(s.get("index", 0)) < question.index
        ]
        return sorted(earlier, key=lambda s: int(s.get("index", 0)))[-limit:]

    def answer(self, question: Question) -> tuple[Any, str] | None:
        """Return only an answer for exactly this version of the question."""
        row = self.db.execute(
            "SELECT answer, source FROM answers "
            "WHERE domain=? AND course=? AND presentation=? AND question=? "
            "AND signature IN (?, ?) "
            "ORDER BY (source='manual') DESC, (signature=?) DESC LIMIT 1",
            (
                self.domain,
                question.course_id,
                question.presentation_id,
                question.question_id,
                question.signature,
                question.legacy_signature,
                question.signature,
            ),
        ).fetchone()
        if row:
            return validate_answer(question, json.loads(row[0])), row[1]
        return None

    def save_answer(self, question: Question, answer: Any, source: str):
        """Store validated answers; AI must not overwrite a manual override."""
        answer = validate_answer(question, answer)
        current = self.answer(question)
        if current and current[1] == "manual" and source != "manual":
            return
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO answers VALUES (?,?,?,?,?,?,?,?)",
                (
                    self.domain,
                    question.course_id,
                    question.presentation_id,
                    question.question_id,
                    question.signature,
                    json.dumps(answer, ensure_ascii=False),
                    source,
                    time.time(),
                ),
            )

    def submission(self, lesson: str, question: str) -> dict | None:
        """Read accepted or unresolved submissions across process restarts."""
        row = self.db.execute(
            "SELECT * FROM submissions "
            "WHERE domain=? AND lesson=? AND question=?",
            (self.domain, lesson, question),
        ).fetchone()
        return dict(row) if row else None

    def mark_submission(
        self, lesson: str, question: Question, answer: Any, status: str
    ):
        """Persist intent before POST and the acknowledgement afterwards."""
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO submissions VALUES (?,?,?,?,?,?,?)",
                (
                    self.domain,
                    lesson,
                    question.question_id,
                    question.signature,
                    json.dumps(answer, ensure_ascii=False),
                    status,
                    time.time(),
                ),
            )

    def export_answers(self, course: str, presentation: str) -> dict:
        """Export the slide-indexed format, with content signatures attached."""
        result: dict[str, Any] = {
            "metadata": {
                "schema_version": 1,
                "domain": self.domain,
                "course_id": course,
                "presentation_id": presentation,
                "signatures": {},
            },
            "answers": {},
            "manual_answers": {},
        }
        for question in self.questions(course, presentation):
            key = str(question.index)
            result["metadata"]["signatures"][key] = question.signature
            cached = self.answer(question)
            if cached:
                answer, source = cached
                target = "manual_answers" if source == "manual" else "answers"
                result[target][key] = answer
        return result

    def import_answers(self, course: str, presentation: str, data: dict) -> int:
        """Bind external caches to slides, preferring manual answers."""
        metadata = data.get("metadata") or {}
        for key, expected in (
            ("domain", self.domain),
            ("course_id", course),
            ("presentation_id", presentation),
        ):
            if key in metadata and identifier(metadata[key]) != expected:
                raise ValueError(f"cache {key} does not match the destination")
        questions = {
            str(q.index): q for q in self.questions(course, presentation)
        }
        if not questions:
            raise ValueError("fetch the presentation before importing answers")
        incoming = dict(data.get("answers") or {})
        manual = {
            k: v for k, v in (data.get("manual_answers") or {}).items() if v
        }
        incoming.update(manual)
        prepared = []
        for key, answer in incoming.items():
            if not answer:
                continue
            question = questions.get(str(key))
            if question is None:
                raise ValueError(f"unknown slide index: {key}")
            signature = (metadata.get("signatures") or {}).get(str(key))
            if signature and not question.matches_signature(signature):
                raise ValueError(f"slide {key} has changed; review the answer")
            prepared.append(
                (question, validate_answer(question, answer), key in manual)
            )
        for question, answer, is_manual in prepared:
            self.save_answer(
                question, answer, "manual" if is_manual else "import"
            )
        return len(prepared)
