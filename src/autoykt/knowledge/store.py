"""SQLite-backed local retrieval for course-specific knowledge."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import threading


_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]+|[A-Za-z0-9_]+")


@dataclass(frozen=True)
class KnowledgeMatch:
    """One locally retrieved course material chunk."""

    content: str
    score: float
    source: str
    title: str


@dataclass(frozen=True)
class IngestResult:
    """Result of adding or de-duplicating one document."""

    document_id: int
    added: bool
    chunks: int


class KnowledgeStore:
    """Store course documents and rank chunks with local lexical similarity."""

    def __init__(
        self,
        database_path: str | Path,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self._path,
            check_same_thread=False,
        )
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._initialize_schema()
        except Exception:
            self._connection.close()
            raise

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(course_id, source)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    course_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    terms_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                        ON DELETE CASCADE
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_course "
                "ON chunks(course_id)"
            )

            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_sources (
                    course_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    document_id INTEGER NOT NULL,
                    PRIMARY KEY(course_id, source),
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                        ON DELETE CASCADE
                )
                """
            )
            # Existing databases retain their original representative sources.
            self._connection.execute(
                "INSERT OR IGNORE INTO document_sources "
                "(course_id, source, title, document_id) "
                "SELECT course_id, source, title, id FROM documents"
            )

    def add_document(
        self,
        course_id: str,
        source: str,
        title: str,
        text: str,
    ) -> IngestResult:
        """Add or replace one source while de-duplicating identical content."""
        normalized = _normalize_text(text)
        if not normalized:
            raise ValueError("knowledge document contains no text")
        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        chunks = _split_chunks(
            normalized,
            size=self._chunk_size,
            overlap=self._chunk_overlap,
        )
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT documents.id, documents.content_hash FROM documents "
                "JOIN document_sources "
                "ON documents.id = document_sources.document_id "
                "WHERE document_sources.course_id = ? "
                "AND document_sources.source = ?",
                (course_id, source),
            ).fetchone()
            if (
                existing is not None
                and existing["content_hash"] == content_hash
            ):
                return IngestResult(
                    document_id=int(existing["id"]),
                    added=False,
                    chunks=0,
                )
            duplicate = self._connection.execute(
                "SELECT id FROM documents "
                "WHERE course_id = ? AND content_hash = ? LIMIT 1",
                (course_id, content_hash),
            ).fetchone()
            if existing is not None:
                self._detach_source(course_id, source, int(existing["id"]))
            if duplicate is not None:
                self._connection.execute(
                    "INSERT INTO document_sources "
                    "(course_id, source, title, document_id) "
                    "VALUES (?, ?, ?, ?)",
                    (course_id, source, title, int(duplicate["id"])),
                )
                return IngestResult(
                    document_id=int(duplicate["id"]),
                    added=False,
                    chunks=0,
                )
            cursor = self._connection.execute(
                """
                INSERT INTO documents
                    (course_id, source, title, content_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    course_id,
                    source,
                    title,
                    content_hash,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite did not return a document id")
            document_id = int(cursor.lastrowid)
            self._connection.execute(
                "INSERT INTO document_sources "
                "(course_id, source, title, document_id) "
                "VALUES (?, ?, ?, ?)",
                (course_id, source, title, document_id),
            )
            self._connection.executemany(
                """
                INSERT INTO chunks
                    (document_id, course_id, ordinal, content, terms_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_id,
                        course_id,
                        index,
                        chunk,
                        json.dumps(_term_counts(chunk), ensure_ascii=False),
                    )
                    for index, chunk in enumerate(chunks)
                ],
            )
        return IngestResult(
            document_id=document_id,
            added=True,
            chunks=len(chunks),
        )

    def _detach_source(
        self, course_id: str, source: str, document_id: int
    ) -> None:
        self._connection.execute(
            "DELETE FROM document_sources WHERE course_id = ? AND source = ?",
            (course_id, source),
        )
        remaining = self._connection.execute(
            "SELECT source, title FROM document_sources "
            "WHERE document_id = ? ORDER BY source LIMIT 1",
            (document_id,),
        ).fetchone()
        if remaining is None:
            self._connection.execute(
                "DELETE FROM documents WHERE id = ?", (document_id,)
            )
        else:
            self._connection.execute(
                "UPDATE documents SET source = ?, title = ? WHERE id = ?",
                (remaining["source"], remaining["title"], document_id),
            )

    def search(
        self,
        course_id: str,
        query: str,
        limit: int = 4,
        minimum_score: float = 0.08,
    ) -> list[KnowledgeMatch]:
        """Rank course chunks using local token and CJK n-gram similarity."""
        query_terms = _term_counts(query)
        if not query_terms or limit < 1:
            return []
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT chunks.content, chunks.terms_json,
                       documents.source, documents.title
                FROM chunks
                JOIN documents ON documents.id = chunks.document_id
                WHERE chunks.course_id = ?
                """,
                (course_id,),
            ).fetchall()
        matches: list[KnowledgeMatch] = []
        for row in rows:
            document_terms = Counter(json.loads(row["terms_json"]))
            score = _cosine_similarity(query_terms, document_terms)
            if score >= minimum_score:
                matches.append(
                    KnowledgeMatch(
                        content=str(row["content"]),
                        score=score,
                        source=str(row["source"]),
                        title=str(row["title"]),
                    )
                )
        matches.sort(key=lambda item: item.score, reverse=True)
        return matches[:limit]

    def document_count(self, course_id: str | None = None) -> int:
        """Return the number of stored documents."""
        with self._lock:
            if course_id is None:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM documents"
                ).fetchone()
            else:
                row = self._connection.execute(
                    "SELECT COUNT(*) AS count FROM documents "
                    "WHERE course_id = ?",
                    (course_id,),
                ).fetchone()
        return int(row["count"] if row is not None else 0)

    def close(self) -> None:
        """Commit and close the database connection."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def format_knowledge_context(
    matches: list[KnowledgeMatch],
    maximum_characters: int,
) -> str:
    """Format ranked chunks for a model prompt within a hard size limit."""
    sections: list[str] = []
    used = 0
    for match in matches:
        section = (
            f"[资料：{match.title}，相关度 {match.score:.2f}]\n"
            f"{match.content}"
        )
        separator = "\n\n" if sections else ""
        remaining = maximum_characters - used - len(separator)
        if remaining <= 0:
            break
        rendered = separator + section[:remaining]
        sections.append(rendered)
        used += len(rendered)
    return "".join(sections)


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _split_chunks(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    return [
        chunk
        for start in range(0, len(text), step)
        if (chunk := text[start : start + size].strip())
    ]


def _term_counts(text: str) -> Counter[str]:
    terms: list[str] = []
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            terms.extend(token)
            terms.extend(
                token[index : index + 2]
                for index in range(max(0, len(token) - 1))
            )
        else:
            terms.append(token)
    return Counter(terms)


def _cosine_similarity(
    left: Counter[str],
    right: Counter[str],
) -> float:
    shared = left.keys() & right.keys()
    numerator = sum(left[term] * right[term] for term in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
