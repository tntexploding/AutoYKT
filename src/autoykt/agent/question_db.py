"""SQLite question cache with fuzzy matching."""

import sqlite3
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from dataclasses import dataclass

logger = logging.getLogger("auto_answer")

DB_PATH = "storage/data/questions.db"


@dataclass
class CachedAnswer:
    """A cached question-answer pair from the database."""
    question_text: str
    options: str
    answer: str
    confidence: float
    source: str
    created_at: str


class QuestionDB:
    """SQLite-backed question cache with fuzzy text matching."""

    def __init__(self, db_path: str = DB_PATH, similarity_threshold: float = 0.9) -> None:
        self._threshold = similarity_threshold
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_table()
        logger.info(f"QuestionDB initialized at {self._db_path}")

    def _init_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT NOT NULL,
                question_text TEXT NOT NULL,
                options TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                source TEXT DEFAULT 'llm',
                created_at TEXT NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_question_hash ON questions(question_hash)
        """)
        self._conn.commit()

    @staticmethod
    def _hash(text: str) -> str:
        """Generate a short hash for quick pre-filtering."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def lookup(self, question_text: str) -> CachedAnswer | None:
        """Look up a question in the cache using fuzzy matching.

        Strategy:
        1. Try exact hash match first (fast path)
        2. Fall back to fuzzy matching against all entries (slow path)
        """
        q_hash = self._hash(question_text)

        # Fast path: exact hash match
        row = self._conn.execute(
            "SELECT * FROM questions WHERE question_hash = ? LIMIT 1",
            (q_hash,),
        ).fetchone()

        if row:
            self._bump_hit_count(row["id"])
            logger.debug(f"Cache hit (exact hash): {question_text[:40]}...")
            return self._row_to_cached(row)

        # Slow path: fuzzy match
        all_rows = self._conn.execute(
            "SELECT * FROM questions ORDER BY hit_count DESC LIMIT 500"
        ).fetchall()

        best_match: sqlite3.Row | None = None
        best_ratio = 0.0

        for r in all_rows:
            ratio = SequenceMatcher(None, question_text, r["question_text"]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = r

        if best_match and best_ratio >= self._threshold:
            self._bump_hit_count(best_match["id"])
            logger.debug(
                f"Cache hit (fuzzy, ratio={best_ratio:.3f}): {question_text[:40]}..."
            )
            return self._row_to_cached(best_match)

        logger.debug(f"Cache miss: {question_text[:40]}...")
        return None

    def store(
        self,
        question_text: str,
        options: str,
        answer: str,
        confidence: float = 0.0,
        source: str = "llm",
    ) -> None:
        """Store a new question-answer pair."""
        self._conn.execute(
            """INSERT INTO questions
               (question_hash, question_text, options, answer, confidence, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                self._hash(question_text),
                question_text,
                options,
                answer,
                confidence,
                source,
                datetime.now().isoformat(),
            ),
        )
        self._conn.commit()
        logger.info(f"Stored answer '{answer}' for: {question_text[:40]}...")

    def _bump_hit_count(self, row_id: int) -> None:
        self._conn.execute(
            "UPDATE questions SET hit_count = hit_count + 1 WHERE id = ?",
            (row_id,),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_cached(row: sqlite3.Row) -> CachedAnswer:
        return CachedAnswer(
            question_text=row["question_text"],
            options=row["options"],
            answer=row["answer"],
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
        )

    def close(self) -> None:
        self._conn.close()