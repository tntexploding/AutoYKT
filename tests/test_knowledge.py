"""Tests for local course knowledge ingestion and retrieval."""

from pathlib import Path
import tempfile
import unittest

from autoykt.knowledge.store import (
    KnowledgeMatch,
    KnowledgeStore,
    _split_chunks,
    format_knowledge_context,
)


class KnowledgeStoreTest(unittest.TestCase):

    def test_ingests_deduplicates_and_retrieves_chinese_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "knowledge.db"
            with KnowledgeStore(
                database, chunk_size=120, chunk_overlap=20
            ) as store:
                first = store.add_document(
                    "physics",
                    "lesson-one",
                    "牛顿定律",
                    "牛顿第二定律说明物体加速度与合外力成正比，"
                    "与质量成反比。公式为 F=ma。" * 3,
                )
                duplicate = store.add_document(
                    "physics",
                    "duplicate-source",
                    "重复资料",
                    "牛顿第二定律说明物体加速度与合外力成正比，"
                    "与质量成反比。公式为 F=ma。" * 3,
                )
                matches = store.search(
                    "physics",
                    "合外力增大时加速度如何变化",
                    limit=2,
                    minimum_score=0.01,
                )
                self.assertTrue(first.added)
                self.assertFalse(duplicate.added)
                self.assertEqual(store.document_count("physics"), 1)
                self.assertTrue(matches)
                self.assertEqual(matches[0].title, "牛顿定律")

    def test_replaces_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with KnowledgeStore(Path(directory) / "knowledge.db") as store:
                store.add_document("course", "slides", "One", "旧内容" * 60)
                store.add_document("course", "slides", "Two", "新内容" * 60)
                self.assertEqual(store.document_count("course"), 1)

    def test_changed_source_can_converge_on_existing_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with KnowledgeStore(Path(directory) / "knowledge.db") as store:
                store.add_document("course", "first", "Old", "旧内容" * 60)
                store.add_document("course", "second", "New", "新内容" * 60)
                result = store.add_document(
                    "course", "first", "Updated", "新内容" * 60
                )
                self.assertFalse(result.added)
                self.assertEqual(store.document_count("course"), 1)
                self.assertFalse(
                    store.search("course", "旧内容", minimum_score=0.9)
                )

    def test_chunk_size_is_a_hard_limit(self) -> None:
        chunks = _split_chunks("段落一\n" * 100, size=120, overlap=20)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))

    def test_formatted_context_respects_hard_limit(self) -> None:
        matches = [
            KnowledgeMatch("甲" * 80, 0.9, "one", "第一章"),
            KnowledgeMatch("乙" * 80, 0.8, "two", "第二章"),
        ]
        context = format_knowledge_context(matches, maximum_characters=100)
        self.assertLessEqual(len(context), 100)


class KnowledgeSourceRegressionTest(unittest.TestCase):
    """Deduplicating content must retain independent source ownership."""

    def test_updating_one_duplicate_source_keeps_the_other_material(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with KnowledgeStore(Path(directory) / "knowledge.db") as store:
                store.add_document(
                    "course", "first", "First", "original material"
                )
                store.add_document(
                    "course", "second", "Second", "original material"
                )
                store.add_document(
                    "course", "first", "Updated", "replacement topic"
                )
                self.assertEqual(store.document_count("course"), 2)
                matches = store.search(
                    "course", "original material", minimum_score=0.99
                )
                self.assertEqual([item.source for item in matches], ["second"])
                self.assertTrue(
                    store.search(
                        "course", "replacement topic", minimum_score=0.99
                    )
                )
                self.assertFalse(
                    store.search("other-course", "original material")
                )

    def test_old_database_sources_are_backfilled_on_open(self) -> None:
        import sqlite3
        from contextlib import closing

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "knowledge.db"
            with KnowledgeStore(database) as store:
                store.add_document(
                    "course", "first", "First", "original material"
                )
            with closing(sqlite3.connect(database)) as connection:
                connection.execute("DROP TABLE document_sources")
            with KnowledgeStore(database) as store:
                result = store.add_document(
                    "course", "first", "First", "original material"
                )
                self.assertFalse(result.added)
                store.add_document(
                    "course", "first", "Updated", "replacement topic"
                )
                self.assertEqual(store.document_count("course"), 1)
                self.assertFalse(
                    store.search(
                        "course", "original material", minimum_score=0.99
                    )
                )


if __name__ == "__main__":
    unittest.main()
