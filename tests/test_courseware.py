"""Exercise report-derived field compatibility and offline courseware export."""

import asyncio
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoykt.server.cache import ClassroomCache
from autoykt.server.cli import main
from autoykt.server.models import Question, SIGNED_SLIDE_CDN, validate_answer
from autoykt.server.solver import AssetStore


FIXTURES = Path(__file__).parent / "fixtures/server"


class CoursewareTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.origin = "https://changjiang.yuketang.cn"
        self.cache = ClassroomCache(
            self.root / "classroom.sqlite3", self.origin
        )
        self.addCleanup(self.cache.close)

    def test_real_projection_uses_result_not_submit_layout_or_correct_answers(
        self,
    ):
        projection = json.loads(
            (FIXTURES / "changjiang_answered_20261105.json").read_text(
                encoding="utf-8"
            )
        )
        document = projection["response"]["data"]
        self.cache.save_presentation("course", "deck", document)
        questions = self.cache.questions("course", "deck")
        self.assertEqual(len(questions), 2)
        self.assertEqual(
            [validate_answer(q, q.result) for q in questions], [["B"], ["A"]]
        )
        for question, slide in zip(questions, document["slides"]):
            self.assertEqual(set(question.options), {"A", "B", "C", "D"})
            self.assertTrue(slide["problem"]["submit"])
            self.assertFalse(slide["problem"]["answers"])
            self.assertIsNone(self.cache.answer(question))
            slide["problem"]["result"] = None
            unanswered = Question.from_slide("course", "deck", slide)
            self.assertIsNone(unanswered.result)

    def test_reference_aliases_relative_cover_and_incomplete_page(self):
        document = {
            "Title": "Deck",
            "custom": {"keep": True},
            "slides": [
                None,
                {
                    "id": "s2",
                    "cover": "folder/page.png",
                    "problemData": {
                        "prob": "q2",
                        "problemtype": "multiple",
                        "content": "题干",
                        "bullets": [
                            {"label": "A", "content": "甲"},
                            {"label": "B", "content": "乙"},
                        ],
                    },
                },
                {"id": "s3", "hasProblem": True},
                {
                    "id": "s4",
                    "hasProblem": True,
                    "prob": "q4",
                    "problem_type": "fillblank",
                    "stem": "填空",
                    "blanks": [{}, {}],
                },
            ],
        }
        self.cache.save_presentation("course", "deck", document)
        saved = self.cache.presentation("course", "deck")
        self.assertEqual(saved["custom"], {"keep": True})
        self.assertEqual(len(saved["slides"]), 4)
        questions = self.cache.questions("course", "deck")
        self.assertEqual(
            [question.question_id for question in questions], ["q2", "q4"]
        )
        self.assertEqual(questions[0].index, 2)
        self.assertEqual(
            questions[0].image_url,
            "https://qn-scd1.yuketang.cn/folder/page.png",
        )
        self.assertEqual(questions[0].options, {"A": "甲", "B": "乙"})
        self.assertEqual(questions[1].blanks, 2)

    def test_unlimited_polling_from_sample_accepts_multiple_keys(self):
        question = Question.from_slide(
            "c",
            "p",
            {
                "problemData": {
                    "id": "q",
                    "problem_type": "polling",
                    "body": "投票",
                    "options": [{"key": "A"}, {"key": "B"}],
                    "polling_count": -1,
                }
            },
        )
        self.assertEqual(validate_answer(question, ["B", "A"]), ["A", "B"])

    def test_offline_cli_exports_original_pages_and_prepared_answer(self):
        page_url = SIGNED_SLIDE_CDN + "slide.png"
        url = page_url + "?e=1&token=first"
        document = {
            "title": "One",
            "slides": [
                {
                    "id": "s",
                    "cover": url,
                    "problemData": {
                        "id": "q",
                        "problemType": 1,
                        "body": "题干",
                        "options": [{"key": "A", "value": "甲"}],
                    },
                }
            ],
        }
        self.cache.save_presentation("course", "deck", document)
        question = self.cache.questions("course", "deck")[0]
        self.cache.save_answer(question, ["A"], "manual")
        with self.cache.db:
            self.cache.db.execute(
                "UPDATE answers SET signature=?", (question.legacy_signature,)
            )
        upgraded = ClassroomCache(
            self.root / "classroom.sqlite3", self.origin
        )
        upgraded.close()
        images = self.root / "images"
        images.mkdir()
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jSucAAAAASUVORK5CYII="
        )
        (images / hashlib.sha256(url.encode()).hexdigest()).write_bytes(png)
        assets = AssetStore(images, None)
        self.assertEqual(asyncio.run(assets.image(url)), (png, "image/png"))
        refreshed_url = page_url + "?e=2&token=second"
        document["slides"][0]["cover"] = refreshed_url
        self.cache.save_presentation("course", "deck", document)
        refreshed = self.cache.questions("course", "deck")[0]
        self.assertEqual(refreshed.signature, question.signature)
        self.assertIsNone(assets.cached_image(refreshed_url + "&crop=2"))
        config = self.root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "data_dir": str(self.root),
                    "platform": {"base_url": self.origin},
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "export"
        with (
            patch("autoykt.server.cli.logging.basicConfig"),
            patch(
                "autoykt.server.client.YuketangClient.__aenter__",
                side_effect=AssertionError("offline export must not connect"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            result = main(
                [
                    "--config",
                    str(config),
                    "courseware",
                    "export",
                    str(output),
                    "--course",
                    "course",
                ]
            )
        self.assertEqual(result, 0)
        manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        deck = output / manifest["presentations"][0]["directory"]
        self.assertEqual((deck / "slide-0001.png").read_bytes(), png)
        exported = json.loads(
            (deck / "questions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(exported[0]["prepared_answer"], ["A"])
        self.assertEqual(exported[0]["answer_source"], "manual")
        self.assertEqual(
            json.loads(
                (deck / "presentation.json").read_text(encoding="utf-8")
            )["slides"][0]["problemData"],
            document["slides"][0]["problemData"],
        )
