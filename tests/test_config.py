"""Tests for v2 validation and in-memory legacy migration."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
import warnings

import yaml

from autoykt.cli import _build_parser, _check_command, _check_config, _migrate_config
from autoykt.core.config import ConfigError, load_config


def _write_yaml(directory: str, document: dict[str, object]) -> Path:
    path = Path(directory) / "config.yaml"
    path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    return path


class ConfigurationTest(unittest.TestCase):

    def test_check_profile_checks_disabled_candidate_assets_without_enabling_it(
        self,
    ):
        config = load_config("config.example.yaml")
        original = config.model_dump()
        args = _build_parser().parse_args(
            ["check", "--profile", "example_yuketang"]
        )
        with redirect_stdout(io.StringIO()) as output:
            result = _check_command(config, args)
        self.assertEqual(result, 1)
        self.assertIn(
            "profile example_yuketang option A file is missing",
            output.getvalue(),
        )
        self.assertEqual(config.model_dump(), original)

    def test_empty_inline_key_is_rejected_without_exposing_secrets(self):
        document = yaml.safe_load(
            Path("config.example.yaml").read_text(encoding="utf-8")
        )
        document["answering"]["providers"][0]["api_key"] = "  "
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ConfigError, "api_key cannot be empty"):
                load_config(_write_yaml(directory, document))
            document["answering"]["providers"][0]["api_key"] = " private-test "
            config = load_config(_write_yaml(directory, document))
            self.assertEqual(
                config.answering.providers[0].resolve_api_key(), "private-test"
            )
            self.assertNotIn("private-test", str(config.model_dump()))

    def test_loads_public_v2_template(self) -> None:
        config = load_config("config.example.yaml")
        self.assertEqual(config.version, 2)
        self.assertFalse(config.migrated_from_legacy)
        self.assertEqual(config.pages[0].id, "example_yuketang")
        packaged = Path(
            "src/autoykt/resources/config.example.yaml"
        ).read_bytes()
        self.assertEqual(Path("config.example.yaml").read_bytes(), packaged)

    def test_migrates_legacy_without_modifying_file(self) -> None:
        legacy = {
            "monitor": {
                "roi": [0, 0, 100, 100],
                "question_roi": [0, 0, 200, 200],
                "verification_roi": [5, 5, 80, 80],
                "rearm_roi": [10, 10, 90, 90],
            },
            "detector": {
                "question_feature_template_path": "question.png",
                "success_template_path": "success.png",
                "option_templates": {"A": "a.png", "B": "b.png"},
            },
            "agent": {
                "api_key": "${TEST_API_KEY}",
                "model": "vision-model",
                "auto_click": False,
            },
            "clicker": {"options_positions": {}},
            "notifier": {"enabled": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = _write_yaml(directory, legacy)
            before = path.read_bytes()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                config = load_config(path)
            self.assertTrue(config.migrated_from_legacy)
            self.assertEqual(config.pages[0].id, "legacy")
            self.assertEqual(
                config.pages[0].regions.verification,
                (5, 5, 80, 80),
            )
            self.assertEqual(
                config.pages[0].verification.success_templates[0].path,
                "success.png",
            )
            self.assertEqual(path.read_bytes(), before)

    def test_rejects_non_positive_region_size(self) -> None:
        document = yaml.safe_load(
            Path("config.example.yaml").read_text(encoding="utf-8")
        )
        document["pages"][0]["regions"]["question"] = [0, 0, 0, 100]
        with tempfile.TemporaryDirectory() as directory:
            path = _write_yaml(directory, document)
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_legacy_secret_is_excluded_from_export(self) -> None:
        legacy = {
            "monitor": {"roi": [0, 0, 100, 100]},
            "detector": {
                "question_feature_template_path": "question.png",
                "option_templates": {"A": "a.png"},
            },
            "agent": {"api_key": "private-value", "model": "vision"},
            "notifier": {"enabled": []},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = _write_yaml(directory, legacy)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                config = load_config(path)
            output = Path(directory) / "config.v2.yaml"
            with redirect_stdout(io.StringIO()):
                _migrate_config(config, output)
            exported = output.read_text(encoding="utf-8")
            self.assertNotIn("private-value", exported)
            self.assertNotIn("api_key:", exported)
            self.assertIn("api_key_env: OPENAI_API_KEY", exported)

    def test_rejects_path_like_trigger_and_option_names(self) -> None:
        document = yaml.safe_load(
            Path("config.example.yaml").read_text(encoding="utf-8")
        )
        document["pages"][0]["triggers"][0]["name"] = "../question"
        document["pages"][0]["answer_style"]["option_templates"] = {
            "../../A": "a.png"
        }
        with tempfile.TemporaryDirectory() as directory:
            path = _write_yaml(directory, document)
            with self.assertRaises(ConfigError):
                load_config(path)

    def test_check_rejects_provider_without_key_source(self) -> None:
        document = yaml.safe_load(
            Path("config.example.yaml").read_text(encoding="utf-8")
        )
        document["answering"]["providers"][0]["api_key_env"] = None
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(_write_yaml(directory, document))
            with redirect_stdout(io.StringIO()) as output:
                result = _check_config(config)
        self.assertEqual(result, 1)
        self.assertIn("has no API key environment", output.getvalue())


class ConfigurationRegressionTest(unittest.TestCase):
    """Private input and moved configuration keep their intended boundaries."""

    def test_invalid_config_does_not_print_secrets(self) -> None:
        document = yaml.safe_load(
            Path("config.example.yaml").read_text(encoding="utf-8")
        )
        document["answering"]["providers"][0]["api_key"] = "private-test-value"
        document["answering"]["minimum_responses"] = 100
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ConfigError) as caught:
                load_config(_write_yaml(directory, document))
        self.assertNotIn("private-test-value", str(caught.exception))

    def test_invalid_yaml_does_not_print_source_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("api_key: [private-test-value\n", encoding="utf-8")
            with self.assertRaises(ConfigError) as caught:
                load_config(path)
        self.assertNotIn("private-test-value", str(caught.exception))
        self.assertIn("line", str(caught.exception))

    def test_rejects_infinite_timeouts_and_zero_change_thresholds(self) -> None:
        for section, field, value in (
            ("question_ready", "timeout_seconds", float("inf")),
            ("verification", "minimum_change_ratio", 0),
            ("rearm", "minimum_change_ratio", 0),
        ):
            with self.subTest(section=section):
                document = yaml.safe_load(
                    Path("config.example.yaml").read_text(encoding="utf-8")
                )
                document["pages"][0][section][field] = value
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaises(ConfigError):
                        load_config(_write_yaml(directory, document))

    def test_migration_to_another_directory_preserves_path_targets(
        self,
    ) -> None:
        document = yaml.safe_load(
            Path("config.example.yaml").read_text(encoding="utf-8")
        )
        document["answering"]["prompt_template"] = "prompts/answer.j2"
        document["pages"][0]["verification"]["success_templates"] = [
            {"path": "templates/success.png"}
        ]
        document["pages"][0]["page_flow"] = {
            "after_submit": [
                {
                    "name": "next",
                    "when": {
                        "path": "templates/next.png",
                        "region": [0, 0, 10, 10],
                    },
                    "target": {"point": [2, 3]},
                }
            ],
            "manual": [
                {"path": "templates/expired.png", "region": [0, 0, 10, 10]}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = load_config(_write_yaml(directory, document))
            output = root / "private" / "config.yaml"
            with redirect_stdout(io.StringIO()):
                _migrate_config(original, output)
            migrated = load_config(output)
            old_page, new_page = original.pages[0], migrated.pages[0]
            for old, new in (
                (original.storage.log_dir, migrated.storage.log_dir),
                (
                    original.knowledge.database_path,
                    migrated.knowledge.database_path,
                ),
                (
                    original.answering.prompt_template,
                    migrated.answering.prompt_template,
                ),
                (old_page.triggers[0].path, new_page.triggers[0].path),
                (
                    old_page.page_flow.after_submit[0].when.path,
                    new_page.page_flow.after_submit[0].when.path,
                ),
                (
                    old_page.page_flow.manual[0].path,
                    new_page.page_flow.manual[0].path,
                ),
                (
                    old_page.answer_style.option_templates["A"],
                    new_page.answer_style.option_templates["A"],
                ),
                (
                    old_page.verification.success_templates[0].path,
                    new_page.verification.success_templates[0].path,
                ),
            ):
                assert isinstance(old, str) and isinstance(new, str)
                self.assertEqual(
                    original.resolve_path(old), migrated.resolve_path(new)
                )

    def test_check_rejects_images_without_spatial_detail(self) -> None:
        import numpy as np
        from autoykt.monitor.image_utils import write_png

        document = yaml.safe_load(
            Path("config.example.yaml").read_text(encoding="utf-8")
        )
        document["pages"][0]["enabled"] = True
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "flat.png"
            write_png(image, np.zeros((8, 8, 3), dtype=np.uint8))
            document["pages"][0]["triggers"][0]["path"] = str(image)
            config = load_config(_write_yaml(directory, document))
            with redirect_stdout(io.StringIO()) as output:
                result = _check_config(config)
            self.assertEqual(result, 1)
            self.assertIn("spatial detail", output.getvalue())


if __name__ == "__main__":
    unittest.main()
