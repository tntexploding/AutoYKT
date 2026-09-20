"""Startup cannot silently activate a legacy desktop configuration."""

import argparse
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoykt.cli import _run_command
from autoykt.core.config import ConfigError, discover_config_path, load_config
from autoykt.monitor.preflight import live_configuration_issues


class PreflightTest(unittest.TestCase):

    def test_local_legacy_config_is_not_an_implicit_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.yaml").write_text("agent: {}", encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "autoykt.core.config.default_user_config_path",
                    return_value=root / "private" / "config.yaml",
                ),
                patch("autoykt.core.config.Path.cwd", return_value=root),
            ):
                with self.assertRaises(ConfigError):
                    discover_config_path()
                self.assertEqual(
                    discover_config_path(str(root / "config.yaml")),
                    (root / "config.yaml").resolve(),
                )

    def test_missing_binding_blocks_before_native_capture_initialization(self):
        config = load_config("config.example.yaml")
        config.runtime.dry_run = False
        config.answering.auto_apply = True
        config.pages[0].enabled = True
        config.pages[0].target_window = None
        args = argparse.Namespace(
            profile=None, dry_run=False, detect_only=False, once=True
        )
        with patch("autoykt.cli.ScreenWatcher") as watcher:
            with self.assertRaisesRegex(ConfigError, "target_window"):
                _run_command(config, args)
        watcher.assert_not_called()

    def test_live_run_checks_samples_before_starting_capture(self):
        config = load_config("config.example.yaml")
        config.runtime.dry_run = False
        config.answering.auto_apply = True
        args = argparse.Namespace(
            profile=None, dry_run=False, detect_only=False, once=True
        )
        with (
            patch("autoykt.cli.require_live_configuration"),
            patch("autoykt.cli._check_config", return_value=1) as check,
            patch("autoykt.cli.ScreenWatcher") as watcher,
        ):
            with self.assertRaisesRegex(
                ConfigError, "calibration checks failed"
            ):
                _run_command(config, args)
        check.assert_called_once()
        watcher.assert_not_called()

    def test_text_provider_requires_question_ocr_and_explicit_submission_behavior(
        self,
    ):
        config = load_config("config.example.yaml")
        config.knowledge.question_ocr_enabled = False
        profile = config.pages[0]
        profile.submit_target = None
        profile.submit_on_select = False
        issues = "\n".join(live_configuration_issues(config, profile.id))
        self.assertIn("question_ocr_enabled", issues)
        self.assertIn("submit_target", issues)
        profile.submit_on_select = True
        profile.triggers[0].question_type = "multiple"
        self.assertIn(
            "submit_target",
            "\n".join(live_configuration_issues(config, profile.id)),
        )
