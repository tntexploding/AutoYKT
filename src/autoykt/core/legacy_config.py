"""Pure conversion helpers for the unversioned prototype configuration."""

from __future__ import annotations

import copy
import re
from typing import Any


class LegacyConfigError(ValueError):
    """Raised when a prototype configuration cannot be migrated safely."""


def migrate_legacy_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy document without mutating the caller's data."""
    legacy = copy.deepcopy(raw)
    monitor = _section(legacy, "monitor")
    detector = _section(legacy, "detector")
    agent = _section(legacy, "agent")
    clicker = _section(legacy, "clicker")
    notifier = _section(legacy, "notifier")
    logging_config = _section(legacy, "logging")

    feature_region = monitor.get("feature_roi") or monitor.get("roi")
    question_region = (
        monitor.get("task_roi") or monitor.get("question_roi") or feature_region
    )
    if feature_region is None or question_region is None:
        raise LegacyConfigError(
            "legacy config does not define usable screen regions"
        )

    auto_apply = agent.get("auto_click", False)
    if not isinstance(auto_apply, bool):
        raise LegacyConfigError("legacy agent.auto_click must be a boolean")
    return {
        "version": 2,
        "runtime": {
            "active_profiles": ["legacy"],
            "poll_interval_seconds": monitor.get("poll_interval", 0.5),
            "dry_run": not auto_apply,
        },
        "storage": {
            "data_dir": "storage/data",
            "log_dir": logging_config.get("log_dir", "storage/logs"),
            "screenshot_dir": "storage/screenshots",
        },
        "answering": {
            "providers": [_legacy_provider(agent)],
            "minimum_responses": agent.get("min_response_count", 1),
            "minimum_agreement": 1,
            "auto_apply": auto_apply,
            "prompt_template": agent.get("prompt_template"),
        },
        "knowledge": {
            "enabled": False,
            "question_ocr_enabled": True,
        },
        "pages": [
            _legacy_page(
                monitor,
                detector,
                clicker,
                feature_region,
                question_region,
            )
        ],
        "notifier": _legacy_notifier(notifier),
        "logging": {"level": logging_config.get("level", "INFO")},
    }


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name) or {}
    if not isinstance(value, dict):
        raise LegacyConfigError(f"legacy '{name}' section must be a mapping")
    return value


def _legacy_page(
    monitor: dict[str, Any],
    detector: dict[str, Any],
    clicker: dict[str, Any],
    feature_region: Any,
    question_region: Any,
) -> dict[str, Any]:
    return {
        "id": "legacy",
        "display_name": "Migrated legacy profile",
        "course_id": "default",
        "monitor_index": monitor.get("monitor_index", 1),
        "regions": {
            "detection": feature_region,
            "question": question_region,
            "answers": question_region,
            "verification": monitor.get("verification_roi") or question_region,
            "rearm": monitor.get("rearm_roi") or question_region,
        },
        "triggers": _legacy_triggers(detector, monitor),
        "answer_style": {
            "option_templates": detector.get("option_templates", {}),
            "option_match_threshold": detector.get(
                "option_match_threshold", 0.85
            ),
            "fallback_positions": clicker.get("options_positions", {}),
            "fallback_coordinate_space": "monitor",
        },
        "entry_target": _legacy_click_target(monitor.get("entry_roi")),
        "submit_target": _legacy_click_target(monitor.get("finish_task_roi")),
        "question_ready": {
            "delay_seconds": 0.3,
            "timeout_seconds": monitor.get("task_ready_timeout", 3.0),
            "stable_frames": monitor.get("task_ready_stable_frames", 2),
        },
        "verification": {
            "success_templates": _legacy_success_templates(
                detector,
                monitor,
            ),
            "timeout_seconds": 5.0,
            "poll_interval_seconds": 0.25,
            "stable_hits": 2,
            "minimum_change_ratio": 0.01,
        },
        "rearm": {
            "minimum_change_ratio": monitor.get(
                "post_answer_resume_change_ratio", 0.18
            ),
            "stable_hits": monitor.get("post_answer_resume_change_hits", 2),
        },
    }


def _legacy_success_templates(
    detector: dict[str, Any],
    monitor: dict[str, Any],
) -> list[dict[str, Any]]:
    path = detector.get("success_template_path")
    if not path:
        return []
    return [
        {
            "path": path,
            "threshold": monitor.get("success_match_threshold", 0.90),
        }
    ]


def _legacy_triggers(
    detector: dict[str, Any], monitor: dict[str, Any]
) -> list[dict[str, Any]]:
    configured_rules = detector.get("detection_rules") or []
    if not isinstance(configured_rules, list):
        raise LegacyConfigError("legacy detection_rules must be a list")
    triggers: list[dict[str, Any]] = []
    for index, rule in enumerate(configured_rules):
        if not isinstance(rule, dict):
            raise LegacyConfigError("legacy detection rule must be a mapping")
        old_action = str(rule.get("action", "question_detected"))
        triggers.append(
            {
                "name": rule.get("name") or f"legacy_rule_{index + 1}",
                "path": rule.get("template_path", ""),
                "threshold": rule.get(
                    "threshold", monitor.get("match_threshold", 0.85)
                ),
                "consecutive_hits": rule.get(
                    "debounce_frames", monitor.get("debounce_frames", 3)
                ),
                "action": (
                    "notify" if old_action == "notify_only" else "answer"
                ),
            }
        )
    if triggers:
        return triggers
    return [
        {
            "name": "question",
            "path": detector.get("question_feature_template_path")
            or monitor.get("template_path")
            or "assets/templates/question_region.png",
            "threshold": monitor.get("match_threshold", 0.85),
            "consecutive_hits": monitor.get("debounce_frames", 3),
            "action": "answer",
        }
    ]


def _legacy_provider(agent: dict[str, Any]) -> dict[str, Any]:
    raw_models = agent.get("models") or []
    if not isinstance(raw_models, list):
        raise LegacyConfigError("legacy agent.models must be a list")
    model_names = list(
        dict.fromkeys(
            str(model).strip() for model in raw_models if str(model).strip()
        )
    )
    if not model_names:
        model_names = [str(agent.get("model", "gpt-4o")).strip()]
    try:
        answer_count = int(agent.get("answer_count", len(model_names)))
    except (TypeError, ValueError) as error:
        raise LegacyConfigError(
            "legacy agent.answer_count must be an integer"
        ) from error
    provider: dict[str, Any] = {
        "name": str(agent.get("provider", "openai")),
        "base_url": agent.get("base_url", "https://api.openai.com/v1"),
        "models": model_names[: max(1, answer_count)],
        "timeout_seconds": agent.get("timeout", 30),
    }
    api_key = agent.get("api_key")
    environment_name = _placeholder_environment_name(api_key)
    if environment_name:
        provider["api_key_env"] = environment_name
    elif api_key:
        provider["api_key_env"] = None
        provider["api_key"] = api_key
    else:
        provider["api_key_env"] = "OPENAI_API_KEY"
    return provider


def _legacy_notifier(notifier: dict[str, Any]) -> dict[str, Any]:
    qq_config = _section(notifier, "qq")
    telegram_config = _section(notifier, "telegram")
    enabled = notifier.get("enabled") or []
    if isinstance(enabled, str):
        enabled = [enabled]
    return {
        "enabled": [item for item in enabled if item in {"qq", "telegram"}],
        "qq": {
            "onebot_url": qq_config.get("onebot_url", "http://127.0.0.1:3001"),
            "target_env": _placeholder_environment_name(
                qq_config.get("target_qq")
            )
            or "AUTOYKT_QQ_TARGET",
            "access_token_env": _placeholder_environment_name(
                qq_config.get("access_token")
            )
            or "AUTOYKT_ONEBOT_ACCESS_TOKEN",
        },
        "telegram": {
            "token_env": _placeholder_environment_name(
                telegram_config.get("token")
            )
            or "AUTOYKT_TELEGRAM_TOKEN",
            "chat_id_env": _placeholder_environment_name(
                telegram_config.get("chat_id")
            )
            or "AUTOYKT_TELEGRAM_CHAT_ID",
        },
    }


def _legacy_click_target(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, int) for item in value):
        return None
    if value[2] <= 0 or value[3] <= 0:
        return None
    return {"region": list(value), "coordinate_space": "monitor"}


def _placeholder_environment_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value.strip())
    return match.group(1) if match else None
