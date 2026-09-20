"""Configuration requirements before a real desktop answer can be entered."""

from autoykt.core.config import AppConfig, ConfigError, PageProfileConfig


def live_configuration_issues(
    config: AppConfig, profile_id: str | None = None
) -> list[str]:
    """Return actionable setup errors without capturing or clicking anything."""
    issues: list[str] = []
    if config.migrated_from_legacy:
        issues.append("legacy configuration must be migrated before live input")
    profiles = (
        [config.page_profile(profile_id)]
        if profile_id
        else [
            config.page_profile(key) for key in config.runtime.active_profiles
        ]
        if config.runtime.active_profiles
        else config.active_page_profiles()
    )
    if not profiles:
        issues.append(
            "enable a calibrated page and select runtime.active_profiles"
        )
    for profile in profiles:
        issues.extend(_profile_issues(profile))
    if (
        any(
            provider.input_mode == "text"
            for provider in config.answering.providers
        )
        and not config.knowledge.question_ocr_enabled
    ):
        issues.append("text providers require knowledge.question_ocr_enabled")
    for provider in config.answering.providers:
        try:
            provider.resolve_api_key()
        except ConfigError as error:
            issues.append(str(error))
    return issues


def _profile_issues(profile: PageProfileConfig) -> list[str]:
    issues: list[str] = []
    if not profile.enabled:
        issues.append("profile is disabled")
    if profile.target_window is None:
        issues.append("calibrate target_window and window-relative regions")
    if profile.page_guard is None:
        issues.append("calibrate page_guard to identify the classroom page")
    if not profile.verification.success_templates:
        issues.append(
            "calibrate verification.success_templates for submission feedback"
        )
    expected_types = {
        item.question_type
        for item in profile.calibration_samples
        if item.question_type is not None
    }
    configured_types = {
        item.question_type
        for item in profile.triggers
        if item.action == "answer"
    }
    if expected_types - configured_types:
        issues.append(
            "missing question types required by samples: "
            f"{sorted(expected_types - configured_types)}"
        )
    style = profile.answer_style
    if not style.option_templates or style.button_colors is None:
        issues.append(
            "calibrate letter templates and selected/unselected button_colors"
        )
    if style.fallback_positions:
        issues.append(
            "remove fixed fallback_positions; locate each visible letter"
        )
    multiple = any(
        trigger.question_type == "multiple" for trigger in profile.triggers
    )
    if profile.submit_target is None and (
        multiple or not profile.submit_on_select
    ):
        issues.append(
            "calibrate submit_target; only confirmed single-choice auto-submit "
            "pages may set submit_on_select: true"
        )
    if profile.submit_target is not None and profile.submit_on_select:
        issues.append("choose submit_target or submit_on_select, not both")
    return [f"{profile.id}: {issue}" for issue in issues]


def require_live_configuration(config: AppConfig) -> None:
    """Reject incomplete setup before native capture or mouse input."""
    issues = live_configuration_issues(config)
    if issues:
        raise ConfigError("live preflight failed:\n- " + "\n- ".join(issues))
