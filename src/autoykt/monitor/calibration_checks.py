"""Validate private labeled screenshots against the current calibration."""

from autoykt.core.config import AppConfig, PageProfileConfig
from autoykt.monitor.image_inspection import analyze_image
from autoykt.monitor.image_utils import read_image


def calibration_sample_issues(
    config: AppConfig, profile: PageProfileConfig
) -> list[str]:
    """Report mismatches without desktop, network, or output writes."""
    issues = []
    for sample in profile.calibration_samples:
        prefix = f"{profile.id} sample {sample.path}"
        try:
            actual = analyze_image(
                config, profile, read_image(config.resolve_path(sample.path))
            )
        except (OSError, ValueError) as error:
            issues.append(f"{prefix}: {error}")
            continue
        if profile.page_guard and not actual["page_identity_matches"]:
            issues.append(f"{prefix}: page identity does not match")
        if actual["page_state"] != sample.expected_state:
            issues.append(
                f"{prefix}: expected {sample.expected_state}, "
                f"got {actual['page_state']}"
            )
        if sample.question_type and actual["question_types"] != [
            sample.question_type
        ]:
            issues.append(
                f"{prefix}: expected {sample.question_type} trigger, "
                f"got {actual['question_types']}"
            )
        options = actual["options"]
        if sample.expected_options is not None:
            if set(options) != set(sample.expected_options):
                issues.append(
                    f"{prefix}: expected options {sample.expected_options}, "
                    f"got {sorted(options)}"
                )
            if any(
                item["ambiguous"] or item["selected"] is None
                for item in options.values()
            ):
                issues.append(
                    f"{prefix}: ambiguous letter or unknown selection color"
                )
        if sample.selected_options is not None:
            selected = {
                key for key, item in options.items() if item["selected"] is True
            }
            if selected != set(sample.selected_options):
                issues.append(
                    f"{prefix}: selected options disagree with sample"
                )
    return issues
