"""Inspect calibrated regions on a local image without desktop access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from autoykt.core.config import (
    AppConfig,
    ImageTemplateConfig,
    PageProfileConfig,
    Region,
)
from autoykt.monitor.detector import (
    ImageTemplateMatcher,
    MultiTemplateDetector,
    OptionTemplateDetector,
)
from autoykt.monitor.image_utils import crop_image, read_image
from autoykt.monitor.operations import ProfileSession
from autoykt.monitor.page_flow import PageFlow

# OpenCV exports its native API dynamically.
# pylint: disable=no-member


def analyze_image(
    config: AppConfig, profile: PageProfileConfig, frame: np.ndarray
) -> dict[str, Any]:
    """Evaluate stored pixels without constructing desktop or model clients."""
    if profile.target_window and profile.target_window.client_size != (
        frame.shape[1],
        frame.shape[0],
    ):
        raise ValueError(
            "image size does not match calibrated window client size"
        )
    style = profile.answer_style
    detector = OptionTemplateDetector.from_config(config, style)
    return {
        **_inspect_states(config, profile, frame),
        "options": detector.detect(
            crop_image(frame, profile.regions.answer_region)
        ),
    }


def inspect_image(config: AppConfig, profile_id: str, image_path: Path) -> Path:
    """Save regions, option states, and annotations without planning input."""
    profile = config.page_profile(profile_id)
    frame = read_image(image_path)
    height, width = frame.shape[:2]
    if profile.target_window and profile.target_window.client_size != (
        width,
        height,
    ):
        raise ValueError(
            "image size does not match calibrated window client size"
        )
    session = ProfileSession(config, profile)
    session.start("image_inspection")
    try:
        session.save_frame("full", frame)
        session.record(
            "source",
            image_path=str(image_path.resolve()),
            image_size=[width, height],
            coordinate_space="image",
            live_geometry_verified=False,
        )
        annotated = frame.copy()
        region_names: dict[Region, list[str]] = {}
        for name, region in profile.regions.model_dump().items():
            if region is not None:
                session.save_frame(name, crop_image(frame, region))
                region_names.setdefault(tuple(region), []).append(name)
        for region, names in region_names.items():
            _draw_region(annotated, region, " / ".join(names))
        observed = _inspect_options(session, frame, annotated)
        session.record(
            "inspection",
            **_inspect_states(config, profile, frame),
            options=observed,
            missing_options=sorted(
                profile.answer_style.option_keys - observed.keys()
            ),
            ai_calls=0,
            mouse_clicks=0,
        )
        session.save_frame("annotated", annotated)
        session.finish("inspected")
    except BaseException:
        session.finish("inspection_failed")
        raise
    assert session.report_path is not None
    return session.report_path


def _inspect_options(
    session: ProfileSession, frame: np.ndarray, annotated: np.ndarray
) -> dict[str, Any]:
    region = session.profile.regions.answer_region
    matches = session.inspect_options(crop_image(frame, region))
    observed = {}
    for key, match in matches.items():
        x, y = (
            region[0] + match["top_left"][0],
            region[1] + match["top_left"][1],
        )
        selected = match["selected"]
        state = (
            "ambiguous"
            if match["ambiguous"]
            else "unknown"
            if selected is None
            else "selected"
            if selected
            else "unselected"
        )
        observed[key] = {
            **match,
            "state": state,
            "top_left": [x, y],
            "center": [
                region[0] + match["center"][0],
                region[1] + match["center"][1],
            ],
        }
        _draw_region(
            annotated,
            (x, y, *match["size"]),
            f"{key}: {state} ({match['score']:.3f})",
            beside=True,
        )
    return observed


def _inspect_states(
    config: AppConfig, profile: PageProfileConfig, frame: np.ndarray
) -> dict[str, Any]:
    rules = [
        {**trigger.model_dump(), "path": str(config.resolve_path(trigger.path))}
        for trigger in profile.triggers
    ]
    observation = MultiTemplateDetector(rules).observe(
        crop_image(frame, profile.regions.detection)
    )
    triggers = observation.present
    verification = crop_image(frame, profile.regions.verification_region)
    success = _matches(
        config, profile.verification.success_templates, verification
    )
    failure = _matches(
        config, profile.verification.failure_templates, verification
    )
    regional = PageFlow(
        config, profile.page_flow, lambda region: crop_image(frame, region)
    )
    identity = (
        regional.matches(profile.page_guard) if profile.page_guard else None
    )
    blocking = [
        state
        for state in ("manual", "finished", "loading")
        if any(
            regional.matches(item) for item in getattr(profile.page_flow, state)
        )
    ]
    notices = [
        item.name
        for item in profile.page_flow.before_question
        if regional.locate(item.when, unique=item.click_match) is not None
    ]
    page_state = (
        "ambiguous"
        if (success and failure)
        or observation.ambiguous
        or len(blocking) > 1
        or len(notices) > 1
        else blocking[0]
        if blocking
        else "notice"
        if notices
        else "completed"
        if success
        else "failure"
        if failure
        else "question"
        if triggers
        else "unrecognized"
    )
    return {
        "page_state": page_state,
        "page_identity_matches": identity,
        "triggers": [item.name for item in triggers],
        "question_types": sorted(
            {item.question_type for item in triggers if item.action == "answer"}
        ),
        "notice_matches": notices,
        "success_visible": success,
        "failure_visible": failure,
    }


def _matches(
    config: AppConfig, templates: list[ImageTemplateConfig], frame: np.ndarray
) -> bool:
    matcher = ImageTemplateMatcher(
        [
            (str(index), str(config.resolve_path(item.path)), item.threshold)
            for index, item in enumerate(templates)
        ]
    )
    return matcher.best(frame) is not None


def _draw_region(
    frame: np.ndarray, region: Region, label: str, *, beside: bool = False
) -> None:
    x, y, width, height = region
    color = (0, 120, 255)
    cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), color, 2)
    position = (max(0, x), max(22, y - 8))
    if beside:
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        label_x = x - text_size[0] - 14
        if label_x < 0:
            label_x = x + width + 14
        position = (label_x, y + height // 2 + text_size[1] // 2)
    cv2.putText(
        frame,
        label,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        1,
        cv2.LINE_AA,
    )
