"""Explicit, non-interacting page capture and click-preview commands."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from autoykt.core.config import AppConfig, ConfigError
from autoykt.monitor.operations import ActionPlan, ProfileSession
from autoykt.monitor.screen_capture import ScreenCapture
from autoykt.monitor.windows import Win32Desktop


def show_windows() -> None:
    """List private selectors and client geometry without changing focus."""
    for window in Win32Desktop().windows():
        if window.title:
            print(json.dumps(asdict(window), ensure_ascii=False))


def capture_state(config: AppConfig, profile_id: str, label: str) -> Path:
    """Save a labeled full surface and configured crops without any input."""
    profile = config.page_profile(profile_id)
    session = ProfileSession(config, profile, load_options=False)
    session.start("state_capture")
    try:
        with ScreenCapture(
            screenshot_dir=None,
            monitor_index=profile.monitor_index,
            window_guard=session.window,
        ) as capture:
            region = capture.surface_region
            frame = capture.grab_frame()
            if region != capture.surface_region:
                raise RuntimeError("capture surface moved during state capture")
        session.save_frame("full", frame)
        session.record("state", label=label, surface_region=region)
        for name, roi in profile.regions.model_dump().items():
            if roi is None:
                continue
            x, y, width, height = roi
            if (
                x < 0
                or y < 0
                or x + width > frame.shape[1]
                or y + height > frame.shape[0]
            ):
                raise RuntimeError(
                    f"{name} region is outside the capture surface"
                )
            session.save_frame(name, frame[y : y + height, x : x + width])
        session.finish("captured", label=label)
    except BaseException:
        session.finish("capture_failed")
        raise
    assert session.report_path is not None
    return session.report_path


def preview_option(config: AppConfig, profile_id: str, option: str) -> Path:
    """Preview a user-specified option without calling AI or clicking."""
    profile = config.page_profile(profile_id)
    session = ProfileSession(config, profile)

    def factory(region):
        return ScreenCapture(
            roi=region,
            screenshot_dir=None,
            monitor_index=profile.monitor_index,
            window_guard=session.window,
        )

    session.attach_capture_factory(factory)
    session.start("manual_preview")
    try:
        session.check_page()
        with factory(profile.regions.answer_region) as capture:
            frame = capture.grab_frame()
            point, annotated = session.locate_answer(
                option.strip().upper(), frame, capture
            )
            session.save_frame("located_option", annotated)
            session.plan(option.strip().upper(), point, capture)
        session.finish("preview_only")
    except BaseException:
        session.finish("preview_failed")
        raise
    assert session.report_path is not None
    return session.report_path


def recover_current(config: AppConfig, profile_id: str) -> Path:
    """Record a human-reviewed current page to skip, without any mouse input."""
    profile = config.page_profile(profile_id)
    session = ProfileSession(config, profile, load_options=False)
    previous = session.checkpoint.read()

    def factory(region):
        return ScreenCapture(
            roi=region,
            screenshot_dir=None,
            monitor_index=profile.monitor_index,
            window_guard=session.window,
        )

    session.attach_capture_factory(factory)
    session.start("manual_recovery")
    session.check_page()
    frame = session.capture(profile.regions.rearm_region)
    session.save_frame("rearm_before", frame)
    session.record(
        "manual_review", previous_checkpoint=previous, decision="skip_current"
    )
    session.finish("reviewed")
    assert session.report_path is not None
    session.checkpoint.review(session.report_path, previous)
    return session.report_path


def preview_step(config: AppConfig, profile_id: str, name: str) -> Path:
    """Preview a result button without AI, input, or restart changes."""
    profile = config.page_profile(profile_id)
    step = next(
        (
            item
            for item in [
                *profile.page_flow.before_question,
                *profile.page_flow.after_submit,
            ]
            if item.name == name
        ),
        None,
    )
    if step is None:
        raise ConfigError("result action is not configured for this profile")
    session = ProfileSession(config, profile, load_options=False)

    def factory(region):
        return ScreenCapture(
            roi=region,
            screenshot_dir=None,
            monitor_index=profile.monitor_index,
            window_guard=session.window,
        )

    session.attach_capture_factory(factory)
    session.start("navigation_preview")
    try:
        session.check_page()
        if not session.flow.matches(step.when):
            raise RuntimeError("configured result control is not visible")
        capture = factory(profile.regions.rearm_region)
        try:
            action = session.step_target("advance_" + name, step)
            session.preview(
                ActionPlan("", (action,)), capture, label="plan_preview"
            )
        finally:
            capture.close()
        session.finish("preview_only")
    except BaseException:
        session.finish("preview_failed")
        raise
    assert session.report_path is not None
    return session.report_path
