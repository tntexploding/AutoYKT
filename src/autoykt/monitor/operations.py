"""Shared click plans, page guards, and private evidence for live operation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Literal, Protocol
from uuid import uuid4

import cv2
import numpy as np

from autoykt.core.config import (
    AppConfig,
    ClickTargetConfig,
    PageAdvanceConfig,
    ConfigError,
    PageProfileConfig,
    PageGuardConfig,
    Region,
)
from autoykt.monitor.clicker import MouseController
from autoykt.monitor.detector import (
    ImageTemplateMatcher,
    OptionMatch,
    OptionTemplateDetector,
)
from autoykt.monitor.image_utils import (
    frame_change_ratio,
    read_image,
    write_png,
)
from autoykt.monitor.checkpoint import OperationCheckpoint
from autoykt.monitor.page_flow import PageFlow
from autoykt.monitor.windows import WindowGuard, WindowUnavailable
from autoykt.monitor.selection import (
    assert_same_layout,
    selected_options,
    selection_change_ratio,
    selection_updates,
)

# OpenCV exports its native API dynamically.
# pylint: disable=no-member


class CaptureDevice(Protocol):
    """Screen-capture surface consumed by a page automation."""

    def grab_frame(self) -> np.ndarray:
        """Capture one frame."""
        raise NotImplementedError

    def save_screenshot(self, frame: np.ndarray, prefix: str) -> Path:
        """Persist one frame and return its path."""
        raise NotImplementedError

    def absolute_from_frame(self, point: tuple[int, int]) -> tuple[int, int]:
        """Convert a frame-relative point to a desktop point."""
        raise NotImplementedError

    def absolute_from_monitor(self, point: tuple[int, int]) -> tuple[int, int]:
        """Convert a monitor-relative point to a desktop point."""
        raise NotImplementedError

    def close(self) -> None:
        """Release capture resources."""
        raise NotImplementedError


CaptureFactory = Callable[[Region], CaptureDevice]


@dataclass(frozen=True)
class PlannedClick:
    """A resolved click, kept window-relative until the final input check."""

    label: str
    point: tuple[int, int]
    coordinate_space: Literal["window", "screen"]


@dataclass(frozen=True)
class ActionPlan:
    """The exact option and optional submit targets used by execution."""

    option: str
    clicks: tuple[PlannedClick, ...]


# The shared planning and evidence API keeps input checks in one place.
# pylint: disable-next=too-many-public-methods
class ProfileSession:
    """Keep one profile's preview, window guard, and operation evidence."""

    def __init__(
        self,
        config: AppConfig,
        profile: PageProfileConfig,
        *,
        load_options: bool = True,
    ) -> None:
        self._config = config
        self.profile = profile
        self.window = (
            WindowGuard(profile.target_window)
            if profile.target_window
            else None
        )
        self._factory: CaptureFactory | None = None
        self._options = OptionTemplateDetector(
            {
                key: str(config.resolve_path(path))
                for key, path in profile.answer_style.option_templates.items()
                if load_options
            },
            threshold=profile.answer_style.option_match_threshold,
            match_grayscale=profile.answer_style.match_grayscale,
            button_colors=profile.answer_style.button_colors,
        )
        self.report_path: Path | None = None
        self._report: dict[str, Any] = {}
        self.completed_cycles = 0
        self.pause_reason = ""
        self._cancelled = threading.Event()
        self._report_lock = threading.RLock()
        self.flow = PageFlow(
            config, profile.page_flow, self.capture, enabled=load_options
        )
        self.checkpoint = OperationCheckpoint(
            config.resolve_path(config.storage.data_dir) / "runs" / profile.id
        )
        self._live = False
        self._deadline = float("inf")
        self.answer_input_attempted = False

    @property
    def outcome(self) -> str:
        """Return the current report outcome."""
        return str(self._report.get("status", "waiting"))

    def attach_capture_factory(self, factory: CaptureFactory) -> None:
        """Use the same capture coordinate system as the answer workflow."""
        self._factory = factory

    def check_live_configuration(self) -> None:
        """Require explicit page and success evidence for bound live runs."""
        self._live = True
        profile = self.profile
        if any(
            trigger.question_type == "multiple" for trigger in profile.triggers
        ):
            if (
                profile.answer_style.button_colors is None
                or not profile.answer_style.option_templates
                or profile.answer_style.fallback_positions
            ):
                raise ConfigError(
                    "multiple choice requires state-aware button templates"
                )
            if profile.submit_target is None:
                raise ConfigError(
                    "multiple choice requires an explicit submit target"
                )
        if profile.target_window and profile.page_guard is None:
            raise ConfigError(
                "live window operation requires a page_guard template"
            )
        required = (
            profile.target_window is not None
            or profile.verification.require_success_template
            or bool(
                profile.page_flow.after_submit
                or profile.page_flow.before_question
            )
            or any(
                trigger.question_type == "multiple"
                for trigger in profile.triggers
            )
        )
        if required and not profile.verification.success_templates:
            raise ConfigError(
                "live operation requires explicit success templates"
            )
        templates = [
            *profile.verification.success_templates,
            *profile.verification.failure_templates,
            *profile.page_flow.templates,
        ]
        if profile.page_guard:
            templates.append(profile.page_guard)
        for template in templates:
            path = self._config.resolve_path(template.path)
            if not path.is_file():
                raise ConfigError(f"operation template is missing: {path}")
            ImageTemplateMatcher([("check", str(path), template.threshold)])

    def check_page(self) -> None:
        """Check the visible window and the configured page identity."""
        if self.window:
            self.window.check(for_capture=True)
        self.flow.check_blocking()
        guard = self.profile.page_guard
        if guard is None:
            return
        if not self.flow.matches(guard):
            raise WindowUnavailable(
                "page identity does not match; restore the calibrated page"
            )

    def capture(self, region: Region) -> np.ndarray:
        """Capture and release a temporary evidence or identity region."""
        if self._factory is None:
            raise RuntimeError("capture factory is not attached")
        device = self._factory(region)
        try:
            return device.grab_frame()
        finally:
            device.close()

    def start(self, kind: str) -> None:
        """Create a private report before any question interaction."""
        self.report_path = None
        self._report = {}
        self._cancelled = threading.Event()
        self.answer_input_attempted = False
        self.flow.reset()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory = (
            self._config.resolve_path(self._config.storage.data_dir)
            / "runs"
            / self.profile.id
            / (stamp + "_" + uuid4().hex[:8])
        )
        directory.mkdir(parents=True)
        self.report_path = directory / "report.json"
        self._report = {
            "profile_id": self.profile.id,
            "kind": kind,
            "status": "running",
            "regions": self.profile.regions.model_dump(mode="json"),
            "target_window": self.profile.target_window.model_dump(mode="json")
            if self.profile.target_window
            else None,
            "profile_config": self.profile.model_dump(mode="json"),
            "steps": [],
        }
        self.record("started")

    def restore(self) -> np.ndarray | None:
        """Resume verified or manually skipped pages, never unresolved input."""
        if not self._live:
            return None
        pending = self.checkpoint.read()
        if pending is None:
            return None
        path = self.checkpoint.directory / pending["report"]
        if pending["status"] == "input_pending":
            raise ConfigError(
                f"previous input needs manual review: {path}; "
                "stop other runs, then "
                f"use recover --profile {self.profile.id} --skip-current "
                "after reviewing the page"
            )
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("profile_config") != self.profile.model_dump(
                mode="json"
            ):
                raise ValueError("profile configuration changed")
            baseline = read_image(path.parent / "rearm_before.png")
            expected = self.profile.regions.rearm_region
            if baseline.shape[:2] != (expected[3], expected[2]):
                raise ValueError("rearm evidence dimensions changed")
            self.report_path, self._report = path, report
            used = {
                step["label"][len("advance_") :]
                for step in report["steps"]
                if step["step"] == "click_result"
                and step.get("applied")
                and step.get("label", "").startswith("advance_")
            }
            self.flow.reset(used)
            self.flow.verified = pending["status"] == "verified"
            self.record("resumed", checkpoint_status=pending["status"])
            return baseline
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ConfigError(
                f"previous operation cannot be resumed: {path}; "
                "review it and use recover --skip-current"
            ) from error

    def set_deadline(self, deadline: float) -> None:
        """Reject input after the active question's total time budget."""
        self._deadline = deadline

    def mark_verified(self) -> None:
        """Persist verified submission or completed result navigation."""
        if self._live and self.report_path:
            self.checkpoint.write(self.report_path, "verified")
        self.flow.verified = True
        self._deadline = float("inf")

    def release_question(self) -> None:
        """Release restart protection once the page is ready."""
        if self._live:
            self.checkpoint.clear()
        self.flow.reset()

    def record(self, step: str, **details: Any) -> None:
        """Persist each completed step, including interrupted run evidence."""
        with self._report_lock:
            if self.report_path is None:
                return
            self._report["steps"].append(
                {
                    "step": step,
                    "time": datetime.now(timezone.utc).isoformat(),
                    **details,
                }
            )
            temporary = self.report_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self._report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.report_path)

    def finish(self, status: str, **details: Any) -> None:
        """Write the terminal outcome without implying submission success."""
        with self._report_lock:
            self._report["status"] = status
            self.record("finished", **details)

    def save_frame(self, label: str, frame: np.ndarray) -> Path:
        """Save an image beside the current private operation report."""
        if self.report_path is None:
            raise RuntimeError("operation report has not been started")
        path = self.report_path.parent / (label + ".png")
        write_png(path, frame)
        self.record(label, screenshot_path=str(path))
        return path

    def inspect_options(self, frame: np.ndarray) -> dict[str, OptionMatch]:
        """Observe option locations and colors without planning input."""
        return self._options.detect(frame)

    def locate_answer(
        self, option: str, frame: np.ndarray, capture: CaptureDevice
    ) -> tuple[tuple[int, int], np.ndarray]:
        """Resolve the same template or fallback in preview and live runs."""
        if option not in self.profile.answer_style.option_keys:
            raise RuntimeError(f"option {option} is not configured")
        matches = self._options.detect(frame)
        annotated = _annotate_option_matches(frame, matches, option)
        located = matches.get(option)
        if located is not None:
            if located.get("ambiguous"):
                raise RuntimeError(
                    f"option {option} matches multiple locations"
                )
            center = located.get("center")
            if isinstance(center, tuple):
                if self.window:
                    x, y, _, _ = self.profile.regions.answer_region
                    return (x + center[0], y + center[1]), annotated
                return capture.absolute_from_frame(center), annotated
        style = self.profile.answer_style
        fallback = style.fallback_positions.get(option)
        if fallback is None:
            raise RuntimeError(
                f"option {option} was not located and has no fallback"
            )
        if style.fallback_coordinate_space in {"screen", "window"}:
            return fallback, annotated
        return capture.absolute_from_monitor(fallback), annotated

    def target(
        self, label: str, target: ClickTargetConfig, capture: CaptureDevice
    ) -> PlannedClick:
        """Resolve a configured entry or submit target into the shared plan."""
        point = target.point
        if point is None and target.region:
            x, y, width, height = target.region
            point = (x + width // 2, y + height // 2)
        if point is None:
            raise RuntimeError("click target has no usable coordinates")
        if target.coordinate_space == "monitor":
            point = capture.absolute_from_monitor(point)
        return PlannedClick(label, point, "window" if self.window else "screen")

    def step_target(self, label: str, step: PageAdvanceConfig) -> PlannedClick:
        """Resolve fixed controls or a unique visible template body."""
        if self._factory is None:
            raise RuntimeError("capture factory is not attached")
        capture = self._factory(step.when.region)
        try:
            if step.target is not None:
                return self.target(label, step.target, capture)
            match = self.flow.locate(step.when, unique=True)
            if match is None:
                raise WindowUnavailable("page control has no unique location")
            x, y = step.when.region[:2]
            point = (x + match.center[0], y + match.center[1])
            if not self.window:
                point = capture.absolute_from_monitor(point)
            return PlannedClick(
                label, point, "window" if self.window else "screen"
            )
        finally:
            capture.close()

    def plan(
        self,
        option: str,
        point: tuple[int, int] | None,
        capture: CaptureDevice,
        *,
        frame: np.ndarray | None = None,
        multiple: bool = False,
    ) -> ActionPlan:
        """Plan observed selection changes or a legacy single-option click."""
        if frame is not None:
            clicks = self._selection_clicks(
                option, frame, capture, multiple=multiple
            )
        elif point is not None and not multiple:
            clicks = [
                PlannedClick(
                    "option_" + option,
                    point,
                    "window" if self.window else "screen",
                )
            ]
        else:
            raise RuntimeError(
                "selection planning requires an observed answer frame"
            )
        if self.profile.submit_target:
            clicks.append(
                self.target("submit", self.profile.submit_target, capture)
            )
        plan = ActionPlan(option, tuple(clicks))
        self.preview(plan, capture)
        return plan

    def _selection_clicks(
        self,
        answer: str,
        frame: np.ndarray,
        capture: CaptureDevice,
        *,
        multiple: bool,
    ) -> list[PlannedClick]:
        matches = self.inspect_options(frame)
        updates = selection_updates(answer, matches, multiple=multiple)
        clicks = []
        for key in updates:
            center = matches[key]["center"]
            if self.window:
                x, y, _, _ = self.profile.regions.answer_region
                point = (x + center[0], y + center[1])
            else:
                point = capture.absolute_from_frame(center)
            clicks.append(
                PlannedClick(
                    "option_" + key,
                    point,
                    "window" if self.window else "screen",
                )
            )
        self.save_frame(
            "located_option", _annotate_option_matches(frame, matches, answer)
        )
        return clicks

    def preview(
        self,
        plan: ActionPlan,
        capture: CaptureDevice,
        *,
        label: str | None = None,
    ) -> Path:
        """Save a screenshot with the exact planned click locations marked."""
        origin = (
            (0, 0) if self.window else capture.absolute_from_monitor((0, 0))
        )
        points = [
            (click.point[0] - origin[0], click.point[1] - origin[1])
            for click in plan.clicks
        ]
        regions = [
            value
            for value in self.profile.regions.model_dump().values()
            if value
        ]
        x = min(
            [region[0] for region in regions] + [point[0] for point in points]
        )
        y = min(
            [region[1] for region in regions] + [point[1] for point in points]
        )
        right = max(
            [region[0] + region[2] for region in regions]
            + [point[0] + 1 for point in points]
        )
        bottom = max(
            [region[1] + region[3] for region in regions]
            + [point[1] + 1 for point in points]
        )
        region = (x, y, right - x, bottom - y)
        frame = self.capture(region)
        for index, (click, point) in enumerate(
            zip(plan.clicks, points, strict=True), 1
        ):
            local = (point[0] - x, point[1] - y)
            cv2.drawMarker(frame, local, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(
                frame,
                f"{index}: {click.label}",
                (max(0, local[0] - 70), max(16, local[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
        label = label or (
            "entry_preview" if not plan.option else "plan_preview"
        )
        path = self.save_frame(label, frame)
        self.record(
            "plan",
            plan=asdict(plan),
            preview_region=region,
            preview_origin=origin,
            submit_condition="only if success is not already visible",
        )
        return path

    def cancel(self) -> None:
        """Revoke input that has not yet reached the mouse controller."""
        self._cancelled.set()

    def click(
        self,
        action: PlannedClick,
        clicker: MouseController,
        *,
        question_frame: np.ndarray | None = None,
        answer_frame: np.ndarray | None = None,
        required_state: PageGuardConfig | None = None,
        required_step: PageAdvanceConfig | None = None,
        selection_reference: dict[str, OptionMatch] | None = None,
    ) -> bool:
        """Recheck page and window immediately before delivering input."""
        cancelled, deadline = self._cancelled, self._deadline
        if cancelled.is_set() or time.monotonic() >= deadline:
            return False
        if self._live and self.window:
            if self.window.prepare_input(clicker.click_point):
                self.record("window_activated")
                time.sleep(0.1)
        self.record("click_attempt", action=asdict(action))
        self.check_page()
        if (required_state and not self.flow.matches(required_state)) or (
            required_step
            and self.step_target(action.label, required_step) != action
        ):
            raise WindowUnavailable(
                "page control changed immediately before input"
            )
        if selection_reference is not None:
            current = self.inspect_options(
                self.capture(self.profile.regions.answer_region)
            )
            assert_same_layout(selection_reference, current)
            if selected_options(current) != selected_options(
                selection_reference
            ):
                raise WindowUnavailable(
                    "selection changed immediately before input"
                )
        for baseline, region in (
            (question_frame, self.profile.regions.question),
            (answer_frame, self.profile.regions.answer_region),
        ):
            if baseline is None:
                continue
            current_frame = self.capture(region)
            difference = (
                selection_change_ratio(
                    baseline,
                    current_frame,
                    region,
                    self.profile.regions.answer_region,
                    selection_reference,
                )
                if selection_reference is not None
                else frame_change_ratio(baseline, current_frame)
            )
            if difference > self.profile.question_ready.maximum_change_ratio:
                raise WindowUnavailable("page changed immediately before input")
        point = (
            self.window.resolve_point(action.point)
            if self.window
            else action.point
        )
        if cancelled.is_set() or time.monotonic() >= deadline:
            return False
        if self._live and self.report_path:
            self.checkpoint.write(
                self.report_path, "input_pending", action.label
            )
        if action.label.startswith("option_") or action.label == "submit":
            self.answer_input_attempted = True
        applied = clicker.click_point(*point)
        self.record(
            "click_result",
            label=action.label,
            applied=applied,
            absolute_point=point,
        )
        return applied


def _annotate_option_matches(
    frame: np.ndarray,
    matches: dict[str, OptionMatch],
    selected: str,
) -> np.ndarray:
    annotated = frame.copy()
    for key, information in matches.items():
        top_left = information.get("top_left")
        size = information.get("size")
        score = information.get("score")
        if not isinstance(top_left, tuple) or not isinstance(size, tuple):
            continue
        color = (0, 255, 0) if key in selected.split(",") else (255, 255, 0)
        x, y = int(top_left[0]), int(top_left[1])
        width, height = int(size[0]), int(size[1])
        score_value = float(score) if isinstance(score, (int, float)) else 0.0
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            color,
            2,
        )
        cv2.putText(
            annotated,
            f"{key}:{score_value:.2f}",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return annotated


def open_capture_devices(
    factory: CaptureFactory,
    regions: tuple[Region, Region, Region, Region, Region],
) -> tuple[
    CaptureDevice, CaptureDevice, CaptureDevice, CaptureDevice, CaptureDevice
]:
    """Open one device per region, cleaning up partial construction."""
    devices: list[CaptureDevice] = []
    try:
        devices.extend(factory(region) for region in regions)
    except BaseException:  # pylint: disable=broad-exception-caught
        close_capture_devices(devices)
        raise
    return (devices[0], devices[1], devices[2], devices[3], devices[4])


def close_capture_devices(
    devices: tuple[CaptureDevice, ...] | list[CaptureDevice],
) -> None:
    """Release every device even if a native handle fails to close."""
    for device in devices:
        try:
            device.close()
        except Exception as error:  # pylint: disable=broad-exception-caught
            logging.getLogger("autoykt").warning(
                "Closing a capture device failed: %s", error
            )
