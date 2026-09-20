"""Interactive calibration for one private AutoYKT page profile."""

from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile
import string
from typing import Annotated, Any

from pydantic import Field, model_validator

import cv2
import mss
import numpy as np
import yaml

from autoykt.core.config import (
    AppConfig,
    ButtonColorsConfig,
    ConfigModel,
    PageRegionsConfig,
    Region,
    WindowTargetConfig,
)
from autoykt.core.legacy_config import migrate_legacy_config
from autoykt.monitor.image_utils import read_image, write_png
from autoykt.monitor.windows import WindowGuard, enable_dpi_awareness


# OpenCV exports its native API dynamically.
# pylint: disable=no-member

enable_dpi_awareness()


class ImageSelections(ConfigModel):
    """Explicit image-pixel crops for repeatable, headless calibration."""

    regions: PageRegionsConfig
    trigger: Region
    success: Region | None = None
    page_guard: Region | None = None
    options: dict[
        Annotated[str, Field(pattern=r"^[A-Z0-9_-]{1,16}$")], Region
    ] = Field(min_length=1)
    selected_sample: str | None = None
    unselected_sample: str | None = None

    @model_validator(mode="after")
    def _validate_samples(self) -> "ImageSelections":
        samples = (self.selected_sample, self.unselected_sample)
        if any(samples) and (
            not all(samples)
            or any(sample not in self.options for sample in samples)
        ):
            raise ValueError("both color samples must name calibrated options")
        return self


def _sample_button_colors(
    frame: np.ndarray, selections: ImageSelections
) -> ButtonColorsConfig | None:
    if not selections.selected_sample or not selections.unselected_sample:
        return None
    colors = []
    for key in (selections.selected_sample, selections.unselected_sample):
        x, y, width, height = selections.options[key]
        crop = frame[y : y + height, x : x + width]
        rgb = np.median(crop, axis=(0, 1))[::-1]
        colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return ButtonColorsConfig(selected_rgb=colors[0], unselected_rgb=colors[1])


class Calibrator:
    """Capture regions, click targets, and templates for one profile."""

    WINDOW_NAME = "AutoYKT calibration"

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        profile_id: str = "legacy",
        monitor_index: int | None = None,
        *,
        from_state: Path | None = None,
        from_image: Path | None = None,
    ) -> None:
        self._config_path = Path(config_path).expanduser().resolve()
        self._document = self._load_document()
        self._legacy = "version" not in self._document
        self._profile_id = profile_id
        configured_monitor = self._configured_monitor_index()
        self._monitor_index = (
            configured_monitor if monitor_index is None else monitor_index
        )
        profile = next(
            (
                item
                for item in self._document.get("pages", [])
                if item["id"] == profile_id
            ),
            {},
        )
        self._window_target = (
            WindowTargetConfig.model_validate(profile["target_window"])
            if profile.get("target_window")
            else None
        )
        if from_state is not None and from_image is not None:
            raise ValueError("choose from_state or from_image, not both")
        self._image_only = from_image is not None
        self._source_geometry: dict[str, int] | None = None
        self._state_frame = self._load_state(from_state) if from_state else None
        if from_image is not None:
            self._state_frame = read_image(from_image)
            height, width = self._state_frame.shape[:2]
            if self._window_target and self._window_target.client_size != (
                width,
                height,
            ):
                raise ValueError("image size does not match window client size")
            self._source_geometry = {
                "left": 0,
                "top": 0,
                "width": width,
                "height": height,
            }
        self._screen = (
            mss.mss() if from_state is None and from_image is None else None
        )
        self._guard_template_path: str | None = None
        if (
            self._screen is not None
            and self._window_target is None
            and not 0 <= self._monitor_index < len(self._screen.monitors)
        ):
            self._screen.close()
            raise ValueError(
                f"monitor index is unavailable: {self._monitor_index}"
            )
        self._template_directory = (
            self._config_path.parent / "templates" / self._profile_id
        )
        self._regions: dict[str, list[int]] = {}
        self._fallback_positions: dict[str, list[int]] = {}
        self._option_template_paths: dict[str, str] = {}
        self._trigger_template_path: str | None = None
        self._success_template_path: str | None = None
        self._points: list[tuple[int, int]] = []
        self._mode = "detection"
        self._current_option = "A"
        self._original_frame: np.ndarray | None = None
        self._display_frame: np.ndarray | None = None
        self._scale = 1.0

    def apply_selections(self, selections: ImageSelections) -> None:
        """Save reviewed image crops without opening a GUI or screen capture."""
        if self._legacy:
            raise ValueError("headless selections require a v2 configuration")
        if not self._image_only or self._state_frame is None:
            raise ValueError("headless selections require from_image")
        frame = self._state_frame
        self._original_frame = frame.copy()
        height, width = frame.shape[:2]
        regions = selections.regions.model_dump(mode="json", exclude_none=True)
        crops = {"question": selections.trigger, **selections.options}
        if selections.success:
            crops["success"] = selections.success
        if selections.page_guard:
            crops["page_guard"] = selections.page_guard
        for x, y, crop_width, crop_height in [
            *regions.values(),
            *crops.values(),
        ]:
            if (
                x < 0
                or y < 0
                or x + crop_width > width
                or y + crop_height > height
            ):
                raise ValueError("calibration region lies outside the image")
        for x, y, crop_width, crop_height in crops.values():
            crop = frame[y : y + crop_height, x : x + crop_width]
            if not np.any(np.std(crop, axis=(0, 1)) > 0):
                raise ValueError("template crop has no spatial detail")
        profile = next(
            item
            for item in self._document["pages"]
            if item["id"] == self._profile_id
        )
        style = profile["answer_style"]
        button_colors = _sample_button_colors(frame, selections)
        style["button_colors"] = (
            button_colors.model_dump(mode="json") if button_colors else None
        )
        # These selections are a complete inventory; never retain old fallback
        # coordinates or templates for options absent from the supplied image.
        style["option_templates"] = {}
        style["fallback_positions"] = {}
        style["match_grayscale"] = True
        self._regions = regions
        paths = {}
        for name, (x, y, crop_width, crop_height) in crops.items():
            directory = self._template_directory
            if name in selections.options:
                directory /= "options"
            output = directory / (name + ".png")
            self._write_crop((x, y), (x + crop_width, y + crop_height), output)
            paths[name] = self._relative_path(output)
        self._option_template_paths = {
            key: paths[key] for key in selections.options
        }
        self._trigger_template_path = paths["question"]
        self._success_template_path = paths.get("success")
        self._guard_template_path = paths.get("page_guard")
        if selections.page_guard:
            self._regions["page_guard"] = list(selections.page_guard)
        self._save()

    def _load_state(self, report_path: Path) -> np.ndarray:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_window = (
            self._window_target.model_dump(mode="json")
            if self._window_target
            else None
        )
        if (
            report.get("kind") != "state_capture"
            or report.get("status") != "captured"
            or report.get("profile_id") != self._profile_id
            or report.get("target_window") != expected_window
        ):
            raise ValueError(
                "state capture does not match this profile and window "
                "configuration"
            )
        state = next(
            (step for step in report["steps"] if step["step"] == "state"), None
        )
        if state is None:
            raise ValueError("state report has no capture geometry")
        geometry: dict[str, int] = state["surface_region"]
        self._source_geometry = geometry
        frame = read_image(report_path.parent / "full.png")
        if frame is None or frame.shape[:2] != (
            geometry["height"],
            geometry["width"],
        ):
            raise ValueError("state image does not match its capture geometry")
        return frame

    def _load_document(self) -> dict[str, Any]:
        with self._config_path.open("r", encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
        if not isinstance(document, dict):
            raise ValueError("configuration root must be a mapping")
        return document

    def _configured_monitor_index(self) -> int:
        if self._legacy:
            return int(
                (self._document.get("monitor") or {}).get("monitor_index", 1)
            )
        for profile in self._document.get("pages") or []:
            if profile.get("id") == self._profile_id:
                return int(profile.get("monitor_index", 1))
        raise ValueError(f"page profile not found: {self._profile_id}")

    def run(self) -> None:
        """Open the calibration window until the user saves or quits."""
        try:
            self._refresh_frame()
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)
            self._print_help()
            while True:
                if self._display_frame is None:
                    break
                cv2.imshow(self.WINDOW_NAME, self._display_frame)
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    self._save()
                elif key == ord("r"):
                    self._refresh_frame()
                elif key in tuple(map(ord, "0123456789vgbuc")):
                    self._select_mode(chr(key))
        finally:
            cv2.destroyAllWindows()
            if self._screen is not None:
                self._screen.close()

    @staticmethod
    def _print_help() -> None:
        print("1 detection ROI   2 question ROI   3 answers ROI")
        print("4 trigger template  5 option templates A-Z  6 fallback points")
        print("7 entry region    8 submit region")
        print("9 verification ROI  0 rearm ROI  v success template")
        print("g page identity template (include stable page-specific content)")
        print("b selected color ROI  u unselected color ROI  c full button ROI")
        print("r refresh   s save private config   q quit")

    def _select_mode(self, key: str) -> None:
        modes = {
            "1": "detection",
            "2": "question",
            "3": "answers",
            "4": "trigger_template",
            "5": "option_template",
            "6": "fallback",
            "7": "entry",
            "8": "submit",
            "9": "verification",
            "0": "rearm",
            "v": "success_template",
            "g": "page_guard",
            "b": "selected_rgb",
            "u": "unselected_rgb",
            "c": "button_size",
        }
        self._mode = modes[key]
        self._points.clear()
        if self._mode in {"option_template", "fallback"}:
            self._current_option = "A"
        print(f"Mode: {self._mode}")

    def _refresh_frame(self) -> None:
        if self._state_frame is not None:
            frame = self._state_frame.copy()
        else:
            if self._screen is None:
                raise RuntimeError("calibration capture is closed")
            if self._window_target:
                if self._original_frame is not None:
                    print(
                        "For another window state, close and reopen "
                        "calibration or use --from-state."
                    )
                    return
                guard = WindowGuard(self._window_target)
                x, y, width, height = guard.check().client
                region = {"left": x, "top": y, "width": width, "height": height}
                raw = self._screen.grab(region)
                if guard.check().client != (x, y, width, height):
                    raise RuntimeError(
                        "window moved during calibration capture"
                    )
            else:
                region = self._screen.monitors[self._monitor_index]
                raw = self._screen.grab(region)
            self._source_geometry = dict(region)
            frame = np.asarray(raw)[:, :, :3].copy()
        self._original_frame = frame
        height, width = frame.shape[:2]
        self._scale = min(1.0, 1920 / width, 1080 / height)
        display_size = (
            int(width * self._scale),
            int(height * self._scale),
        )
        self._display_frame = cv2.resize(  # pyright: ignore[reportCallIssue]
            frame,
            display_size,
        )

    def _mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter: object,
    ) -> None:
        del flags, parameter
        if event != cv2.EVENT_LBUTTONDOWN or self._display_frame is None:
            return
        original = (int(x / self._scale), int(y / self._scale))
        if self._mode == "fallback":
            self._fallback_positions[self._current_option] = list(original)
            self._draw_option(x, y, self._current_option)
            self._advance_option()
            return
        self._points.append(original)
        cv2.circle(self._display_frame, (x, y), 5, (0, 255, 255), -1)
        if len(self._points) < 2:
            return
        first = self._points[0]
        second = self._points[1]
        self._points.clear()
        region = _region_from_points(first, second)
        if region[2] <= 0 or region[3] <= 0:
            print("Ignored an empty selection; choose two distinct points.")
            return
        if self._mode in {
            "detection",
            "question",
            "answers",
            "verification",
            "rearm",
        }:
            self._regions[self._mode] = region
            print(f"{self._mode}: {region}")
        elif self._mode in {"selected_rgb", "unselected_rgb", "button_size"}:
            self._regions[self._mode] = region
            print(f"{self._mode}: {region}")
        elif self._mode in {"entry", "submit"}:
            self._regions[self._mode] = region
            print(f"{self._mode} target: {region}")
        elif self._mode == "trigger_template":
            output = self._template_directory / "question.png"
            self._write_crop(first, second, output)
            self._trigger_template_path = self._relative_path(output)
        elif self._mode == "success_template":
            output = self._template_directory / "success.png"
            self._write_crop(first, second, output)
            self._success_template_path = self._relative_path(output)
        elif self._mode == "page_guard":
            output = self._template_directory / "page_guard.png"
            self._write_crop(first, second, output)
            self._guard_template_path = self._relative_path(output)
            self._regions["page_guard"] = region
        elif self._mode == "option_template":
            output = (
                self._template_directory
                / "options"
                / f"{self._current_option}.png"
            )
            self._write_crop(first, second, output)
            self._option_template_paths[self._current_option] = (
                self._relative_path(output)
            )
            self._advance_option()
        self._draw_region(first, second)

    def _write_crop(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
        output: Path,
    ) -> None:
        if self._original_frame is None:
            return
        x1, y1 = min(first[0], second[0]), min(first[1], second[1])
        x2, y2 = max(first[0], second[0]), max(first[1], second[1])
        crop = self._original_frame[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError("selected template region is empty")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_png(output, crop)
        print(f"Template: {output}")

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self._config_path.parent).as_posix()

    def _draw_region(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> None:
        if self._display_frame is None:
            return
        display_first = (
            int(first[0] * self._scale),
            int(first[1] * self._scale),
        )
        display_second = (
            int(second[0] * self._scale),
            int(second[1] * self._scale),
        )
        cv2.rectangle(
            self._display_frame,
            display_first,
            display_second,
            (0, 255, 0),
            2,
        )

    def _draw_option(self, x: int, y: int, option: str) -> None:
        if self._display_frame is None:
            return
        cv2.circle(self._display_frame, (x, y), 8, (255, 0, 0), -1)
        cv2.putText(
            self._display_frame,
            option,
            (x + 10, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    def _advance_option(self) -> None:
        options = string.ascii_uppercase
        self._current_option = options[
            (options.index(self._current_option) + 1) % len(options)
        ]
        print(f"Next option: {self._current_option}")

    def _save_button_colors(self, style: dict[str, Any]) -> None:
        """Update only measured colors, preserving a previous state's sample."""
        fields = {"selected_rgb", "unselected_rgb", "button_size"}
        if not fields.intersection(self._regions):
            return
        if self._original_frame is None:
            raise ValueError("button calibration requires a captured frame")
        colors = dict(style.get("button_colors") or {})
        for name in fields.intersection(self._regions):
            x, y, width, height = self._regions[name]
            if name == "button_size":
                colors[name] = [width, height]
            else:
                crop = self._original_frame[y : y + height, x : x + width]
                colors[name] = [
                    int(channel)
                    for channel in np.median(crop, axis=(0, 1))[::-1]
                ]
        if not {"selected_rgb", "unselected_rgb"} <= colors.keys():
            raise ValueError(
                "calibrate both b (selected) and u (unselected) colors "
                "before the first save"
            )
        style["button_colors"] = ButtonColorsConfig.model_validate(
            colors
        ).model_dump(mode="json")
        style["match_grayscale"] = True

    def _save(self) -> None:
        if self._legacy:
            self._save_legacy()
        else:
            self._save_v2()
        validation_document = (
            migrate_legacy_config(self._document)
            if self._legacy
            else self._document
        )
        AppConfig.model_validate(validation_document)
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._config_path.parent,
                prefix=f".{self._config_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as config_file:
                temporary_path = Path(config_file.name)
                yaml.safe_dump(
                    self._document,
                    config_file,
                    allow_unicode=True,
                    sort_keys=False,
                )
            os.replace(temporary_path, self._config_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        print(f"Saved: {self._config_path}")

    def _save_v2(self) -> None:  # pylint: disable=too-many-branches
        profile = next(
            (
                item
                for item in self._document.get("pages") or []
                if item.get("id") == self._profile_id
            ),
            None,
        )
        if profile is None:
            raise ValueError(f"page profile not found: {self._profile_id}")
        coordinate_space = "window" if self._window_target else "monitor"
        if self._guard_template_path:
            profile["page_guard"] = {
                "path": self._guard_template_path,
                "region": self._regions["page_guard"],
                "threshold": 0.90,
            }
        profile["monitor_index"] = self._monitor_index
        # A standalone image cannot establish live window geometry.
        profile["enabled"] = not self._image_only
        regions = profile.setdefault("regions", {})
        old_question = regions.get("question")
        for name in (
            "detection",
            "question",
            "answers",
            "verification",
            "rearm",
        ):
            if name in self._regions:
                regions[name] = self._regions[name]
        if "question" in self._regions:
            for name in ("verification", "rearm"):
                current_region = regions.get(name)
                follows_question = (
                    current_region is None or current_region == old_question
                )
                if name not in self._regions and follows_question:
                    regions[name] = self._regions["question"]
        if self._trigger_template_path:
            profile["triggers"][0]["path"] = self._trigger_template_path
        answer_style = profile.setdefault("answer_style", {})
        self._save_button_colors(answer_style)
        self._save_options(answer_style, coordinate_space)
        if "entry" in self._regions:
            profile["entry_target"] = {
                "region": self._regions["entry"],
                "coordinate_space": coordinate_space,
            }
        if "submit" in self._regions:
            profile["submit_target"] = {
                "region": self._regions["submit"],
                "coordinate_space": coordinate_space,
            }
        if self._success_template_path:
            verification = profile.setdefault("verification", {})
            templates = verification.setdefault("success_templates", [])
            if templates:
                templates[0]["path"] = self._success_template_path
            else:
                templates.append(
                    {"path": self._success_template_path, "threshold": 0.90}
                )
        active = self._document.setdefault("runtime", {}).setdefault(
            "active_profiles", []
        )
        if active and self._profile_id not in active:
            active.append(self._profile_id)

    def _save_options(
        self, answer_style: dict[str, Any], coordinate_space: str
    ) -> None:
        """Merge calibrated option templates and legacy fallback coordinates."""
        existing_templates = dict(answer_style.get("option_templates") or {})
        if self._option_template_paths:
            existing_templates.update(self._option_template_paths)
        elif self._fallback_positions:
            existing_templates = {
                option: path
                for option, path in existing_templates.items()
                if option not in self._fallback_positions
                or (self._config_path.parent / path).is_file()
            }
        answer_style["option_templates"] = existing_templates
        if self._fallback_positions:
            old_positions = answer_style.get("fallback_positions") or {}
            if answer_style.get("fallback_coordinate_space") == "screen":
                if self._source_geometry is None:
                    raise ValueError(
                        "capture geometry is required to convert screen points"
                    )
                monitor = self._source_geometry
                old_positions = {
                    key: [point[0] - monitor["left"], point[1] - monitor["top"]]
                    for key, point in old_positions.items()
                }
            answer_style["fallback_positions"] = {
                **old_positions,
                **self._fallback_positions,
            }
            answer_style["fallback_coordinate_space"] = coordinate_space

    def _save_legacy(self) -> None:
        monitor = self._document.setdefault("monitor", {})
        detector = self._document.setdefault("detector", {})
        if "detection" in self._regions:
            monitor["feature_roi"] = self._regions["detection"]
            monitor["roi"] = self._regions["detection"]
        if "question" in self._regions:
            monitor["task_roi"] = self._regions["question"]
            monitor["question_roi"] = self._regions["question"]
        if "verification" in self._regions:
            monitor["verification_roi"] = self._regions["verification"]
        if "rearm" in self._regions:
            monitor["rearm_roi"] = self._regions["rearm"]
        if "entry" in self._regions:
            monitor["entry_roi"] = self._regions["entry"]
        if "submit" in self._regions:
            monitor["finish_task_roi"] = self._regions["submit"]
        if self._trigger_template_path:
            detector["question_feature_template_path"] = (
                self._trigger_template_path
            )
            monitor["template_path"] = self._trigger_template_path
        if self._success_template_path:
            detector["success_template_path"] = self._success_template_path
        if self._option_template_paths:
            detector["option_templates"] = self._option_template_paths
        if self._fallback_positions:
            self._document.setdefault("clicker", {})[
                "options_positions"
            ] = self._fallback_positions


def _region_from_points(
    first: tuple[int, int],
    second: tuple[int, int],
) -> list[int]:
    x = min(first[0], second[0])
    y = min(first[1], second[1])
    return [x, y, abs(second[0] - first[0]), abs(second[1] - first[1])]


if __name__ == "__main__":
    Calibrator().run()
