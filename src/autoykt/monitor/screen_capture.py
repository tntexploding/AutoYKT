"""DPI-aware screen capture with explicit coordinate conversion."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import mss
import numpy as np

from autoykt.monitor.image_utils import write_png
from autoykt.monitor.windows import (
    WindowGuard,
    WindowUnavailable,
    enable_dpi_awareness,
)


enable_dpi_awareness()


class CaptureError(RuntimeError):
    """Raised when a screen capture cannot be configured or saved."""


class ScreenCapture:
    """Capture one monitor or a monitor-relative region."""

    def __init__(
        self,
        roi: tuple[int, int, int, int] | list[int] | None = None,
        screenshot_dir: str | Path | None = "storage/screenshots",
        monitor_index: int = 1,
        window_guard: WindowGuard | None = None,
    ) -> None:
        self._window_guard = window_guard
        self._sct = mss.mss()
        try:
            if window_guard is None and (
                monitor_index < 0 or monitor_index >= len(self._sct.monitors)
            ):
                raise CaptureError(
                    f"monitor index {monitor_index} is unavailable; valid "
                    f"range is 0..{len(self._sct.monitors) - 1}"
                )
            self._monitor_index = monitor_index
            self._roi = self._normalize_region(roi)
            self._screenshot_dir = (
                Path(screenshot_dir) if screenshot_dir is not None else None
            )
            if self._screenshot_dir is not None:
                self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._sct.close()
            raise
        self._closed = False

    @staticmethod
    def _normalize_region(
        region: tuple[int, int, int, int] | list[int] | None,
    ) -> tuple[int, int, int, int] | None:
        if region is None:
            return None
        if len(region) != 4:
            raise CaptureError(
                "a screen region must contain x, y, width, height"
            )
        normalized = (
            int(region[0]),
            int(region[1]),
            int(region[2]),
            int(region[3]),
        )
        if normalized[2] <= 0 or normalized[3] <= 0:
            raise CaptureError(
                "screen region width and height must be positive"
            )
        return normalized

    @property
    def monitor_region(self) -> dict[str, int]:
        """Return the absolute MSS rectangle for this capture."""
        base = self.surface_region
        if self._roi is None:
            return {
                "left": int(base["left"]),
                "top": int(base["top"]),
                "width": int(base["width"]),
                "height": int(base["height"]),
            }
        x, y, width, height = self._roi
        if self._window_guard and (
            x < 0
            or y < 0
            or x + width > base["width"]
            or y + height > base["height"]
        ):
            raise WindowUnavailable(
                "capture region lies outside the window client area"
            )
        return {
            "left": int(base["left"]) + x,
            "top": int(base["top"]) + y,
            "width": width,
            "height": height,
        }

    @property
    def surface_region(self) -> dict[str, int]:
        """Return the current monitor or guarded window client rectangle."""
        if self._window_guard is not None:
            x, y, width, height = self._window_guard.check(
                for_capture=True
            ).client
            return {"left": x, "top": y, "width": width, "height": height}
        return dict(self._sct.monitors[self._monitor_index])

    @property
    def monitor_origin(self) -> tuple[int, int]:
        """Return the selected monitor's absolute desktop origin."""
        monitor = self.surface_region
        return int(monitor["left"]), int(monitor["top"])

    def absolute_from_frame(self, point: tuple[int, int]) -> tuple[int, int]:
        """Convert a capture-frame point to absolute desktop coordinates."""
        region = self.monitor_region
        return region["left"] + point[0], region["top"] + point[1]

    def absolute_from_monitor(self, point: tuple[int, int]) -> tuple[int, int]:
        """Convert a monitor-relative point to absolute desktop coordinates."""
        origin_x, origin_y = self.monitor_origin
        return origin_x + point[0], origin_y + point[1]

    def grab_frame(self) -> np.ndarray:
        """Return one BGR OpenCV frame."""
        self._ensure_open()
        region = self.monitor_region
        raw = self._sct.grab(region)
        if self._window_guard and region != self.monitor_region:
            raise WindowUnavailable(
                "window moved during capture; retry when stationary"
            )
        return np.asarray(raw)[:, :, :3].copy()

    def grab_full_screen(self) -> np.ndarray:
        """Capture the selected monitor regardless of the configured ROI."""
        self._ensure_open()
        region = self.surface_region
        raw = self._sct.grab(region)
        if self._window_guard and region != self.surface_region:
            raise WindowUnavailable(
                "window moved during capture; retry when stationary"
            )
        return np.asarray(raw)[:, :, :3].copy()

    def save_screenshot(
        self,
        frame: np.ndarray,
        prefix: str = "screenshot",
    ) -> Path:
        """Save a frame as PNG and return its absolute path."""
        if self._screenshot_dir is None:
            raise CaptureError("this capture has no screenshot directory")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        safe_prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix)
        safe_prefix = safe_prefix.strip("._")[:80] or "screenshot"
        path = (
            self._screenshot_dir / f"{safe_prefix}_{timestamp}.png"
        ).resolve()
        try:
            write_png(path, frame)
        except OSError as error:
            raise CaptureError(f"failed to save screenshot: {path}") from error
        return path

    def update_roi(self, roi: tuple[int, int, int, int] | list[int]) -> None:
        """Replace the capture region."""
        self._roi = self._normalize_region(roi)

    def _ensure_open(self) -> None:
        if self._closed:
            raise CaptureError("screen capture is already closed")

    def close(self) -> None:
        """Release the MSS handle; repeated calls are safe."""
        if not self._closed:
            self._sct.close()
            self._closed = True

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
