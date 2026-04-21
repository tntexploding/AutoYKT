"""Calibration tool - helps users mark ROI and option positions on screen.

Run this before first use to configure screen coordinates.
Usage: python -m scripts.calibrate
"""

import sys
import ctypes

# Enable DPI awareness on Windows so mss captures at full resolution
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import cv2
import numpy as np
import mss
import yaml
from pathlib import Path


class Calibrator:
    """Interactive calibration: click on screen to mark regions and positions."""

    WINDOW_NAME = "Calibration - Press key to switch mode"

    def __init__(self, monitor_index: int = 1) -> None:
        self._sct = mss.mss()
        self._monitor_index = min(monitor_index, len(self._sct.monitors) - 1)
        self._points: list[tuple[int, int]] = []
        self._roi: list[int] | None = None
        self._option_positions: dict[str, list[int]] = {}
        self._mode = "roi"  # roi / options / template
        self._current_option = "A"
        self._frame_orig: np.ndarray | None = None  # full-res for cropping
        self._frame_display: np.ndarray | None = None  # scaled for display
        self._scale = 1.0
        mon = self._sct.monitors[self._monitor_index]
        print(f"Using monitor [{self._monitor_index}]: {mon['width']}x{mon['height']} at ({mon['left']}, {mon['top']})")

    def _grab_screen(self) -> np.ndarray:
        raw = self._sct.grab(self._sct.monitors[self._monitor_index])
        return np.array(raw)[:, :, :3].copy()

    def _to_display(self, x: int, y: int) -> tuple[int, int]:
        """Convert original coords to display coords."""
        return int(x * self._scale), int(y * self._scale)

    def _to_original(self, x: int, y: int) -> tuple[int, int]:
        """Convert display coords to original coords."""
        return int(x / self._scale), int(y / self._scale)

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self._frame_display is None:
            return

        # Display coords for drawing, original coords for data
        dx, dy = x, y
        ox, oy = self._to_original(x, y)

        if self._mode == "roi":
            self._points.append((ox, oy))
            cv2.circle(self._frame_display, (dx, dy), 5, (0, 0, 255), -1)
            if len(self._points) == 2:
                p1, p2 = self._points
                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                w = abs(p2[0] - p1[0])
                h = abs(p2[1] - p1[1])
                self._roi = [x1, y1, w, h]
                dp1 = self._to_display(*p1)
                dp2 = self._to_display(*p2)
                cv2.rectangle(self._frame_display, dp1, dp2, (0, 255, 0), 2)
                print(f"  ROI set: {self._roi}")
                self._points.clear()

        elif self._mode == "options":
            opt = self._current_option
            self._option_positions[opt] = [ox, oy]
            cv2.circle(self._frame_display, (dx, dy), 8, (255, 0, 0), -1)
            cv2.putText(self._frame_display, opt, (dx + 10, dy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            print(f"  Option {opt} -> ({ox}, {oy})")
            next_map = {"A": "B", "B": "C", "C": "D", "D": "A"}
            self._current_option = next_map[opt]

        elif self._mode == "template":
            self._points.append((ox, oy))
            cv2.circle(self._frame_display, (dx, dy), 5, (0, 255, 255), -1)
            if len(self._points) == 2:
                p1, p2 = self._points
                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                template = self._frame_orig[y1:y2, x1:x2]
                out_path = Path("assets/templates/question_region.png")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), template)
                dp1 = self._to_display(*p1)
                dp2 = self._to_display(*p2)
                cv2.rectangle(self._frame_display, dp1, dp2, (0, 255, 255), 2)
                print(f"  Template saved to {out_path} ({template.shape})")
                self._points.clear()

    def run(self) -> None:
        """Run the interactive calibration window."""
        print("=== Auto-Answer Calibration Tool ===")
        print("Modes: [1] ROI  [2] Options  [3] Template  [s] Save  [q] Quit")
        print("Current mode: ROI - click two corners of the question area.\n")

        self._frame_orig = self._grab_screen()
        h, w = self._frame_orig.shape[:2]
        print(f"Screenshot size: {w}x{h}")

        # Auto-scale if too large for display
        self._scale = 1.0
        if w > 1920 or h > 1080:
            self._scale = min(1920 / w, 1080 / h)
        new_w, new_h = int(w * self._scale), int(h * self._scale)
        self._frame_display = cv2.resize(self._frame_orig, (new_w, new_h))
        if self._scale < 1.0:
            print(f"Scaled to {new_w}x{new_h} for display (scale={self._scale:.2f})")

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        while True:
            cv2.imshow(self.WINDOW_NAME, self._frame_display)
            key = cv2.waitKey(50) & 0xFF

            if key == ord("1"):
                self._mode = "roi"
                self._points.clear()
                print("Mode: ROI - click two corners of the question area.")
            elif key == ord("2"):
                self._mode = "options"
                self._current_option = "A"
                print("Mode: Options - click position of A, B, C, D in order.")
            elif key == ord("3"):
                self._mode = "template"
                self._points.clear()
                print("Mode: Template - click two corners to crop template image.")
            elif key == ord("r"):
                # Refresh screenshot
                self._frame_orig = self._grab_screen()
                h, w = self._frame_orig.shape[:2]
                new_w, new_h = int(w * self._scale), int(h * self._scale)
                self._frame_display = cv2.resize(self._frame_orig, (new_w, new_h))
                print("Screenshot refreshed.")
            elif key == ord("s"):
                self._save_config()
            elif key == ord("q"):
                break

        cv2.destroyAllWindows()
        self._sct.close()

    def _save_config(self) -> None:
        """Write calibrated values to config file."""
        config_path = Path("config.example.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        if self._roi:
            cfg.setdefault("monitor", {})["roi"] = self._roi
        if self._option_positions:
            cfg.setdefault("clicker", {})["options_positions"] = self._option_positions

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

        print(f"\nConfig saved to {config_path}")
        if self._roi:
            print(f"  ROI: {self._roi}")
        if self._option_positions:
            print(f"  Options: {self._option_positions}")


if __name__ == "__main__":
    Calibrator().run()