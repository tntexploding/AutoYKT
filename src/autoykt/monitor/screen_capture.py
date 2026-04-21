"""Screen capture using mss - fast, cross-platform screenshot with ROI support."""

import sys
import ctypes

# Enable DPI awareness on Windows so mss captures at full resolution
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

import numpy as np
import mss
import mss.tools
from pathlib import Path
from datetime import datetime


class ScreenCapture:
    """Captures screen frames, supports full screen or ROI (region of interest)."""

    def __init__(self, roi: list[int] | None = None, screenshot_dir: str = "storage/screenshots", monitor_index: int = 1) -> None:
        """
        Args:
            roi: [x, y, width, height] region to capture, relative to the selected monitor.
            screenshot_dir: directory to save screenshot files.
            monitor_index: mss monitor index (0=all, 1=primary, 2=secondary...).
        """
        self._roi = roi
        self._monitor_index = monitor_index
        self._screenshot_dir = Path(screenshot_dir)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._sct = mss.mss()

    @property
    def _base_monitor(self) -> dict:
        """Get the selected monitor's geometry."""
        idx = min(self._monitor_index, len(self._sct.monitors) - 1)
        return self._sct.monitors[idx]

    @property
    def monitor_region(self) -> dict:
        """Build mss monitor dict from ROI (relative to selected monitor) or full monitor."""
        base = self._base_monitor
        if self._roi:
            x, y, w, h = self._roi
            return {
                "left": base["left"] + x,
                "top": base["top"] + y,
                "width": w,
                "height": h,
            }
        return base

    def grab_frame(self) -> np.ndarray:                                                 #获取指定区域截图，返回openCV格式的颜色矩阵
        """Capture a single frame and return as BGR numpy array (OpenCV format)."""
        raw = self._sct.grab(self.monitor_region)                                       #指定区域截图
        # mss returns BGRA, convert to BGR for OpenCV
        img = np.array(raw)[:, :, :3].copy()                                            #截图色域转换为BGR
        return img

    def grab_full_screen(self) -> np.ndarray:
        """Capture the full selected monitor regardless of ROI setting."""
        raw = self._sct.grab(self._base_monitor)
        img = np.array(raw)[:, :, :3].copy()
        return img

    def save_screenshot(self, frame: np.ndarray, prefix: str = "screenshot") -> Path:   #用openCV保存截图
        """Save a frame to disk as PNG. Returns the file path."""
        import cv2
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.png"
        filepath = self._screenshot_dir / filename
        cv2.imwrite(str(filepath), frame)
        return filepath

    def update_roi(self, roi: list[int]) -> None:                                       #热重载ROI
        """Update the capture region at runtime."""
        self._roi = roi

    def close(self) -> None:                                                            #关闭并释放mss资源
        """Release mss resources."""
        self._sct.close()