"""Simulate mouse clicks on answer options and confirm result."""

import time
import logging

import cv2
import numpy as np
import pyautogui

from autoykt.monitor.screen_capture import ScreenCapture

logger = logging.getLogger("auto_answer")

# pyautogui safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.3


class Clicker:
    """Clicks answer options on screen and verifies the result."""

    def __init__(
        self,
        options_positions: dict[str, list[int]],
        confirm_delay: float = 1.0,
        screen_capture: ScreenCapture | None = None,
    ) -> None:
        """
        Args:
            options_positions: mapping of option key -> [x, y] screen coordinates.
            confirm_delay: seconds to wait after click before checking result.
            screen_capture: ScreenCapture instance for taking confirmation screenshots.
        """
        self._positions = options_positions
        self._confirm_delay = confirm_delay
        self._capture = screen_capture or ScreenCapture()

    def click_option(self, option: str) -> bool:
        """Click the specified option (A/B/C/D).

        Returns:
            True if the option position is known and click was performed.
        """
        key = option.strip().upper()
        pos = self._positions.get(key)
        if not pos:
            logger.error(f"No position configured for option '{key}'")
            return False

        x, y = pos
        logger.info(f"Clicking option {key} at ({x}, {y})")
        pyautogui.click(x, y)
        return True

    def click_point(self, x: int, y: int) -> bool:
        """Click an absolute screen coordinate."""
        logger.info(f"Clicking point at ({x}, {y})")
        pyautogui.click(x, y)
        return True

    def confirm_result(self, before: np.ndarray | None = None) -> tuple[bool, np.ndarray]:
        """Wait and take a screenshot to confirm the answer was submitted.

        Args:
            before: pre-click screenshot for comparison. If None, skips diff check.

        Returns:
            (success, screenshot_frame)
            success is True if the screen changed after clicking (basic pixel diff check).
        """
        time.sleep(self._confirm_delay)
        after = self._capture.grab_frame()

        if before is None:
            return True, after  # no baseline to compare, assume success

        # Simple change detection: if enough pixels changed, assume submission happened
        diff = cv2.absdiff(before, after)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        changed_pixels = np.count_nonzero(gray_diff > 30)
        total_pixels = gray_diff.size
        change_ratio = changed_pixels / total_pixels

        success = bool(change_ratio > 0.01)  # >1% pixels changed
        logger.info(
            f"Result confirmation: change_ratio={change_ratio:.4f}, "
            f"success={success}"
        )
        return success, after

    def click_and_confirm(self, option: str) -> tuple[bool, np.ndarray | None]:
        """Click an option and confirm the result in one call.

        Returns:
            (success, confirmation_screenshot)
        """
        # Capture BEFORE clicking for comparison
        before = self._capture.grab_frame()

        clicked = self.click_option(option)
        if not clicked:
            return False, None

        success, frame = self.confirm_result(before=before)
        return success, frame

    def click_point_and_confirm(self, x: int, y: int) -> tuple[bool, np.ndarray | None]:
        """Click an absolute point and confirm via change detection."""
        before = self._capture.grab_frame()
        clicked = self.click_point(x, y)
        if not clicked:
            return False, None
        success, frame = self.confirm_result(before=before)
        return success, frame