"""Mouse interaction with a fail-safe and a testable interface."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Protocol


logger = logging.getLogger("autoykt")


class MouseController(Protocol):
    """Interface consumed by the automation workflow."""

    def click_point(self, x: int, y: int) -> bool:
        """Click one absolute desktop coordinate."""
        raise NotImplementedError


class Clicker:
    """Perform absolute clicks through PyAutoGUI."""

    def __init__(
        self,
        options_positions: (
            Mapping[str, tuple[int, int] | list[int]] | None
        ) = None,
        confirm_delay: float = 1.0,
        screen_capture: object | None = None,
    ) -> None:
        del screen_capture
        self._positions = {
            key.strip().upper(): (int(value[0]), int(value[1]))
            for key, value in (options_positions or {}).items()
        }
        self.confirm_delay = max(0.0, float(confirm_delay))

    def click_point(self, x: int, y: int) -> bool:
        """Click an absolute point and propagate PyAutoGUI's fail-safe."""
        import pyautogui  # pylint: disable=import-outside-toplevel

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.3
        logger.info("Clicking absolute point (%d, %d)", x, y)
        pyautogui.click(x, y)
        return True

    def click_option(self, option: str) -> bool:
        """Click a pre-resolved absolute fallback option position."""
        key = option.strip().upper()
        point = self._positions.get(key)
        if point is None:
            logger.error(
                "No fallback position is configured for option '%s'", key
            )
            return False
        return self.click_point(*point)
