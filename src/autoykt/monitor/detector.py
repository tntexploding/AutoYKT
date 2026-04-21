"""Question detector using OpenCV template matching with debounce."""

import cv2
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("auto_answer")


class QuestionDetector:
    """Detects whether a question dialog/frame has appeared on screen.

    Uses OpenCV template matching against a reference image of the question frame.
    Includes debounce logic: only triggers after N consecutive positive frames.
    """

    def __init__(
        self,
        template_path: str,
        threshold: float = 0.90,
        debounce_frames: int = 3,
    ) -> None:
        """
        Args:
            template_path: path to the template image (question frame reference).
            threshold: match confidence threshold (0~1).
            debounce_frames: consecutive positive frames required before triggering.
        """
        self._threshold = threshold
        self._debounce_required = debounce_frames
        self._consecutive_hits = 0
        self._triggered = False  # prevents re-trigger on same question

        tpl_path = Path(template_path)
        if not tpl_path.exists():
            logger.warning(f"Template not found: {template_path}. Detector will use fallback mode.")
            self._template = None
        else:
            self._template = cv2.imread(str(tpl_path), cv2.IMREAD_COLOR)
            if self._template is not None:
                logger.info(f"Template loaded: {tpl_path} ({self._template.shape})")

    def detect(self, frame: np.ndarray) -> tuple[bool, float, tuple[int, int] | None]:
        """Check if the question frame is present in the given screen frame.

        Returns:
            (is_detected, confidence, match_location)
            match_location is (x, y) of the top-left corner of the best match.
        """
        if self._template is None:
            return False, 0.0, None

        result = cv2.matchTemplate(frame, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        is_match = max_val >= self._threshold

        if is_match:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0
            self._triggered = False  # reset trigger when question disappears

        # Debounce: only fire after N consecutive hits, and only once per appearance
        detected = (
            self._consecutive_hits >= self._debounce_required
            and not self._triggered
        )

        if detected:
            self._triggered = True
            logger.info(f"Question detected! confidence={max_val:.3f} at {max_loc}")

        location: tuple[int, int] | None = max_loc if is_match else None
        return detected, max_val, location

    def reset(self) -> None:
        """Reset detector state. Call after question is answered."""
        self._consecutive_hits = 0
        self._triggered = False
        logger.debug("Detector state reset.")

    def update_template(self, template_path: str) -> None:
        """Hot-reload template image."""
        tpl_path = Path(template_path)
        if tpl_path.exists():
            self._template = cv2.imread(str(tpl_path), cv2.IMREAD_COLOR)
            logger.info(f"Template reloaded: {tpl_path}")
        else:
            logger.error(f"Template not found: {template_path}")