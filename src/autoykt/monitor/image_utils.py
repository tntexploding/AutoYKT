"""Small image-comparison helpers shared by monitoring workflows."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from autoykt.core.config import Region


# OpenCV exports its native API dynamically.
# pylint: disable=no-member


def frame_change_ratio(before: np.ndarray, after: np.ndarray) -> float:
    """Return the fraction of pixels with a meaningful visual change."""
    if before.shape != after.shape or before.size == 0:
        return 1.0
    difference = cv2.absdiff(before, after)
    grayscale = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(grayscale > 30) / grayscale.size)


def read_image(path: str | Path) -> np.ndarray:
    """Read a BGR image using Unicode-safe filesystem access."""
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None or image.size == 0:
        raise ValueError(f"image cannot be decoded: {path}")
    return image


def write_png(path: str | Path, frame: np.ndarray) -> None:
    """Write a PNG using Unicode-safe filesystem access."""
    encoded, data = cv2.imencode(".png", frame)
    if not encoded:
        raise OSError(f"image cannot be encoded: {path}")
    Path(path).write_bytes(data.tobytes())


def crop_image(frame: np.ndarray, region: Region) -> np.ndarray:
    """Return an image view, rejecting truncation or negative offsets."""
    x, y, width, height = region
    if (
        min(x, y) < 0
        or min(width, height) <= 0
        or x + width > frame.shape[1]
        or y + height > frame.shape[0]
    ):
        raise ValueError("configured region lies outside the supplied image")
    return frame[y : y + height, x : x + width]
