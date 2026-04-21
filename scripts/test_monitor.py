"""Quick test script to verify each monitor component independently.

Usage:
    python -m scripts.test_monitor --capture       # Test screen capture
    python -m scripts.test_monitor --detect        # Test template detection
    python -m scripts.test_monitor --ocr           # Test OCR on a screenshot
    python -m scripts.test_monitor --click A       # Test clicking option A
    python -m scripts.test_monitor --all           # Run all tests in sequence
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def test_capture() -> np.ndarray:
    """Test screen capture - grabs a frame and saves it."""
    from autoykt.monitor.screen_capture import ScreenCapture

    print("[1/4] Testing screen capture...")
    cap = ScreenCapture()
    frame = cap.grab_frame()
    path = cap.save_screenshot(frame, prefix="test_capture")
    cap.close()
    print(f"  OK - captured {frame.shape}, saved to {path}")
    return frame


def test_detect(frame: np.ndarray | None = None) -> None:
    """Test template detection on a frame."""
    from autoykt.monitor.detector import QuestionDetector

    print("[2/4] Testing question detector...")
    template_path = "assets/templates/question_region.png"
    if not Path(template_path).exists():
        print(f"  SKIP - template not found at {template_path}")
        print("  Run 'python -m scripts.calibrate' first to create a template.")
        return

    if frame is None:
        from autoykt.monitor.screen_capture import ScreenCapture
        cap = ScreenCapture()
        frame = cap.grab_frame()
        cap.close()

    if frame is None:
        print("  SKIP - could not obtain a frame.")
        return

    detector = QuestionDetector(template_path=template_path, debounce_frames=1)
    detected, confidence, location = detector.detect(frame)
    print(f"  detected={detected}, confidence={confidence:.3f}, location={location}")


def test_ocr(frame: np.ndarray | None = None) -> None:
    """Test OCR on a frame or saved screenshot."""
    from autoykt.monitor.ocr_engine import create_ocr_engine

    print("[3/4] Testing OCR engine...")

    if frame is None:
        # Try to use the most recent screenshot
        screenshots = sorted(Path("storage/screenshots").glob("*.png"))
        if screenshots:
            img_path = screenshots[-1]
            print(f"  Using saved screenshot: {img_path}")
            frame = cv2.imread(str(img_path))
        else:
            from autoykt.monitor.screen_capture import ScreenCapture
            cap = ScreenCapture()
            frame = cap.grab_frame()
            cap.close()

    if frame is None:
        print("  SKIP - could not obtain a frame.")
        return

    try:
        engine = create_ocr_engine("rapidocr")
        result = engine.recognize(frame)
        print(f"  Question: {result.question}")
        print(f"  Options:  {result.options}")
        print(f"  Raw text ({len(result.raw_text)} chars): {result.raw_text[:200]}")
    except ImportError:
        print("  SKIP - rapidocr-onnxruntime not installed.")
        print("  Run: pip install rapidocr-onnxruntime")


def test_click(option: str = "A") -> None:
    """Test clicking an option (dry run - moves mouse but doesn't click)."""
    print(f"[4/4] Testing clicker (option {option})...")
    print("  NOTE: This will actually move your mouse and click.")
    print("  Move mouse to top-left corner to abort (pyautogui failsafe).")

    import pyautogui
    pos = pyautogui.position()
    print(f"  Current mouse position: {pos}")
    print(f"  Skipping actual click in test mode. Use --force-click to execute.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test monitor components")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--click", type=str, default=None, metavar="OPTION")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.capture, args.detect, args.ocr, args.click, args.all]):
        parser.print_help()
        sys.exit(0)

    frame = None

    if args.all or args.capture:
        frame = test_capture()
    if args.all or args.detect:
        test_detect(frame)
    if args.all or args.ocr:
        test_ocr(frame)
    if args.click:
        test_click(args.click)
    elif args.all:
        test_click("A")

    print("\nDone.")


if __name__ == "__main__":
    main()