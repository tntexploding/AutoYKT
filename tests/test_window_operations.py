"""Window and action-plan checks without a real desktop or mouse."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from autoykt.core.config import (
    ClickTargetConfig,
    ConfigError,
    PageGuardConfig,
    PageProfileConfig,
    WindowTargetConfig,
)
from autoykt.monitor.image_utils import read_image, write_png
from autoykt.monitor.operations import ProfileSession
from autoykt.monitor.screen_capture import ScreenCapture
from autoykt.monitor.windows import WindowGuard, WindowInfo, WindowUnavailable
from tests.test_profile_runtime import _configuration, _trigger_image, _FakeCapture, _FakeClicker


class FakeDesktop:
    """An independently movable window and an input hit-test surface."""

    def __init__(self):
        self.items = [
            WindowInfo(
                100,
                200,
                "Quiz",
                "Browser",
                (40, 80, 400, 300),
                (38, 78, 404, 304),
            )
        ]
        self.focus = 100
        self.activations = []
        self.blocked_point: tuple[int, int] | None = None

    def windows(self):
        return list(self.items)

    def foreground(self):
        return self.focus

    def activate(self, handle):
        self.activations.append(handle)
        self.focus = handle

    def root_at(self, point):
        return 999 if point == self.blocked_point else 100


def make_guard(desktop):
    return WindowGuard(
        WindowTargetConfig(
            title_pattern="^Quiz$", class_name="Browser", client_size=(400, 300)
        ),
        desktop,
    )


class WindowGuardTest(unittest.TestCase):
    """Window changes cannot redirect a planned click to another surface."""

    def test_move_recomputes_absolute_point(self):
        desktop = FakeDesktop()
        guard = make_guard(desktop)
        self.assertEqual(guard.resolve_point((12, 20)), (52, 100))
        desktop.items[0] = replace(
            desktop.items[0], client=(-600, 20, 400, 300)
        )
        self.assertEqual(guard.resolve_point((12, 20)), (-588, 40))

    def test_focus_minimize_resize_and_ambiguity_block(self):
        for reason in ("focus", "minimize", "resize", "duplicate"):
            with self.subTest(reason=reason):
                desktop = FakeDesktop()
                if reason == "focus":
                    desktop.focus = 999
                elif reason == "minimize":
                    desktop.items[0] = replace(desktop.items[0], minimized=True)
                elif reason == "resize":
                    desktop.items[0] = replace(
                        desktop.items[0], client=(40, 80, 500, 300)
                    )
                else:
                    desktop.items.append(replace(desktop.items[0], handle=101))
                with self.assertRaises(WindowUnavailable):
                    make_guard(desktop).check()

    def test_overlay_blocks_even_when_target_keeps_focus(self):
        desktop = FakeDesktop()
        desktop.items.insert(
            0,
            WindowInfo(
                999, 888, "Overlay", "Popup", (70, 90, 10, 10), (70, 90, 10, 10)
            ),
        )
        desktop.blocked_point = (70, 90)
        with self.assertRaisesRegex(WindowUnavailable, "covered"):
            make_guard(desktop).check()

    def test_replaced_process_requires_restart(self):
        desktop = FakeDesktop()
        guard = make_guard(desktop)
        guard.check()
        desktop.items[0] = replace(desktop.items[0], process_id=201)
        with self.assertRaisesRegex(WindowUnavailable, "replaced"):
            guard.check()

    def test_input_hit_test_and_bounds_are_checked(self):
        desktop = FakeDesktop()
        guard = make_guard(desktop)
        desktop.blocked_point = (52, 100)
        with self.assertRaisesRegex(WindowUnavailable, "receive"):
            guard.resolve_point((12, 20))
        with self.assertRaisesRegex(WindowUnavailable, "outside"):
            guard.resolve_point((400, 20))

    def test_background_capture_never_activates_but_input_requires_focus(self):
        desktop = FakeDesktop()
        desktop.focus = 999
        guard = make_guard(desktop)
        with self.assertRaisesRegex(WindowUnavailable, "foreground"):
            guard.check(for_capture=True)
        guard.target.background_monitoring = True
        self.assertEqual(guard.check(for_capture=True).handle, 100)
        self.assertEqual(desktop.activations, [])
        with self.assertRaisesRegex(WindowUnavailable, "foreground"):
            guard.resolve_point((12, 20))
        self.assertTrue(guard.prepare_input())
        self.assertEqual(desktop.activations, [100])
        self.assertEqual(guard.resolve_point((12, 20)), (52, 100))
        self.assertFalse(guard.prepare_input())

    def test_activation_refusal_and_window_replacement_block_input(self):
        for replace_window in (False, True):
            with self.subTest(replace_window=replace_window):
                desktop = FakeDesktop()
                desktop.focus = 999
                guard = make_guard(desktop)
                guard.target.background_monitoring = True

                def activate(handle):
                    del handle
                    if replace_window:
                        desktop.focus = 100
                        desktop.items[0] = replace(
                            desktop.items[0], process_id=201
                        )

                desktop.activate = activate
                with self.assertRaises(WindowUnavailable):
                    guard.prepare_input()

    def test_calibrated_focus_click_is_checked_and_focus_is_verified(self):
        desktop = FakeDesktop()
        desktop.focus = 999
        desktop.activate = Mock(
            side_effect=WindowUnavailable("OS denied focus")
        )
        guard = make_guard(desktop)
        guard.target.background_monitoring = True
        guard.target.focus_point = (12, 20)
        points = []

        def focus(x, y):
            points.append((x, y))
            desktop.focus = 100
            return True

        desktop.blocked_point = (52, 100)
        with self.assertRaisesRegex(
            WindowUnavailable, "focus click is covered"
        ):
            guard.prepare_input(focus)
        self.assertEqual(points, [])
        desktop.blocked_point = None
        self.assertTrue(guard.prepare_input(focus))
        self.assertEqual(points, [(52, 100)])
        self.assertEqual(guard.check().handle, 100)

    def test_invisible_overlap_is_allowed_but_actual_overlap_blocks(self):
        desktop = FakeDesktop()
        desktop.focus = 999
        desktop.items.insert(
            0,
            WindowInfo(
                999,
                888,
                "Other screen",
                "Browser",
                (440, 80, 400, 300),
                (432, 78, 408, 304),
            ),
        )
        guard = make_guard(desktop)
        guard.target.background_monitoring = True
        self.assertEqual(guard.check(for_capture=True).handle, 100)
        desktop.blocked_point = (432, 80)
        with self.assertRaisesRegex(WindowUnavailable, "covered"):
            guard.check(for_capture=True)
        self.assertEqual(desktop.activations, [])

    def test_bound_capture_ignores_an_old_monitor_index(self):
        native = Mock(monitors=[{}])
        native.grab.return_value = np.zeros((20, 20, 4), dtype=np.uint8)
        with patch(
            "autoykt.monitor.screen_capture.mss.mss", return_value=native
        ):
            with ScreenCapture(
                (2, 3, 20, 20),
                None,
                monitor_index=9,
                window_guard=make_guard(FakeDesktop()),
            ) as capture:
                self.assertEqual(capture.grab_frame().shape, (20, 20, 3))
                native.grab.assert_called_once_with(
                    {"left": 42, "top": 83, "width": 20, "height": 20}
                )

    def test_movement_during_capture_discards_frame(self):
        desktop = FakeDesktop()
        native = Mock(
            monitors=[{}, {"left": 0, "top": 0, "width": 800, "height": 600}]
        )

        def grab(_region):
            desktop.items[0] = replace(
                desktop.items[0], client=(41, 80, 400, 300)
            )
            return np.zeros((20, 20, 4), dtype=np.uint8)

        native.grab.side_effect = grab
        with patch(
            "autoykt.monitor.screen_capture.mss.mss", return_value=native
        ):
            with ScreenCapture(
                (0, 0, 20, 20), None, window_guard=make_guard(desktop)
            ) as capture:
                with self.assertRaisesRegex(WindowUnavailable, "moved"):
                    capture.grab_frame()
        native.close.assert_called_once()


class PlanTest(unittest.TestCase):
    """Plans carry visible, checked targets and stay in private storage."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        template, frame = _trigger_image(self.root)
        self.config = _configuration(
            template, self.root / "private" / "config.yaml"
        )
        self.clicker = _FakeClicker()
        self.capture = _FakeCapture(
            (0, 0, 40, 40), self.root, frame, self.clicker
        )
        self.session = ProfileSession(self.config, self.config.pages[0])
        self.session.attach_capture_factory(
            lambda region: _FakeCapture(region, self.root, frame, self.clicker)
        )
        self.session.start("test")

    def test_fallback_and_submit_preview_are_the_executed_points(self):
        self.config.pages[0].submit_target = ClickTargetConfig(
            point=(500, 650), coordinate_space="screen"
        )
        point, _ = self.session.locate_answer(
            "B", np.zeros((40, 40, 3), dtype=np.uint8), self.capture
        )
        plan = self.session.plan("B", point, self.capture)
        self.assertFalse(self.clicker.points)
        for click in plan.clicks:
            self.assertTrue(self.session.click(click, self.clicker))
        self.assertEqual(self.clicker.points, [(321, 654), (500, 650)])
        assert self.session.report_path is not None
        self.assertTrue(
            self.session.report_path.is_relative_to(
                (self.root / "private" / "data").resolve()
            )
        )
        report = json.loads(
            self.session.report_path.read_text(encoding="utf-8")
        )
        step = next(item for item in report["steps"] if item["step"] == "plan")
        preview = read_image(
            self.session.report_path.parent / "plan_preview.png"
        )
        origin_x, origin_y = step["preview_origin"]
        region_x, region_y = step["preview_region"][:2]
        for click in plan.clicks:
            x, y = (
                click.point[0] - origin_x - region_x,
                click.point[1] - origin_y - region_y,
            )
            self.assertEqual(preview[y, x].tolist(), [0, 0, 255])

    def test_ambiguous_marker_refuses_even_with_fallback(self):
        marker = np.random.default_rng(9).integers(
            0, 256, (6, 6, 3), dtype=np.uint8
        )
        path = self.root / "option.png"
        write_png(path, marker)
        self.config.pages[0].answer_style.option_templates = {"B": str(path)}
        session = ProfileSession(self.config, self.config.pages[0])
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        frame[2:8, 2:8] = marker
        frame[25:31, 25:31] = marker
        with self.assertRaisesRegex(RuntimeError, "multiple locations"):
            session.locate_answer("B", frame, self.capture)
        self.assertFalse(self.clicker.points)

    def test_page_identity_blocks_a_click(self):
        marker = self.root / "page.png"
        write_png(
            marker,
            np.random.default_rng(4).integers(
                0, 256, (8, 8, 3), dtype=np.uint8
            ),
        )
        self.config.pages[0].page_guard = PageGuardConfig(
            path=str(marker), region=(0, 0, 8, 8), threshold=0.99
        )
        action = self.session.target(
            "submit", ClickTargetConfig(point=(2, 3)), self.capture
        )
        with self.assertRaisesRegex(WindowUnavailable, "identity"):
            self.session.click(action, self.clicker)
        self.assertFalse(self.clicker.points)

    def test_cancellation_during_final_check_revokes_input(self):
        action = self.session.target(
            "submit", ClickTargetConfig(point=(2, 3)), self.capture
        )
        with patch.object(
            self.session, "check_page", side_effect=self.session.cancel
        ):
            self.assertFalse(self.session.click(action, self.clicker))
        self.assertFalse(self.clicker.points)

    def test_log_failure_prevents_input(self):
        action = self.session.target(
            "submit", ClickTargetConfig(point=(2, 3)), self.capture
        )
        with patch.object(
            self.session, "record", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                self.session.click(action, self.clicker)
        self.assertFalse(self.clicker.points)

    def test_bound_profile_requires_relative_coordinates(self):
        document = self.config.pages[0].model_dump()
        document["target_window"] = {
            "title_pattern": "Quiz",
            "client_size": [400, 300],
        }
        with self.assertRaisesRegex(ValueError, "window coordinates"):
            PageProfileConfig.model_validate(document)
        document["answer_style"]["fallback_coordinate_space"] = "window"
        with self.assertRaisesRegex(ValueError, "outside"):
            PageProfileConfig.model_validate(document)
        document["answer_style"]["fallback_positions"] = {"B": [100, 100]}
        profile = PageProfileConfig.model_validate(document)
        with patch("autoykt.monitor.operations.WindowGuard"):
            session = ProfileSession(self.config, profile)
        with self.assertRaisesRegex(ConfigError, "page_guard"):
            session.check_live_configuration()
        profile.page_guard = PageGuardConfig(
            path="missing.png", region=(0, 0, 10, 10)
        )
        with self.assertRaisesRegex(ConfigError, "success templates"):
            session.check_live_configuration()

    def test_bound_plan_tracks_movement_after_preview(self):
        desktop = FakeDesktop()
        self.session.window = make_guard(desktop)
        self.session.window.check()
        # Use a client-relative point as required by a bound profile.
        plan = self.session.plan("B", (100, 110), self.capture)
        desktop.items[0] = replace(
            desktop.items[0], client=(300, 200, 400, 300)
        )
        self.session.click(plan.clicks[0], self.clicker)
        self.assertEqual(self.clicker.points, [(400, 310)])
