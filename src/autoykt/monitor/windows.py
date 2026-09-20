"""Windows-specific process setup kept separate from capture logic."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
import sys
from ctypes import wintypes
from dataclasses import dataclass
import re
from typing import Protocol

from autoykt.core.config import WindowTargetConfig


def enable_dpi_awareness() -> None:
    """Request physical-pixel coordinates when running on Windows."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


class WindowUnavailable(RuntimeError):
    """The bound window cannot currently be operated safely."""


@dataclass(frozen=True)
class WindowInfo:
    """One top-level window, using physical desktop pixels."""

    handle: int
    process_id: int
    title: str
    class_name: str
    client: tuple[int, int, int, int]
    rectangle: tuple[int, int, int, int]
    minimized: bool = False


class DesktopWindows(Protocol):
    """Window information and explicit activation for the input guard."""

    def windows(self) -> list[WindowInfo]:
        """Return visible, uncloaked windows from top to bottom."""
        raise NotImplementedError

    def foreground(self) -> int:
        """Return the foreground top-level window handle."""
        raise NotImplementedError

    def activate(self, handle: int) -> None:
        """Request foreground input for the validated window."""
        raise NotImplementedError

    def root_at(self, point: tuple[int, int]) -> int:
        """Return the top-level window receiving input at a point."""
        raise NotImplementedError


class Win32Desktop:
    """Query windows and request foreground input without moving them."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise WindowUnavailable(
                "window binding is available on Windows only"
            )
        enable_dpi_awareness()
        self._api = ctypes.WinDLL("user32", use_last_error=True)
        self._dwm = ctypes.WinDLL("dwmapi", use_last_error=True)
        self._callback_type = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )
        signatures = {
            "EnumWindows": (
                [self._callback_type, wintypes.LPARAM],
                wintypes.BOOL,
            ),
            "GetForegroundWindow": ([], wintypes.HWND),
            "SetForegroundWindow": ([wintypes.HWND], wintypes.BOOL),
            "GetAncestor": ([wintypes.HWND, wintypes.UINT], wintypes.HWND),
            "WindowFromPoint": ([wintypes.POINT], wintypes.HWND),
            "IsWindowVisible": ([wintypes.HWND], wintypes.BOOL),
            "IsIconic": ([wintypes.HWND], wintypes.BOOL),
            "GetWindowTextLengthW": ([wintypes.HWND], ctypes.c_int),
            "GetWindowTextW": (
                [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int],
                ctypes.c_int,
            ),
            "GetClassNameW": (
                [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int],
                ctypes.c_int,
            ),
            "GetClientRect": (
                [wintypes.HWND, ctypes.POINTER(wintypes.RECT)],
                wintypes.BOOL,
            ),
            "GetWindowRect": (
                [wintypes.HWND, ctypes.POINTER(wintypes.RECT)],
                wintypes.BOOL,
            ),
            "ClientToScreen": (
                [wintypes.HWND, ctypes.POINTER(wintypes.POINT)],
                wintypes.BOOL,
            ),
            "GetWindowThreadProcessId": (
                [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)],
                wintypes.DWORD,
            ),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self._api, name)
            function.argtypes = arguments
            function.restype = result
        self._dwm.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self._dwm.DwmGetWindowAttribute.restype = ctypes.c_long

    def windows(self) -> list[WindowInfo]:
        """Enumerate visible desktop windows in their current Z order."""
        result = []

        def collect(handle: int, _parameter: int) -> bool:
            info = self._describe(handle)
            if info is not None:
                result.append(info)
            return True

        callback = self._callback_type(collect)
        if not self._api.EnumWindows(callback, 0):
            raise WindowUnavailable("could not enumerate desktop windows")
        return result

    def _describe(self, handle: int) -> WindowInfo | None:
        if not self._api.IsWindowVisible(handle):
            return None
        cloaked = wintypes.DWORD()
        if (
            self._dwm.DwmGetWindowAttribute(
                handle, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked)
            )
            == 0
            and cloaked.value
        ):
            return None
        client = wintypes.RECT()
        rectangle = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if not (
            self._api.GetClientRect(handle, ctypes.byref(client))
            and self._api.ClientToScreen(handle, ctypes.byref(origin))
            and self._api.GetWindowRect(handle, ctypes.byref(rectangle))
        ):
            return None
        visible = wintypes.RECT()
        if (
            self._dwm.DwmGetWindowAttribute(
                handle, 9, ctypes.byref(visible), ctypes.sizeof(visible)
            )
            == 0
        ):
            rectangle = visible
        title = ctypes.create_unicode_buffer(
            self._api.GetWindowTextLengthW(handle) + 1
        )
        class_name = ctypes.create_unicode_buffer(256)
        self._api.GetWindowTextW(handle, title, len(title))
        self._api.GetClassNameW(handle, class_name, len(class_name))
        process_id = wintypes.DWORD()
        self._api.GetWindowThreadProcessId(handle, ctypes.byref(process_id))
        return WindowInfo(
            handle,
            process_id.value,
            title.value,
            class_name.value,
            (origin.x, origin.y, client.right, client.bottom),
            (
                rectangle.left,
                rectangle.top,
                rectangle.right - rectangle.left,
                rectangle.bottom - rectangle.top,
            ),
            bool(self._api.IsIconic(handle)),
        )

    def foreground(self) -> int:
        """Return the foreground window handle."""
        return int(self._api.GetForegroundWindow() or 0)

    def activate(self, handle: int) -> None:
        """Request foreground input, respecting a denied Windows request."""
        if not self._api.SetForegroundWindow(handle):
            raise WindowUnavailable(
                "Windows could not activate the bound window"
            )

    def root_at(self, point: tuple[int, int]) -> int:
        """Resolve a child hit-test result to its root window."""
        handle = self._api.WindowFromPoint(wintypes.POINT(*point))
        return int(self._api.GetAncestor(handle, 2) or 0)


def _overlaps(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> bool:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    return max(x1, x2) < min(x1 + w1, x2 + w2) and max(y1, y2) < min(
        y1 + h1, y2 + h2
    )


class WindowGuard:
    """Bind one window; refuse ambiguity, occlusion, focus loss or resizing."""

    def __init__(
        self, target: WindowTargetConfig, desktop: DesktopWindows | None = None
    ) -> None:
        self.target = target
        self._desktop = desktop if desktop is not None else Win32Desktop()
        self._identity: tuple[int, int] | None = None

    def check(self, *, for_capture: bool = False) -> WindowInfo:
        """Return a safe, current client rectangle without stealing focus."""
        windows = self._desktop.windows()
        candidates = [
            item
            for item in windows
            if re.search(self.target.title_pattern, item.title)
            and (
                not self.target.class_name
                or item.class_name == self.target.class_name
            )
        ]
        if len(candidates) != 1:
            raise WindowUnavailable(
                "window selector must match exactly one visible window"
            )
        selected = candidates[0]
        identity = (selected.handle, selected.process_id)
        if self._identity is not None and identity != self._identity:
            raise WindowUnavailable(
                "bound window was replaced; restart the run"
            )
        if selected.minimized:
            raise WindowUnavailable("bound window is minimized")
        if self._desktop.foreground() != selected.handle and not (
            for_capture and self.target.background_monitoring
        ):
            raise WindowUnavailable("bound window is not in the foreground")
        if selected.client[2:] != self.target.client_size:
            raise WindowUnavailable(
                "window client size changed; recalibrate before running"
            )
        for item in windows[: windows.index(selected)]:
            if not item.minimized and _overlaps(
                item.rectangle, selected.client
            ):
                left = max(item.rectangle[0], selected.client[0])
                top = max(item.rectangle[1], selected.client[1])
                right = min(
                    item.rectangle[0] + item.rectangle[2],
                    selected.client[0] + selected.client[2],
                )
                bottom = min(
                    item.rectangle[1] + item.rectangle[3],
                    selected.client[1] + selected.client[3],
                )
                # Some borderless apps retain invisible resize margins even
                # in their DWM bounds. Hit-test the overlap, not that margin.
                if any(
                    self._desktop.root_at(point) != selected.handle
                    for point in (
                        (left, top),
                        (right - 1, top),
                        (left, bottom - 1),
                        (right - 1, bottom - 1),
                        ((left + right - 1) // 2, (top + bottom - 1) // 2),
                    )
                ):
                    raise WindowUnavailable(
                        "bound window is covered by another window"
                    )
        x, y, width, height = selected.client
        for point in (
            (x, y),
            (x + width - 1, y),
            (x, y + height - 1),
            (x + width - 1, y + height - 1),
            (x + width // 2, y + height // 2),
        ):
            if self._desktop.root_at(point) != selected.handle:
                raise WindowUnavailable(
                    "window client area is not fully available for input"
                )
        self._identity = identity
        return selected

    def prepare_input(
        self, focus_click: Callable[[int, int], bool] | None = None
    ) -> bool:
        """Activate an opted-in visible window, then validate it again."""
        info = self.check(for_capture=True)
        activated = self._desktop.foreground() != info.handle
        if activated:
            try:
                self._desktop.activate(info.handle)
            except WindowUnavailable:
                point = self.target.focus_point
                if point is None or focus_click is None:
                    raise
                # A calibrated neutral click is ordinary user input, not a
                # workaround for the OS foreground API. Validate its receiver.
                info = self.check(for_capture=True)
                absolute = (
                    info.client[0] + point[0],
                    info.client[1] + point[1],
                )
                if self._desktop.root_at(absolute) != info.handle:
                    raise WindowUnavailable("focus click is covered") from None
                if not focus_click(*absolute):
                    raise WindowUnavailable(
                        "focus click was not applied"
                    ) from None
        self.check()
        return activated

    def resolve_point(self, point: tuple[int, int]) -> tuple[int, int]:
        """Check the live window again and translate a client-relative click."""
        info = self.check()
        x, y, width, height = info.client
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise WindowUnavailable(
                "click lies outside the bound window client area"
            )
        absolute = (x + point[0], y + point[1])
        if self._desktop.root_at(absolute) != info.handle:
            raise WindowUnavailable("another window would receive this click")
        return absolute
