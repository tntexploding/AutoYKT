"""Exercise the production workflow on private images with virtual controls.

Only the answer provider and local OCR are real. The mouse, submit control,
completion feedback, and page transitions are simulated in memory.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np

from autoykt.agent.answer_agent import AnswerCoordinator
from autoykt.agent.models import AnswerProvider
from autoykt.core.config import (
    AppConfig,
    ClickTargetConfig,
    ConfigError,
    PageProfileConfig,
    Region,
)
from autoykt.core.event_bus import EventBus
from autoykt.core.state import WorkflowState
from autoykt.knowledge.store import KnowledgeStore
from autoykt.monitor.detector import (
    ImageTemplateMatcher,
    OptionTemplateDetector,
)
from autoykt.monitor.image_utils import crop_image, read_image, write_png
from autoykt.monitor.profile_runtime import PageAutomation
from autoykt.monitor.selection import selected_options


def _crop(frame: np.ndarray, region: Region) -> np.ndarray:
    try:
        return crop_image(frame, region)
    except ValueError as error:
        raise ConfigError(
            "image does not contain the calibrated region"
        ) from error


class _Scene:
    """An image-backed quiz; this object has no native desktop access."""

    def __init__(self, config: AppConfig, profile: PageProfileConfig):
        self.profile = profile
        style = profile.answer_style
        if style.button_colors is None or not style.option_templates:
            raise ConfigError(
                "rehearsal requires calibrated letter templates and colors"
            )
        if not profile.verification.success_templates:
            raise ConfigError("rehearsal requires a completion template")
        self.colors = style.button_colors
        self.detector = OptionTemplateDetector.from_config(config, style)
        self.triggers = ImageTemplateMatcher(
            [
                (
                    trigger.name,
                    str(config.resolve_path(trigger.path)),
                    trigger.threshold,
                )
                for trigger in profile.triggers
                if trigger.action == "answer"
            ]
        )
        self.success = read_image(
            config.resolve_path(profile.verification.success_templates[0].path)
        )
        self.original = np.empty((0, 0, 3), dtype=np.uint8)
        self.matches = {}
        self.selected: set[str] = set()
        self.submissions: list[list[str]] = []
        self.clicks: list[dict[str, object]] = []
        self.completed = False
        self.waiting = False
        self.multiple = False
        x, y, width, height = profile.regions.verification_region
        if self.success.shape[1] > width or self.success.shape[0] > height:
            raise ConfigError(
                "completion template does not fit verification region"
            )
        self.submit_point = (x + width // 2, y + height // 2)

    def load(self, path: Path) -> None:
        """Load the supplied layout and text, clearing historical selection."""
        self.original = read_image(path)
        trigger = self.triggers.best(
            _crop(self.original, self.profile.regions.detection)
        )
        if trigger is None:
            raise ConfigError(f"no question trigger found in image: {path}")
        rule = next(
            item for item in self.profile.triggers if item.name == trigger.name
        )
        self.multiple = rule.question_type == "multiple"
        self.matches = self.detector.detect(
            _crop(self.original, self.profile.regions.answer_region)
        )
        selected_options(self.matches)
        if len(self.matches) < 2:
            raise ConfigError(
                "rehearsal needs at least two recognizable option buttons"
            )
        self.selected = set()
        self.submissions = []
        self.clicks = []
        self.completed = False
        self.waiting = False

    def render(self) -> np.ndarray:
        """Paint virtual controls without altering source files."""
        frame = self.original.copy()
        answers = _crop(frame, self.profile.regions.answer_region)
        selected = np.array(self.colors.selected_rgb[::-1], dtype=np.int16)
        unselected = np.array(self.colors.unselected_rgb[::-1], dtype=np.int16)
        for key, match in self.matches.items():
            x, y = match["top_left"]
            width, height = match["size"]
            button = answers[y : y + height, x : x + width]
            pixels = button.astype(np.int16)
            mask = (
                np.max(np.abs(pixels - selected), axis=2)
                <= self.colors.tolerance
            ) | (
                np.max(np.abs(pixels - unselected), axis=2)
                <= self.colors.tolerance
            )
            button[mask] = selected if key in self.selected else unselected
        feedback = _crop(frame, self.profile.regions.verification_region)
        feedback[:] = 230
        if self.completed:
            height, width = self.success.shape[:2]
            feedback[:height, :width] = self.success
        if self.waiting:
            _crop(frame, self.profile.regions.detection)[:] = 255
        return frame

    def click_point(self, x: int, y: int) -> bool:
        """Accept only a virtual letter center or the virtual submit center."""
        if self.completed or self.waiting:
            raise RuntimeError(
                "attempted input on an inactive virtual question"
            )
        if (x, y) == self.submit_point:
            if not self.selected:
                raise RuntimeError(
                    "attempted virtual submission without an answer"
                )
            self.submissions.append(sorted(self.selected))
            self.completed = True
            self.clicks.append({"target": "submit", "point": [x, y]})
            return True
        offset_x, offset_y = self.profile.regions.answer_region[:2]
        for key, match in self.matches.items():
            if (x, y) != (
                offset_x + match["center"][0],
                offset_y + match["center"][1],
            ):
                continue
            if self.multiple:
                self.selected.symmetric_difference_update({key})
            else:
                self.selected = {key}
            self.clicks.append({"target": key, "point": [x, y]})
            return True
        raise RuntimeError(
            "virtual click did not target a recognized letter or submit control"
        )


class _Capture:
    """The same capture interface as production, using only scene pixels."""

    def __init__(self, scene: _Scene, region: Region, directory: Path):
        self.scene = scene
        self.region = region
        self.directory = directory

    def grab_frame(self) -> np.ndarray:
        """Return the currently rendered region."""
        return _crop(self.scene.render(), self.region).copy()

    def save_screenshot(self, frame: np.ndarray, prefix: str) -> Path:
        """Persist private replay evidence."""
        path = self.directory / f"{prefix}_{time.monotonic_ns()}.png"
        write_png(path, frame)
        return path

    def absolute_from_frame(self, point: tuple[int, int]) -> tuple[int, int]:
        """Map crop coordinates into the source image."""
        return self.region[0] + point[0], self.region[1] + point[1]

    def absolute_from_monitor(self, point: tuple[int, int]) -> tuple[int, int]:
        """The virtual monitor is the image itself."""
        return point

    def close(self) -> None:
        """The file backend owns no screen handles."""


def _rehearsal_config(
    config: AppConfig, profile_id: str, output: Path
) -> AppConfig:
    derived = config.model_copy(deep=True)
    # Convert window binding and all dependent click targets atomically.
    # Validating partial assignments would reject the old window coordinates.
    document = derived.page_profile(profile_id).model_dump()
    document.update(
        target_window=None,
        enabled=True,
        entry_target=None,
        submit_target=None,
        submit_on_select=False,
        page_flow={},
    )
    document["answer_style"].update(
        fallback_positions={}, fallback_coordinate_space="monitor"
    )
    document["verification"]["require_success_template"] = True
    profile = PageProfileConfig.model_validate(document)
    derived.runtime.active_profiles = [profile_id]
    derived.pages = [profile]
    derived.runtime.dry_run = False
    derived.answering.auto_apply = True
    derived.notifier.enabled = []
    derived.storage.data_dir = str(output / "data")
    derived.storage.log_dir = str(output / "logs")
    derived.storage.screenshot_dir = str(output / "screenshots")
    return derived


async def _replay_question(
    runtime: PageAutomation,
    scene: _Scene,
    path: Path,
    directory: Path,
) -> dict[str, object]:
    scene.load(path)
    write_png(directory / "virtual_before.png", scene.render())
    cycle_count = runtime.completed_cycles
    started = time.monotonic()
    polls = (
        scene.profile.rearm.stable_hits
        + max(trigger.consecutive_hits for trigger in scene.profile.triggers)
        + 5
    )
    for _ in range(polls):
        await runtime.poll_once()
        if runtime.completed_cycles > cycle_count or runtime.needs_attention:
            break
        await asyncio.sleep(0.1)
    elapsed = time.monotonic() - started
    report = runtime.report_path
    verified = (
        runtime.completed_cycles == cycle_count + 1
        and runtime.outcome == "verified"
    )
    input_count = len(scene.clicks)
    cycles_after = runtime.completed_cycles
    for _ in range(3):
        await runtime.poll_once()
    no_duplicates = (
        len(scene.clicks) == input_count
        and runtime.completed_cycles == cycles_after
    )
    write_png(directory / "virtual_after.png", scene.render())
    return {
        "source_image": str(path.resolve()),
        "verified": verified,
        "elapsed_seconds": round(elapsed, 3),
        "outcome": (
            runtime.outcome
            if runtime.completed_cycles > cycle_count
            else "not_detected"
        ),
        "virtual_clicks": scene.clicks,
        "virtual_submissions": scene.submissions,
        "no_duplicate_input": no_duplicates,
        "state_after": runtime.state.value,
        "operation_report": (
            str(report)
            if report and runtime.completed_cycles > cycle_count
            else None
        ),
        "passed": verified and no_duplicates and len(scene.submissions) == 1,
    }


async def _replay_images(
    runtime: PageAutomation,
    scene: _Scene,
    images: list[Path],
    output: Path,
) -> dict[str, object]:
    prepared_at = time.monotonic()
    await runtime.prepare()
    preparation_seconds = time.monotonic() - prepared_at
    scene.waiting = True
    await runtime.poll_once()
    await runtime.poll_once()
    waiting_passed = runtime.state == WorkflowState.WAITING and not scene.clicks
    results = []
    for index, path in enumerate(images, start=1):
        directory = output / f"question-{index}"
        directory.mkdir()
        result = await _replay_question(runtime, scene, path, directory)
        results.append(result)
        if not result["passed"]:
            break
    return {
        "preparation_seconds": round(preparation_seconds, 3),
        "waiting_passed": waiting_passed,
        "questions": results,
        "passed": waiting_passed
        and len(results) == len(images)
        and all(item["passed"] for item in results),
    }


async def run_rehearsal(
    config: AppConfig,
    profile_id: str,
    images: list[Path],
    output: Path | None = None,
    *,
    answerer: AnswerProvider | None = None,
) -> Path:
    """Run OCR/answering with virtual selection, submit, verify, and rearm.

    The source config is never changed. A supplied answerer permits offline
    regression tests; otherwise this calls the explicitly configured providers.
    Native window binding and website server responses are not exercised.
    """
    if not images:
        raise ConfigError("rehearsal requires at least one image")
    output = (
        output
        or config.resolve_path(config.storage.data_dir)
        / "rehearsals"
        / datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    derived = _rehearsal_config(config, profile_id, output)
    profile = derived.page_profile(profile_id)
    scene = _Scene(derived, profile)
    scene.load(images[0])
    profile.submit_target = ClickTargetConfig(
        point=scene.submit_point, coordinate_space="monitor"
    )
    screenshots = output / "screenshots"
    screenshots.mkdir()
    knowledge = (
        KnowledgeStore(derived.resolve_path(derived.knowledge.database_path))
        if derived.knowledge.enabled
        else None
    )
    coordinator = None
    runtime = None
    results = {}
    try:
        if answerer is None:
            prompt = derived.answering.prompt_template
            coordinator = AnswerCoordinator(
                derived.answering,
                derived.resolve_path(prompt) if prompt else None,
            )
            answerer = coordinator
        runtime = PageAutomation(
            derived,
            profile,
            EventBus(),
            answerer,
            scene,
            knowledge,
            capture_factory=lambda region: _Capture(scene, region, screenshots),
        )
        results = await _replay_images(runtime, scene, images, output)
    finally:
        if runtime is not None:
            await runtime.stop()
        if coordinator is not None:
            await coordinator.close()
        if knowledge is not None:
            knowledge.close()
    report = output / "report.json"
    report.write_text(
        json.dumps(
            {
                "simulation": True,
                "actual_desktop_clicks": 0,
                "actual_website_submissions": 0,
                "simulated_parts": [
                    "waiting",
                    "unselected_state",
                    "submit_control",
                    "completion_feedback",
                ],
                "uses_configured_models": coordinator is not None,
                **results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
