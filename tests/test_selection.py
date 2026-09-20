"""Exercise dynamic button state through a complete visual answer cycle."""

import numpy as np

from autoykt.agent.models import ConsensusAnswer
from autoykt.core.config import (
    AnswerStyleConfig,
    ButtonColorsConfig,
    ClickTargetConfig,
    ImageTemplateConfig,
)
from autoykt.core.state import WorkflowState
from autoykt.monitor.image_utils import read_image, write_png
from tests.test_image_calibration import BLUE, GRAY, _button
from tests.test_profile_runtime import WorkflowScene


class SelectionWorkflowTest(WorkflowScene):
    """Real template matching plus simulated, observable mouse effects."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        profile = self.config.pages[0]
        profile.regions.question = (0, 0, 500, 500)
        profile.regions.answers = (0, 0, 500, 500)
        profile.regions.rearm = (0, 0, 500, 500)
        profile.triggers[0].question_type = "multiple"
        profile.question_ready.maximum_change_ratio = 0.001
        templates = {}
        for key in "ABCDE":
            path = self.root / f"{key}.png"
            write_png(path, _button(key, GRAY))
            templates[key] = str(path)
        profile.answer_style = AnswerStyleConfig(
            option_templates=templates,
            option_match_threshold=0.94,
            match_grayscale=True,
            button_colors=ButtonColorsConfig(
                selected_rgb=(BLUE[2], BLUE[1], BLUE[0]),
                unselected_rgb=GRAY,
            ),
            selection_timeout_seconds=0.3,
        )
        profile.submit_target = ClickTargetConfig(
            point=(11, 12),
            coordinate_space="screen",
        )
        profile.verification.success_templates = [
            ImageTemplateConfig(
                path=str(self.root / "trigger.png"),
                threshold=0.99,
            )
        ]
        self.positions = {
            "A": (30, 50),
            "B": (240, 65),
            "C": (80, 200),
            "D": (280, 235),
            "E": (160, 360),
        }
        self.selected = {"A", "B"}
        self.desired = {"A", "C", "E"}
        self.answer_request.return_value = ConsensusAnswer(
            "A,C,E",
            True,
            1.0,
            votes={"A,C,E": 2},
        )
        self.ignore_click = False
        self.change_after_click = None
        self.applied = []
        self.submissions = []
        self.clicker.click_point = self._apply_click
        self._render_buttons()

    def _render_buttons(self):
        frame = np.full((500, 500, 3), 255, dtype=np.uint8)
        frame[8:22, 8:320] = 0  # Immutable question text surrogate.
        for key, (x, y) in self.positions.items():
            frame[y : y + 60, x : x + 60] = _button(
                key,
                BLUE if key in self.selected else GRAY,
            )
        self.scene[500] = frame

    def _apply_click(self, x, y):
        self.clicker.points.append((x, y))
        if (x, y) == (11, 12):
            self.submissions.append(self.selected.copy())
            self.scene[50][2:8, 2:8] = read_image(self.root / "trigger.png")
            return True
        for key, (left, top) in self.positions.items():
            if (x, y) != (left + 130, top + 230):
                continue
            self.applied.append(key)
            if not self.ignore_click:
                if self.config.pages[0].triggers[0].question_type == "single":
                    self.selected = {key}
                else:
                    self.selected.symmetric_difference_update({key})
                self._render_buttons()
            if self.change_after_click:
                self.change_after_click()
            return True
        self.fail(f"unexpected click: {(x, y)}")

    async def test_multiselect_changes_only_needed_keys_then_submits(self):
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.applied, ["B", "C", "E"])
        self.assertEqual(self.submissions, [self.desired])
        self.assertEqual(automation.outcome, "verified")
        self.assertTrue(self.answer_request.call_args.kwargs["multiple"])
        self.assertEqual(
            self.answer_request.call_args.args[1], frozenset("ABCDE")
        )
        await automation.poll_once()
        self.assertEqual(len(self.submissions), 1)

    async def test_small_text_change_rearms_but_selection_colors_do_not(self):
        self.config.pages[0].rearm.minimum_change_ratio = 0.02
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(automation.outcome, "verified")
        # A transiently hidden completion label must not repeat the same answer.
        self.scene[50][:] = 0
        await automation.poll_once()
        self.assertEqual(automation.state, WorkflowState.REARMING)
        self.assertEqual(self.answer_request.await_count, 1)
        # A new short line changes under 10% of the large white question area.
        self.scene[500][24:50, 8:320] = 0
        await automation.poll_once()
        self.assertEqual(automation.state, WorkflowState.WAITING)
        await automation.poll_once()
        self.assertEqual(self.answer_request.await_count, 2)
        self.assertEqual(automation.outcome, "verified")
        self.assertEqual(len(self.submissions), 2)

    def _use_glyph_templates(self):
        style = self.config.pages[0].answer_style
        assert style.button_colors is not None
        style.button_colors.button_size = (60, 60)
        for path in style.option_templates.values():
            write_png(path, read_image(path)[10:50, 14:46])

    async def test_glyph_crops_mask_whole_buttons_during_selection(self):
        self._use_glyph_templates()
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.applied, ["B", "C", "E"])
        self.assertEqual(self.submissions, [self.desired])
        self.assertEqual(automation.outcome, "verified")

    async def test_glyph_crops_still_detect_unmapped_physical_button(self):
        self._use_glyph_templates()
        del self.config.pages[0].answer_style.option_templates["E"]
        automation = self._automation()
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)

    async def test_already_correct_set_only_submits(self):
        self.selected = self.desired.copy()
        self._render_buttons()
        automation = self._automation()
        await automation.poll_once()
        self.assertFalse(self.applied)
        self.assertEqual(self.submissions, [self.desired])
        self.assertEqual(automation.outcome, "verified")

    async def test_single_choice_replaces_selection(self):
        self.config.pages[0].triggers[0].question_type = "single"
        self.selected = {"A"}
        self.desired = {"D"}
        self.answer_request.return_value = ConsensusAnswer("D", True, 1.0)
        self._render_buttons()
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.applied, ["D"])
        self.assertEqual(self.submissions, [{"D"}])
        self.assertFalse(self.answer_request.call_args.kwargs["multiple"])

    async def test_disappeared_options_are_excluded_from_model_request(self):
        self.positions = {
            key: value for key, value in self.positions.items() if key in "ABC"
        }
        self.desired = {"A", "C"}
        self.answer_request.return_value = ConsensusAnswer("A,C", True, 1.0)
        self._render_buttons()
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(
            self.answer_request.call_args.args[1], frozenset("ABC")
        )
        self.assertEqual(self.submissions, [self.desired])

    async def test_click_without_state_change_stops_before_submission(self):
        self.ignore_click = True
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.applied, ["B"])
        self.assertFalse(self.submissions)
        self.assertTrue(automation.needs_attention)
        self.assertEqual(automation.state, WorkflowState.ERROR)

    async def test_layout_change_stops_remaining_toggles(self):
        def move():
            self.positions["C"] = (100, 200)
            self._render_buttons()

        self.change_after_click = move
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.applied, ["B"])
        self.assertFalse(self.submissions)
        self.assertTrue(automation.needs_attention)

    async def test_question_change_is_not_hidden_by_button_masks(self):
        def replace_text():
            self.scene[500][8:22, 8:320] = 255

        self.change_after_click = replace_text
        automation = self._automation()
        await automation.poll_once()
        self.assertEqual(self.applied, ["B"])
        self.assertFalse(self.submissions)
        self.assertTrue(automation.needs_attention)

    async def test_unknown_button_color_stops_before_answering(self):
        self.scene[500][50:110, 30:90] = _button("A", (80, 170, 90))
        automation = self._automation()
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)

    async def test_unmapped_visible_letter_cannot_be_silently_ignored(self):
        del self.config.pages[0].answer_style.option_templates["E"]
        automation = self._automation()
        await automation.poll_once()
        self.answer_request.assert_not_awaited()
        self.assertFalse(self.clicker.points)
