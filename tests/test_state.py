"""Tests for workflow state transition validation."""

import unittest

from autoykt.core.state import (
    InvalidStateTransition,
    WorkflowState,
    WorkflowStateMachine,
)


class WorkflowStateMachineTest(unittest.TestCase):

    def test_accepts_normal_answer_path(self) -> None:
        machine = WorkflowStateMachine("page")
        for state in (
            WorkflowState.PREPARING,
            WorkflowState.CAPTURING,
            WorkflowState.ANSWERING,
            WorkflowState.APPLYING,
            WorkflowState.SUBMITTING,
            WorkflowState.VERIFYING,
            WorkflowState.REARMING,
            WorkflowState.WAITING,
        ):
            machine.transition(state)
        self.assertEqual(machine.state, WorkflowState.WAITING)

    def test_rejects_skipping_directly_to_submit(self) -> None:
        machine = WorkflowStateMachine("page")
        with self.assertRaises(InvalidStateTransition):
            machine.transition(WorkflowState.SUBMITTING)

    def test_waiting_can_fail_closed(self) -> None:
        machine = WorkflowStateMachine("page")
        machine.transition(WorkflowState.ERROR)
        self.assertEqual(machine.state, WorkflowState.ERROR)


if __name__ == "__main__":
    unittest.main()
