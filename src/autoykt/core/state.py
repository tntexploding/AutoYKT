"""Explicit workflow states and validated transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkflowState(str, Enum):
    """States in one page profile's answer lifecycle."""

    WAITING = "waiting"
    PREPARING = "preparing"
    CAPTURING = "capturing"
    ANSWERING = "answering"
    APPLYING = "applying"
    SUBMITTING = "submitting"
    VERIFYING = "verifying"
    REARMING = "rearming"
    ERROR = "error"
    STOPPED = "stopped"


_ALLOWED_TRANSITIONS: dict[WorkflowState, frozenset[WorkflowState]] = {
    WorkflowState.WAITING: frozenset(
        {
            WorkflowState.PREPARING,
            WorkflowState.REARMING,
            WorkflowState.ERROR,
            WorkflowState.STOPPED,
        }
    ),
    WorkflowState.PREPARING: frozenset(
        {
            WorkflowState.CAPTURING,
            WorkflowState.REARMING,
            WorkflowState.ERROR,
            WorkflowState.STOPPED,
        }
    ),
    WorkflowState.CAPTURING: frozenset(
        {
            WorkflowState.ANSWERING,
            WorkflowState.REARMING,
            WorkflowState.ERROR,
            WorkflowState.STOPPED,
        }
    ),
    WorkflowState.ANSWERING: frozenset(
        {
            WorkflowState.APPLYING,
            WorkflowState.REARMING,
            WorkflowState.ERROR,
            WorkflowState.STOPPED,
        }
    ),
    WorkflowState.APPLYING: frozenset(
        {
            WorkflowState.SUBMITTING,
            WorkflowState.VERIFYING,
            WorkflowState.ERROR,
            WorkflowState.STOPPED,
        }
    ),
    WorkflowState.SUBMITTING: frozenset(
        {WorkflowState.VERIFYING, WorkflowState.ERROR, WorkflowState.STOPPED}
    ),
    WorkflowState.VERIFYING: frozenset(
        {WorkflowState.REARMING, WorkflowState.ERROR, WorkflowState.STOPPED}
    ),
    WorkflowState.REARMING: frozenset(
        {WorkflowState.WAITING, WorkflowState.ERROR, WorkflowState.STOPPED}
    ),
    WorkflowState.ERROR: frozenset(
        {WorkflowState.REARMING, WorkflowState.WAITING, WorkflowState.STOPPED}
    ),
    WorkflowState.STOPPED: frozenset(),
}


class InvalidStateTransition(RuntimeError):
    """Raised when workflow code attempts an impossible transition."""


@dataclass
class WorkflowStateMachine:
    """Track and validate one profile's current workflow state."""

    profile_id: str
    state: WorkflowState = WorkflowState.WAITING

    def transition(self, new_state: WorkflowState) -> WorkflowState:
        """Move to ``new_state`` and return the previous state."""
        if new_state == self.state:
            return self.state
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidStateTransition(
                f"profile '{self.profile_id}' cannot transition from "
                f"{self.state.value} to {new_state.value}"
            )
        previous = self.state
        self.state = new_state
        return previous
