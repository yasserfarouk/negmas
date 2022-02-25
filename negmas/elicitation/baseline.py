from __future__ import annotations

import time

from ..common import MechanismState, Value
from ..outcomes import Outcome
from .base import BaseElicitor

__all__ = ["DummyElicitor", "FullKnowledgeElicitor"]


class DummyElicitor(BaseElicitor):
    """
    A dummy elicitation algorithm that does not do any elicitation.
    """

    def utility_on_rejection(self, outcome: Outcome, state: MechanismState) -> Value:
        return self.reserved_value

    def can_elicit(self) -> bool:
        return True

    def elicit_single(self, state: MechanismState):
        return False

    def init_elicitation(
        self,
        preferences: IPUtilityFunction | Distribution | None,
        **kwargs,
    ):
        super().init_elicitation(preferences=preferences, **kwargs)
        strt_time = time.perf_counter()
        self.offerable_outcomes = self._nmi.outcomes
        self._elicitation_time += time.perf_counter() - strt_time


class FullKnowledgeElicitor(BaseElicitor):
    """
    An elicitor that does not *need* to do any elicitation because it has full access
    to the user ufun.
    """

    def utility_on_rejection(self, outcome: Outcome, state: MechanismState) -> Value:
        return self.reserved_value

    def can_elicit(self) -> bool:
        return True

    def elicit_single(self, state: MechanismState):
        return False

    def init_elicitation(
        self,
        preferences: IPUtilityFunction | Distribution | None,
        **kwargs,
    ):
        super().init_elicitation(preferences=self.user.ufun)
        strt_time = time.perf_counter()
        self.offerable_outcomes = self._nmi.outcomes
        self._elicitation_time += time.perf_counter() - strt_time
