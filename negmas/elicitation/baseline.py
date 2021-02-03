import time

from typing import Optional, Union

from .base import BaseElicitor
from ..common import MechanismState
from ..outcomes import Outcome
from ..utilities import UtilityValue

__all__ = ["DummyElicitor", "FullKnowledgeElicitor"]


class DummyElicitor(BaseElicitor):
    """
    A dummy elicitation algorithm that does not do any elicitation.
    """

    def utility_on_rejection(
        self, outcome: "Outcome", state: MechanismState
    ) -> UtilityValue:
        return self.reserved_value

    def can_elicit(self) -> bool:
        return True

    def elicit_single(self, state: MechanismState):
        return False

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        **kwargs,
    ):
        super().init_elicitation(ufun=ufun, **kwargs)
        strt_time = time.perf_counter()
        self.offerable_outcomes = self._ami.outcomes
        self._elicitation_time += time.perf_counter() - strt_time


class FullKnowledgeElicitor(BaseElicitor):
    """
    An elicitor that does not *need* to do any elicitation because it has full access
    to the user ufun.
    """

    def utility_on_rejection(
        self, outcome: "Outcome", state: MechanismState
    ) -> UtilityValue:
        return self.reserved_value

    def can_elicit(self) -> bool:
        return True

    def elicit_single(self, state: MechanismState):
        return False

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        **kwargs,
    ):
        super().init_elicitation(ufun=self.user.ufun)
        strt_time = time.perf_counter()
        self.offerable_outcomes = self._ami.outcomes
        self._elicitation_time += time.perf_counter() - strt_time
