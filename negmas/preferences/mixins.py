from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from negmas.common import MechanismState, NegotiatorMechanismInterface
    from negmas.negotiators import Negotiator
    from negmas.outcomes.common import Outcome

__all__ = [
    "VolatileUFunMixin",
    "SessionDependentUFunMixin",
    "StateDependentUFunMixin",
    "StationaryMixin",
]


class VolatileUFunMixin:
    """
    Indicates that the ufun is volatile
    """

    def is_volatile(self):
        return True


class SessionDependentUFunMixin:
    """
    Indicates that the ufun is session-dependent (i.e. utility value of outcomes depend on the NMI)
    """

    @abstractmethod
    def eval_on_session(
        self, offer: Outcome, nmi: NegotiatorMechanismInterface | None = None
    ):
        """Evaluates the offer given a session"""

    def eval(self, offer: Outcome):
        if not self.owner or not self.owner.nmi:
            return self.eval_on_session(offer, None)
        self.owner: Negotiator
        return self.eval_on_session(offer, self.owner.nmi)

    def is_session_dependent(self):
        """
        Does the utiltiy of an outcome depend on the `NegotiatorMechanismInterface`?
        """
        return True

    def is_state_dependent(self):
        """
        Does the utiltiy of an outcome depend on the negotiation state?
        """
        return False


class StateDependentUFunMixin:
    """
    Indicates that the ufun is state-dependent (i.e. utility value of outcomes depend on the mechanism state)
    """

    @abstractmethod
    def eval_on_state(
        self,
        offer: Outcome,
        nmi: NegotiatorMechanismInterface | None = None,
        state: MechanismState | None = None,
    ):
        """Evaluates the offer given a session and state"""

    def eval(self, offer: Outcome):
        if not self.owner or not self.owner.nmi:
            return self.eval_on_state(offer, None, None)
        self.owner: Negotiator
        return self.eval_on_state(offer, self.owner.nmi, self.owner.nmi.state)

    def is_session_dependent(self):
        """
        Does the utiltiy of an outcome depend on the `NegotiatorMechanismInterface`?
        """
        return True

    def is_state_dependent(self):
        """
        Does the utiltiy of an outcome depend on the negotiation state?
        """
        return True


class StationaryMixin:
    """
    Indicates that the ufun is stationary which means that it is not session or state dependent and not volatile.

    Negotiators using this type of ufuns can assume that if they call it twice with the same outcome, it will always return the same value.
    """

    def is_session_dependent(self) -> bool:
        return False

    def is_volatile(self) -> bool:
        return False

    def is_state_dependent(self) -> bool:
        return False

    def is_stationary(self) -> bool:
        return True

    def to_stationary(self):
        return self

    # @lru_cache(maxsize=100)
    # def eval_normalized(
    #     self: BaseUtilityFunction, # type: ignore
    #     offer: Outcome | None,
    #     above_reserve: bool = True,
    #     expected_limits: bool = True,
    # ) -> Value:
    #     """
    #     Caches the top 100 results from the ufun because we know they are never going to change.
    #     """
    #     return super().eval_normalized(offer, above_reserve, expected_limits)
