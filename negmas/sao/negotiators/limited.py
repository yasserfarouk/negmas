from __future__ import annotations

from typing import List, Optional, Union

from negmas import warnings

from ...common import MechanismState
from ...outcomes import Outcome
from ..common import ResponseType
from ..components import LimitedOutcomesAcceptorMixin, LimitedOutcomesMixin
from .base import SAONegotiator

__all__ = [
    "LimitedOutcomesNegotiator",
    "LimitedOutcomesAcceptor",
]


class LimitedOutcomesNegotiator(LimitedOutcomesMixin, SAONegotiator):
    """
    A negotiation agent that uses a fixed set of outcomes in a single
    negotiation.

    Args:
        acceptable_outcomes: the set of acceptable outcomes. If None then it is assumed to be all the outcomes of
                             the negotiation.
        acceptance_probabilities: probability of accepting each acceptable outcome. If None then it is assumed to
                                  be unity.
        proposable_outcomes: the set of outcomes from which the agent is allowed to propose. If None, then it is
                             the same as acceptable outcomes with nonzero probability
        p_no_response: probability of refusing to respond to offers
        p_ending: probability of ending negotiation

    Remarks:
        - The ufun inputs to the constructor and join are ignored. A ufun will be generated that gives a utility equal to
          the probability of choosing a given outcome.
        - If `proposable_outcomes` is passed as None, it is considered the same as `acceptable_outcomes`

    """

    def __init__(
        self,
        acceptable_outcomes: Optional[List["Outcome"]] = None,
        acceptance_probabilities: Optional[Union[float, List[float]]] = None,
        proposable_outcomes: Optional[List["Outcome"]] = None,
        p_ending=0.0,
        p_no_response=0.0,
        preferences=None,
        ufun=None,
        **kwargs,
    ) -> None:
        if ufun:
            preferences = ufun
        if preferences is not None:
            warnings.warn(
                f"LimitedOutcomesAcceptor negotiators ignore preferences but they are given",
                warnings.NegmasUnusedValueWarning,
            )
        super().__init__(**kwargs)
        if proposable_outcomes is None:
            proposable_outcomes = acceptable_outcomes
        self.init_limited_outcomes(
            p_ending=p_ending,
            p_no_response=p_no_response,
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            proposable_outcomes=proposable_outcomes,
        )


# noinspection PyCallByClass
class LimitedOutcomesAcceptor(LimitedOutcomesAcceptorMixin, SAONegotiator):
    """A negotiation agent that uses a fixed set of outcomes in a single
    negotiation.

    Remarks:
        - The ufun inputs to the constructor and join are ignored. A ufun will be generated that gives a utility equal to
          the probability of choosing a given outcome.

    """

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return LimitedOutcomesAcceptorMixin.respond(self, state=state, offer=offer)

    def __init__(
        self,
        acceptable_outcomes: Optional[List["Outcome"]] = None,
        acceptance_probabilities: Optional[List[float]] = None,
        p_ending=0.0,
        p_no_response=0.0,
        preferences=None,  # type: ignore This type of negotiator does not take preferences on construction
        **kwargs,
    ) -> None:
        if preferences is not None:
            warnings.warn(
                f"LimitedOutcomesAcceptor negotiators ignore preferences but they are given",
                warnings.NegmasIgnoredValueWarning,
            )
        super().__init__(self, **kwargs)
        self.init_limited_outcomes_acceptor(
            p_ending=p_ending,
            p_no_response=p_no_response,
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
        )
        self.add_capabilities({"propose": False})

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Always refuses to propose"""
        return None
