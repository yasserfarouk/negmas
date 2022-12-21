from __future__ import annotations

from negmas import warnings
from negmas.gb.components.acceptance import LimitedOutcomesAcceptancePolicy
from negmas.gb.components.offering import LimitedOutcomesOfferingPolicy

from ...outcomes import Outcome
from .base import GBNegotiator
from .modular.mapneg import MAPNegotiator

__all__ = [
    "LimitedOutcomesNegotiator",
    "LimitedOutcomesAcceptor",
]


class LimitedOutcomesNegotiator(MAPNegotiator):
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
        acceptable_outcomes: list[Outcome] | None = None,
        acceptance_probabilities: float | list[float] | None = None,
        proposable_outcomes: list[Outcome] | None = None,
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
        if not proposable_outcomes:
            proposable_outcomes = acceptable_outcomes
        offering = LimitedOutcomesOfferingPolicy(
            outcomes=proposable_outcomes if proposable_outcomes else [],
            p_ending=p_no_response,
        )
        if acceptance_probabilities is None and acceptable_outcomes is None:
            acceptance = LimitedOutcomesAcceptancePolicy(
                prob=0.5,
                p_ending=p_ending,
            )
        elif acceptance_probabilities is None and acceptable_outcomes is not None:
            acceptance = LimitedOutcomesAcceptancePolicy.from_outcome_list(
                acceptable_outcomes,
                p_ending=p_ending,
            )
        elif isinstance(acceptance_probabilities, float):
            acceptance = LimitedOutcomesAcceptancePolicy(
                prob=acceptance_probabilities,
                p_ending=p_ending,
            )
        elif acceptable_outcomes is None:
            warnings.warn(
                "No outcomes are given but we have a list of acceptance probabilities for a limited negotiatoor!! Will just reject everything and offer nothing",
                warnings.NegmasUnexpectedValueWarning,
            )
            acceptance = LimitedOutcomesAcceptancePolicy(
                prob=0.0,
                p_ending=p_ending,
            )
        else:
            if acceptance_probabilities is None:
                acceptance_probabilities = [1.0] * len(acceptable_outcomes)
            acceptance = LimitedOutcomesAcceptancePolicy(
                prob=dict(zip(acceptable_outcomes, acceptance_probabilities)),
                p_ending=p_ending,
            )
        super().__init__(acceptance=acceptance, offering=offering, **kwargs)


# noinspection PyCallByClass
class LimitedOutcomesAcceptor(MAPNegotiator, GBNegotiator):
    """A negotiation agent that uses a fixed set of outcomes in a single
    negotiation.

    Remarks:
        - The ufun inputs to the constructor and join are ignored. A ufun will be generated that gives a utility equal to
          the probability of choosing a given outcome.

    """

    def __init__(
        self,
        acceptable_outcomes: list[Outcome] | None = None,
        acceptance_probabilities: list[float] | None = None,
        p_ending=0.0,
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
        if acceptance_probabilities is None and acceptable_outcomes is None:
            acceptance = LimitedOutcomesAcceptancePolicy(
                prob=None,
                p_ending=p_ending,
            )
        elif acceptance_probabilities is None and acceptable_outcomes is not None:
            acceptance = LimitedOutcomesAcceptancePolicy.from_outcome_list(
                acceptable_outcomes,
                p_ending=p_ending,
            )
        elif isinstance(acceptance_probabilities, float):
            acceptance = LimitedOutcomesAcceptancePolicy(
                prob=acceptance_probabilities,
                p_ending=p_ending,
            )
        elif acceptable_outcomes is None:
            warnings.warn(
                "No outcomes are given but we have a list of acceptance probabilities for a limited negotiatoor!! Will just reject everything and offer nothing",
                warnings.NegmasUnexpectedValueWarning,
            )
            acceptance = LimitedOutcomesAcceptancePolicy(
                prob=0.0,
                p_ending=p_ending,
            )
        else:
            acceptance = LimitedOutcomesAcceptancePolicy(
                prob=dict(zip(acceptable_outcomes, acceptance_probabilities)),  # type: ignore I know that acceptance probabilitis is not none by now
                p_ending=p_ending,
            )
        super().__init__(models=None, acceptance=acceptance, **kwargs)
        self.capabilities["propose"] = False
