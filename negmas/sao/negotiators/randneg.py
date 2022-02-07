from __future__ import annotations

import numpy as np

from negmas.preferences.preferences import Preferences

from ...common import MechanismState, NegotiatorMechanismInterface
from ...outcomes import CartesianOutcomeSpace, Outcome
from ...preferences import LinearUtilityFunction, MappingUtilityFunction
from ..common import ResponseType
from ..mixins import RandomProposalMixin, RandomResponseMixin
from .base import SAONegotiator

__all__ = [
    "RandomNegotiator",
]


class RandomNegotiator(RandomResponseMixin, RandomProposalMixin, SAONegotiator):
    """
    A negotiation agent that responds randomly in a single negotiation.

    Args:
        p_acceptance: Probability of accepting an offer
        p_rejection:  Probability of rejecting an offer
        p_ending: Probability of ending the negotiation at any round
        can_propose: Whether the agent can propose or not
        **kwargs: Passed to the SAONegotiator

    Remarks:
        - If p_acceptance + p_rejection + p_ending < 1, the rest is the probability of no-response.
    """

    def propose_(self, state: MechanismState) -> Outcome | None:
        return RandomProposalMixin.propose(self, state)  # type: ignore

    def respond_(self, state: MechanismState, offer: Outcome) -> "ResponseType":
        return RandomResponseMixin.respond(self, state, offer)

    def __init__(
        self,
        p_acceptance=0.15,
        p_rejection=0.75,
        p_ending=0.1,
        can_propose=True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.init_random_response(
            p_acceptance=p_acceptance, p_rejection=p_rejection, p_ending=p_ending
        )
        self.init_random_proposal()
        self.capabilities["propose"] = can_propose

    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: "Preferences" | None = None,
        role: str = "negotiator",
    ) -> bool:
        """
        Will create a random utility function to be used by the negotiator.

        Args:
            nmi: The AMI
            state: The current mechanism state
            preferences: IGNORED.
            role: IGNORED.
        """
        result = super().join(nmi, state, preferences=preferences, role=role)
        if not result:
            return False
        if nmi.outcome_space is None:
            raise ValueError(
                "Cannot generate a random ufun without knowing the issue space"
            )
        if nmi.outcome_space.is_numeric() and isinstance(
            nmi.outcome_space, CartesianOutcomeSpace
        ):
            self.set_preferences(
                LinearUtilityFunction(
                    weights=np.random.random(len(nmi.outcome_space.issues)).tolist(),
                    outcome_space=nmi.outcome_space,
                )
            )
        else:
            outcomes = list(nmi.discrete_outcomes())
            self.set_preferences(
                MappingUtilityFunction(
                    dict(zip(outcomes, np.random.rand(len(outcomes)))),
                    outcome_space=nmi.outcome_space,
                )
            )
        return True
