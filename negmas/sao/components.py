"""
Implements basic components that can be used by `SAOSAONegotiator` s.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable

from negmas.warnings import NegmasUnexpectedValueWarning

from ..common import MechanismState, PreferencesChange
from ..outcomes import Outcome
from ..preferences import MappingUtilityFunction
from .common import ResponseType

if TYPE_CHECKING:
    from .negotiators import SAONegotiator

__all__ = [
    "LimitedOutcomesAcceptorMixin",
    "LimitedOutcomesProposerMixin",
    "LimitedOutcomesMixin",
    "RandomResponseMixin",
    "RandomProposalMixin",
]


class RandomResponseMixin:
    """
    A mixin that adds the ability to respond randomly to offers

    Remarks:
        - Call `init_random_response` to initialize the mixin.
    """

    def init_random_response(
        self,
        p_acceptance: float = 0.15,
        p_rejection: float = 0.25,
        p_ending: float = 0.1,
    ) -> None:
        """
        Initializes the mixin.

        Args:
            p_acceptance: probability of accepting offers
            p_rejection: probability of rejecting offers
            p_ending: probability of ending negotiation

        Remarks:
            - If the summation of acceptance, rejection and ending probabilities
                is less than 1.0 then with the remaining probability a
                NO_RESPONSE is returned from respond()
        """
        self.add_capabilities({"respond": True})  # type: ignore (This mixin will always be added to a SAONegotiator which has this)
        self.p_acceptance = p_acceptance
        self.p_rejection = p_rejection
        self.p_ending = p_ending
        self.wheel: list[tuple[float, ResponseType]] = [(0.0, ResponseType.NO_RESPONSE)]
        if self.p_acceptance > 0.0:
            self.wheel.append(
                (self.wheel[-1][0] + self.p_acceptance, ResponseType.ACCEPT_OFFER)
            )
        if self.p_rejection > 0.0:
            self.wheel.append(
                (self.wheel[-1][0] + self.p_rejection, ResponseType.REJECT_OFFER)
            )
        if self.p_ending > 0.0:
            self.wheel.append(
                (self.wheel[-1][0] + self.p_ending, ResponseType.REJECT_OFFER)
            )
        if self.wheel[-1][0] > 1.0:
            raise ValueError("Probabilities of acceptance+rejection+ending>1")

        self.wheel = self.wheel[1:]

    def respond(self, state: MechanismState, offer: Outcome) -> "ResponseType":
        r = random.random()
        for w in self.wheel:
            if w[0] >= r:
                return w[1]
        return ResponseType.NO_RESPONSE


class RandomProposalMixin:
    """
    A mixin that adds the ability to propose random offers

    Remarks:
        - call `init_random_proposal` to initialize the mixin
    """

    def init_random_proposal(self: SAONegotiator):  # type: ignore
        """Initializes the mixin"""
        self.add_capabilities(
            {
                "propose": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def propose(self: SAONegotiator, state: MechanismState) -> Outcome | None:  # type: ignore
        """
        Proposes a random offer

        Args:
            state: The mechanism state

        Returns:
            A proposed outcome or None to refuse to offer.

        Remarks:
            - Does not use the state.
        """
        return self.nmi.random_outcomes(1)[0]


class LimitedOutcomesAcceptorMixin:
    """
    A mixin that adds the ability to accept random offers
    """

    def init_limited_outcomes_acceptor(
        self: SAONegotiator,  # type: ignore
        acceptable_outcomes: Iterable[Outcome] | None = None,
        acceptance_probabilities: list[float] | None = None,
        time_factor: float | list[float] = None,
        p_ending: float = 0.05,
        p_no_response: float = 0.0,
    ) -> None:
        """Initializes the mixin

        Args:
            acceptable_outcomes: the set of acceptable outcomes. If None then it is assumed to be all the outcomes of the negotiation.
            acceptance_probabilities: probability of accepting each acceptable outcome. If None then it is assumed to be unity.
            time_factor: If given, the acceptance probability will go up with time by this factor
            p_no_response: probability of refusing to respond to offers
            p_ending: probability of ending negotiation

        Returns:
            None
        """
        self.add_capabilities({"respond": True})
        self.acceptable_outcomes, self.acceptance_probabilities = (  # type: ignore
            acceptable_outcomes,
            acceptance_probabilities,
        )
        self.p_no_response = p_no_response  # type: ignore
        self.p_ending = p_ending + p_no_response  # type: ignore
        self.time_factor = time_factor  # type: ignore

    def join(
        self,
        nmi,
        state,
        *,
        preferences=None,
        ufun=None,
        role: str = "negotiator",
        **kwargs,
    ) -> bool:
        if preferences or ufun:
            warnings.warn(
                f"Preferences or ufun are given but {self.name}  [id: {self.id}] of type {self.__class__.__name__} will ignore it",
                NegmasUnexpectedValueWarning,
            )
        ufun = self._make_preferences(nmi.issues)
        return super().join(nmi, state, ufun=ufun, role=role, **kwargs)

    def _make_preferences(self, issues):  # type: ignore
        """Generates a ufun that maps acceptance probability to the utility"""
        if self.acceptable_outcomes is None:
            self.acceptable_outcomes = self.nmi.discrete_outcomes()  # type: ignore

        if self.acceptance_probabilities is None:
            self.acceptance_probabilities = [1.0] * len(self.acceptable_outcomes)  # type: ignore
        if not isinstance(self.acceptance_probabilities, Iterable):
            self.acceptance_probabilities = [self.acceptance_probabilities] * len(
                self.acceptable_outcomes  # type: ignore
            )
        self.mapping = defaultdict(float)
        for p, o in zip(self.acceptance_probabilities, self.acceptable_outcomes):
            self.mapping[o] = p
        return MappingUtilityFunction(self.mapping)

    def respond(self, state: MechanismState, offer: Outcome) -> "ResponseType":  # type: ignore
        """Respond to an offer.

        Args:
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

        """
        if not hasattr(self, "mapping"):
            self.set_preferences(self._make_preferences(self.nmi.issues))  # type: ignore
        r = random.random()
        if r < self.p_no_response:
            return ResponseType.NO_RESPONSE

        if r < self.p_ending:
            return ResponseType.END_NEGOTIATION

        if random.random() < self.mapping[offer]:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class LimitedOutcomesProposerMixin:
    """
    A mixin that adds the ability to propose an outcome from a set of outcomes randomly.
    """

    def init_limited_outcomes_proposer(
        self, proposable_outcomes: list[Outcome] | None = None
    ) -> None:
        """
        Initializes the mixin

        Args:
            proposable_outcomes: the set of prooposable outcomes. If None then it is assumed to be all the outcomes of the negotiation

        """
        self.add_capabilities(  # type: ignore
            {
                "propose": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self._offerable_outcomes = proposable_outcomes
        if proposable_outcomes is not None:
            self._offerable_outcomes = list(proposable_outcomes)

    def propose(self, state: MechanismState) -> Outcome | None:
        """Proposes one of the proposable_outcomes"""
        if self._offerable_outcomes is None:
            return self.nmi.random_outcomes(1)[0]  # type: ignore
        else:
            return random.sample(self._offerable_outcomes, 1)[0]


class LimitedOutcomesMixin(LimitedOutcomesAcceptorMixin, LimitedOutcomesProposerMixin):
    """
    A mixin that adds the ability to propose and respond a limited set of outcomes.
    """

    def init_limited_outcomes(
        self,
        acceptable_outcomes: Iterable[Outcome] | None = None,
        acceptance_probabilities: float | list[float] | None = None,
        proposable_outcomes: list[Outcome] | None = None,
        p_ending=0.0,
        p_no_response=0.0,
    ) -> None:
        """Initializes the mixin.

        Args:
            acceptable_outcomes: the set of acceptable outcomes. If None then it is assumed to be all the outcomes of
                                 the negotiation.
            acceptance_probabilities: probability of accepting each acceptable outcome. If None then it is assumed to
                                      be unity.
            proposable_outcomes: the set of outcomes from which the agent is allowed to propose. If None, then it is
                                 the same as acceptable outcomes with nonzero probability
            p_no_response: probability of refusing to respond to offers
            p_ending: probability of ending negotiation
        """
        if (
            isinstance(acceptance_probabilities, float)
            and acceptable_outcomes is not None
        ):
            acceptance_probabilities = [acceptance_probabilities] * len(
                acceptable_outcomes
            )
        self.init_limited_outcomes_acceptor(  # type: ignore
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            p_ending=p_ending,
            p_no_response=p_no_response,
        )

        if proposable_outcomes is None and self.acceptable_outcomes is not None:
            if isinstance(self.acceptance_probabilities, Iterable):
                proposable_outcomes = [
                    _
                    for _, p in zip(
                        self.acceptable_outcomes, self.acceptance_probabilities
                    )
                    if p > 1e-9
                ]
        self.init_limited_outcomes_proposer(proposable_outcomes=proposable_outcomes)
