"""
Implements basic components that can be used by `SAONegotiator` s.
"""
import random
from collections import defaultdict

from typing import (
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

from ..common import *
from ..negotiators import Negotiator
from ..outcomes import (
    Outcome,
    ResponseType,
    outcome_as_tuple,
)
from ..utilities import MappingUtilityFunction

__all__ = [
    "LimitedOutcomesAcceptorMixin",
    "LimitedOutcomesProposerMixin",
    "LimitedOutcomesMixin",
    "RandomResponseMixin",
    "RandomProposalMixin",
]


class RandomResponseMixin:
    def init_random_response(
        self,
        p_acceptance: float = 0.15,
        p_rejection: float = 0.25,
        p_ending: float = 0.1,
    ) -> None:
        """Constructor

        Args:
            p_acceptance (float): probability of accepting offers
            p_rejection (float): probability of rejecting offers
            p_ending (float): probability of ending negotiation

        Returns:
            None

        Remarks:
            - If the summation of acceptance, rejection and ending probabilities
                is less than 1.0 then with the remaining probability a
                NO_RESPONSE is returned from respond()
        """
        self.add_capabilities({"respond": True})
        self.p_acceptance = p_acceptance
        self.p_rejection = p_rejection
        self.p_ending = p_ending
        self.wheel: List[Tuple[float, ResponseType]] = [(0.0, ResponseType.NO_RESPONSE)]
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

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        r = random.random()
        for w in self.wheel:
            if w[0] >= r:
                return w[1]
        return ResponseType.NO_RESPONSE


class RandomProposalMixin:
    """The simplest possible agent.

    It just generates random offers and respond randomly to offers.
    """

    def init_random_proposal(self: Negotiator):
        self.add_capabilities(
            {
                "propose": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if (
            hasattr(self, "_offerable_outcomes")
            and self._offerable_outcomes is not None
        ):
            return random.sample(self._offerable_outcomes, 1)[0]
        return self._ami.random_outcomes(1)[0]


class LimitedOutcomesAcceptorMixin:
    """An agent the accepts a limited set of outcomes.

    The agent accepts any of the given outcomes with the given probabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acceptance_probabilities = self.acceptable_outcomes = None
        self.p_ending = self.p_no_response = self.time_factor = None

    def init_limited_outcomes_acceptor(
        self,
        acceptable_outcomes: Optional[Iterable["Outcome"]] = None,
        acceptance_probabilities: Optional[List[float]] = None,
        time_factor: Union[float, List[float]] = None,
        p_ending: float = 0.05,
        p_no_response: float = 0.0,
    ) -> None:
        """Constructor

        Args:
            acceptable_outcomes: the set of acceptable
                outcomes. If None then it is assumed to be all the outcomes of
                the negotiation.
            acceptance_probabilities: probability of accepting
                each acceptable outcome. If None then it is assumed to be unity.
            time_factor: If given, the acceptance probability will go up with time by this factor
            p_no_response: probability of refusing to respond to offers
            p_ending: probability of ending negotiation

        Returns:
            None

        """
        self.add_capabilities({"respond": True})
        self.acceptable_outcomes, self.acceptance_probabilities = (
            acceptable_outcomes,
            acceptance_probabilities,
        )
        self.p_no_response = p_no_response
        self.p_ending = p_ending + p_no_response
        self.time_factor = time_factor

    def join(self, *args, **kwargs):
        if not super().join(*args, **kwargs):
            return False

        self._make_ufun()
        return True

    def _make_ufun(self):
        """Generates a ufun that maps acceptance probability to the utility"""
        if self.acceptable_outcomes is None:
            self.acceptable_outcomes = self.ami.discrete_outcomes()

        self.acceptable_outcomes = [
            outcome_as_tuple(_) for _ in self.acceptable_outcomes
        ]

        if self.acceptance_probabilities is None:
            self.acceptance_probabilities = [1.0] * len(self.acceptable_outcomes)
        if not isinstance(self.acceptance_probabilities, Iterable):
            self.acceptance_probabilities = [self.acceptance_probabilities] * len(
                self.acceptable_outcomes
            )
        self.mapping = defaultdict(float)
        for p, o in zip(self.acceptance_probabilities, self.acceptable_outcomes):
            self.mapping[o] = p
        self.utility_function = MappingUtilityFunction(self.mapping)
        self.on_ufun_changed()

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Respond to an offer.

        Args:
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

        """
        # offer = outcome_as_tuple(offer)
        if not hasattr(self, "mapping"):
            self._make_ufun()
        r = random.random()
        if r < self.p_no_response:
            return ResponseType.NO_RESPONSE

        if r < self.p_ending:
            return ResponseType.END_NEGOTIATION

        if random.random() < self.mapping[outcome_as_tuple(offer)]:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class LimitedOutcomesProposerMixin:
    """An agent the accepts a limited set of outcomes.

    The agent proposes randomly from the given set of outcomes.

    Args:
        proposable_outcomes (Optional[Outcomes]): the set of prooposable
            outcomes. If None then it is assumed to be all the outcomes of
            the negotiation


    """

    def init_limited_outcomes_proposer(
        self: Negotiator, proposable_outcomes: Optional[List["Outcome"]] = None
    ) -> None:
        self.add_capabilities(
            {
                "propose": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self._offerable_outcomes = proposable_outcomes
        if proposable_outcomes is not None:
            self._offerable_outcomes = list(proposable_outcomes)

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self._offerable_outcomes is None:
            return self._ami.random_outcomes(1)[0]
        else:
            return random.sample(self._offerable_outcomes, 1)[0]


class LimitedOutcomesMixin(LimitedOutcomesAcceptorMixin, LimitedOutcomesProposerMixin):
    """An agent the accepts a limited set of outcomes.

    The agent accepts any of the given outcomes with the given probabilities.
    """

    def init_limited_outcomes(
        self,
        acceptable_outcomes: Optional[Iterable["Outcome"]] = None,
        acceptance_probabilities: Optional[Union[float, List[float]]] = None,
        proposable_outcomes: Optional[Iterable["Outcome"]] = None,
        p_ending=0.0,
        p_no_response=0.0,
    ) -> None:
        """Constructor

        Args:
            acceptable_outcomes (Optional[Outcomes]): the set of acceptable
                outcomes. If None then it is assumed to be all the outcomes of
                the negotiation.
            acceptance_probabilities (Sequence[Float]): probability of accepting
                each acceptable outcome. If None then it is assumed to be unity.
            proposable_outcomes (Optional[Outcomes]): the set of outcomes from which the agent is allowed
                to propose. If None, then it is the same as acceptable outcomes with nonzero probability
            p_no_response (float): probability of refusing to respond to offers
            p_ending (float): probability of ending negotiation

        Returns:
            None

        """
        self.init_limited_outcomes_acceptor(
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            p_ending=p_ending,
            p_no_response=p_no_response,
        )
        if proposable_outcomes is None and self.acceptable_outcomes is not None:
            if not isinstance(self.acceptance_probabilities, float):
                proposable_outcomes = [
                    _
                    for _, p in zip(
                        self.acceptable_outcomes, self.acceptance_probabilities
                    )
                    if p > 1e-9
                ]
        self.init_limited_outcomes_proposer(proposable_outcomes=proposable_outcomes)
