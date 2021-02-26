"""
Implements negotiators for the SAO mechanism.
"""
import math
import random
import warnings
from abc import abstractmethod
from typing import List, Optional, Type, Union

import numpy as np

from .common import SAOResponse
from .components import (
    LimitedOutcomesAcceptorMixin,
    LimitedOutcomesMixin,
    RandomProposalMixin,
    RandomResponseMixin,
)
from ..common import (
    AgentMechanismInterface,
    MechanismState,
    _ShadowAgentMechanismInterface,
)
from ..events import Notification
from ..java import (
    JavaCallerMixin,
    JNegmasGateway,
    from_java,
    java_link,
    python_identifier,
    to_dict,
    to_java,
)
from ..negotiators import AspirationMixin, Controller, Negotiator
from ..outcomes import Issue, Outcome, ResponseType, outcome_as_dict, outcome_as_tuple
from ..utilities import (
    JavaUtilityFunction,
    LinearUtilityFunction,
    MappingUtilityFunction,
    UtilityFunction,
    outcome_with_utility,
    utility_range,
)

__all__ = [
    "SAONegotiator",
    "RandomNegotiator",
    "LimitedOutcomesNegotiator",
    "LimitedOutcomesAcceptor",
    "AspirationNegotiator",
    "ToughNegotiator",
    "OnlyBestNegotiator",
    "NaiveTitForTatNegotiator",
    "SimpleTitForTatNegotiator",
    "NiceNegotiator",
    "JavaSAONegotiator",
    "PassThroughSAONegotiator",
    "_ShadowSAONegotiator",
]


class SAONegotiator(Negotiator):
    """
    Base class for all SAO negotiators.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         ufun: The ufun of the negotiator
         assume_normalized: If true, the negotiator can assume that the ufun is normalized between zreo and one.
         rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                            reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
         owner: The `Agent` that owns the negotiator.

    Remarks:
        - The only method that **must** be implemented by any SAONegotiator is `propose`.
        - The default `respond` method, accepts offers with a utility value no less than whatever `propose` returns
          with the same mechanism state.

    """

    def __init__(
        self,
        assume_normalized=True,
        ufun: Optional[UtilityFunction] = None,
        name: Optional[str] = None,
        parent: Controller = None,
        owner: "Agent" = None,
        id: Optional[str] = None,
        rational_proposal=True,
    ):
        super().__init__(name=name, ufun=ufun, parent=parent, owner=owner, id=id)
        self.assume_normalized = assume_normalized
        self.__end_negotiation = False
        self.my_last_proposal: Optional["Outcome"] = None
        self.my_last_proposal_utility: float = None
        self.rational_proposal = rational_proposal
        self.add_capabilities({"respond": True, "propose": True, "max-proposals": 1})

    def on_notification(self, notification: Notification, notifier: str):
        """
        Called whenever a notification is received

        Args:
            notification: The notification
            notifier: The notifier entity

        Remarks:
            - The default implementation only responds to end_negotiation by ending the negotiation
        """
        if notification.type == "end_negotiation":
            self.__end_negotiation = True

    def propose_(self, state: MechanismState) -> Optional["Outcome"]:
        """
        The method directly called by the mechanism (through `counter` ) to ask for a proposal

        Args:
            state: The mechanism state

        Returns:
            An outcome to offer or None to refuse to offer

        Remarks:
            - Depending on the `SAOMechanism` settings, refusing to offer may be interpreted as ending the negotiation
            - The negotiator will only receive this call if it has the 'propose' capability.
            - Rational proposal is implemented in this method.

        """
        if not self._capabilities["propose"] or self.__end_negotiation:
            return None
        if self._ufun_modified:
            self.on_ufun_changed()
        proposal = self.propose(state=state)

        # never return a proposal that is less than the reserved value
        if self.rational_proposal:
            utility = None
            if proposal is not None and self._utility_function is not None:
                utility = self._utility_function(proposal)
                if utility is not None and utility < self.reserved_value:
                    return None

            if utility is not None:
                self.my_last_proposal = proposal
                self.my_last_proposal_utility = utility

        return proposal

    def respond_(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """The method to be called directly by the mechanism (through `counter` ) to respond to an offer.

        Args:
            state: a `MechanismState` giving current state of the negotiation.
            offer: the offer being responded to.

        Returns:
            ResponseType: The response to the offer. Possible values are:

                - NO_RESPONSE: refuse to offer. Depending on the mechanism settings this may be interpreted as ending
                               the negotiation.
                - ACCEPT_OFFER: Accepting the offer.
                - REJECT_OFFER: Rejecting the offer. The negotiator will be given the chance to counter this
                                offer through a call of `propose_` later if this was not the last offer to be evaluated
                                by the mechanism.
                - END_NEGOTIATION: End the negotiation
                - WAIT: Instructs the mechanism to wait for this negotiator more. It may lead to cycles so use with care.

        Remarks:
            - The default implementation never ends the negotiation except if an earler end_negotiation notification is
              sent to the negotiator
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).

        """
        if self.__end_negotiation:
            return ResponseType.END_NEGOTIATION
        if self._ufun_modified:
            self.on_ufun_changed()
        return self.respond(state=state, offer=offer)

    def counter(
        self, state: MechanismState, offer: Optional["Outcome"]
    ) -> "SAOResponse":
        """
        Called by the mechanism to counter the offer. It just calls `respond_` and `propose_` as needed.

        Args:
            state: `MechanismState` giving current state of the negotiation.
            offer: The offer to be countered. None means no offer and the agent is requested to propose an offer

        Returns:
            Tuple[ResponseType, Outcome]: The response to the given offer with a counter offer if the response is REJECT

        """
        if self.__end_negotiation:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if offer is None:
            return SAOResponse(ResponseType.REJECT_OFFER, self.propose_(state=state))
        response = self.respond_(state=state, offer=offer)
        if response != ResponseType.REJECT_OFFER:
            return SAOResponse(response, None)
        return SAOResponse(response, self.propose_(state=state))

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Called to respond to an offer. This is the method that should be overriden to provide an acceptance strategy.

        Args:
            state: a `MechanismState` giving current state of the negotiation.
            offer: offer being tested

        Returns:
            ResponseType: The response to the offer

        Remarks:
            - The default implementation never ends the negotiation
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).

        """
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        if offer is None:
            return ResponseType.REJECT_OFFER
        if self._utility_function(offer) < self.reserved_value:
            return ResponseType.REJECT_OFFER
        utility = None
        if self.my_last_proposal_utility is not None:
            utility = self.my_last_proposal_utility
        if utility is None:
            myoffer = self.propose_(state=state)
            if myoffer is None:
                return ResponseType.NO_RESPONSE
            utility = self._utility_function(myoffer)
        if utility is not None and self._utility_function(offer) >= utility:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def on_partner_proposal(
        self, state: MechanismState, partner_id: str, offer: "Outcome"
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something

        Args:
            state: `MechanismState` giving the state of the negotiation when the offer was porposed.
            partner_id: The ID of the agent who proposed
            offer: The proposal.

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """

    def on_partner_refused_to_propose(
        self, state: MechanismState, agent_id: str
    ) -> None:
        """
        A callback called by the mechanism when a partner refuses to propose

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner refused to offer.
            agent_id: The ID of the agent who refused to propose

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """

    def on_partner_response(
        self,
        state: MechanismState,
        partner_id: str,
        outcome: "Outcome",
        response: "ResponseType",
    ) -> None:
        """
        A callback called by the mechanism when a partner responds to some offer

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner responded.
            partner_id: The ID of the agent who responded
            outcome: The proposal being responded to.
            response: The response

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """

    @abstractmethod
    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Propose an offer or None to refuse.

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """

    class Java:
        implements = ["jnegmas.sao.SAONegotiator"]


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
        - This negotiator ignores the `rational_proposal` parameter.
    """

    def propose_(self, state: MechanismState) -> Optional["Outcome"]:
        return RandomProposalMixin.propose(self, state)

    def respond_(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
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
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        """
        Will create a random utility function to be used by the negotiator.

        Args:
            ami: The AMI
            state: The current mechanism state
            ufun: IGNORED.
            role: IGNORED.
        """
        result = super().join(ami, state, ufun=ufun, role=role)
        if not result:
            return False
        if ami.issues is not None and any(_.is_continuous() for _ in ami.issues):
            self._utility_function = LinearUtilityFunction(
                weights=np.random.random(len(ami.issues))
            )
        else:
            outcomes = [outcome_as_tuple(_) for _ in ami.discrete_outcomes()]
            self._utility_function = MappingUtilityFunction(
                dict(zip(outcomes, np.random.rand(len(outcomes)))),
            )
        return True


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
        **kwargs,
    ) -> None:
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
        ufun=None,
        **kwargs,
    ) -> None:
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


class AspirationNegotiator(SAONegotiator, AspirationMixin):
    """
    Represents a time-based negotiation strategy that is independent of the offers received during the negotiation.

    Args:
        name: The agent name
        ufun:  The utility function to attache with the agent
        max_aspiration: The aspiration level to use for the first offer (or first acceptance decision).
        aspiration_type: The polynomial aspiration curve type. Here you can pass the exponent as a real value or
                         pass a string giving one of the predefined types: linear, conceder, boulware.
        dynamic_ufun: If True, the utility function will be assumed to be changing over time. This is depricated.
        randomize_offer: If True, the agent will propose outcomes with utility >= the current aspiration level not
                         outcomes just above it.
        can_propose: If True, the agent is allowed to propose
        assume_normalized: If True, the ufun will just be assumed to have the range [0, 1] inclusive
        ranking: If True, the aspiration level will not be based on the utility value but the ranking of the outcome
                 within the presorted list. It is only effective when presort is set to True
        ufun_max: The maximum utility value (used only when `presort` is True)
        ufun_min: The minimum utility value (used only when `presort` is True)
        presort: If True, the negotiator will catch a list of outcomes, presort them and only use them for offers
                 and responses. This is much faster then other option for general continuous utility functions
                 but with the obvious problem of only exploring a discrete subset of the issue space (Decided by
                 the `discrete_outcomes` property of the `AgentMechanismInterface` . If the number of outcomes is
                 very large (i.e. > 10000) and discrete, presort will be forced to be True. You can check if
                 presorting is active in realtime by checking the "presorted" attribute.
        tolerance: A tolerance used for sampling of outcomes when `presort` is set to False
        assume_normalized: If true, the negotiator can assume that the ufun is normalized.
        rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                        reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
        owner: The `Agent` that owns the negotiator.
        parent: The parent which should be an `SAOController`

    """

    def __init__(
        self,
        max_aspiration=1.0,
        aspiration_type="boulware",
        dynamic_ufun=True,
        randomize_offer=False,
        can_propose=True,
        assume_normalized=False,
        ranking=False,
        ufun_max=None,
        ufun_min=None,
        presort: bool = True,
        tolerance: float = 0.01,
        **kwargs,
    ):
        self.ordered_outcomes = []
        self.ufun_max = ufun_max
        self.ufun_min = ufun_min
        self.ranking = ranking
        self.tolerance = tolerance
        if assume_normalized:
            self.ufun_max, self.ufun_min = 1.0, 0.0
        super().__init__(
            assume_normalized=assume_normalized, **kwargs,
        )
        self.aspiration_init(
            max_aspiration=max_aspiration, aspiration_type=aspiration_type
        )
        if not dynamic_ufun:
            warnings.warn(
                "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
        self.randomize_offer = randomize_offer
        self._max_aspiration = self.max_aspiration
        self.best_outcome, self.worst_outcome = None, None
        self.presort = presort
        self.presorted = False
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self.__last_offer_util, self.__last_offer = float("inf"), None
        self.n_outcomes_to_force_presort = 10000
        self.n_trials = 1

    def on_ufun_changed(self):
        super().on_ufun_changed()
        if self.ufun is None or self._ami is None:
            self.ufun_max = self.ufun_min = None
            return
        presort = self.presort
        if (
            not presort
            and all(i.is_countable() for i in self._ami.issues)
            and Issue.num_outcomes(self._ami.issues) >= self.n_outcomes_to_force_presort
        ):
            presort = True
        if presort:
            outcomes = self._ami.discrete_outcomes()
            uvals = self.utility_function.eval_all(outcomes)
            uvals_outcomes = [
                (u, o)
                for u, o in zip(uvals, outcomes)
                if u >= self.utility_function.reserved_value
            ]
            self.ordered_outcomes = sorted(
                uvals_outcomes,
                key=lambda x: float(x[0]) if x[0] is not None else float("-inf"),
                reverse=True,
            )
            if self.assume_normalized:
                self.ufun_min, self.ufun_max = 0.0, 1.0
            elif len(self.ordered_outcomes) < 1:
                self.ufun_max = self.ufun_min = self.utility_function.reserved_value
            else:
                if self.ufun_max is None:
                    self.ufun_max = self.ordered_outcomes[0][0]

                if self.ufun_min is None:
                    # we set the minimum utility to the minimum finite value above both reserved_value
                    for j in range(len(self.ordered_outcomes) - 1, -1, -1):
                        self.ufun_min = self.ordered_outcomes[j][0]
                        if self.ufun_min is not None and self.ufun_min > float("-inf"):
                            break
                    if (
                        self.ufun_min is not None
                        and self.ufun_min < self.reserved_value
                    ):
                        self.ufun_min = self.reserved_value
        else:
            if (
                self.ufun_min is None
                or self.ufun_max is None
                or self.best_outcome is None
                or self.worst_outcome is None
            ):
                mn, mx, self.worst_outcome, self.best_outcome = utility_range(
                    self.ufun, return_outcomes=True, issues=self._ami.issues
                )
                if self.ufun_min is None:
                    self.ufun_min = mn
                if self.ufun_max is None:
                    self.ufun_max = mx

        if self.ufun_min < self.reserved_value:
            self.ufun_min = self.reserved_value
        if self.ufun_max < self.ufun_min:
            self.ufun_max = self.ufun_min

        self.presorted = presort
        self.n_trials = 10

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if self.ufun_max is None or self.ufun_min is None:
            self.on_ufun_changed()
        if (
            self._utility_function is None
            or self.ufun_max is None
            or self.ufun_min is None
        ):
            return ResponseType.REJECT_OFFER
        u = self._utility_function(offer)
        if u is None or u < self.reserved_value:
            return ResponseType.REJECT_OFFER
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if u >= asp and u > self.reserved_value:
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.ufun_max is None or self.ufun_min is None:
            self.on_ufun_changed()
        if (
            self._utility_function is None
            or self.ufun_max is None
            or self.ufun_min is None
        ):
            return None
        if self.ufun_max < self.reserved_value:
            return None
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if asp < self.reserved_value:
            return None
        if self.presorted:
            if len(self.ordered_outcomes) < 1:
                return None
            for i, (u, _) in enumerate(self.ordered_outcomes):
                if u is None:
                    continue
                if u < asp:
                    if u < self.reserved_value:
                        return None
                    if i == 0:
                        return self.ordered_outcomes[i][1]
                    if self.randomize_offer:
                        return random.sample(self.ordered_outcomes[:i], 1)[0][1]
                    return self.ordered_outcomes[i - 1][1]
            if self.randomize_offer:
                return random.sample(self.ordered_outcomes, 1)[0][1]
            return self.ordered_outcomes[-1][1]
        else:
            if asp >= 0.99999999999 and self.best_outcome is not None:
                return self.best_outcome
            if self.randomize_offer:
                return outcome_with_utility(
                    ufun=self._utility_function,
                    rng=(asp, float("inf")),
                    issues=self._ami.issues,
                )
            tol = self.tolerance
            for _ in range(self.n_trials):
                rng = self.ufun_max - self.ufun_min
                mx = min(asp + tol * rng, self.__last_offer_util)
                outcome = outcome_with_utility(
                    ufun=self._utility_function, rng=(asp, mx), issues=self._ami.issues,
                )
                if outcome is not None:
                    break
                tol = math.sqrt(tol)
            else:
                outcome = (
                    self.best_outcome
                    if self.__last_offer is None
                    else self.__last_offer
                )
            self.__last_offer_util = self.utility_function(outcome)
            self.__last_offer = outcome
            return outcome


class NiceNegotiator(SAONegotiator, RandomProposalMixin):
    """
    Offers and accepts anything.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         ufun: The ufun of the negotiator
         assume_normalized: If true, the negotiator can assume that the ufun is normalized.
         rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                            reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        SAONegotiator.__init__(self, *args, **kwargs)
        self.init_random_proposal()

    def respond_(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return ResponseType.ACCEPT_OFFER

    def propose_(self, state: MechanismState) -> Optional["Outcome"]:
        return RandomProposalMixin.propose(self, state)

    propose = propose_


class ToughNegotiator(SAONegotiator):
    """
    Accepts and proposes only the top offer (i.e. the one with highest utility).

    Args:
         name: Negotiator name
         parent: Parent controller if any
         dynamic_ufun: If `True`, assumes a dynamic ufun that can change during the negotiation
         can_propose: If `False` the negotiator will never propose but can only accept
         ufun: The ufun of the negotiator
         rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                            reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
         owner: The `Agent` that owns the negotiator.

    Remarks:
        - If there are multiple outcome with the same maximum utility, only one of them will be used.

    """

    def __init__(
        self,
        name=None,
        parent: Controller = None,
        dynamic_ufun=True,
        can_propose=True,
        ufun=None,
        **kwargs,
    ):
        super().__init__(name=name, parent=parent, ufun=ufun, **kwargs)
        self.best_outcome = None
        self._offerable_outcomes = None
        if not dynamic_ufun:
            warnings.warn(
                "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def on_ufun_changed(self):
        super().on_ufun_changed()
        if self._utility_function is None:
            return
        _, _, _, self.best_outcome = utility_range(
            self.utility_function, issues=self.ami.issues, return_outcomes=True,
        )

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if offer == self.best_outcome:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if not self._capabilities["propose"]:
            return None
        return self.best_outcome


class OnlyBestNegotiator(SAONegotiator):
    """
    Offers and accepts only one of the top outcomes for the negotiator.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         dynamic_ufun: If `True`, assumes a dynamic ufun that can change during the negotiation
         can_propose: If `False` the negotiator will never propose but can only accept
         ufun: The ufun of the negotiator
         min_utility: The minimum utility to offer or accept
         top_fraction: The fraction of the outcomes (ordered decreasingly by utility) to offer or accept
         best_first: Guarantee offering will non-increasing in terms of utility value
         probabilistic_offering: Offer randomly from the outcomes selected based on `top_fraction` and `min_utility`
         rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                            reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
         owner: The `Agent` that owns the negotiator.
    """

    def __init__(
        self,
        name=None,
        parent: Controller = None,
        dynamic_ufun=True,
        min_utility=0.95,
        top_fraction=0.05,
        best_first=False,
        probabilistic_offering=True,
        can_propose=True,
        ufun=None,
        **kwargs,
    ):
        self._offerable_outcomes = None
        self.best_outcome = []
        self.ordered_outcomes = []
        self.acceptable_outcomes = []
        self.wheel = np.array([])
        self.offered = set([])
        super().__init__(name=name, parent=parent, ufun=ufun, **kwargs)
        if not dynamic_ufun:
            warnings.warn(
                "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
        self.top_fraction = top_fraction
        self.min_utility = min_utility
        self.best_first = best_first
        self.probabilistic_offering = probabilistic_offering
        self.add_capabilities(
            {
                "respond": True,
                "propose": can_propose,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def on_ufun_changed(self):
        super().on_ufun_changed()
        outcomes = (
            self._ami.discrete_outcomes()
            if self._offerable_outcomes is None
            else self._offerable_outcomes
        )
        eu_outcome = list(zip(self.utility_function.eval_all(outcomes), outcomes))
        self.ordered_outcomes = sorted(eu_outcome, key=lambda x: x[0], reverse=True)
        if self.min_utility is None:
            selected, selected_utils = [], []
        else:
            util_limit = self.min_utility * self.ordered_outcomes[0][0]
            selected, selected_utils = [], []
            for u, o in self.ordered_outcomes:
                if u >= util_limit:
                    selected.append(o)
                    selected_utils.append(u)
                else:
                    break
        if self.top_fraction is not None:
            frac_limit = max(1, round(self.top_fraction * len(self.ordered_outcomes)))
        else:
            frac_limit = len(outcomes)

        if frac_limit >= len(selected) > 0:
            sum = np.asarray(selected_utils).sum()
            if sum > 0.0:
                selected_utils /= sum
                selected_utils = np.cumsum(selected_utils)
            else:
                selected_utils = np.linspace(0.0, 1.0, len(selected_utils))
            self.acceptable_outcomes, self.wheel = selected, selected_utils
            return
        if frac_limit > 0:
            n_sel = len(selected)
            fsel = [_[1] for _ in self.ordered_outcomes[n_sel:frac_limit]]
            futil = [_[0] for _ in self.ordered_outcomes[n_sel:frac_limit]]
            selected_utils = selected_utils + futil
            sum = np.asarray(selected_utils).sum()
            if sum > 0.0:
                selected_utils /= sum
                selected_utils = np.cumsum(selected_utils)
            else:
                selected_utils = np.linspace(0.0, 1.0, len(selected_utils))
            self.acceptable_outcomes, self.wheel = selected + fsel, selected_utils
            return
        self.acceptable_outcomes, self.wheel = [], []
        return

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if offer in self.acceptable_outcomes:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if not self._capabilities["propose"]:
            return None
        if self.best_first:
            for o in self.acceptable_outcomes:
                if o not in self.offered:
                    self.offered.add(o)
                    return o
        if len(self.acceptable_outcomes) > 0:
            if self.probabilistic_offering:
                r = random.random()
                for o, w in zip(self.acceptable_outcomes, self.wheel):
                    if w > r:
                        return o
                return random.sample(self.acceptable_outcomes, 1)[0]
            return random.sample(self.acceptable_outcomes, 1)[0]
        return None


class NaiveTitForTatNegotiator(SAONegotiator):
    """
    Implements a naive tit-for-tat strategy that does not depend on the availability of an opponent model.

    Args:
        name: Negotiator name
        ufun: negotiator ufun
        parent: A controller
        kindness: How 'kind' is the agent. A value of zero is standard tit-for-tat. Positive values makes the negotiator
                  concede faster and negative values slower.
        randomize_offer: If `True`, the offers will be randomized above the level determined by the current concession
                        which in turn reflects the opponent's concession.
        always_concede: If `True` the agent will never use a negative concession rate
        initial_concession: How much should the agent concede in the beginning in terms of utility. Should be a number
                            or the special string value 'min' for minimum concession

    Remarks:
        - This negotiator does not keep an opponent model. It thinks only in terms of changes in its own utility.
          If the opponent's last offer was better for the negotiator compared with the one before it, it considers
          that the opponent has conceded by the difference. This means that it implicitly assumes a zero-sum
          situation.
    """

    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        ufun: Optional["UtilityFunction"] = None,
        kindness=0.0,
        randomize_offer=False,
        always_concede=True,
        initial_concession: Union[float, str] = "min",
        **kwargs,
    ):
        self.received_utilities = []
        self.proposed_utility = None
        self.ordered_outcomes = None
        self.sent_offer_index = None
        self.n_sent = 0
        super().__init__(name=name, ufun=ufun, parent=parent, **kwargs)
        self.kindness = kindness
        self.initial_concession = initial_concession
        self.randomize_offer = randomize_offer
        self.always_concede = always_concede

    def on_ufun_changed(self):
        super().on_ufun_changed()
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        offered_utility = self._utility_function(offer)
        if len(self.received_utilities) < 2:
            self.received_utilities.append(offered_utility)
        else:
            self.received_utilities[0] = self.received_utilities[1]
            self.received_utilities[-1] = offered_utility
        indx = self._propose(state=state)
        my_utility, _ = self.ordered_outcomes[indx]
        if offered_utility >= my_utility:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _outcome_just_below(self, ulevel: float) -> int:
        for i, (u, o) in enumerate(self.ordered_outcomes):
            if u is None:
                continue
            if u < ulevel:
                if self.randomize_offer:
                    return random.randint(0, i)
                return i
        if self.randomize_offer:
            return random.randint(0, len(self.ordered_outcomes) - 1)
        return -1

    def _propose(self, state: MechanismState) -> int:
        if self.proposed_utility is None:
            return 0
        if len(self.received_utilities) < 2:
            if (
                isinstance(self.initial_concession, str)
                and self.initial_concession == "min"
            ):
                return self._outcome_just_below(ulevel=self.ordered_outcomes[0][0])
            else:
                asp = self.ordered_outcomes[0][0] * (1.0 - self.initial_concession)
            return self._outcome_just_below(ulevel=asp)

        if self.always_concede:
            opponent_concession = max(
                0.0, self.received_utilities[1] - self.received_utilities[0]
            )
        else:
            opponent_concession = (
                self.received_utilities[1] - self.received_utilities[0]
            )
        indx = self._outcome_just_below(
            ulevel=self.proposed_utility
            - opponent_concession
            - self.kindness * max(0.0, opponent_concession)
        )
        return indx

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        indx = self._propose(state)
        self.proposed_utility = self.ordered_outcomes[indx][0]
        return self.ordered_outcomes[indx][1]


class PassThroughSAONegotiator(SAONegotiator):
    """
    A negotiator that acts as an end point to a parent Controller.

    This negotiator simply calls its controler for everything.
    """

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Calls parent controller"""
        if self._Negotiator__parent:
            return self._Negotiator__parent.propose(self.id, state)

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Calls parent controller"""
        if self._Negotiator__parent:
            return self._Negotiator__parent.respond(self.id, state, offer)
        return ResponseType.REJECT_OFFER

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Calls parent controller"""
        if self._Negotiator__parent:
            return self._Negotiator__parent.on_negotiation_start(self.id, state)

    def on_negotiation_end(self, state: MechanismState) -> None:
        """Calls parent controller"""
        if self._Negotiator__parent:
            return self._Negotiator__parent.on_negotiation_end(self.id, state)

    def join(self, ami, state, *, ufun=None, role="agent",) -> bool:
        """
        Joins a negotiation.

        Remarks:

            This method first gets permission from the parent controller by
            calling `before_join` on it and confirming the result is `True`,
            it then joins the negotiation and calls `after_join` of the
            controller to inform it that joining is completed if joining was
            successful.
        """
        permission = (
            self._Negotiator__parent is None
            or self._Negotiator__parent.before_join(
                self.id, ami, state, ufun=ufun, role=role
            )
        )
        if not permission:
            return False
        if super().join(ami, state, ufun=ufun, role=role):
            if self._Negotiator__parent:
                self._Negotiator__parent.after_join(
                    self.id, ami, state, ufun=ufun, role=role
                )
            return True
        return False


def _to_java_response(response: ResponseType) -> int:
    if response == ResponseType.ACCEPT_OFFER:
        return 0
    if response == ResponseType.REJECT_OFFER:
        return 1
    if response == ResponseType.END_NEGOTIATION:
        return 2
    if response == ResponseType.NO_RESPONSE:
        return 3
    raise ValueError(f"Unknown response{response}")


def _from_java_response(response: int) -> ResponseType:
    if response == 0:
        return ResponseType.ACCEPT_OFFER
    if response == 1:
        return ResponseType.REJECT_OFFER
    if response == 2:
        return ResponseType.END_NEGOTIATION
    if response == 3:
        return ResponseType.NO_RESPONSE
    raise ValueError(
        f"Unknown response type {response} returned from the Java underlying negotiator"
    )


class JavaSAONegotiator(SAONegotiator, JavaCallerMixin):
    """
    Represents a negotiator running on JNegMAS (depricated)


    Args:
        java_object: The Java object if known
        java_class_name: Name of the java negotiator class
        auto_load_java: If true, jnemgas will be loaded if not running.
        outcome_type: Type of outcome to use

    Remarks:
        - You cannot pass a ufun here.
    """

    def __init__(
        self,
        java_object,
        java_class_name: Optional[str],
        auto_load_java: bool = False,
        outcome_type: Type = dict,
    ):
        if java_class_name is not None or java_object is not None:
            self.init_java_bridge(
                java_object=java_object,
                java_class_name=java_class_name,
                auto_load_java=auto_load_java,
                python_shadow_object=None,
            )
            if java_object is None:
                self._java_object.fromMap(to_java(self))
        d = {
            python_identifier(k): v
            for k, v in JNegmasGateway.gateway.entry_point.toMap(
                self._java_object
            ).items()
        }
        ufun = d.get("utility_function", None)
        ufun = JavaUtilityFunction(ufun, None) if ufun is not None else None
        super().__init__(
            name=d.get("name", None),
            assume_normalized=d.get("assume_normalized", False),
            ufun=ufun,
            rational_proposal=d.get("rational_proposal", True),
            parent=d.get("parent", None),
        )
        self._outcome_type = outcome_type
        self.add_capabilities(
            {
                "respond": True,
                "propose": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )

    def on_partner_proposal(
        self, state: MechanismState, partner_id: str, offer: "Outcome"
    ) -> None:
        self._java_object.onPartnerProposal(to_java(state), partner_id, to_java(offer))

    def on_partner_refused_to_propose(
        self, state: MechanismState, agent_id: str
    ) -> None:
        self._java_object.onPartnerRefusedToPropose(to_java(state), agent_id)

    def on_partner_response(
        self,
        state: MechanismState,
        partner_id: str,
        outcome: "Outcome",
        response: "ResponseType",
    ) -> None:
        self._java_object.onPartnerResponse(
            to_java(state),
            partner_id,
            to_java(outcome),
            self._to_java_response(response),
        )

    def isin(self, negotiation_id: Optional[str]) -> bool:
        return self._java_object.isIn(negotiation_id)

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        return self._java_object.join(
            java_link(_ShadowAgentMechanismInterface(ami)), to_java(state), ufun, role
        )

    def on_negotiation_start(self, state: MechanismState) -> None:
        self._java_object.onNegotiationStart(to_java(state))

    def on_round_start(self, state: MechanismState) -> None:
        self._java_object.onRoundStart(to_java(state))

    def on_mechanism_error(self, state: MechanismState) -> None:
        self._java_object.onMechanismError(to_java(state))

    def on_round_end(self, state: MechanismState) -> None:
        self._java_object.onRoundEnd(to_java(state))

    def on_leave(self, state: MechanismState) -> None:
        self._java_object.onLeave(to_java(state))

    def on_negotiation_end(self, state: MechanismState) -> None:
        self._java_object.onNegotiationEnd(to_java(state))

    def on_ufun_changed(self):
        self._java_object.onUfunChanged()

    @classmethod
    def from_dict(
        cls, java_object, *args, parent: Controller = None, **kwargs
    ) -> "JavaSAONegotiator":
        """Creates a Java negotiator from an object returned from the JVM implementing PySAONegotiator"""
        ufun = java_object.getUtilityFunction()
        if ufun is not None:
            ufun = JavaUtilityFunction.from_dict(java_object=ufun)
        return JavaCallerMixin.from_dict(
            java_object,
            name=java_object.getName(),
            assume_normalized=java_object.getAssumeNormalized(),
            rational_proposal=java_object.getRationalProposal(),
            parent=parent,
            ufun=ufun,
        )

    def on_notification(self, notification: Notification, notifier: str):
        super().on_notification(notification=notification, notifier=notifier)
        jnotification = {"type": notification.type, "data": to_java(notification.data)}
        self._java_object.on_notification(jnotification, notifier)

    def respond(self, state: MechanismState, offer: "Outcome"):
        return _from_java_response(
            self._java_object.respond(to_java(state), outcome_as_dict(offer))
        )

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        outcome = from_java(self._java_object.propose(to_java(state)))
        if outcome is None:
            return None
        if self._outcome_type == dict:
            return outcome
        if self._outcome_type == tuple:
            return tuple(outcome.values())
        return self._outcome_type(outcome)

    # class Java:
    #    implements = ['jnegmas.sao.SAONegotiator']


class _ShadowSAONegotiator:
    """A python shadow to a java negotiator"""

    class Java:
        implements = ["jnegmas.sao.SAONegotiator"]

    def to_java(self):
        return to_dict(self.shadow)

    def __init__(self, negotiator: SAONegotiator):
        self.shadow = negotiator

    def respond(self, state, outcome):
        return _to_java_response(
            self.shadow.respond(from_java(state), from_java(outcome))
        )

    def propose(self, state):
        return to_java(self.shadow.propose(from_java(state)))

    def isIn(self, negotiation_id):
        return to_java(self.shadow.isin(negotiation_id=negotiation_id))

    def join(self, ami, state, ufun, role):
        return to_java(
            self.shadow.join(
                ami=from_java(ami),
                state=from_java(state),
                ufun=JavaUtilityFunction(ufun, None) if ufun is not None else None,
                role=role,
            )
        )

    def onNegotiationStart(self, state):
        return to_java(self.shadow.on_negotiation_start(from_java(state)))

    def onNegotiationEnd(self, state):
        return to_java(self.shadow.on_negotiation_end(from_java(state)))

    def onMechanismError(self, state):
        return to_java(self.shadow.on_mechanism_error(from_java(state)))

    def onRoundStart(self, state):
        return to_java(self.shadow.on_round_start(from_java(state)))

    def onRoundEnd(self, state):
        return to_java(self.shadow.on_round_end(from_java(state)))

    def onLeave(self, state):
        return to_java(self.shadow.on_leave(from_java(state)))

    def onPartnerProposal(self, state, agent_id, offer):
        return to_java(
            self.shadow.on_partner_proposal(
                state=from_java(state), partner_id=agent_id, offer=from_java(offer)
            )
        )

    def onUfunChanged(self):
        return to_java(self.shadow.on_ufun_changed())

    def onPartnerRefusedToPropose(self, state, agent_id):
        return to_java(
            self.shadow.on_partner_refused_to_propose(
                state=from_java(state), agent_id=agent_id
            )
        )

    def onPartnerResponse(self, state, agent_id, offer, response: int, counter_offer):
        # TODO check that jnemgas is expecting ResponseType not SAOResponse here.
        return to_java(
            self.shadow.on_partner_response(
                state=from_java(state),
                partner_id=agent_id,
                outcome=from_java(offer),
                response=ResponseType(response),
            )
        )

    def onNotification(self, notification, notifier):
        return to_java(self.shadow.on_notification(from_java(notification), notifier))

    def setUtilityFunction(self, ufun):
        self.shadow.utility_function = (
            ufun if ufun is None else JavaUtilityFunction(ufun, None)
        )

    def getUtilityFunction(self):
        return to_java(self.shadow._utility_function)

    def getID(self):
        return self.shadow.id

    def setID(self):
        return self.shadow.id


SimpleTitForTatNegotiator = NaiveTitForTatNegotiator
"""A simple tit-for-tat negotiator"""
