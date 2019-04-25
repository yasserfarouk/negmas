"""
Implements Stacked Alternating Offers (SAO) mechanism and basic negotiators.
"""
import random
import time
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Callable, Type
from typing import Sequence, Optional, List, Tuple, Iterable, Union

import numpy as np

from negmas.common import *
from negmas.common import _ShadowAgentMechanismInterface
from negmas.events import Notification
from negmas.java import (
    JavaCallerMixin,
    to_java,
    from_java,
    java_link,
    to_dict,
    JNegmasGateway,
    python_identifier,
)
from negmas.mechanisms import MechanismRoundResult, Mechanism
from negmas.negotiators import Negotiator, AspirationMixin, Controller
from negmas.outcomes import (
    sample_outcomes,
    Outcome,
    outcome_is_valid,
    ResponseType,
    outcome_as_dict,
    outcome_as_tuple,
)
from negmas.utilities import (
    MappingUtilityFunction,
    normalize,
    UtilityFunction,
    UtilityValue,
    JavaUtilityFunction,
)

__all__ = [
    "SAOState",
    "SAOMechanism",
    "SAOProtocol",
    "SAONegotiator",
    "RandomNegotiator",
    "LimitedOutcomesNegotiator",
    "LimitedOutcomesAcceptor",
    "AspirationNegotiator",
    "ToughNegotiator",
    "OnlyBestNegotiator",
    "NaiveTitForTatNegotiator",
    "SimpleTitForTatNegotiator",  # @todo remove this in future versions
    "NiceNegotiator",
    "SAOController",
    "JavaSAONegotiator",
]


@dataclass
class SAOResponse:
    """A response to an offer given by an agent in the alternating offers protocol"""

    response: ResponseType = ResponseType.NO_RESPONSE
    outcome: Optional["Outcome"] = None


@dataclass
class SAOState(MechanismState):
    current_offer: Optional["Outcome"] = None
    current_proposer: Optional[str] = None
    n_acceptances: int = 0


@dataclass
class SAOAMI(AgentMechanismInterface):
    end_on_no_response: bool = True
    publish_proposer: bool = True
    publish_n_acceptances: bool = False


class SAOMechanism(Mechanism):
    def __init__(
        self,
        issues=None,
        outcomes=None,
        n_steps=None,
        time_limit=None,
        step_time_limit=None,
        max_n_agents=None,
        dynamic_entry=True,
        keep_issue_names=True,
        cache_outcomes=True,
        max_n_outcomes: int = 1000000,
        annotation: Optional[Dict[str, Any]] = None,
        end_on_no_response=True,
        publish_proposer=True,
        publish_n_acceptances=False,
        enable_callbacks=False,
        avoid_ultimatum=True,
        name: Optional[str] = None,
    ):
        super().__init__(
            issues=issues,
            outcomes=outcomes,
            n_steps=n_steps,
            time_limit=time_limit,
            step_time_limit=step_time_limit,
            max_n_agents=max_n_agents,
            dynamic_entry=dynamic_entry,
            keep_issue_names=keep_issue_names,
            cache_outcomes=cache_outcomes,
            max_n_outcomes=max_n_outcomes,
            annotation=annotation,
            state_factory=SAOState,
            enable_callbacks=enable_callbacks,
            name=name,
        )
        self._current_offer = None
        self._current_proposer = None
        self._current_proposer_index = -1
        self._n_accepting = 0
        self._avoid_ultimatum = n_steps is not None and avoid_ultimatum
        self.end_negotiation_on_refusal_to_propose = end_on_no_response
        self.publish_proposer = publish_proposer
        self.publish_n_acceptances = publish_n_acceptances

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        if not super().join(ami, state, ufun=ufun, role=role):
            return False
        if not self.ami.dynamic_entry and not any(
            [a.capabilities.get("propose", False) for a in self.negotiators]
        ):
            self._current_proposer = None
            self._current_proposer_index = -1
            self._current_offer = None
            self._n_accepting = 0
            return False
        return True

    def extra_state(self):
        return SAOState(
            current_offer=self._current_offer,
            current_proposer=self._current_proposer.id
            if self._current_proposer and self.publish_proposer
            else None,
            n_acceptances=self._n_accepting if self.publish_n_acceptances else 0,
        )

    def round(self) -> MechanismRoundResult:
        """implements a round of the Stacked Alternating Offers Protocol.


        """
        negotiators: List[SAONegotiator] = self.negotiators
        n_negotiators = len(negotiators)

        # if this is the first step which means that there is no _current_offer
        if self.ami.state.step == 0:
            assert self._current_offer is None
            assert self._current_proposer is None
            assert self._current_proposer_index == -1

            # choose a random negotiator and set it as the current negotiator
            _first_proposer = random.randint(0, n_negotiators - 1)
            if self._avoid_ultimatum:
                # if we are trying to avoid an ultimatum, we take an offer from everyone and ignore them but one.
                # this way, the agent cannot know its order. For example, if we have two agents and 3 steps, this will
                # be the situation after each step:
                #
                # Case 1: Assume that it ignored the offer from agent 1
                # Step, Agent 0 calls received  , Agent 1 calls received    , relative time during last call
                # 0   , counter(None)->offer1*  , counter(None) -> offer0   , 0/3
                # 1   , counter(offer2)->offer3 , counter(offer1) -> offer2 , 1/3
                # 2   , counter(offer4)->offer5 , counter(offer3) -> offer4 , 2/3
                # 3   ,                         , counter(offer5)->offer6   , 3/3
                #
                # Case 2: Assume that it ignored the offer from agent 0
                # Step, Agent 0 calls received  , Agent 1 calls received    , relative time during last call
                # 0   , counter(None)->offer1   , counter(None) -> offer0*  , 0/3
                # 1   , counter(offer0)->offer2 , counter(offer2) -> offer3 , 1/3
                # 2   , counter(offer3)->offer4 , counter(offer4) -> offer5 , 2/3
                # 3   , counter(offer5)->offer6 ,                           , 3/3
                #
                # in both cases, the agent cannot know whether its last offer going to be passed to the other agent
                # (the ultimatum scenario) or not.
                responses = []
                for neg in negotiators:
                    if not neg.capabilities.get("propose", False):
                        continue
                    strt = time.perf_counter()
                    resp = neg.counter(state=self.state, offer=None)
                    if (
                        self.ami.step_time_limit is not None
                        and time.perf_counter() - strt > self.ami.step_time_limit
                    ):
                        return MechanismRoundResult(
                            broken=False, timedout=True, agreement=None
                        )
                    if self._enable_callbacks:
                        for other in self.negotiators:
                            if other is not neg:
                                other.on_partner_response(
                                    state=self.state,
                                    agent_id=neg.id,
                                    outcome=None,
                                    response=resp,
                                )
                    if (
                        resp.response == ResponseType.END_NEGOTIATION
                        or resp.response == ResponseType.NO_RESPONSE
                        or (
                            resp.outcome is None
                            and neg.capabilities.get("propose", False)
                        )
                    ):
                        return MechanismRoundResult(
                            broken=True, timedout=False, agreement=None
                        )
                    responses.append(resp)
                if len(responses) < 1:
                    if not self.dynamic_entry:
                        return MechanismRoundResult(
                            broken=True,
                            timedout=False,
                            agreement=None,
                            error=True,
                            error_details="No proposers and no dynamic entry",
                        )
                    else:
                        return MechanismRoundResult(
                            broken=False, timedout=False, agreement=None
                        )
                resp = (
                    responses[_first_proposer]
                    if len(responses) > _first_proposer
                    else responses[0]
                )
                neg = (
                    negotiators[_first_proposer]
                    if len(negotiators) > _first_proposer
                    else negotiators[0]
                )
            else:
                # when there is no risk of ultimatum (n_steps is not known), we just take one first offer.
                neg = negotiators[_first_proposer]
                strt = time.perf_counter()
                resp = neg.counter(state=self.state, offer=None)
                if (
                    self.ami.step_time_limit is not None
                    and time.perf_counter() - strt > self.ami.step_time_limit
                ):
                    return MechanismRoundResult(
                        broken=False, timedout=True, agreement=None
                    )
                if self._enable_callbacks:
                    for other in self.negotiators:
                        if other is not neg:
                            other.on_partner_response(
                                state=self.state,
                                agent_id=neg.id,
                                outcome=None,
                                response=resp,
                            )
                if (
                    resp.response == ResponseType.END_NEGOTIATION
                    or resp.response == ResponseType.NO_RESPONSE
                    or resp.outcome is None
                ):
                    return MechanismRoundResult(
                        broken=True, timedout=False, agreement=None
                    )
            self._n_accepting = 1
            self._current_offer = resp.outcome
            self._current_proposer = neg
            self._current_proposer_index = _first_proposer
            return MechanismRoundResult(broken=False, timedout=False, agreement=None)

        # this is not the first round. A round will get n_negotiators steps
        ordered_indices = [
            (_ + self._current_proposer_index + 1) % n_negotiators
            for _ in range(n_negotiators)
        ]
        for neg_indx in ordered_indices:
            neg = self.negotiators[neg_indx]
            strt = time.perf_counter()
            resp = neg.counter(state=self.state, offer=self._current_offer)
            if (
                self.ami.step_time_limit is not None
                and time.perf_counter() - strt > self.ami.step_time_limit
            ):
                return MechanismRoundResult(broken=False, timedout=True, agreement=None)
            if self._enable_callbacks:
                for other in self.negotiators:
                    if other is not neg:
                        other.on_partner_response(
                            state=self.state,
                            agent_id=neg.id,
                            outcome=self._current_offer,
                            response=resp,
                        )
            if resp.response == ResponseType.NO_RESPONSE:
                continue
            if resp.response == ResponseType.END_NEGOTIATION:
                return MechanismRoundResult(broken=True, timedout=False, agreement=None)
            if resp.response == ResponseType.ACCEPT_OFFER:
                self._n_accepting += 1
                if self._n_accepting == n_negotiators:
                    return MechanismRoundResult(
                        broken=False, timedout=False, agreement=self._current_offer
                    )
            if resp.response == ResponseType.REJECT_OFFER:
                proposal = resp.outcome
                if proposal is None:
                    if self.end_negotiation_on_refusal_to_propose:
                        return MechanismRoundResult(
                            broken=True, timedout=False, agreement=None
                        )
                    elif self._enable_callbacks:
                        for other in self.negotiators:
                            if other is not neg:
                                other.on_partner_refused_to_propose(
                                    agent_id=neg, state=self.state
                                )
                        continue
                else:
                    self._current_offer = proposal
                    self._current_proposer = neg
                    self._current_proposer_index = neg_indx
                    self._n_accepting = 1
                    if self._enable_callbacks:
                        for other in self.negotiators:
                            if other is neg:
                                continue
                            other.on_partner_proposal(
                                agent_id=neg.id, offer=proposal, state=self.state
                            )
                    return MechanismRoundResult(
                        broken=False, timedout=False, agreement=None
                    )
        # we can arrive here only if all agents either refused to response or to offer.
        if not self.ami.dynamic_entry:
            raise RuntimeError(
                "No negotiators can propose. I cannot run a meaningful negotiation"
            )
        return MechanismRoundResult(
            broken=False,
            timedout=False,
            agreement=None,
            error=True,
            error_details="No negotiators can propose in a static_entry"
            " negotiation!!",
        )


class RandomResponseMixin(object):
    def init_ranodm_response(
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
        if not hasattr(self, "add_capabilities"):
            raise ValueError(
                f"self.__class__.__name__ is just a mixin for class Negotiator. You must inherit from Negotiator or"
                f" one of its descendents before inheriting from self.__class__.__name__"
            )
        self.add_capabilities({"respond": True})
        self.p_acceptance = p_acceptance
        self.p_rejection = p_rejection
        self.p_ending = p_ending
        self.wheel: List[Tuple[float, ResponseType]] = [(0.0, ResponseType.NO_RESPONSE)]
        if self.p_acceptance > 0.0:
            self.wheel = [
                (self.wheel[-1][0] + self.p_acceptance, ResponseType.ACCEPT_OFFER)
            ]
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

    # noinspection PyUnusedLocal,PyUnusedLocal
    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        r = random.random()
        for w in self.wheel:
            if w[0] >= r:
                return w[1]

        return ResponseType.NO_RESPONSE


# noinspection PyAttributeOutsideInit
class RandomProposalMixin(object):
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
        return self._ami.random_outcomes(1, astype=dict)[0]


class LimitedOutcomesAcceptorMixin(object):
    """An agent the accepts a limited set of outcomes.

    The agent accepts any of the given outcomes with the given probabilities.
    """

    def init_limited_outcomes_acceptor(
        self,
        outcomes: Optional[Union[int, Iterable["Outcome"]]] = None,
        acceptable_outcomes: Optional[Iterable["Outcome"]] = None,
        acceptance_probabilities: Optional[List[float]] = None,
        p_ending=0.05,
        p_no_response=0.0,
    ) -> None:
        """Constructor

        Args:
            acceptable_outcomes (Optional[Floats]): the set of acceptable
                outcomes. If None then it is assumed to be all the outcomes of
                the negotiation.
            acceptance_probabilities (Sequence[int]): probability of accepting
                each acceptable outcome. If None then it is assumed to be unity.
            p_no_response (float): probability of refusing to respond to offers
            p_ending (float): probability of ending negotiation

        Returns:
            None

        """
        self.add_capabilities({"respond": True})
        if acceptable_outcomes is not None and outcomes is None:
            raise ValueError(
                "If you are passing acceptable outcomes explicitly then outcomes must also be passed"
            )
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        self.outcomes = outcomes
        if acceptable_outcomes is not None:
            acceptable_outcomes = list(acceptable_outcomes)
        self.acceptable_outcomes = acceptable_outcomes
        if acceptable_outcomes is None:
            if acceptance_probabilities is None:
                acceptance_probabilities = [0.5] * len(outcomes)
                self.acceptable_outcomes = outcomes
            else:
                self.acceptable_outcomes = outcomes
        elif acceptance_probabilities is None:
            acceptance_probabilities = [1.0] * len(acceptable_outcomes)
            if outcomes is None:
                self.outcomes = [(_,) for _ in range(len(acceptable_outcomes))]
        if self.outcomes is None:
            raise ValueError(
                "Could not calculate all the outcomes. It is needed to assign a utility function"
            )
        self.acceptance_probabilities = acceptance_probabilities
        u = [0.0] * len(self.outcomes)
        for p, o in zip(self.acceptance_probabilities, self.acceptable_outcomes):
            u[self.outcomes.index(o)] = p
        self._utility_function = MappingUtilityFunction(dict(zip(self.outcomes, u)))
        self.p_no_response = p_no_response
        self.p_ending = p_ending + p_no_response

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """Respond to an offer.

        Args:
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

        """
        ami = self._ami
        r = random.random()
        if r < self.p_no_response:
            return ResponseType.NO_RESPONSE

        if r < self.p_ending:
            return ResponseType.END_NEGOTIATION

        if self.acceptable_outcomes is None:
            if (
                outcome_is_valid(offer, ami.issues)
                and random.random() < self.acceptance_probabilities
            ):
                return ResponseType.ACCEPT_OFFER

            else:
                return ResponseType.REJECT_OFFER

        try:
            indx = self.acceptable_outcomes.index(offer)
        except ValueError:
            return ResponseType.REJECT_OFFER
        prob = self.acceptance_probabilities[indx]
        if indx < 0 or not outcome_is_valid(offer, ami.issues):
            return ResponseType.REJECT_OFFER

        if random.random() < prob:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class LimitedOutcomesProposerMixin(object):
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
        outcomes: Optional[Union[int, Iterable["Outcome"]]] = None,
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
            outcomes=outcomes,
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


class SAONegotiator(Negotiator):
    def __init__(
        self,
        assume_normalized=True,
        ufun: Optional[UtilityFunction] = None,
        name: Optional[str] = None,
        rational_proposal=True,
        parent: Controller = None,
    ):
        super().__init__(name=name, ufun=ufun, parent=parent)
        self.assume_normalized = assume_normalized
        self.__end_negotiation = False
        self.my_last_proposal: Optional["Outcome"] = None
        self.my_last_proposal_utility: float = None
        self.rational_proposal = rational_proposal
        self.add_capabilities({"respond": True, "propose": True, "max-proposals": 1})

    def on_notification(self, notification: Notification, notifier: str):
        if notification.type == "end_negotiation":
            self.__end_negotiation = True

    def propose_(self, state: MechanismState) -> Optional["Outcome"]:
        if not self._capabilities["propose"] or self.__end_negotiation:
            return None
        if self._ufun_modified:
            self.on_ufun_changed()
        proposal = self.propose(state=state)

        # never return a proposal that is less than the reserved value
        if self.rational_proposal and self.reserved_value is not None:
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
        """Respond to an offer.

        Args:
            state: `MechanismState` giving current state of the negotiation.
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

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
        Called to counter an offer

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
        """Respond to an offer.

        Args:
            state: `MechanismState` giving current state of the negotiation.
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

        Remarks:
            - The default implementation never ends the negotiation
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).

        """
        if self._utility_function is None:
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

    @property
    def eu(self) -> Callable[["Outcome"], Optional["UtilityValue"]]:
        """
        The utility function in the given negotiation taking opponent model into account.

        Remarks:
            - If no utility_function is internally stored, `eu` still returns a valid callable that returns None for
              everything.
        """
        if hasattr(self, "opponent_model"):
            return lambda x: self._utility_function(
                x
            ) * self.opponent_model.probability_of_acceptance(x)
        else:
            return self._utility_function

    # CALLBACK
    def on_partner_proposal(
        self, state: MechanismState, agent_id: str, offer: "Outcome"
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something

        Args:
            state: `MechanismState` giving the state of the negotiation when the offer was porposed.
            agent_id: The ID of the agent who proposed
            offer: The proposal.

        Returns:
            None

        """

    def on_partner_refused_to_propose(
        self, state: MechanismState, agent_id: str
    ) -> None:
        """
        A callback called by the mechanism when a partner refuses to propose

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner refused to offer.
            agent_id: The ID of the agent who refused to propose

        Returns:
            None

        """

    def on_partner_response(
        self,
        state: MechanismState,
        agent_id: str,
        outcome: "Outcome",
        response: "SAOResponse",
    ) -> None:
        """
        A callback called by the mechanism when a partner responds to some offer

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner responded.
            agent_id: The ID of the agent who responded
            outcome: The proposal being responded to.
            response: The response

        Returns:
            None

        """

    @abstractmethod
    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        """Propose a set of offers

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """

    class Java:
        implements = ["jnegmas.sao.SAONegotiator"]


class RandomNegotiator(Negotiator, RandomResponseMixin, RandomProposalMixin):
    """A negotiation agent that responds randomly in a single negotiation."""

    def __init__(
        self,
        outcomes: Union[int, List["Outcome"]],
        name: str = None,
        parent: Controller = None,
        reserved_value: float = 0.0,
        p_acceptance=0.15,
        p_rejection=0.25,
        p_ending=0.1,
        can_propose=True,
        ufun=None,
    ) -> None:
        super().__init__(name=name, parent=parent)
        # noinspection PyCallByClass
        self.init_ranodm_response(
            p_acceptance=p_acceptance, p_rejection=p_rejection, p_ending=p_ending
        )
        self.init_random_proposal()
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        self.capabilities["propose"] = can_propose
        self._utility_function = MappingUtilityFunction(
            dict(zip(outcomes, np.random.rand(len(outcomes)))),
            reserved_value=reserved_value,
        )


# noinspection PyCallByClass
class LimitedOutcomesNegotiator(LimitedOutcomesMixin, SAONegotiator):
    """A negotiation agent that uses a fixed set of outcomes in a single
    negotiation."""

    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        outcomes: Optional[Union[int, Iterable["Outcome"]]] = None,
        acceptable_outcomes: Optional[List["Outcome"]] = None,
        acceptance_probabilities: Optional[Union[float, List[float]]] = None,
        p_ending=0.0,
        p_no_response=0.0,
        ufun=None,
    ) -> None:
        super().__init__(name=name, parent=parent)
        self.init_limited_outcomes(
            p_ending=p_ending,
            p_no_response=p_no_response,
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            outcomes=outcomes,
        )


# noinspection PyCallByClass
class LimitedOutcomesAcceptor(SAONegotiator, LimitedOutcomesAcceptorMixin):
    """A negotiation agent that uses a fixed set of outcomes in a single
    negotiation."""

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return LimitedOutcomesAcceptorMixin.respond(self, state=state, offer=offer)

    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        outcomes: Optional[Union[int, Iterable["Outcome"]]] = None,
        acceptable_outcomes: Optional[List["Outcome"]] = None,
        acceptance_probabilities: Optional[List[float]] = None,
        p_ending=0.0,
        p_no_response=0.0,
        ufun=None,
    ) -> None:
        SAONegotiator.__init__(self, name=name, parent=parent)
        self.init_limited_outcomes_acceptor(
            p_ending=p_ending,
            p_no_response=p_no_response,
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            outcomes=outcomes,
        )
        self.add_capabilities({"propose": False})

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        return None


class AspirationNegotiator(SAONegotiator, AspirationMixin):
    def __init__(
        self,
        name=None,
        ufun=None,
        parent: Controller = None,
        max_aspiration=0.95,
        aspiration_type="boulware",
        dynamic_ufun=True,
        randomize_offer=False,
        can_propose=True,
        assume_normalized=False,
    ):
        self.ordered_outcomes = []
        self.ufun_max = None
        self.ufun_min = None
        if assume_normalized:
            self.ufun_max, self.ufun_min = 1.0, 0.0
        super().__init__(
            name=name, assume_normalized=assume_normalized, parent=parent, ufun=ufun
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
        outcomes = self._ami.discrete_outcomes()
        self.ordered_outcomes = sorted(
            [(self._utility_function(outcome), outcome) for outcome in outcomes],
            key=lambda x: x[0],
            reverse=True,
        )
        if not self.assume_normalized:
            self.ufun_max = self.ordered_outcomes[0][0]
            self.ufun_min = self.ordered_outcomes[-1][0]
            if self.reserved_value is not None and self.ufun_min < self.reserved_value:
                self.ufun_min = self.reserved_value

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if self._utility_function is None:
            return ResponseType.REJECT_OFFER
        u = self._utility_function(offer)
        if u is None:
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
        asp = (
            self.aspiration(state.relative_time) * (self.ufun_max - self.ufun_min)
            + self.ufun_min
        )
        if asp < self.reserved_value:
            return None
        for i, (u, o) in enumerate(self.ordered_outcomes):
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


class NiceNegotiator(SAONegotiator, RandomProposalMixin):
    def __init__(self, *args, **kwargs):
        SAONegotiator.__init__(self, *args, **kwargs)
        self.init_random_proposal()

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return ResponseType.ACCEPT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        return RandomProposalMixin.propose(self=self, state=state)


class ToughNegotiator(SAONegotiator):
    def __init__(
        self,
        name=None,
        parent: Controller = None,
        dynamic_ufun=True,
        can_propose=True,
        ufun=None,
    ):
        super().__init__(name=name, parent=parent)
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
        outcomes = (
            self._ami.outcomes
            if self._offerable_outcomes is None
            else self._offerable_outcomes
        )
        self.best_outcome = max(
            [(self._utility_function(outcome), outcome) for outcome in outcomes]
        )[1]

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        if offer == self.best_outcome:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if not self._capabilities["propose"]:
            return None
        return self.best_outcome


class OnlyBestNegotiator(SAONegotiator):
    def __init__(
        self,
        name=None,
        parent: Controller = None,
        dynamic_ufun=True,
        min_utility=0.95,
        top_fraction=0.05,
        best_first=False,
        probabilisic_offering=True,
        can_propose=True,
        ufun=None,
    ):
        self._offerable_outcomes = None
        self.best_outcome = []
        self.ordered_outcomes = []
        self.acceptable_outcomes = []
        self.wheel = np.array([])
        self.offered = set([])
        super().__init__(name=name, parent=parent)
        if not dynamic_ufun:
            warnings.warn(
                "dynamic_ufun is deprecated. All Aspiration negotiators assume a dynamic ufun"
            )
        self.top_fraction = top_fraction
        self.min_utility = min_utility
        self.best_first = best_first
        self.probabilisic_offering = probabilisic_offering
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
        eu_outcome = [(self.eu(outcome), outcome) for outcome in outcomes]
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
            frac_limit = max(
                1, int(round(self.top_fraction * len(self.ordered_outcomes)))
            )
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
            if self.probabilisic_offering:
                r = random.random()
                for o, w in zip(self.acceptable_outcomes, self.wheel):
                    if w > r:
                        return o
                return random.sample(self.acceptable_outcomes, 1)[0]
            return random.sample(self.acceptable_outcomes, 1)[0]
        return None


class NaiveTitForTatNegotiator(SAONegotiator):
    """Implements a generalized tit-for-tat strategy"""

    def __init__(
        self,
        name: str = None,
        parent: Controller = None,
        ufun: Optional["UtilityFunction"] = None,
        kindness=0.0,
        randomize_offer=False,
        always_concede=True,
        initial_concession: Union[float, str] = "min",
    ):
        self.received_utilities = []
        self.proposed_utility = None
        self.ordered_outcomes = None
        self.sent_offer_index = None
        self.n_sent = 0
        super().__init__(name=name, ufun=ufun, parent=parent)
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
        my_utility, my_offer = self.ordered_outcomes[indx]
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

    def propose(self, state: MechanismState) -> Optional[Outcome]:
        indx = self._propose(state)
        self.proposed_utility = self.ordered_outcomes[indx][0]
        return self.ordered_outcomes[indx][1]


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
        self, state: MechanismState, agent_id: str, offer: "Outcome"
    ) -> None:
        self._java_object.onPartnerProposal(to_java(state), agent_id, to_java(offer))

    def on_partner_refused_to_propose(
        self, state: MechanismState, agent_id: str
    ) -> None:
        self._java_object.onPartnerRefusedToPropose(to_java(state), agent_id)

    def on_partner_response(
        self,
        state: MechanismState,
        agent_id: str,
        outcome: "Outcome",
        response: "SAOResponse",
    ) -> None:
        self._java_object.onPartnerResponse(
            to_java(state), agent_id, to_java(outcome), self._to_java_response(response)
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
        cls, java_object, *args, parent: Controller = None
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
                state=from_java(state), agent_id=agent_id, offer=from_java(offer)
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
        return to_java(
            self.shadow.on_partner_response(
                state=from_java(state),
                agent_id=agent_id,
                outcome=from_java(offer),
                response=SAOResponse(
                    response=ResponseType(response), outcome=from_java(counter_offer)
                ),
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


class SAOController(Controller):
    def propose_(
        self, negotiator_id: str, state: MechanismState
    ) -> Optional["Outcome"]:
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "propose", state=state)

    def respond_(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "respond", state=state, offer=offer)


SAOProtocol = SAOMechanism
"""An alias for `SAOMechanism object"""

SimpleTitForTatNegotiator = NaiveTitForTatNegotiator
"""A simple tit-for-tat negotiator"""
