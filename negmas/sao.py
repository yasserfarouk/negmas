import random
from abc import abstractmethod
from typing import Dict, Any, Callable, Type
from typing import Sequence, Optional, List, Tuple, Iterable, Union

import numpy as np
from dataclasses import dataclass

from negmas.common import *
from negmas.events import Notification
from negmas.java import JNegmasGateway, JavaCallerMixin, to_java
from negmas.mechanisms import MechanismRoundResult, Mechanism
from negmas.negotiators import Negotiator, AspirationMixin, Controller
from negmas.outcomes import sample_outcomes, Outcome, outcome_is_valid, ResponseType, outcome_as_dict
from negmas.utilities import MappingUtilityFunction, normalize, UtilityFunction, UtilityValue, JavaUtilityFunction

__all__ = [
    'SAOState',
    'SAOMechanism',
    'SAOProtocol',
    'SAONegotiator',
    'RandomNegotiator',
    'LimitedOutcomesNegotiator',
    'LimitedOutcomesAcceptor',
    'AspirationNegotiator',
    'ToughNegotiator',
    'OnlyBestNegotiator',
    'SimpleTitForTatNegotiator',
    'NiceNegotiator',
    'SAOController',
    'SAOController',
]


@dataclass
class SAOResponse:
    """A response to an offer given by an agent in the alternating offers protocol"""
    response: ResponseType = ResponseType.NO_RESPONSE
    outcome: Optional['Outcome'] = None


@dataclass
class SAOState(MechanismState):
    current_offer: Optional['Outcome'] = None
    current_offerer: Optional[str] = None
    n_acceptances: int = 0


@dataclass
class SAOInfo(MechanismInfo):
    end_negotiation_on_refusal_to_propose: bool = False


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
        end_negotiation_on_refusal_to_propose=True,
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
            name=name,
        )
        self._current_offer = None
        self._current_offerer = None
        self._n_accepting = 0
        self._current_negotiator = 0
        self.end_negotiation_on_refusal_to_propose = end_negotiation_on_refusal_to_propose

    def on_negotiation_start(self):
        if not self.info.dynamic_entry and not any([a.capabilities.get('propose', False) for a in self.negotiators]):
            self._current_offerer = None
            self._current_offer = None
            self._n_accepting = 0
            return False
        return True

    def extra_state(self):
        return SAOState(
            current_offer=self._current_offer
            , current_offerer=self._current_offerer.id if self._current_offerer else None
            , n_acceptances=self._n_accepting,
        )

    def step_(self) -> MechanismRoundResult:
        n_agents = len(self.negotiators)
        accepted = False
        negotiator = self.negotiators[self._current_negotiator]
        self._current_negotiator = (self._current_negotiator + 1) % n_agents
        if self._current_offer is None:
            response = ResponseType.NO_RESPONSE
        elif negotiator is not self._current_offerer:
            response = negotiator.respond_(state=self.state, offer=self._current_offer)
            for other in self.negotiators:
                if other is negotiator:
                    continue
                other.on_partner_response(state=self.state, agent_id=negotiator.id, outcome=self._current_offer
                                          , response=response)
        else:
            response = ResponseType.NO_RESPONSE
        if response == ResponseType.END_NEGOTIATION:
            self._current_offerer = negotiator
            self._current_offer = None
        else:
            if response == ResponseType.ACCEPT_OFFER:
                if negotiator is not self._current_offerer:
                    self._n_accepting += 1
                if self._n_accepting == n_agents:
                    accepted = True
            else:
                started_at = self._current_negotiator
                while not negotiator.capabilities.get('propose', False):
                    negotiator = self.negotiators[self._current_negotiator]
                    self._current_negotiator = (self._current_negotiator + 1) % n_agents
                    if self._current_negotiator == started_at:
                        if not self.info.dynamic_entry:
                            raise RuntimeError('No negotiators can propose. I cannot run a meaningful negotiation')
                        else:
                            return MechanismRoundResult(broken=False, timedout=False, agreement=None, error=True
                                                        , error_details='No negotiators can propose in a static_entry'
                                                                        ' negotiation!!')
                state = self.state
                proposal = negotiator.propose(state=state)
                if proposal is None:
                    if self.end_negotiation_on_refusal_to_propose:
                        response = ResponseType.END_NEGOTIATION
                        accepted = False
                    else:
                        for other in self.negotiators:
                            if other is negotiator:
                                continue
                            other.on_partner_refused_to_propose(agent_id=negotiator, state=state)
                else:
                    self._current_offer = proposal
                    self._current_offerer = negotiator
                    self._n_accepting = 1
                    for other in self.negotiators:
                        if other is negotiator:
                            continue
                        other.on_partner_proposal(agent_id=negotiator.id, offer=proposal, state=state)
        return MechanismRoundResult(broken=response == ResponseType.END_NEGOTIATION, timedout=False
                                    , agreement=self._current_offer if accepted else None)


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
        if not hasattr(self, 'add_capabilities'):
            raise ValueError(
                f"self.__class__.__name__ is just a mixin for class Negotiator. You must inherit from Negotiator or"
                f" one of its descendents before inheriting from self.__class__.__name__")
        self.add_capabilities({'respond': True})
        self.p_acceptance = p_acceptance
        self.p_rejection = p_rejection
        self.p_ending = p_ending
        self.wheel: List[Tuple[float, ResponseType]] = [
            (0.0, ResponseType.NO_RESPONSE)
        ]
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
            raise ValueError('Probabilities of acceptance+rejection+ending>1')

        self.wheel = self.wheel[1:]

    # noinspection PyUnusedLocal,PyUnusedLocal
    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
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
                'propose': True,
                'propose-with-value': False,
                'max-proposals': None,  # indicates infinity
            }
        )

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        if hasattr(self, 'offerable_outcomes') and self._offerable_outcomes is not None:
            return random.sample(self._offerable_outcomes, 1)[0]
        return self._mechanism_info.random_outcomes(1, astype=dict)[0]


class LimitedOutcomesAcceptorMixin(object):
    """An agent the accepts a limited set of outcomes.

    The agent accepts any of the given outcomes with the given probabilities.
    """

    def init_limited_outcomes_acceptor(self,
                                       outcomes: Optional[Union[int, Iterable['Outcome']]] = None,
                                       acceptable_outcomes: Optional[Iterable['Outcome']] = None,
                                       acceptance_probabilities: Optional[List[float]] = None,
                                       p_ending=.05,
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
        self.add_capabilities(
            {
                'respond': True,
            }
        )
        if acceptable_outcomes is not None and outcomes is None:
            raise ValueError('If you are passing acceptable outcomes explicitly then outcomes must also be passed')
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
            raise ValueError('Could not calculate all the outcomes. It is needed to assign a utility function')
        self.acceptance_probabilities = acceptance_probabilities
        u = [0.0] * len(self.outcomes)
        for p, o in zip(self.acceptance_probabilities, self.acceptable_outcomes):
            u[self.outcomes.index(o)] = p
        self.utility_function = MappingUtilityFunction(
            dict(zip(self.outcomes, u))
        )
        self.p_no_response = p_no_response
        self.p_ending = p_ending + p_no_response

    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
        """Respond to an offer.

        Args:
            offer (Outcome): offer being tested

        Returns:
            ResponseType: The response to the offer

        """
        info = self._mechanism_info
        r = random.random()
        if r < self.p_no_response:
            return ResponseType.NO_RESPONSE

        if r < self.p_ending:
            return ResponseType.END_NEGOTIATION

        if self.acceptable_outcomes is None:
            if outcome_is_valid(offer, info.issues) and random.random() < self.acceptance_probabilities:
                return ResponseType.ACCEPT_OFFER

            else:
                return ResponseType.REJECT_OFFER

        try:
            indx = self.acceptable_outcomes.index(offer)
        except ValueError:
            return ResponseType.REJECT_OFFER
        prob = self.acceptance_probabilities[indx]
        if indx < 0 or not outcome_is_valid(offer, info.issues):
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

    def init_limited_outcomes_proposer(self: Negotiator, proposable_outcomes: Optional[List['Outcome']] = None) -> None:
        self.add_capabilities(
            {
                'propose': True,
                'propose-with-value': False,
                'max-proposals': None,  # indicates infinity
            }
        )
        self._offerable_outcomes = proposable_outcomes
        if proposable_outcomes is not None:
            self._offerable_outcomes = list(proposable_outcomes)

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        if self._offerable_outcomes is None:
            return self._mechanism_info.random_outcomes(1)[0]
        else:
            return random.sample(self._offerable_outcomes, 1)[0]


class LimitedOutcomesMixin(LimitedOutcomesAcceptorMixin, LimitedOutcomesProposerMixin):
    """An agent the accepts a limited set of outcomes.

    The agent accepts any of the given outcomes with the given probabilities.
    """

    def init_limited_outcomes(self,
                              outcomes: Optional[Union[int, Iterable['Outcome']]] = None,
                              acceptable_outcomes: Optional[Iterable['Outcome']] = None,
                              acceptance_probabilities: Optional[Union[float, List[float]]] = None,
                              proposable_outcomes: Optional[Iterable['Outcome']] = None,
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
        self.init_limited_outcomes_acceptor(outcomes=outcomes, acceptable_outcomes=acceptable_outcomes
                                            , acceptance_probabilities=acceptance_probabilities
                                            , p_ending=p_ending, p_no_response=p_no_response)
        if proposable_outcomes is None and self.acceptable_outcomes is not None:
            if not isinstance(self.acceptance_probabilities, float):
                proposable_outcomes = [_ for _, p in zip(self.acceptable_outcomes, self.acceptance_probabilities)
                                       if p > 1e-9
                                       ]
        self.init_limited_outcomes_proposer(proposable_outcomes=proposable_outcomes)


class SAONegotiator(Negotiator):
    def __init__(self, assume_normalized=True, ufun: Optional[UtilityFunction] = None, name: Optional[str] = None
                 , rational_proposal=True, parent: Controller = None):
        super().__init__(name=name, ufun=ufun, parent=parent)
        self.assume_normalized = assume_normalized
        self.__end_negotiation = False
        self.my_last_proposal: Optional['Outcome'] = None
        self.my_last_proposal_utility: float = None
        self.rational_proposal = rational_proposal
        self.add_capabilities(
            {
                'respond': True,
                'propose': True,
                'max-proposals': 1,
            }
        )

    def on_notification(self, notification: Notification, notifier: str):
        if notification.type == 'end_negotiation':
            self.__end_negotiation = True
        elif notification.type == 'propose' and notifier == self._mechanism_id:
            return self.propose(state=notification.data['state'])
        elif notification.type == 'respond' and notifier == self._mechanism_id:
            return self.respond(state=notification.data['state'], offer=notification.data('offer', None))
        elif notification.type == 'counter' and notifier == self._mechanism_id:
            return self.counter(state=notification.data['state'], offer=notification.data('offer', None))

    def propose(self, state: MechanismState) -> Optional['Outcome']:
        if not self._capabilities['propose'] or self.__end_negotiation:
            return None
        proposal = self.propose_(state=state)

        # never return a proposal that is less than the reserved value
        if self.rational_proposal:
            utility = None
            if proposal is not None:
                utility = self.ufun(proposal)
                if utility is not None and utility < self.reserved_value:
                    return None

            if utility is not None:
                self.my_last_proposal = proposal
                self.my_last_proposal_utility = utility

        return proposal

    def respond(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
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
        return self.respond_(state=state, offer=offer)

    def counter(self, state: MechanismState, offer: Optional['Outcome']) -> 'SAOResponse':
        """
        Called to counter an offer

        Args:
            state: `MechanismState` giving current state of the negotiation.
            offer: The offer to be countered. None means no offer and the agent is requested to propose an offer

        Returns:
            Tuple[ResponseType, Outcome]: The response to the given offer with a counter offer if the response is REJECT

        """
        if offer is None:
            return SAOResponse(ResponseType.REJECT_OFFER, self.propose(state=state))
        response = self.respond(state=state, offer=offer)
        if response != ResponseType.REJECT_OFFER:
            return SAOResponse(response, None)
        return SAOResponse(response, self.propose(state=state))

    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
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
        if self.ufun(offer) < self.reserved_value:
            return ResponseType.REJECT_OFFER
        utility = None
        if self.my_last_proposal_utility is not None:
            utility = self.my_last_proposal_utility
        if utility is None:
            myoffer = self.propose(state=state)
            if myoffer is None:
                return ResponseType.NO_RESPONSE
            utility = self.ufun(myoffer)
        if utility is not None and self.ufun(offer) >= utility:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    # CALLBACK
    def on_partner_proposal(self, state: MechanismState, agent_id: str, offer: 'Outcome') -> None:
        """
        A callback called by the mechanism when a partner proposes something

        Args:
            state: `MechanismState` giving the state of the negotiation when the offer was porposed.
            agent_id: The ID of the agent who proposed
            offer: The proposal.

        Returns:
            None

        """

    def on_partner_refused_to_propose(self, state: MechanismState, agent_id: str) -> None:
        """
        A callback called by the mechanism when a partner refuses to propose

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner refused to offer.
            agent_id: The ID of the agent who refused to propose

        Returns:
            None

        """

    def on_partner_response(self, state: MechanismState, agent_id: str, outcome: 'Outcome'
                            , response: 'SAOResponse') -> None:
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
    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        """Propose a set of offers

        Args:
            state: `MechanismState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """

    class Java:
        implements = ['jnegmas.sao.SAONegotiator']


class RandomNegotiator(Negotiator, RandomResponseMixin, RandomProposalMixin):
    """A negotiation agent that responds randomly in a single negotiation."""

    def __init__(
        self,
        outcomes: Union[int, List['Outcome']],
        name: str = None, parent: Controller = None,
        reserved_value: float = 0.0,
        p_acceptance=0.15,
        p_rejection=0.25,
        p_ending=0.1,
        can_propose=True
    ) -> None:
        super().__init__(name=name, parent=parent)
        # noinspection PyCallByClass
        self.init_ranodm_response(p_acceptance=p_acceptance, p_rejection=p_rejection, p_ending=p_ending)
        self.init_random_proposal()
        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        self.capabilities['propose'] = can_propose
        self.utility_function = MappingUtilityFunction(dict(zip(outcomes, np.random.rand(len(outcomes))))
                                                       , reserved_value=reserved_value)


# noinspection PyCallByClass
class LimitedOutcomesNegotiator(LimitedOutcomesMixin, SAONegotiator):
    """A negotiation agent that uses a fixed set of outcomes in a single
    negotiation."""

    def __init__(
        self, name: str = None
        , parent: Controller = None,
        outcomes: Optional[Union[int, Iterable['Outcome']]] = None,
        acceptable_outcomes: Optional[List['Outcome']] = None,
        acceptance_probabilities: Optional[Union[float, List[float]]] = None,
        p_ending=0.0,
        p_no_response=0.0,
    ) -> None:
        super().__init__(name=name, parent=parent)
        self.init_limited_outcomes(p_ending=p_ending, p_no_response=p_no_response
                                   , acceptable_outcomes=acceptable_outcomes
                                   , acceptance_probabilities=acceptance_probabilities, outcomes=outcomes)


# noinspection PyCallByClass
class LimitedOutcomesAcceptor(SAONegotiator, LimitedOutcomesAcceptorMixin):
    """A negotiation agent that uses a fixed set of outcomes in a single
    negotiation."""

    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
        return LimitedOutcomesAcceptorMixin.respond_(self, state=state, offer=offer)

    def __init__(
        self,
        name: str = None, parent: Controller = None,
        outcomes: Optional[Union[int, Iterable['Outcome']]] = None,
        acceptable_outcomes: Optional[List['Outcome']] = None,
        acceptance_probabilities: Optional[List[float]] = None,
        p_ending=0.0,
        p_no_response=0.0,
    ) -> None:
        SAONegotiator.__init__(self, name=name, parent=parent)
        self.init_limited_outcomes_acceptor(
            p_ending=p_ending,
            p_no_response=p_no_response,
            acceptable_outcomes=acceptable_outcomes,
            acceptance_probabilities=acceptance_probabilities,
            outcomes=outcomes
        )
        self.add_capabilities(
            {
                'propose': False,
            }
        )

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        return None


class AspirationNegotiator(SAONegotiator, AspirationMixin):
    def __init__(self, name=None, ufun=None, parent: Controller = None, max_aspiration=0.95, aspiration_type='boulware'
                 , dynamic_ufun=True, randomize_offer=False, can_propose=True, assume_normalized=True):
        super().__init__(name=name, assume_normalized=assume_normalized, parent=parent, ufun=ufun)
        self.aspiration_init(max_aspiration=max_aspiration, aspiration_type=aspiration_type)
        self.ordered_outcomes = []
        self.dynamic_ufun = dynamic_ufun
        self.randomize_offer = randomize_offer
        self._max_aspiration = self.max_aspiration
        self.__ufun_modified = False
        self.add_capabilities(
            {
                'respond': True,
                'propose': can_propose,
                'propose-with-value': False,
                'max-proposals': None,  # indicates infinity
            }
        )

    @property
    def eu(self) -> Callable[['Outcome'], Optional['UtilityValue']]:
        """
        The utility function in the given negotiation taking opponent model into account.

        Remarks:
            - If no utility_function is internally stored, `eu` still returns a valid callable that returns None for
              everything.
        """
        if hasattr(self, 'opponent_model'):
            return lambda x: self.utility_function(x) * self.opponent_model.probability_of_acceptance(x)
        else:
            return self.utility_function

    def _update_ordered_outcomes(self):
        outcomes = self._mechanism_info.discrete_outcomes()
        if not self.assume_normalized:
            self.utility_function = normalize(self.utility_function, outcomes=outcomes, infeasible_cutoff=-1e-6)
        self.ordered_outcomes = sorted([(self.ufun(outcome), outcome) for outcome in outcomes]
                                       , key=lambda x: x[0], reverse=True)

    def on_notification(self, notification: Notification, notifier: str):
        super().on_notification(notification, notifier)
        if notification.type == 'ufun_modified':
            if self.dynamic_ufun:
                self.__ufun_modified = True
                self._update_ordered_outcomes()

    def on_negotiation_start(self, state: MechanismState):
        self.__ufun_modified = False
        self._update_ordered_outcomes()

    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
        u = self.ufun(offer)
        if u is None:
            return ResponseType.REJECT_OFFER
        asp = self.aspiration(state.relative_time)
        if u >= asp and u >= self.reserved_value:
            return ResponseType.ACCEPT_OFFER
        if asp < self.reserved_value:
            return ResponseType.END_NEGOTIATION
        return ResponseType.REJECT_OFFER

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        asp = self.aspiration(state.relative_time)
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

    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
        return ResponseType.ACCEPT_OFFER

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        return RandomProposalMixin.propose_(self=self, state=state)


class ToughNegotiator(SAONegotiator):
    def __init__(self, name=None, parent: Controller = None, dynamic_ufun=True, can_propose=True):
        super().__init__(name=name, parent=parent)
        self.best_outcome = None
        self.dynamic_ufun = dynamic_ufun
        self._offerable_outcomes = None
        self.add_capabilities(
            {
                'respond': True,
                'propose': can_propose,
                'propose-with-value': False,
                'max-proposals': None,  # indicates infinity
            }
        )

    @property
    def eu(self) -> Callable[['Outcome'], Optional['UtilityValue']]:
        """
        The utility function in the given negotiation taking opponent model into account.

        Remarks:
            - If no utility_function is internally stored, `eu` still returns a valid callable that returns None for
              everything.
        """
        if hasattr(self, 'opponent_model'):
            return lambda x: self.utility_function(x) * self.opponent_model.probability_of_acceptance(x)
        else:
            return self.utility_function

    def on_negotiation_start(self, state: MechanismState):
        outcomes = self._mechanism_info.outcomes if self._offerable_outcomes is None else self._offerable_outcomes
        self.best_outcome = max([(self.ufun(outcome), outcome) for outcome in outcomes])[1]

    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
        if self.dynamic_ufun:
            outcomes = self._mechanism_info.outcomes if self._offerable_outcomes is None else self._offerable_outcomes
            self.best_outcome = max([(self.ufun(outcome), outcome) for outcome in outcomes])[1]
        if offer == self.best_outcome:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        if not self._capabilities['propose']:
            return None
        if self.dynamic_ufun:
            outcomes = self._mechanism_info.outcomes if self._offerable_outcomes is None else self._offerable_outcomes
            self.best_outcome = max([(self.eu(outcome), outcome) for outcome in outcomes])[1]
        return self.best_outcome


class OnlyBestNegotiator(SAONegotiator):
    def __init__(self, name=None, parent: Controller = None, dynamic_ufun=True
                 , min_utility=0.95, top_fraction=0.05, best_first=False
                 , probabilisic_offering=True, can_propose=True
                 ):
        super().__init__(name=name, parent=parent)
        self._offerable_outcomes = None
        self.best_outcome = []
        self.dynamic_ufun = dynamic_ufun
        self.top_fraction = top_fraction
        self.min_utility = min_utility
        self.best_first = best_first
        self.probabilisit_offering = probabilisic_offering
        self.ordered_outcomes = []
        self.acceptable_outcomes = []
        self.wheel = np.array([])
        self.offered = set([])
        self.add_capabilities(
            {
                'respond': True,
                'propose': can_propose,
                'propose-with-value': False,
                'max-proposals': None,  # indicates infinity
            }
        )

    @property
    def eu(self) -> Callable[['Outcome'], Optional['UtilityValue']]:
        """
        The utility function in the given negotiation taking opponent model into account.

        Remarks:
            - If no utility_function is internally stored, `eu` still returns a valid callable that returns None for
              everything.
        """
        if hasattr(self, 'opponent_model'):
            return lambda x: self.utility_function(x) * self.opponent_model.probability_of_acceptance(x)
        else:
            return self.utility_function

    def acceptable(self):
        outcomes = self._mechanism_info.outcomes if self._offerable_outcomes is None else self._offerable_outcomes
        eu_outcome = [(self.eu(outcome), outcome) for outcome in outcomes]
        self.ordered_outcomes = sorted(eu_outcome
                                       , key=lambda x: x[0], reverse=True)
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
            frac_limit = max(1, int(round(self.top_fraction * len(self.ordered_outcomes))))
        else:
            frac_limit = len(outcomes)

        if frac_limit >= len(selected) > 0:
            sum = np.asarray(selected_utils).sum()
            if sum > 0.0:
                selected_utils /= sum
                selected_utils = np.cumsum(selected_utils)
            else:
                selected_utils = np.linspace(0.0, 1.0, len(selected_utils))
            return selected, selected_utils
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
            return selected + fsel, selected_utils
        return [], []

    def on_negotiation_start(self, state: MechanismState):
        self.acceptable_outcomes, self.wheel = self.acceptable()

    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
        if self.dynamic_ufun:
            self.acceptable_outcomes, self.wheel = self.acceptable()
        if offer in self.acceptable_outcomes:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        if not self._capabilities['propose']:
            return None
        if self.dynamic_ufun:
            self.acceptable_outcomes, self.wheel = self.acceptable()
        if self.best_first:
            for o in self.acceptable_outcomes:
                if o not in self.offered:
                    self.offered.add(o)
                    return o
        if len(self.acceptable_outcomes) > 0:
            if self.probabilisit_offering:
                r = random.random()
                for o, w in zip(self.acceptable_outcomes, self.wheel):
                    if w > r:
                        return o
                return random.sample(self.acceptable_outcomes, 1)[0]
            return random.sample(self.acceptable_outcomes, 1)[0]
        return None


class SimpleTitForTatNegotiator(SAONegotiator):
    """Implements a generalized tit-for-tat strategy"""

    def __init__(self, name: str = None, parent: Controller = None
                 , ufun: Optional['UtilityFunction'] = None, kindness=0.0
                 , randomize_offer=False, initial_concession: Union[float, str] = 'min'):
        super().__init__(name=name, ufun=ufun, parent=parent)
        self._offerable_outcomes = None
        self.received_utilities = []
        self.proposed_utility = None
        self.kindness = kindness
        self.ordered_outcomes = None
        self.initial_concession = initial_concession
        self.randomize_offer = randomize_offer

    def on_negotiation_start(self, state: MechanismState):
        outcomes = self._mechanism_info.outcomes if self._offerable_outcomes is None else self._offerable_outcomes
        if outcomes is None:
            outcomes = sample_outcomes(self._mechanism_info.issues, keep_issue_names=True)
        self.ordered_outcomes = sorted([(self.ufun(outcome), outcome) for outcome in outcomes]
                                       , key=lambda x: x[0], reverse=True)

    def respond_(self, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
        my_offer = self.propose(state=state)
        u = self.ufun(offer)
        if len(self.received_utilities) < 2:
            self.received_utilities.append(u)
        else:
            self.received_utilities[-1] = u
        if u >= self.ufun(my_offer):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def _outcome_at_utility(self, asp: float, n: int) -> List['Outcome']:
        for i, (u, o) in enumerate(self.ordered_outcomes):
            if u is None:
                continue
            if u < asp:
                if i == 0:
                    return [self.ordered_outcomes[0][1]] * n
                if self.randomize_offer:
                    if n < i:
                        return [random.sample(self.ordered_outcomes, 1)[0][1]] * n
                    else:
                        return [_[1] for _ in self.ordered_outcomes[: i]]
                else:
                    return [self.ordered_outcomes[i][1]] * n

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        if len(self.received_utilities) < 2 or self.proposed_utility is None:
            if len(self.received_utilities) < 1:
                return self.ordered_outcomes[0][1]
            if isinstance(self.initial_concession, str) and self.initial_concession == 'min':
                asp = None
                for u, o in self.ordered_outcomes:
                    if u is None:
                        continue
                    if asp is not None and u < asp:
                        break
                    asp = u
                if asp is None:
                    return self.ordered_outcomes[0][1]
            else:
                asp = self.ordered_outcomes[0][0] * (1 - self.initial_concession)
            return self._outcome_at_utility(asp=asp, n=1)[0]
        opponent_concession = self.received_utilities[1] - self.received_utilities[0]
        asp = self.proposed_utility - opponent_concession * (1 + self.kindness)
        return self._outcome_at_utility(asp=asp, n=1)[0]


class JavaSAONegotiator(SAONegotiator, JavaCallerMixin):

    def __init__(self, java_class_name: Optional[str]
                 , auto_load_java: bool = False, outcome_type: Type = dict
                 , name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._outcome_type = outcome_type
        self.add_capabilities(
            {
                'respond': True,
                'propose': True,
                'propose-with-value': False,
                'max-proposals': None,  # indicates infinity
            }
        )
        if java_class_name is not None:
            self.init_java_bridge(java_class_name=java_class_name, auto_load_java=auto_load_java)
            self.java_object.fromMap(to_java(self))

    @classmethod
    def from_dict(cls, java_object, *args, parent: Controller = None) -> 'JavaSAONegotiator':
        """Creates a Java negotiator from an object returned from the JVM implementing PySAONegotiator"""
        ufun = java_object.getUtilityFunction()
        if ufun is not None:
            ufun = JavaUtilityFunction.from_dict(java_object=ufun)
        return JavaCallerMixin.from_dict(java_object, name=java_object.getName()
                                         , assume_normalized=java_object.getAssumeNormalized()
                                         , rational_proposal=java_object.getRationalProposal()
                                         , parent=parent
                                         , ufun=ufun)

    def on_notification(self, notification: Notification, notifier: str):
        super().on_notification(notification=notification, notifier=notifier)
        jnotification = {'type': notification.type, 'data': to_java(notification.data)}
        self.java_object.on_notification(jnotification, notifier)

    def respond_(self, state: MechanismState, offer: 'Outcome'):
        response = self.java_object.respond(state, outcome_as_dict(offer))
        if response == 0:
            return ResponseType.ACCEPT_OFFER
        if response == 1:
            return ResponseType.REJECT_OFFER
        if response == 2:
            return ResponseType.END_NEGOTIATION
        if response == 3:
            return ResponseType.NO_RESPONSE
        raise ValueError(f'Unknown response type {response} returned from the Java underlying negotiator')

    def propose_(self, state: MechanismState) -> Optional['Outcome']:
        outcome = self.java_object.propose(state)
        if outcome is None:
            return None
        if self._outcome_type == dict:
            return outcome
        if self._outcome_type == tuple:
            return tuple(outcome.values())
        return self._outcome_type(outcome)

    class Java:
        implements = ['jnegmas.sao.SAONegotiator']


class SAOController(Controller):

    def propose_(self, negotiator_id: str, state: MechanismState) -> Optional['Outcome']:
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'propose_', state=state)

    def respond_(self, negotiator_id: str, state: MechanismState, offer: 'Outcome') -> 'ResponseType':
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f'Unknown negotiator {negotiator_id}')
        return self.call(negotiator, 'respond_', state=state,  offer=offer)


SAOProtocol = SAOMechanism
"""An alias for `SAOMechanism object"""
