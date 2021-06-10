"""
Implements controllers for the SAO mechanism.
"""
import itertools
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache

from typing import Dict, List, Optional, Tuple, Union

from .common import SAOResponse, SAOState
from .negotiators import AspirationNegotiator, PassThroughSAONegotiator, SAONegotiator
from ..common import AgentMechanismInterface, MechanismState
from ..negotiators import AspirationMixin, Controller
from ..outcomes import Outcome, ResponseType, outcome_is_valid
from ..utilities import utility_range

__all__ = [
    "SAOController",
    "SAORandomController",
    "SAOSyncController",
    "SAORandomSyncController",
    "SAOSingleAgreementController",
    "SAOSingleAgreementRandomController",
    "SAOSingleAgreementAspirationController",
    "SAOMetaNegotiatorController",
]


class SAOController(Controller):
    """
    A controller that can manage multiple negotiators taking full or partial control from them.

    Args:
         default_negotiator_type: Default type to use when creating negotiators using this controller. The default type is
                                  `PassThroughSAONegotiator` which passes *full control* to the controller.
         default_negotiator_params: Default paramters to pass to the default controller.
         auto_kill: Automatically kill the negotiator once its negotiation session is ended.
         name: Controller name
         ufun: The ufun of the controller.
    """

    def __init__(
        self,
        default_negotiator_type=PassThroughSAONegotiator,
        default_negotiator_params=None,
        auto_kill=False,
        name=None,
        ufun=None,
    ):
        super().__init__(
            default_negotiator_type=default_negotiator_type,
            default_negotiator_params=default_negotiator_params,
            auto_kill=auto_kill,
            name=name,
            ufun=ufun,
        )

    def before_join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> bool:
        """
        Called by children negotiators to get permission to join negotiations

        Args:
            negotiator_id: The negotiator ID
            ami  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.

        Returns:
            True if the negotiator is allowed to join the negotiation otherwise
            False

        """
        return True

    def after_join(
        self,
        negotiator_id: str,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
    ) -> None:
        """
        Called by children negotiators after joining a negotiation to inform
        the controller

        Args:
            negotiator_id: The negotiator ID
            ami  (AgentMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.
        """

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:

        negotiator, _ = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "propose", state=state)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "respond", state=state, offer=offer)

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        if self._auto_kill:
            self.kill_negotiator(negotiator_id, True)

    def on_negotiation_start(self, negotiator_id: str, state: MechanismState) -> None:
        pass


class SAORandomController(SAOController):
    """
    A controller that returns random offers.

    Args:
        p_acceptance: The probability of accepting an offer.

    """

    def __init__(self, *args, p_acceptance: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._p_acceptance = p_acceptance

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:

        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        if negotiator.ami is None:
            return None
        return negotiator.ami.random_outcomes(1)[0]

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        if random.random() > self._p_acceptance:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class SAOSyncController(SAOController):
    """
    A controller that can manage multiple negotiators synchronously.

    Remarks:
        - The controller waits for an offer from each one of its negotiators before deciding what to do.
        - Loops may happen if multiple controllers of this type negotiate with each other. For example controller A
          is negotiating with B, C, while B is also negotiating with C. These loops are broken by the `SAOMechanism`
          by **forcing** some controllers to respond before they have all of the offers. In this case, `counter_all`
          will receive offers from one or more negotiators but not all of them.

    """

    def __init__(self, *args, global_ufun=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.offers: Dict[str, "Outcome"] = {}
        """Keeps the last offer received for each negotiation"""
        self.responses: Dict[str, ResponseType] = {}
        """Keeps the next response type for each negotiation"""
        self.proposals: Dict[str, "Outcome"] = {}
        """Keeps the next proposal for each negotiation"""
        self.offer_states: Dict[str, "SAOState"] = {}
        """Keeps the last state received for each negotiation"""
        self.n_waits: Dict[str, int] = defaultdict(int)
        self.global_ufun = global_ufun

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        # if there are no proposals yet, get first proposals
        if not self.proposals:
            self.proposals = self.first_proposals()
            # print(f"{self.name} first proposals {self.proposals}")
        # get the saved proposal if it exists and return it
        proposal = self.proposals.get(negotiator_id, None)
        # if some proposal was there, delete it to force the controller to get a new one
        if proposal is not None:
            self.proposals[negotiator_id] = None
            # print(f"{self.name} found proposal {proposal} for {negotiator_id}")
        else:
            # print(f"{self.name} found {None} for {negotiator_id}")
            # if the proposal that was there was None, just offer the best offer
            if self.global_ufun:
                self.proposals = self.first_proposals()
                proposal = self.proposals.get(negotiator_id, None)
                self.proposals[negotiator_id] = None
            else:
                proposal = self.first_offer(negotiator_id)
            # print(f"{self.name} generated proposal {proposal} for {negotiator_id}")
        # print(f"{self.name} sent proposal {proposal} through {negotiator_id}")
        return proposal

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        # get the saved response to this negotiator if any
        # print(f"{self.name} received offer {offer} through {negotiator_id}")
        response = self.responses.get(negotiator_id, None)
        if response is not None:
            # print(f"{self.name} found response {response} for {negotiator_id}")
            # remove the response and return it
            del self.responses[negotiator_id]
            self.n_waits[negotiator_id] = 0
            return response

        # we get here if there was no saved response or if the saved response was None

        # set the saved offer for this negotiator
        self.offers[negotiator_id] = offer
        self.offer_states[negotiator_id] = state
        n_negotiators = len(self.active_negotiators)
        # if we got all the offers or waited long enough, counter all the offers so-far
        if (
            len(self.offers) == n_negotiators
            or self.n_waits[negotiator_id] >= n_negotiators
        ):
            responses = self.counter_all(offers=self.offers, states=self.offer_states)
            # print(f"{self.name} responded to {self.offers} with {responses}: s: { {k: v.step for k, v in self.offer_states.items()}  }")
            for nid in responses.keys():
                # register the responses for next time for all other negotiators
                if nid != negotiator_id:
                    self.responses[nid] = responses[nid].response
                # register the proposals to be sent to all agents including this one
                self.proposals[nid] = responses[nid].outcome
            self.offers = dict()
            self.offer_states = dict()
            self.n_waits[negotiator_id] = 0
            resp =  responses[negotiator_id].response
            self.responses[negotiator_id] = None
            return resp
        self.n_waits[negotiator_id] += 1
        # print(f"controller {self.id}: {self.n_waits}")
        return ResponseType.WAIT

    def first_proposals(self) -> Dict[str, "Outcome"]:
        """Gets a set of proposals to use for initializing the negotiation. To avoid offering anything, just return None
        for all of them. That is the default"""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.first_offer(_) for _ in self.negotiators.keys()),
            )
        )

    @abstractmethod
    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.
            - If the system determines that a loop is formed, the agent may receive this call for a subset of
              negotiations not all of them.

        """

    # @lru_cache(maxsize=100)
    def first_offer(self, negotiator_id: str) -> Optional["Outcome"]:
        """
        Finds the first offer for this given negotiator. By default it will be the best offer

        Args:
            negotiator_id: The ID of the negotiator

        Returns:
            The first offer to use.

        Remarks:
            Default behavior is to use the ufun defined for the controller if any then try the ufun
            defined for the negotiator. If neither exists, the first offer will be None.
        """
        negotiator, _ = self.negotiators.get(negotiator_id, (None, None))
        if negotiator is None or negotiator.ami is None:
            return None
        # if the controller has a ufun, use it otherwise use the negotiator ufun
        if self.ufun is not None:
            _, _, _, best = utility_range(
                self.ufun,
                issues=negotiator.ami.issues,
                return_outcomes=True,
                ami=negotiator.ami,
            )
        elif negotiator.ufun is not None:
            _, _, _, best = utility_range(
                negotiator.ufun,
                issues=negotiator.ami.issues,
                return_outcomes=True,
                ami=negotiator.ami,
            )
        else:
            best = None
        return best


class SAORandomSyncController(SAOSyncController):
    """
    A sync controller that returns random offers. (See `SAOSyncController` ).

    Args:
        p_acceptance: The probability that an offer will be accepted
        p_rejection: The probability that an offer will be rejected
        p_ending: The probability of ending the negotiation at any negotiation round.

    Remarks:
        - If probability of acceptance, rejection and ending sum to less than 1.0, the agent will return NO_RESPONSE
          with the remaining probability. Depending on the settings of the `SAOMechanism` this may be treated as
          ending the negotiation.

    """

    def __init__(
        self, *args, p_acceptance=0.15, p_rejection=0.85, p_ending=0.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
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

    def make_response(self) -> "ResponseType":
        r = random.random()
        for w in self.wheel:
            if w[0] >= r:
                return w[1]
        return ResponseType.NO_RESPONSE

    def counter_all(self, offers, states):
        result = {}
        for negotiator in offers.keys():
            response = self.make_response()
            if response == ResponseType.REJECT_OFFER:
                result[negotiator] = SAOResponse(
                    response, self.negotiators[negotiator][0].ami.random_outcomes(1)[0]
                )
            else:
                result[negotiator] = SAOResponse(response, None)
        return result

    def first_proposals(self):
        return dict(
            zip(
                self.negotiators.keys(),
                [n[0].ami.random_outcomes(1)[0] for n in self.negotiators.values()],
            )
        )


class SAOSingleAgreementController(SAOSyncController, ABC):
    """
    A synchronized controller that tries to get no more than one agreeement.

    This controller manages a set of negotiations from which only a single one
    -- at most -- is likely to result in an agreement. An example of a case in which
    it is useful is an agent negotiating to buy a car from multiple suppliers.
    It needs a single car at most but it wants the best one from all of those
    suppliers. To guarentee a single agreement, pass strict=True

    The general algorithm for this controller is something like this:

        - Receive offers from all partners.
        - Find the best offer among them by calling the abstract `best_offer`
          method.
        - Check if this best offer is acceptable using the abstract `is_acceptable`
          method.

            - If the best offer is acceptable, accept it and end all other negotiations.
            - If the best offer is still not acceptable, then all offers are rejected
              and with the partner who sent it receiving the result of `best_outcome`
              while the rest of the partners receive the result of `make_outcome`.

        - The default behavior of `best_outcome` is to return the outcome with
          maximum utility.
        - The default behavior of `make_outcome` is to return the best offer
          received in this round if it is valid for the respective negotiation
          and the result of `best_outcome` otherwise.

    Args:
        strict: If True the controller is **guaranteed** to get a single agreement but it will have to send
                no-response repeatedly so there is a higher chance of never getting an agreement when two of those controllers
                negotiate with each other
    """

    def __init__(self, *args, strict=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_outcomes = dict()
        self.strict = strict
        self.__end_all = False

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        super().on_negotiation_end(negotiator_id, state)
        if state.agreement is not None:
            self.__end_all = True

    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """
        Counters all responses

        Args:
            offers: A dictionary mapping partner ID to offer
            states: A dictionary mapping partner ID to mechanism state

        Returns:
            A dictionary mapping partner ID to a response

        Remarks:

            The agent will counter all offers by either ending all negotiations except one which is accepted or by
            rejecting all offers and countering them using the `make_offer` method.

        """
        # if one of the negotiations ended in a contract, end all others
        partners = list(offers.keys())

        if self.__end_all:
            return dict(
                zip(
                    partners,
                    itertools.repeat(SAOResponse(ResponseType.END_NEGOTIATION, None)),
                )
            )

        # find the partner who sent the best offer (partner is represented by the ID of the negotiator engaged with it)
        partner = self.best_offer(offers)
        if partner is None:
            # if there is no best-partner, make an offer for each partner (non-strict) or for any random partner
            # (strict)
            if self.strict:
                selected = random.sample(partners, 1)[0]
                current_offers = [
                    self.make_offer(
                        negotiator=nid,
                        state=states[nid],
                        best_offer=None,
                        best_from=None,
                    )
                    if nid == selected
                    else None
                    for nid in partners
                ]
            else:
                current_offers = [
                    self.make_offer(
                        negotiator=nid,
                        state=states[nid],
                        best_offer=None,
                        best_from=None,
                    )
                    for nid in partners
                ]
            return dict(
                zip(
                    offers.keys(),
                    [SAOResponse(ResponseType.REJECT_OFFER, o) for o in current_offers],
                )
            )
        # if the best offer is acceptable, accept it and end all other negotiations
        acceptable = self.is_acceptable(offers[partner], partner, states[partner])
        if acceptable:
            responses = dict(
                zip(
                    partners,
                    itertools.repeat(SAOResponse(ResponseType.END_NEGOTIATION, None)),
                )
            )
            responses[partner] = SAOResponse(ResponseType.ACCEPT_OFFER, None)
            return responses
        # We know that there is a best-offer and that it is still unacceptable.
        if self.strict:
            # if strict, then just select one partner and make an offer for it sending everyone else None
            selected = random.sample(partners, 1)[0]
            current_offers = [
                self.make_offer(
                    negotiator=nid,
                    state=states[nid],
                    best_offer=offers[partner],
                    best_from=partner,
                )
                if nid == selected
                else None
                for nid in partners
            ]
        else:
            # if not strict, make an offer to each partner including the best one.
            current_offers = [
                self.make_offer(
                    negotiator=nid,
                    state=states[nid],
                    best_offer=offers[partner],
                    best_from=partner,
                )
                for nid in partners
            ]
        return dict(
            zip(
                partners,
                [SAOResponse(ResponseType.REJECT_OFFER, o) for o in current_offers],
            )
        )

    def first_proposals(self) -> Dict[str, "Outcome"]:
        if not self.strict:
            return super().first_proposals()

        partners = list(self.negotiators.keys())
        selected = random.sample(partners, 1)[0]
        return dict(
            zip(
                partners,
                (self.first_offer(_) if _ == selected else None for _ in partners),
            )
        )

    def response_to_best_offer(
        self, negotiator: str, state: SAOState, offer: "Outcome"
    ) -> Optional["Outcome"]:
        """
        Return a response to the partner from which the best current offer was
        received

        Args:
            negotiator: The negotiator from which the best offer was received
            state: The state of the corresponding negotiation
            offer: The best offer received at this round.

        Returns:
            The offer to be sent back to `negotiator`

        """
        return self._best_outcomes[negotiator][0]

    def after_join(self, negotiator_id, ami, state, *, ufun=None, role="agent"):
        super().after_join(negotiator_id, ami, state, ufun=ufun, role=role)
        self._best_outcomes[negotiator_id] = self.best_outcome(negotiator_id)

    def best_outcome(
        self, negotiator: str, state: Optional[SAOState] = None
    ) -> Optional["Outcome"]:
        """
        The best outcome for the negotiation `negotiator` engages in given the
        `state` .

        Args:
            negotiator: The negotiator for which the best outcome is to be found
            state: If given, the state of the negotiation. If None, should
                   return the absolute best outcome

        Return:
            The outcome with maximum utility.

        Remarks:

            - The default implementation, just returns the best outcome for
              this negotiation without considering the `state` or returns None
              if it is not possible to find this best outcome.
            - If the negotiator defines a ufun, it is used otherwise the ufun
              defined for the controller if used (if any)
        """
        neg = self.negotiators[negotiator][0]
        if neg is None:
            return None
        if neg.ami is None:
            return None
        ufun = neg.ufun
        if ufun is None and hasattr(self, "ufun"):
            ufun = self.ufun
        if ufun is None:
            return None
        _, _, _, top_outcome = ufun.utility_range(
            issues=neg.ami.issues, return_outcomes=True, ami=neg.ami
        )
        return top_outcome

    def make_offer(
        self,
        negotiator: str,
        state: SAOState,
        best_offer: Optional["Outcome"],
        best_from: Optional[str],
    ) -> Optional["Outcome"]:
        """Generate an offer for the given partner

        Args:
            negotiator: The ID of the negotiator for who an offer is to be made.
            state: The mechanism state of this partner
            best_offer: The best offer received in this round. None means that
                        no known best offers.
            best_from: The ID of the negotiator that received the best offer

        Returns:
            The outcome to be offered to `negotiator` (None means no-offer)

        Remarks:

            Default behavior is to offer everyone `best_offer` if available
            otherwise, it returns no offers (None). The `best_from` negotiator
            will be sending the result of `best_outcome`.

        """
        current_best = self.best_outcome(negotiator, state)
        if negotiator == best_from:
            return current_best
        ami = self.negotiators[negotiator][0].ami
        if ami is None:
            return None
        if best_offer is not None and outcome_is_valid(best_offer, ami.issues):
            if self.is_better(best_offer, current_best, negotiator, state):
                return best_offer
            return current_best

    @abstractmethod
    def is_acceptable(self, offer: "Outcome", source: str, state: SAOState) -> bool:
        """Should decide if the given offer is acceptable

        Args:
            offer: The offer being tested
            source: The ID of the negotiator that received this offer
            state: The state of the negotiation handled by that negotiator

        Remarks:
            - If True is returned, this offer will be accepted and all other
              negotiations will be ended.
        """

    @abstractmethod
    def best_offer(self, offers: Dict[str, "Outcome"]) -> Optional[str]:
        """
        Return the ID of the negotiator with the best offer

        Args:
            offers: A mapping from negotiator ID to the offer it received

        Returns:
            The ID of the negotiator with best offer. Ties should be broken.
            Return None only if there is no way to calculate the best offer.
        """

    @abstractmethod
    def is_better(self, a: "Outcome", b: "Outcome", negotiator: str, state: SAOState):
        """Compares two outcomes of the same negotiation

        Args:
            a: "Outcome"
            b: "Outcome"
            negotiator: The negotiator for which the comparison is to be made
            state: Current state of the negotiation

        Returns:
            True if utility(a) > utility(b)
        """


class SAOMetaNegotiatorController(SAOController):
    """
    Controls multiple negotiations using a single `meta` negotiator.

    Args:
        - meta_negotiator: The negotiator used for controlling all negotiations.

    Remarks:

        - The controller will use the meta-negotiator to handle all negotiations by
          first setting its AMI to the current negotiation then calling the appropriate
          method in the meta negotiator.
        - If no meta-negotiator is given, an `AspirationNegotiator` will be created
          and used (with default parameters).
        - The meta-negotiator should not internally store information between
          calls to `propose` and `respond` about the negotiation because it
          will be called to propose and respond in *multiple* negotiations. The
          `AspirationNegotiator` can be used but `SimpleTitForTatNegotiator`
          cannot as it stores the past offer.

    """

    def __init__(self, *args, meta_negotiator: SAONegotiator = None, **kwargs):
        super().__init__(*args, **kwargs)
        if meta_negotiator is None:
            meta_negotiator = AspirationNegotiator(
                name=f"{self.name}-negotiator", ufun=kwargs.get("ufun", None)
            )
        self.meta_negotiator = meta_negotiator

    def propose(self, negotiator_id: str, state: MechanismState) -> Optional["Outcome"]:
        """Uses the meta negotiator to propose"""
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        self.meta_negotiator._ami = negotiator.ami
        return self.meta_negotiator.propose(state)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        """Uses the meta negotiator to respond"""
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        self.meta_negotiator._ami = negotiator.ami
        return self.meta_negotiator.respond(state=state, offer=offer)


class SAOSingleAgreementRandomController(SAOSingleAgreementController):
    """
    A single agreement controller that uses a random negotiation strategy.

    Args:
        p_acceptance: The probability that an offer is accepted

    """

    def is_acceptable(self, offer: "Outcome", source: str, state: SAOState) -> bool:
        return random.random() > self.p_acceptance

    def best_offer(self, offers: Dict[str, "Outcome"]) -> Optional[str]:
        return random.sample(list(offers.keys()), 1)[0]

    def is_better(self, a: "Outcome", b: "Outcome", negotiator: str, state: SAOState):
        return random.random() > 0.5

    def __init__(self, *args, p_acceptance=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_acceptance = p_acceptance


class SAOSingleAgreementAspirationController(
    SAOSingleAgreementController, AspirationMixin
):
    """
    A `SAOSingleAgreementController` that uses aspiration level to decide what
    to accept and what to propose.

    Args:
        ufun: The utility function to use for ALL negotiations
        max_aspiration: The maximum aspiration level to start with
        aspiration_type: The aspiration type/ exponent

    Remarks:

        - This controller will get at most one agreement. It uses a concession
          strategy controlled by `max_aspiration` and `aspiration_type` as in
          the `AspirationNegotiator` for all negotiations.
        - The partner who gives the best proposal will receive an offer that
          is guaranteed to have a utility value (for the controller) above
          the current aspiration level.
        - The controller uses a single ufun for all negotiations. This implies
          that all negotiations should have the same outcome space (i.e.
          the same issues).
    """

    def __init__(
        self,
        *args,
        max_aspiration: float = 1.0,
        aspiration_type: Union[str, int, float] = "boulware",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        AspirationMixin.aspiration_init(self, max_aspiration, aspiration_type)

    def after_join(self, negotiator_id, ami, state, *, ufun=None, role="agent"):
        if self.ufun is not None:
            if (
                self.ufun.outcome_type is not None
                and self.ufun.outcome_type != ami.outcome_type
            ):
                raise ValueError(
                    f"Controller's ufun is of type {self.ufun.outcome_type} while the mechanism ufun is of type"
                    f"{ami.outcome_type}"
                )
            self.ufun.outcome_type = ami.outcome_type

    def is_acceptable(self, offer: "Outcome", source: str, state: SAOState):
        return self.ufun(offer) >= self.aspiration(state.relative_time)

    def is_better(self, a: "Outcome", b: "Outcome", negotiator: str, state: SAOState):
        return self.ufun(a) > self.ufun(b)

    def best_offer(self, offers):
        best_val, best_negotiator = float("-inf"), None
        for k, offer in offers.items():
            curr_val = self.ufun(offer)
            if curr_val >= best_val:
                best_val, best_negotiator = curr_val, k
        return best_negotiator

    def best_outcome(self, negotiator, state=None):
        outcome = self.ufun.outcome_with_utility(
            rng=(
                self.aspiration(state.relative_time)
                if state is not None
                else self.ufun(super().best_outcome(negotiator, None)),
                None,
            ),
            issues=self.negotiators[negotiator][0].ami.issues,
        )
        if outcome is None:
            return self._best_outcomes.get(negotiator, None)
        return outcome
