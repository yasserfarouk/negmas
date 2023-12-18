"""
Implements controllers for the SAO mechanism.
"""
from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from negmas.gb.negotiators import AspirationNegotiator
from negmas.preferences.protocols import UFun

from ..common import MechanismState, NegotiatorMechanismInterface
from ..gb.negotiators.base import GBNegotiator
from ..negotiators import Controller
from ..negotiators.helpers import PolyAspiration
from ..outcomes import Outcome, outcome_is_valid
from .common import ResponseType, SAOResponse, SAOState
from .negotiators.base import SAONegotiator
from .negotiators.controlled import ControlledSAONegotiator

if TYPE_CHECKING:
    from negmas.preferences import Preferences

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

ControlledNegotiatorType = TypeVar("ControlledNegotiatorType", bound=SAONegotiator)


class SAOController(Controller):
    """
    A controller that can manage multiple negotiators taking full or partial control from them.

    Args:
         default_negotiator_type: Default type to use when creating negotiators using this controller. The default type is
                                  `ControlledSAONegotiator` which passes *full control* to the controller.
         default_negotiator_params: Default paramters to pass to the default controller.
         auto_kill: Automatically kill the negotiator once its negotiation session is ended.
         name: Controller name
         preferences: The preferences of the controller.
         ufun: The ufun of the controller (overrides `preferences`).
    """

    def __init__(
        self,
        default_negotiator_type: str
        | type[SAONegotiator]
        | None = ControlledSAONegotiator,
        default_negotiator_params: dict[str, Any] | None = None,
        auto_kill=True,
        name=None,
        preferences=None,
        ufun=None,
    ):
        if ufun is not None:
            preferences = ufun
        super().__init__(
            default_negotiator_type=default_negotiator_type,  # type: ignore
            default_negotiator_params=default_negotiator_params,
            auto_kill=auto_kill,
            name=name,
            preferences=preferences,
            ufun=ufun,
        )

    def before_join(
        self,
        negotiator_id: str,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        role: str = "negotiator",
    ) -> bool:
        """
        Called by children negotiators to get permission to join negotiations

        Args:
            negotiator_id: The negotiator ID
            nmi  (NegotiatorMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.

        Returns:
            True if the negotiator is allowed to join the negotiation otherwise
            False

        """
        return True

    def create_negotiator(  # type: ignore
        self,
        negotiator_type: str | ControlledNegotiatorType | None = None,
        name: str | None = None,
        cntxt: Any = None,
        **kwargs,
    ) -> ControlledNegotiatorType:
        return super().create_negotiator(negotiator_type, name, cntxt, **kwargs)  # type: ignore I know that the return type is an SAONegotiator

    def after_join(
        self, negotiator_id, *args, ufun=None, preferences=None, **kwargs
    ) -> None:
        """
        Called by children negotiators after joining a negotiation to inform
        the controller

        Args:
            negotiator_id: The negotiator ID
            nmi  (NegotiatorMechanismInterface): The negotiation.
            state (MechanismState): The current state of the negotiation
            ufun (UtilityFunction): The ufun function to use before any discounting.
            role (str): role of the agent.
        """

    def propose(self, negotiator_id: str, state: MechanismState) -> Outcome | None:
        negotiator, _ = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "propose", state=state)

    def respond(
        self,
        negotiator_id: str,
        state: MechanismState,
        source: str | None = None,
    ) -> ResponseType:
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        return self.call(negotiator, "respond", state=state, source=source)

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        if self._auto_kill:
            self.kill_negotiator(negotiator_id)

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

    def propose(self, negotiator_id: str, state: MechanismState) -> Outcome | None:
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        if negotiator.nmi is None:
            return None
        return negotiator.nmi.random_outcomes(1)[0]

    def respond(
        self,
        negotiator_id: str,
        state: MechanismState,
        source: str | None = None,
    ) -> ResponseType:
        negotiator, cntxt = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        if random.random() > self._p_acceptance:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER


class SAOSyncController(SAOController):
    """
    A controller that can manage multiple negotiators synchronously.

    Args:

        global_ufun: If true, the controller assumes that the ufun is only
                     defined globally for the complete set of negotiations

    Remarks:
        - The controller waits for an offer from each one of its negotiators before deciding what to do.
        - Loops may happen if multiple controllers of this type negotiate with each other. For example controller A
          is negotiating with B, C, while B is also negotiating with C. These loops are broken by the `SAOMechanism`
          by **forcing** some controllers to respond before they have all of the offers. In this case, `counter_all`
          will receive offers from one or more negotiators but not all of them.

    """

    def __init__(self, *args, global_ufun=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.__global_ufun = global_ufun
        self.reset()

    def reset(self):
        self.__offers: dict[str, Outcome] = dict()
        self.__responses: dict[str, ResponseType] = dict()
        self.__proposals: dict[str, Outcome | None] = dict()
        self.__offer_states: dict[str, SAOState] = dict()
        self.__n_waits: dict[str, int] = defaultdict(int)
        self.__first_proposals_collected = False
        super().reset()

    def first_offer(self, negotiator_id: str) -> Outcome | None:
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
        if negotiator is None or negotiator.nmi is None:
            return None
        # if the controller has a ufun, use it otherwise use the negotiator's
        try:
            ufun = self.ufun
        except ValueError:
            ufun = negotiator.ufun
        if isinstance(ufun, UFun):
            _, best = ufun.extreme_outcomes()
        else:
            best = None
        return best

    def first_proposals(self) -> dict[str, Outcome | None]:
        """Gets a set of proposals to use for initializing the negotiation."""
        return dict(
            zip(
                self.negotiators.keys(),
                (self.first_offer(_) for _ in self.negotiators.keys()),
            )
        )

    def _set_first_proposals(self):
        self.__proposals = self.first_proposals()
        if not self.__proposals:
            self.__proposals = dict(
                zip(self.negotiators.keys(), [None] * len(self.negotiators))
            )
        self.__first_proposals_collected = True

    def propose(self, negotiator_id: str, state: MechanismState) -> Outcome | None:
        # if there are no proposals yet, get first proposals
        if not self.__proposals:
            self._set_first_proposals()
        # get the saved proposal if it exists and return it
        if negotiator_id in self.__proposals.keys():
            # if some proposal was there, delete it to force the controller to get a new one
            proposal = self.__proposals.pop(negotiator_id, None)
        else:
            # if there was no proposal, get one. Note that `None` is a valid proposal
            if self.__global_ufun:
                self._set_first_proposals()
                proposal = self.__proposals.pop(negotiator_id, None)
            else:
                proposal = self.first_offer(negotiator_id)

        # report not waiting on this offer because I obviously just sent
        self.__n_waits[negotiator_id] = 0
        self.__responses.pop(negotiator_id, None)
        self.__offers.pop(negotiator_id, None)
        self.__offer_states.pop(negotiator_id, None)
        return proposal

    @abstractmethod
    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

        Args:
            offers: Maps negotiator IDs to offers
            states: Maps negotiator IDs to offers AT the time the offers were made.

        Remarks:
            - The response type CANNOT be WAIT.
            - If the system determines that a loop is formed, the agent may receive this call for a subset of
              negotiations not all of them.

        """

    def respond(  # type: ignore
        self,
        negotiator_id: str,
        state: SAOState,
        source: str | None = None,
    ) -> ResponseType:
        offer = state.current_offer
        # get the saved response to this negotiator if any
        response = self.__responses.pop(negotiator_id, ResponseType.WAIT)

        # if there some non-waiting saved respons return and delete it
        if response != ResponseType.WAIT:
            # remove the response and return it
            self.__n_waits[negotiator_id] = 0
            return response

        # we get here if there was no saved response (WAIT should never be saved)

        # set the saved offer for this negotiator
        self.__offers[negotiator_id] = offer  # type: ignore
        self.__offer_states[negotiator_id] = state
        n_negotiators = len(self.active_negotiators)
        # if we did not get all the offers yet and can still wait, wait
        if (
            len(self.__offers) < n_negotiators
            and self.__n_waits[negotiator_id] < n_negotiators
        ):
            self.__n_waits[negotiator_id] += 1
            return ResponseType.WAIT

        # we arrive here if we already have all the offers to counter. WE may though not have proposed yet
        if not self.__first_proposals_collected:
            self._set_first_proposals()
        responses = self.counter_all(offers=self.__offers, states=self.__offer_states)
        for neg in self.negotiators.keys():
            if neg not in responses:
                responses[neg] = SAOResponse(ResponseType.END_NEGOTIATION, None)
            saved_response = responses.get(neg, None)
            if saved_response is None:
                self.__responses[neg] = ResponseType.REJECT_OFFER
                self.__proposals[neg] = None
                continue
            # register the responses for next time for all other negotiators
            self.__responses[neg] = saved_response.response
            # register the proposals to be sent to all agents including this one
            self.__proposals[neg] = saved_response.outcome
            # register that we are not waiting anymore on any of the offers we received
            self.__n_waits[neg] = 0
        self.__offers = dict()
        self.__offer_states = dict()
        if negotiator_id not in self.__responses:
            return ResponseType.REJECT_OFFER
        return self.__responses.pop(negotiator_id, ResponseType.REJECT_OFFER)

    def on_negotiation_end(self, negotiator_id: str, state: MechanismState) -> None:
        if negotiator_id in self.__offers.keys():
            del self.__offers[negotiator_id]
        if negotiator_id in self.__offer_states.keys():
            del self.__offer_states[negotiator_id]
        if negotiator_id in self.__responses.keys():
            del self.__responses[negotiator_id]
        if negotiator_id in self.__proposals.keys():
            del self.__proposals[negotiator_id]
        if negotiator_id in self.__n_waits.keys():
            del self.__n_waits[negotiator_id]
        results = super().on_negotiation_end(negotiator_id, state)
        if not self.negotiators:
            self.reset()
        return results


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

    def make_response(self) -> ResponseType:
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
                    response, self.negotiators[negotiator][0].nmi.random_outcomes(1)[0]
                )
            else:
                result[negotiator] = SAOResponse(response, None)
        return result

    def first_proposals(self):
        return dict(
            zip(
                self.negotiators.keys(),
                [n[0].nmi.random_outcomes(1)[0] for n in self.negotiators.values()],
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

    def best_outcome(
        self, negotiator: str, state: SAOState | None = None
    ) -> Outcome | None:
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
        if neg.nmi is None:
            return None
        ufun = None
        if self.has_ufun:
            ufun = self.ufun
        elif not self.has_ufun and hasattr(self, "ufun"):
            ufun = self.ufun
        if ufun is None:
            return None
        _, top_outcome = ufun.extreme_outcomes()
        return top_outcome

    @abstractmethod
    def is_better(
        self, a: Outcome | None, b: Outcome | None, negotiator: str, state: SAOState
    ) -> bool:
        """Compares two outcomes of the same negotiation

        Args:
            a: Outcome
            b: Outcome
            negotiator: The negotiator for which the comparison is to be made
            state: Current state of the negotiation

        Returns:
            True if utility(a) > utility(b)
        """

    def make_offer(
        self,
        negotiator: str,
        state: SAOState,
        best_offer: Outcome | None,
        best_from: str | None,
    ) -> Outcome | None:
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
        nmi = self.negotiators[negotiator][0].nmi
        if nmi is None:
            return None
        if best_offer is not None and outcome_is_valid(best_offer, nmi.issues):
            if self.is_better(best_offer, current_best, negotiator, state):
                return best_offer
            return current_best

    @abstractmethod
    def is_acceptable(self, offer: Outcome, source: str, state: SAOState) -> bool:
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
    def best_offer(self, offers: dict[str, Outcome]) -> str | None:
        """
        Return the ID of the negotiator with the best offer

        Args:
            offers: A mapping from negotiator ID to the offer it received

        Returns:
            The ID of the negotiator with best offer. Ties should be broken.
            Return None only if there is no way to calculate the best offer.
        """

    def counter_all(
        self, offers: dict[str, Outcome], states: dict[str, SAOState]
    ) -> dict[str, SAOResponse]:
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
                        negotiator=source,
                        state=states[source],
                        best_offer=None,
                        best_from=None,
                    )
                    if source == selected
                    else None
                    for source in partners
                ]
            else:
                current_offers = [
                    self.make_offer(
                        negotiator=source,
                        state=states[source],
                        best_offer=None,
                        best_from=None,
                    )
                    for source in partners
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
                    negotiator=source,
                    state=states[source],
                    best_offer=offers[partner],
                    best_from=partner,
                )
                if source == selected
                else None
                for source in partners
            ]
        else:
            # if not strict, make an offer to each partner including the best one.
            current_offers = [
                self.make_offer(
                    negotiator=source,
                    state=states[source],
                    best_offer=offers[partner],
                    best_from=partner,
                )
                for source in partners
            ]
        return dict(
            zip(
                partners,
                [SAOResponse(ResponseType.REJECT_OFFER, o) for o in current_offers],
            )
        )

    def first_proposals(self) -> dict[str, Outcome | None]:
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
        self, negotiator: str, state: SAOState, offer: Outcome
    ) -> Outcome | None:
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

    def after_join(  # type: ignore
        self, negotiator_id, nmi, state, *, preferences=None, role="negotiator"
    ):
        super().after_join(
            negotiator_id, nmi, state, preferences=preferences, role=role
        )
        self._best_outcomes[negotiator_id] = self.best_outcome(negotiator_id)


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

    def __init__(self, *args, meta_negotiator: GBNegotiator | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if meta_negotiator is None:
            meta_negotiator = AspirationNegotiator(
                name=f"{self.name}-negotiator", ufun=kwargs.get("ufun", None)
            )
        self.meta_negotiator = meta_negotiator

    def propose(self, negotiator_id: str, state: SAOState) -> Outcome | None:  # type: ignore
        """Uses the meta negotiator to propose"""
        negotiator, _ = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        self.meta_negotiator._nmi = negotiator.nmi
        return self.meta_negotiator.propose(state)

    def respond(
        self,
        negotiator_id: str,
        state: MechanismState,
        source: str | None = None,
    ) -> ResponseType:
        """Uses the meta negotiator to respond"""
        negotiator, _ = self._negotiators.get(negotiator_id, (None, None))
        if negotiator is None:
            raise ValueError(f"Unknown negotiator {negotiator_id}")
        self.meta_negotiator._nmi = negotiator.nmi
        return self.meta_negotiator.respond(state=state, source=source)  # type: ignore


class SAOSingleAgreementRandomController(SAOSingleAgreementController):
    """
    A single agreement controller that uses a random negotiation strategy.

    Args:
        p_acceptance: The probability that an offer is accepted

    """

    def __init__(self, *args, p_acceptance=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_acceptance = p_acceptance

    def is_acceptable(self, offer: Outcome, source: str, state: SAOState) -> bool:
        return random.random() > self.p_acceptance

    def best_offer(self, offers: dict[str, Outcome]) -> str | None:
        return random.sample(list(offers.keys()), 1)[0]

    def is_better(
        self, a: Outcome | None, b: Outcome | None, negotiator: str, state: SAOState
    ) -> bool:
        return random.random() > 0.5


class SAOSingleAgreementAspirationController(SAOSingleAgreementController):
    """
    A `SAOSingleAgreementController` that uses aspiration level to decide what
    to accept and what to propose.

    Args:
        preferences: The utility function to use for ALL negotiations
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
        aspiration_type: Literal["boulware", "conceder", "linear"]
        | int
        | float = "boulware",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.__asp = PolyAspiration(max_aspiration, aspiration_type)

    def utility_at(self, x):
        return self.__asp.utility_at(x)

    def is_acceptable(self, offer: Outcome, source: str, state: SAOState):
        if not self.ufun:
            return False
        return self.ufun(offer) >= self.utility_at(state.relative_time)

    def is_better(
        self, a: Outcome | None, b: Outcome | None, negotiator: str, state: SAOState
    ) -> bool:
        if not self.ufun:
            raise ValueError("No ufun is defined")
        return self.ufun.is_better(a, b)

    def best_offer(self, offers):
        if not self.ufun:
            raise ValueError("No ufun is defined")
        best_negotiator, best_offer = None, None
        for k, offer in offers.items():
            if best_negotiator is None:
                best_negotiator, best_offer = k, offer
                continue
            if self.ufun.is_better(offer, best_offer):
                best_negotiator, best_offer = k, offer
        return best_negotiator

    def best_outcome(self, negotiator, state=None):
        if not self.ufun:
            raise ValueError("No ufun is defined")
        outcome = self.ufun.sample_outcome_with_utility(
            rng=(
                self.utility_at(state.relative_time)
                if state is not None
                else float(self.ufun(super().best_outcome(negotiator, None))),
                float("inf"),
            ),
            issues=self.negotiators[negotiator][0].nmi.issues,
        )
        if outcome is None:
            return self._best_outcomes.get(negotiator, None)
        return outcome
