from __future__ import annotations
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING


from negmas.gb.negotiators.base import GBNegotiator
from negmas.outcomes.common import ExtendedOutcome
from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.preferences import Preferences
from negmas.warnings import warn
from time import sleep

from ...events import Notification
from ...negotiators import Controller
from ...outcomes import Outcome
from ..common import SAONMI, ResponseType, SAOResponse, SAOState

if TYPE_CHECKING:
    from negmas.situated import Agent

__all__ = ["SAONegotiator", "SAOPRNegotiator", "SAOCallNegotiator"]


class SAOPRNegotiator(GBNegotiator[SAONMI, SAOState]):
    """
    Base class for all SAO negotiators. Implemented by implementing propose() and respond() methods.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The utility function of the negotiator (overrides preferences if given)
         owner: The `Agent` that owns the negotiator.

    Remarks:
        - The only method that **must** be implemented by any SAONegotiator is `propose`.
        - The default `respond` method, accepts offers with a utility value no less than whatever `propose` returns
          with the same mechanism state.
        - A default implementation of respond() is provided which simply accepts any offer better than the last
          offer I gave or the next one I would have given in the current state.

    See Also:
        `SAOCallNegotiator`

    """

    def __init__(
        self,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        name: str | None = None,
        parent: Controller | None = None,
        owner: Agent | None = None,
        id: str | None = None,
        type_name: str | None = None,
        can_propose: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=name,
            preferences=preferences,
            ufun=ufun,
            parent=parent,
            owner=owner,
            id=id,
            type_name=type_name,
            **kwargs,
        )
        self.__end_negotiation = False
        self.__my_last_proposal: Outcome | ExtendedOutcome | None = None
        self.__my_last_proposal_time: int = -1
        self.add_capabilities(
            {"respond": True, "propose": can_propose, "max-proposals": 1}
        )

    def on_notification(self, notification: Notification, notifier: str):
        """
        Called whenever a notification is received

        Args:
            notification: The notification
            notifier: The notifier entity

        Remarks:
            - The default implementation only responds to end_negotiation by ending the negotiation
        """
        _ = notifier
        if notification.type == "end_negotiation":
            self.__end_negotiation = True

    def propose_(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """
        The method directly called by the mechanism (through `counter` ) to ask for a proposal

        Args:
            state: The mechanism state
            dest: the destination (can be ignored in AOP, SAOP, MAOP)

        Returns:
            An outcome to offer or None to refuse to offer

        Remarks:
            - Depending on the `SAOMechanism` settings, refusing to offer may be interpreted as ending the negotiation
            - The negotiator will only receive this call if it has the 'propose' capability.

        """
        if not self._capabilities["propose"] or self.__end_negotiation:
            return None
        if not state.running:
            warn(
                f"{self.name} asked to propose in a negotiation that is not running:\n{state}"
            )
            return None
        if (
            self.__my_last_proposal_time == state.step
            and self.__my_last_proposal is not None
        ):
            return self.__my_last_proposal
        try:
            self.__my_last_proposal = self.propose(
                state=state,
                dest=dest
                if dest or self.nmi.n_negotiators > 2
                else state.current_proposer,
            )
        except TypeError:
            self.__my_last_proposal = self.propose(state=state)

        self.__my_last_proposal_time = state.step
        return self.__my_last_proposal

    @abstractmethod
    def propose(  # type: ignore
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        ...

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:  # type: ignore
        """
        Called to respond to an offer. This is the method that should be overriden to provide an acceptance strategy.

        Args:
            state: a `SAOState` giving current state of the negotiation.
            source: The ID of the negotiator that gave this offer

        Returns:
            ResponseType: The response to the offer

        Remarks:
            - The default implementation never ends the negotiation
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).
            - The current offer to respond to can be accessed through `state.current_offer`

        """
        _ = source
        if not isinstance(state, SAOState):
            state = self._sao_state_from_gb_state(state)

        offer = state.current_offer
        # Always reject offers that we do not know or if we have no known preferences
        if offer is None or self.preferences is None:
            return ResponseType.REJECT_OFFER
        # if the offer is worse than the reserved value or its utility is less than that of the reserved outcome, reject it
        if (
            self.reserved_value is not None and self.preferences.is_worse(offer, None)
        ) or (
            self.reserved_outcome is not None
            and self.preferences.is_worse(offer, self.reserved_outcome)
        ):
            return ResponseType.REJECT_OFFER
        # find my last proposal (if any)
        myoffer = self.__my_last_proposal
        # if I never proposed, find what would I have proposed at this state and its utility
        if myoffer is None:
            myoffer = self.propose_(
                state=state, dest=None if self.nmi.n_negotiators > 2 else source
            )
            if myoffer is None:
                return ResponseType.REJECT_OFFER
        # accept only if I know what I would have proposed at this state (or the previous one) and it was worse than what I am about to proposed
        if self.preferences.is_not_worse(
            offer, myoffer.outcome if isinstance(myoffer, ExtendedOutcome) else myoffer
        ):
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER

    def respond_(self, state: SAOState, source: str | None = None) -> ResponseType:
        """The method to be called directly by the mechanism (through `counter` ) to respond to an offer.

        Args:
            state: a `SAOState` giving current state of the negotiation.
            source: The ID of the negotiator that gave the offer.

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
        if not state.running:
            warn(
                f"{self.name} asked to respond to a negotiation that is not running:\n{state}"
            )
            return ResponseType.END_NEGOTIATION
        if self.__end_negotiation:
            return ResponseType.END_NEGOTIATION
        try:
            return self.respond(state=state, source=source)
        except TypeError:
            return self.respond(state=state)

    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Called by `Negotiator.__call__` (which is called by the mechanism) to counter the offer.
        It just calls `respond_` and `propose_` as needed.

        Args:
            state: `SAOState` giving current state of the negotiation.
            dest: The ID of the destination of the response. May be empty under `SAOMechanism`

        Returns:
            Tuple[ResponseType, Outcome]: The response to the given offer with a counter offer if the response is REJECT

        Remarks:
            - The current offer is accessible through state.current_offer

        """
        offer = state.current_offer
        if self.__end_negotiation:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if self.has_ufun:
            changes = self.ufun.changes()  # type: ignore
            if changes:
                self.on_preferences_changed(changes)
        if offer is None:
            proposal = self.propose_(state=state, dest=dest)
            if isinstance(proposal, ExtendedOutcome):
                return SAOResponse(
                    ResponseType.REJECT_OFFER, proposal.outcome, proposal.data
                )
            return SAOResponse(ResponseType.REJECT_OFFER, proposal)
        try:
            response = self.respond_(state=state, source=state.current_proposer)
        except TypeError:
            response = self.respond_(state=state)
        if response != ResponseType.REJECT_OFFER:
            return SAOResponse(response, offer)
        proposal = self.propose_(state=state)
        if isinstance(proposal, ExtendedOutcome):
            return SAOResponse(response, proposal.outcome, proposal.data)
        return SAOResponse(response, proposal)


class SAOCallNegotiator(SAOPRNegotiator, ABC):
    """
    An SAO negotiator implemented by overriding __call__ to return a counter offer.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The utility function of the negotiator (overrides preferences if given)
         owner: The `Agent` that owns the negotiator.

    Remarks:
        - The only method that **must** be implemented by any SAONegotiator is `propose`.
        - The default `respond` method, accepts offers with a utility value no less than whatever `propose` returns
          with the same mechanism state.

    See Also:
        `SAOPRNegotiator`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__last_offer: dict[str | None, Outcome | ExtendedOutcome | None] = dict()
        self.__last_response: dict[str | None, ResponseType] = dict()

    @abstractmethod
    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """Implements counter-offering."""

    def propose(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        if dest not in self.__last_offer:
            resp = self(state, dest)
            self.__last_response[dest] = resp.response
            self.__last_offer[dest] = resp.outcome
        assert not isinstance(self.__last_offer, int)
        return self.__last_offer.pop(dest, None)

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
        _ = source
        if source not in self.__last_response:
            try:
                negotiators = self.nmi.negotiator_ids
                my_index = self.nmi.negotiator_index(self.id)
                next_index = (my_index + 1) % self.nmi.n_negotiators
                dest = negotiators[next_index]
            except Exception as e:
                warn(str(e))
                dest = None
            resp = self(state, dest)
            self.__last_response[source] = resp.response
            self.__last_offer[source] = resp.outcome
        return self.__last_response.pop(source, ResponseType.REJECT_OFFER)


class _InfiniteWaiter(SAOPRNegotiator):
    """Used only for testing: waits forever and never agrees to anything"""

    def __call__(self, state, dest: str | None = None) -> SAOResponse:
        _ = state, dest

        sleep(10000 * 60 * 60)
        return SAOResponse(ResponseType.REJECT_OFFER, self.nmi.random_outcome())


SAONegotiator = SAOPRNegotiator
"""Base of all SAO negotiators"""
