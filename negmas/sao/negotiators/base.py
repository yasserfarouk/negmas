from __future__ import annotations

from abc import abstractmethod
from multiprocessing.process import current_process
from typing import TYPE_CHECKING

from negmas import Value
from negmas.gb.negotiators.base import GBNegotiator
from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.preferences import Preferences
from negmas.warnings import NegmasUnexpectedValueWarning, warn

from ...events import Notification
from ...negotiators import Controller
from ...outcomes import Outcome
from ..common import ResponseType, SAOResponse

if TYPE_CHECKING:
    from negmas.sao import SAOState
    from negmas.situated import Agent

__all__ = [
    "SAONegotiator",
]


class SAONegotiator(GBNegotiator):
    """
    Base class for all SAO negotiators.

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
        )
        self.__end_negotiation = False
        self.__my_last_proposal: Outcome | None = None
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
        if notification.type == "end_negotiation":
            self.__end_negotiation = True

    def propose_(self, state: SAOState) -> Outcome | None:
        """
        The method directly called by the mechanism (through `counter` ) to ask for a proposal

        Args:
            state: The mechanism state

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
        self.__my_last_proposal = self.propose(state=state)
        self.__my_last_proposal_time = state.step
        return self.__my_last_proposal

    def propose(self, state: SAOState) -> Outcome | None:
        return None

    def respond(self, state: SAOState, source: str | None = None) -> ResponseType:
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
        offer = state.current_offer
        # Always reject offers that we do not know or if we have no kknown preferences
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
            myoffer = self.propose_(state=state)
            if myoffer is None:
                return ResponseType.REJECT_OFFER
        # accept only if I know what I would have proposed at this state (or the previous one) and it was worse than what I am about to proposed
        if self.preferences.is_not_worse(offer, myoffer):
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
        offer = state.current_offer
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

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Called by `Negotiator.__call__` (which is called by the mechanism) to counter the offer.
        It just calls `respond_` and `propose_` as needed.

        Args:
            state: `SAOState` giving current state of the negotiation.

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
            return SAOResponse(ResponseType.REJECT_OFFER, self.propose_(state=state))
        try:
            response = self.respond_(
                state=state,
                source=state.current_proposer if state.current_proposer else "",
            )
        except TypeError:
            response = self.respond_(
                state=state,
            )
        if response != ResponseType.REJECT_OFFER:
            return SAOResponse(response, offer)
        return SAOResponse(response, self.propose_(state=state))
