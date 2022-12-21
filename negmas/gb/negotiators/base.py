from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.preferences import Preferences

from ...negotiators import Controller, Negotiator
from ...outcomes import Outcome
from ..common import GBState, ResponseType, ThreadState

if TYPE_CHECKING:
    from negmas.sao.common import SAOResponse, SAOState
    from negmas.situated import Agent

__all__ = [
    "GBNegotiator",
]


class GBNegotiator(Negotiator):
    """
    Base class for all GB negotiators.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The utility function of the negotiator (overrides preferences if given)
         owner: The `Agent` that owns the negotiator.

    Remarks:
        - The only method that **must** be implemented by any GBNegotiator is `propose`.
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
        self.add_capabilities({"respond": True, "propose": True, "max-proposals": 1})
        self.__end_negotiation = False
        self.__received_offer: dict[str | None, Outcome | None] = defaultdict(
            lambda: None
        )

    @abstractmethod
    def propose(self, state: GBState) -> Outcome | None:
        """Propose an offer or None to refuse.

        Args:
            state: `GBState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """

    @abstractmethod
    def respond(self, state: GBState, offer: Outcome, source: str) -> ResponseType:
        """Called to respond to an offer. This is the method that should be overriden to provide an acceptance strategy.

        Args:
            state: a `GBState` giving current state of the negotiation.
            offer: offer being tested

        Returns:
            ResponseType: The response to the offer

        Remarks:
            - The default implementation never ends the negotiation
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).

        """

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something

        Args:
            state: `GBState` giving the state of the negotiation when the offer was porposed.
            partner_id: The ID of the agent who proposed
            offer: The proposal.

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """

    def on_partner_response(
        self,
        state: GBState,
        partner_id: str,
        outcome: Outcome,
        response: ResponseType,
    ) -> None:
        """
        A callback called by the mechanism when a partner responds to some offer

        Args:
            state: `GBState` giving the state of the negotiation when the partner responded.
            partner_id: The ID of the agent who responded
            outcome: The proposal being responded to.
            response: The response

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """

    def on_partner_ended(self, partner: str):
        """
        Called when a partner ends the negotiation.

        Note  that the negotiator owning this component may never receive this
        offer.
        This is only receivd if the mechanism is sending notifications
        on every offer.
        """

    # compatibility with SAOMechanism
    def __call__(self, state: SAOState, offer: Outcome | None) -> SAOResponse:
        """
        Called by the mechanism to counter the offer. It just calls `respond_` and `propose_` as needed.

        Args:
            state: `SAOState` giving current state of the negotiation.
            offer: The offer to be countered. None means no offer and the agent is requested to propose an offer

        Returns:
            Tuple[ResponseType, Outcome]: The response to the given offer with a counter offer if the response is REJECT

        """
        from negmas.sao.common import SAOResponse

        if self.__end_negotiation:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if self.ufun is not None:
            changes = self.ufun.changes()
            if changes:
                self.on_preferences_changed(changes)
        if offer is None:
            return SAOResponse(ResponseType.REJECT_OFFER, self.propose_(state=state))
        response = self.respond_(state=state, offer=offer)
        if response != ResponseType.REJECT_OFFER:
            return SAOResponse(response, None)
        return SAOResponse(response, self.propose_(state=state))

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
        return self.propose(
            state=self._state_from_sao_state(state),
        )

    def respond_(self, state: SAOState, offer: Outcome) -> ResponseType:
        """The method to be called directly by the mechanism (through `counter` ) to respond to an offer.

        Args:
            state: a `SAOState` giving current state of the negotiation.
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
        self.__received_offer[state.current_proposer] = offer

        return self.respond(
            state=self._state_from_sao_state(state),
            offer=offer,
            source=state.current_proposer if state.current_proposer else "",
        )

    def _state_from_sao_state(self, state: SAOState) -> GBState:
        if not self.nmi:
            raise ValueError("No NMI. Cannot convert SAOState to GBState")
        threads = {
            source: ThreadState(
                current_offer=None,
                new_offer=self.__received_offer.get(state.current_proposer, None),
            )
            for source in self.nmi.negotiator_ids
        }

        return GBState(
            running=state.running,
            waiting=state.waiting,
            started=state.started,
            step=state.step,
            time=state.time,
            relative_time=state.relative_time,
            broken=state.broken,
            timedout=state.timedout,
            agreement=state.agreement,
            results=state.results,
            n_negotiators=state.n_negotiators,
            has_error=state.has_error,
            error_details=state.error_details,
            threads=threads,
        )
