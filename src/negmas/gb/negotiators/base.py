from __future__ import annotations
from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Generic, TypeVar

from negmas.outcomes.common import ExtendedOutcome
from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.preferences import Preferences

from ...negotiators import Controller, Negotiator
from ...outcomes import Outcome
from ..common import GBNMI, GBState, ResponseType, ThreadState
from ...common import NegotiatorMechanismInterface, MechanismState

if TYPE_CHECKING:
    from negmas.sao.common import SAOResponse, SAOState
    from negmas.situated import Agent

__all__ = ["GBNegotiator"]

TNMI = TypeVar("TNMI", bound=NegotiatorMechanismInterface)
TState = TypeVar("TState", bound=MechanismState)


def none_return():
    return None


class GBNegotiator(Negotiator[GBNMI, GBState], Generic[TNMI, TState]):
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
        self.add_capabilities({"respond": True, "propose": True, "max-proposals": 1})
        self.__end_negotiation = False
        self.__received_offer: dict[str | None, Outcome | None] = defaultdict(
            none_return
        )

    @abstractmethod
    def propose(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Propose an offer or None to refuse.

        Args:
            state: `GBState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """

    @abstractmethod
    def respond(self, state: GBState, source: str | None) -> ResponseType:
        """Called to respond to an offer. This is the method that should be overriden to provide an acceptance strategy.

        Args:
            state: a `GBState` giving current state of the negotiation.

        Returns:
            ResponseType: The response to the offer

        Remarks:
            - The default implementation never ends the negotiation
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).
            - Current offer is accessible through state.threads[source].current_offer as long as source != None otherwise it is None

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
        self, state: GBState, partner_id: str, outcome: Outcome, response: ResponseType
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

        Note that the negotiator owning this component may never receive this
        offer. This is only received if the mechanism is sending notifications
        on every offer.
        """

    # compatibility with SAOMechanism
    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """
        Called by the mechanism to counter the offer. It just calls `respond_` and `propose_` as needed.

        Args:
            state: `SAOState` giving current state of the negotiation.
            dest: The partner to respond to with a counter offer (for AOP, SAOP, MAOP this can safely be ignored).

        Returns:
            Tuple[ResponseType, Outcome]: The response to the given offer with a counter offer if the response is REJECT

        """
        from negmas.sao.common import SAOResponse

        offer = state.current_offer

        if self.__end_negotiation:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)
        if self.preferences is not None:
            changes = self.preferences.changes()
            if changes:
                self.on_preferences_changed(changes)
        if offer is None:
            proposal = self.propose_(state=state)
            if isinstance(proposal, ExtendedOutcome):
                return SAOResponse(
                    ResponseType.REJECT_OFFER, proposal.outcome, proposal.data
                )
            return SAOResponse(ResponseType.REJECT_OFFER, proposal)
        response = self.respond_(
            state=state, source=state.current_proposer if state.current_proposer else ""
        )
        if response == ResponseType.ACCEPT_OFFER:
            return SAOResponse(response, offer)
        if response != ResponseType.REJECT_OFFER:
            return SAOResponse(response, None)
        proposal = self.propose_(state=state, dest=dest)
        if isinstance(proposal, ExtendedOutcome):
            return SAOResponse(
                ResponseType.REJECT_OFFER, proposal.outcome, proposal.data
            )
        return SAOResponse(response, proposal)

    def propose_(
        self, state: SAOState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        if not self._capabilities["propose"] or self.__end_negotiation:
            return None
        return self.propose(state=self._gb_state_from_sao_state(state), dest=dest)

    def respond_(self, state: SAOState, source: str | None = None) -> ResponseType:
        offer = state.current_offer
        if self.__end_negotiation:
            return ResponseType.END_NEGOTIATION
        self.__received_offer[state.current_proposer] = offer

        return self.respond(state=self._gb_state_from_sao_state(state), source=source)

    def _gb_state_from_sao_state(self, state: SAOState) -> GBState:
        if isinstance(state, GBState):
            return state
        if not self.nmi:
            raise ValueError("No NMI. Cannot convert SAOState to GBState")
        threads = {
            source: ThreadState(
                new_offer=self.__received_offer.get(state.current_proposer, None)
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

    def _sao_state_from_gb_state(self, state: GBState) -> SAOState:
        if isinstance(state, SAOState):
            return state
        if not self.nmi:
            raise ValueError("No NMI. Cannot convert SAOState to GBState")
        # TODO: correct all of this nonsense
        if state.last_thread:
            offerer = state.last_thread
            owner = self.nmi._mechanism._negotiator_map[offerer].owner
            aid = owner.id if owner else None
            current_offer = state.threads[state.last_thread].new_offer
            current_data = state.threads[state.last_thread].new_data
            current_proposer = None
            current_proposer_agent = None
            n_acceptances = len(
                [
                    (offerer, _)
                    for _ in state.threads[state.last_thread].new_responses.values()
                    if _ == ResponseType.ACCEPT_OFFER
                ]
            )
            new_offers = [(offerer, state.threads[state.last_thread].new_offer)]
            new_data = [(offerer, state.threads[state.last_thread].new_data)]
            new_offerer_agents = [aid]
            last_negotiator = None
        else:
            current_offer = None
            current_data = None
            current_proposer = None
            current_proposer_agent = None
            n_acceptances = 0
            new_offers = []
            new_data = []
            new_offerer_agents = []
            last_negotiator = None
        return SAOState(
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
            current_offer=current_offer,
            current_data=current_data,
            current_proposer=current_proposer,
            current_proposer_agent=current_proposer_agent,
            n_acceptances=n_acceptances,
            new_offers=new_offers,
            new_data=new_data,
            new_offerer_agents=new_offerer_agents,
            last_negotiator=last_negotiator,
        )
