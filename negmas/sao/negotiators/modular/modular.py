from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ....negotiators.modular import ModularNegotiator
from ..base import SAONegotiator

if TYPE_CHECKING:
    from ....outcomes import Outcome
    from ...common import ResponseType
    from ...components import SAOComponent

if TYPE_CHECKING:
    from negmas.sao import SAOState

__all__ = ["SAOModularNegotiator"]


class SAOModularNegotiator(ModularNegotiator, SAONegotiator):
    """
    A generic modular SAO negotiator.
    """

    _components: list[SAOComponent]

    def components(self) -> tuple[SAOComponent, ...]:
        return super().components  # type: ignore

    @abstractmethod
    def generate_response(self, state: SAOState, offer: Outcome) -> ResponseType:
        ...

    @abstractmethod
    def generate_proposal(self, state: SAOState) -> Outcome | None:
        ...

    def propose(self, state: SAOState) -> Outcome | None:
        for c in self._components:
            c.before_proposing(state)
        offer = self.generate_proposal(state)
        for c in self._components:
            c.after_proposing(state, offer=offer)
        return offer

    def respond(self, state: SAOState, offer: Outcome) -> ResponseType:
        for c in self._components:
            c.before_responding(state=state, offer=offer)
        response = self.generate_response(state=state, offer=offer)
        for c in self._components:
            c.after_responding(state=state, offer=offer, response=response)
        return response

    def on_partner_joined(self, partner: str):
        """
        Called when a partner joins the negotiation.

        This is only receivd if the mechanism is sending notifications.
        """
        for c in self._components:
            c.on_partner_joined(partner)

    def on_partner_left(self, partner: str):
        """
        Called when a partner leaves the negotiation.

        This is only receivd if the mechanism is sending notifications.
        """
        for c in self._components:
            c.on_partner_left(partner)

    def on_partner_ended(self, partner: str):
        """
        Called when a partner ends the negotiation.

        Note  that the negotiator owning this component may never receive this
        offer.
        This is only receivd if the mechanism is sending notifications
        on every offer.
        """
        for c in self._components:
            c.on_partner_ended(partner)

    def on_partner_proposal(
        self, state: SAOState, partner_id: str, offer: Outcome
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something

        Args:
            state: `SAOState` giving the state of the negotiation when the offer was porposed.
            partner_id: The ID of the agent who proposed
            offer: The proposal.

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """
        for c in self._components:
            c.on_partner_proposal(state=state, partner_id=partner_id, offer=offer)

    def on_partner_refused_to_propose(self, state: SAOState, partner_id: str) -> None:
        """
        A callback called by the mechanism when a partner refuses to propose

        Args:
            state: `SAOState` giving the state of the negotiation when the partner refused to offer.
            agent_id: The ID of the agent who refused to propose

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """
        for c in self._components:
            c.on_partner_refused_to_propose(state=state, partner_id=partner_id)

    def on_partner_response(
        self,
        state: SAOState,
        partner_id: str,
        outcome: Outcome,
        response: ResponseType,
    ) -> None:
        """
        A callback called by the mechanism when a partner responds to some offer

        Args:
            state: `SAOState` giving the state of the negotiation when the partner responded.
            partner_id: The ID of the agent who responded
            outcome: The proposal being responded to.
            response: The response

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """
        for c in self._components:
            c.on_partner_response(
                state=state, partner_id=partner_id, outcome=outcome, response=response
            )
