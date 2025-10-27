"""Module for modular functionality."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable

from negmas.gb.common import get_offer

from ....negotiators.modular import ModularNegotiator
from ..base import GBNegotiator

if TYPE_CHECKING:
    from ....outcomes import Outcome
    from ...common import ResponseType
    from ...components import GBComponent

if TYPE_CHECKING:
    from negmas.gb import GBState

__all__ = ["GBModularNegotiator"]


class GBModularNegotiator(ModularNegotiator, GBNegotiator):
    """
    A generic modular GB negotiator.
    """

    def __init__(
        self,
        *args,
        components: Iterable[GBComponent],
        component_names: Iterable[str] | None = None,
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            *args, components=components, component_names=component_names, **kwargs
        )
        self._components: list[GBComponent]  # type: ignore

    @abstractmethod
    def generate_response(
        self, state: GBState, offer: Outcome | None, source: str | None = None
    ) -> ResponseType:
        """Generate response.

        Args:
            state: Current state.
            offer: Offer being considered.
            source: Source identifier.

        Returns:
            ResponseType: The result.
        """
        ...

    @abstractmethod
    def generate_proposal(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | None:
        """Generate proposal.

        Args:
            state: Current state.
            dest: Dest.

        Returns:
            Outcome | None: The result.
        """
        ...

    def propose(self, state: GBState, dest: str | None = None) -> Outcome | None:
        """Propose.

        Args:
            state: Current state.
            dest: Dest.

        Returns:
            Outcome | None: The result.
        """
        for c in self._components:
            c.before_proposing(state, dest=dest)
        offer = self.generate_proposal(state, dest=dest)
        for c in self._components:
            c.after_proposing(state, offer=offer, dest=dest)
        return offer

    def respond(self, state: GBState, source: str | None = None) -> ResponseType:
        """Respond.

        Args:
            state: Current state.
            source: Source identifier.

        Returns:
            ResponseType: The result.
        """
        offer = get_offer(state, source)
        for c in self._components:
            c.before_responding(state=state, offer=offer, source=source)
        response = self.generate_response(state=state, offer=offer, source=source)
        for c in self._components:
            c.after_responding(
                state=state, offer=offer, response=response, source=source
            )
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
        for c in self._components:
            c.on_partner_proposal(state=state, partner_id=partner_id, offer=offer)

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
        for c in self._components:
            c.on_partner_response(
                state=state, partner_id=partner_id, outcome=outcome, response=response
            )
