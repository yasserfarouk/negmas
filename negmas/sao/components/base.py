from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from negmas.negotiators.components import Component

if TYPE_CHECKING:
    from negmas import ResponseType
    from negmas.outcomes import Outcome
    from negmas.sao import SAONegotiator, SAOState

__all__ = [
    "SAOComponent",
    "SAODoNothingComponent",
]


class SAOComponent(Component, Protocol):
    def before_proposing(self, state: SAOState):
        """
        Called before proposing
        """

    def after_proposing(self, state: SAOState, offer: Outcome):
        """
        Called after proposing
        """

    def before_responding(self, state: SAOState, offer: Outcome):
        """
        Called before offering
        """

    def after_responding(self, state: SAOState, offer: Outcome, response: ResponseType):
        """
        Called before offering
        """

    def on_partner_joined(self, partner: str):
        """
        Called when a partner joins the negotiation.

        This is only receivd if the mechanism is sending notifications.
        """

    def on_partner_left(self, partner: str):
        """
        Called when a partner leaves the negotiation.

        This is only receivd if the mechanism is sending notifications.
        """

    def on_partner_ended(self, partner: str):
        """
        Called when a partner ends the negotiation.

        Note  that the negotiator owning this component may never receive this
        offer.
        This is only receivd if the mechanism is sending notifications
        on every offer.
        """

    def on_partner_offered(
        self,
        partner: str,
        offer: Outcome | None,
        state: SAOState = None,
    ):
        """
        Called when any partner in the negotiation offers.

        Note  that the negotiator owning this component may never receive this
        offer. This is only receivd if the mechanism is sending notifications
        on every offer.
        """

    def on_partner_rejected(
        self,
        partner: str,
        offer: Outcome | None,
        state: SAOState,
    ):
        """
        Called when any partner in the negotiation rejects an offer.

        Note  that the negotiator owning this component may never receive this
        offer. This is only receivd if the mechanism is sending notifications
        on every offer.
        """

    def on_partner_accepted(
        self,
        partner: str,
        offer: Outcome | None,
        state: SAOState,
    ):
        """
        Called when any partner in the negotiation accepts an offer.

        Note  that the negotiator owning this component may never receive this
        offer. This is only receivd if the mechanism is sending notifications
        on every offer.
        """


@dataclass
class SAODoNothingComponent(SAOComponent):
    _negotiator: SAONegotiator | None = field(init=False)

    def set_negotiator(self, negotiator: SAONegotiator) -> None:
        self._negotiator = negotiator
