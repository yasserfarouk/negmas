from __future__ import annotations

from abc import abstractmethod
from collections import namedtuple
from typing import TYPE_CHECKING

from attr import define, field

from ...negotiators.components.component import Component

if TYPE_CHECKING:
    from negmas import ResponseType
    from negmas.gb import GBNegotiator, GBState
    from negmas.outcomes import Outcome

__all__ = [
    "GBComponent",
    "AcceptancePolicy",
    "OfferingPolicy",
    "ProposalPolicy",
    "Model",
]


@define
class GBComponent(Component):
    _negotiator: GBNegotiator | None = field(default=None, kw_only=True)

    def before_proposing(self, state: GBState):
        """
        Called before proposing
        """

    def after_proposing(self, state: GBState, offer: Outcome | None):
        """
        Called after proposing
        """

    def before_responding(self, state: GBState, offer: Outcome | None, source: str):
        """
        Called before offering
        """

    def after_responding(
        self,
        state: GBState,
        offer: Outcome | None,
        response: ResponseType,
        source: str,
    ):
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

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """
        A callback called by the mechanism when a partner proposes something

        Args:
            state: `MechanismState` giving the state of the negotiation when the offer was porposed.
            partner_id: The ID of the agent who proposed
            offer: The proposal.

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """

    def on_partner_refused_to_propose(self, state: GBState, partner_id: str) -> None:
        """
        A callback called by the mechanism when a partner refuses to propose

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner refused to offer.
            partner_id: The ID of the agent who refused to propose

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """

    def on_partner_response(
        self,
        state: GBState,
        partner_id: str,
        outcome: Outcome | None,
        response: ResponseType,
    ) -> None:
        """
        A callback called by the mechanism when a partner responds to some offer

        Args:
            state: `MechanismState` giving the state of the negotiation when the partner responded.
            partner_id: The ID of the agent who responded
            outcome: The proposal being responded to.
            response: The response

        Remarks:
            - Will only be called if `enable_callbacks` is set for the mechanism
        """


@define
class AcceptancePolicy(GBComponent):
    def respond(
        self, state: GBState, offer: Outcome | None, source: str
    ) -> ResponseType:
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
        return self(state, offer, source)

    def __not__(self):
        return RejectionPolicy(self)

    @abstractmethod
    def __call__(
        self, state: GBState, offer: Outcome | None, source: str
    ) -> ResponseType:
        ...

    def __and__(self, s: AcceptancePolicy):
        from .acceptance import AllAcceptanceStrategies

        if self.negotiator != s.negotiator:
            raise ValueError(f"Cannot combine strategies with different negotiators")
        return AllAcceptanceStrategies([self, s])

    def __or__(self, s: AcceptancePolicy):
        from .acceptance import AnyAcceptancePolicy

        if self.negotiator != s.negotiator:
            raise ValueError(f"Cannot combine strategies with different negotiators")
        return AnyAcceptancePolicy([self, s])


@define
class RejectionPolicy(AcceptancePolicy):
    """
    Reverses the decision of an acceptance strategy (Rejects if the original was accepting and Accepts in any other case)
    """

    a: AcceptancePolicy

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str
    ) -> ResponseType:
        response = self.a(state, offer, source)
        if response == ResponseType.ACCEPT_OFFER:
            return ResponseType.REJECT_OFFER
        return ResponseType.ACCEPT_OFFER


@define
class OfferingPolicy(GBComponent):
    _current_offer: tuple[int, int, Outcome | None] = field(
        init=False, default=(-1, None)
    )

    def propose(self, state: GBState) -> Outcome | None:
        """Propose an offer or None to refuse.

        Args:
            state: `GBState` giving current state of the negotiation.
            source: the thread in which I am supposed to offer.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - Caches results for the same thread and step. If called multiple times for the same thread and step, it will do the computations only once.
            - Caching is useful when the acceptance strategy calls the offering strategy

        """
        source = self.negotiator.id
        if self._current_offer[0] != state.step or self._current_offer[1] != source:
            offer = self(state)
            self._current_offer = (state.step, source, offer)
        return self._current_offer[2]

    @abstractmethod
    def __call__(self, state: GBState) -> Outcome | None:
        ...

    def __and__(self, s: OfferingPolicy):
        from .offering import UnanimousConcensusOfferingPolicy

        if self.negotiator != s.negotiator:
            raise ValueError(f"Cannot combine strategies with different negotiators")
        return UnanimousConcensusOfferingPolicy([self, s])

    def __or__(self, s: OfferingPolicy):
        from .offering import RandomConcensusOfferingPolicy

        if self.negotiator != s.negotiator:
            raise ValueError(f"Cannot combine strategies with different negotiators")
        return RandomConcensusOfferingPolicy([self, s])


ProposalPolicy = OfferingPolicy
"""An alias for `OfferingStrategy` """

Model = GBComponent
"""An alias for `GBComponent` """


FilterResult = namedtuple("FilterResult", ["next", "save"])
