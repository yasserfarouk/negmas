from __future__ import annotations

from abc import abstractmethod
from collections import namedtuple
from typing import TYPE_CHECKING

from attr import define, field

from ...negotiators.components.component import Component

if TYPE_CHECKING:
    from negmas import ResponseType
    from negmas.outcomes import Outcome
    from negmas.sao import SAONegotiator, SAOState

__all__ = [
    "SAOComponent",
    "AcceptanceStrategy",
    "OfferingStrategy",
    "ProposalStrategy",
    "Model",
]


@define
class SAOComponent(Component):
    _negotiator: SAONegotiator | None = field(default=None, kw_only=True)

    def before_proposing(self, state: SAOState):
        """
        Called before proposing
        """

    def after_proposing(self, state: SAOState, offer: Outcome | None):
        """
        Called after proposing
        """

    def before_responding(self, state: SAOState, offer: Outcome | None):
        """
        Called before offering
        """

    def after_responding(
        self, state: SAOState, offer: Outcome | None, response: ResponseType
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
        self, state: SAOState, partner_id: str, offer: Outcome
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

    def on_partner_refused_to_propose(self, state: SAOState, partner_id: str) -> None:
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
        state: SAOState,
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
class AcceptanceStrategy(SAOComponent):
    def respond(self, state: SAOState, offer: Outcome | None) -> ResponseType:
        """Called to respond to an offer. This is the method that should be overriden to provide an acceptance strategy.

        Args:
            state: a `SAOState` giving current state of the negotiation.
            offer: offer being tested

        Returns:
            ResponseType: The response to the offer

        Remarks:
            - The default implementation never ends the negotiation
            - The default implementation asks the negotiator to `propose`() and accepts the `offer` if its utility was
              at least as good as the offer that it would have proposed (and above the reserved value).

        """
        return self(state, offer)

    def __not__(self):
        return RejectionStrategy(self)

    @abstractmethod
    def __call__(self, state: SAOState, offer: Outcome | None) -> ResponseType:
        return self.respond(state, offer)

    def __and__(self, s: AcceptanceStrategy):
        from .acceptance import AllAcceptanceStrategies

        if self.negotiator != s.negotiator:
            raise ValueError(f"Cannot combine strategies with different negotiators")
        return AllAcceptanceStrategies([self, s])

    def __or__(self, s: AcceptanceStrategy):
        from .acceptance import AnyAcceptanceStrategy

        if self.negotiator != s.negotiator:
            raise ValueError(f"Cannot combine strategies with different negotiators")
        return AnyAcceptanceStrategy([self, s])


@define
class RejectionStrategy(AcceptanceStrategy):
    """
    Reverses the decision of an acceptance strategy (Rejects if the original was accepting and Accepts in any other case)
    """

    a: AcceptanceStrategy

    def __call__(self, state: SAOState, offer: Outcome | None) -> ResponseType:
        response = self.a(state, offer)
        if response == ResponseType.ACCEPT_OFFER:
            return ResponseType.REJECT_OFFER
        return ResponseType.ACCEPT_OFFER


@define
class OfferingStrategy(SAOComponent):
    _current_offer: tuple[int, Outcome | None] = field(init=False, default=(-1, None))

    def propose(self, state: SAOState) -> Outcome | None:
        """Propose an offer or None to refuse.

        Args:
            state: `SAOState` giving current state of the negotiation.

        Returns:
            The outcome being proposed or None to refuse to propose

        Remarks:
            - This function guarantees that no agents can propose something with a utility value

        """
        if self._current_offer[0] != state.step:
            offer = self(state)
            self._current_offer = (state.step, offer)
        return self._current_offer[1]

    @abstractmethod
    def __call__(self, state: SAOState) -> Outcome | None:
        ...

    def __and__(self, s: OfferingStrategy):
        from .offering import UnanimousConcensusOfferingStrategy

        if self.negotiator != s.negotiator:
            raise ValueError(f"Cannot combine strategies with different negotiators")
        return UnanimousConcensusOfferingStrategy([self, s])

    def __or__(self, s: OfferingStrategy):
        from .offering import RandomConcensusOfferingStrategy

        if self.negotiator != s.negotiator:
            raise ValueError(f"Cannot combine strategies with different negotiators")
        return RandomConcensusOfferingStrategy([self, s])


ProposalStrategy = OfferingStrategy
"""An alias for `OfferingStrategy` """

Model = SAOComponent
"""An alias for `SAOComponent` """


FilterResult = namedtuple("FilterResult", ["next", "save"])
