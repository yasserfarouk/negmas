"""GB Meta-negotiator that combines multiple GB negotiators with aggregation strategies."""

from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Iterable

from negmas.common import MechanismState, NegotiatorMechanismInterface
from negmas.negotiators.meta import MetaNegotiator
from negmas.outcomes import Outcome
from negmas.outcomes.common import ExtendedOutcome

from ..common import GBState, ResponseType, ExtendedResponseType, get_offer
from .base import GBNegotiator

if TYPE_CHECKING:
    from negmas.preferences import BaseUtilityFunction, Preferences

__all__ = ["GBMetaNegotiator"]


def none_return():
    """A helper function that always returns None."""
    return None


class GBMetaNegotiator(MetaNegotiator, GBNegotiator):
    """
    A meta-negotiator for GB (General Bargaining) protocols that aggregates
    multiple `GBNegotiator` instances.

    Unlike `GBModularNegotiator` which uses `GBComponent` behavior pieces,
    `GBMetaNegotiator` works with complete `GBNegotiator` instances. This
    allows for ensemble strategies where multiple negotiators can vote on
    proposals or responses.

    Subclasses must implement `aggregate_proposals` and `aggregate_responses`
    to define how proposals and responses from sub-negotiators are combined.

    Args:
        negotiators: An iterable of `GBNegotiator` instances to manage.
        negotiator_names: Optional names for the negotiators.
        share_ufun: If True (default), sub-negotiators will share the parent's ufun.
        share_nmi: If True (default), sub-negotiators will receive the parent's NMI on join.
        *args: Additional positional arguments passed to the base class.
        **kwargs: Additional keyword arguments passed to the base class.

    Remarks:
        - `propose` collects proposals from all sub-negotiators and aggregates them.
        - `respond` collects responses from all sub-negotiators and aggregates them.
        - All GB-specific callbacks are delegated to all sub-negotiators.
    """

    def __init__(
        self,
        *args,
        negotiators: Iterable[GBNegotiator],
        negotiator_names: Iterable[str] | None = None,
        share_ufun: bool = True,
        share_nmi: bool = True,
        **kwargs,
    ):
        """Initialize the GBMetaNegotiator.

        Args:
            *args: Positional arguments for the base MetaNegotiator.
            negotiators: The GB sub-negotiators to manage.
            negotiator_names: Optional names for the sub-negotiators.
            share_ufun: Whether sub-negotiators should share the parent's ufun.
            share_nmi: Whether sub-negotiators should receive the parent's NMI.
            **kwargs: Keyword arguments for the base MetaNegotiator.
        """
        # Initialize with empty negotiators first, then add them
        super().__init__(
            *args,
            negotiators=[],  # Start with empty, add later
            negotiator_names=None,
            share_ufun=share_ufun,
            share_nmi=share_nmi,
            **kwargs,
        )
        self._negotiators: list[GBNegotiator]  # type: ignore

        # Now add the negotiators
        import itertools

        for neg, name in zip(
            negotiators,
            negotiator_names if negotiator_names else itertools.repeat(None),
        ):
            self.add_negotiator(neg, name=name)

        self.__end_negotiation = False
        self.__received_offer: dict[str | None, Outcome | ExtendedOutcome | None] = (
            defaultdict(none_return)
        )

    @property
    def gb_negotiators(self) -> tuple[GBNegotiator, ...]:
        """Return the tuple of GB sub-negotiators.

        Returns:
            A tuple of all GB sub-negotiators.
        """
        return tuple(self._negotiators)  # type: ignore

    # Abstract aggregation methods that subclasses must implement

    @abstractmethod
    def aggregate_proposals(
        self,
        state: GBState,
        proposals: list[tuple[GBNegotiator, Outcome | ExtendedOutcome | None]],
        dest: str | None = None,
    ) -> Outcome | ExtendedOutcome | None:
        """Aggregate proposals from all sub-negotiators into a single proposal.

        Args:
            state: The current GB state.
            proposals: List of (negotiator, proposal) tuples from sub-negotiators.
            dest: The destination partner ID (if applicable).

        Returns:
            The aggregated proposal, or None to refuse to propose.
        """
        ...

    @abstractmethod
    def aggregate_responses(
        self,
        state: GBState,
        responses: list[tuple[GBNegotiator, ResponseType | ExtendedResponseType]],
        offer: Outcome | None,
        source: str | None = None,
    ) -> ResponseType | ExtendedResponseType:
        """Aggregate responses from all sub-negotiators into a single response.

        Args:
            state: The current GB state.
            responses: List of (negotiator, response) tuples from sub-negotiators.
            offer: The offer being responded to.
            source: The source partner ID (if applicable).

        Returns:
            The aggregated response.
        """
        ...

    # GB protocol methods - override to use aggregation

    def propose(
        self, state: GBState, dest: str | None = None
    ) -> Outcome | ExtendedOutcome | None:
        """Collect proposals from all sub-negotiators and aggregate them.

        Args:
            state: The current GB state.
            dest: The destination partner ID (if applicable).

        Returns:
            The aggregated proposal.
        """
        proposals: list[tuple[GBNegotiator, Outcome | ExtendedOutcome | None]] = []
        for neg in self._negotiators:
            proposal = neg.propose(state, dest=dest)
            proposals.append((neg, proposal))
        return self.aggregate_proposals(state, proposals, dest=dest)

    def respond(
        self, state: GBState, source: str | None = None
    ) -> ResponseType | ExtendedResponseType:
        """Collect responses from all sub-negotiators and aggregate them.

        Args:
            state: The current GB state.
            source: The source partner ID.

        Returns:
            The aggregated response.
        """
        offer = get_offer(state, source)
        responses: list[tuple[GBNegotiator, ResponseType | ExtendedResponseType]] = []
        for neg in self._negotiators:
            response = neg.respond(state, source=source)
            responses.append((neg, response))
        return self.aggregate_responses(state, responses, offer, source=source)

    # Override join to handle sub-negotiators properly
    def join(
        self,
        nmi: NegotiatorMechanismInterface,
        state: MechanismState,
        *,
        preferences: Preferences | None = None,
        ufun: BaseUtilityFunction | None = None,
        role: str = "negotiator",
    ) -> bool:
        """Join a negotiation and have sub-negotiators join too.

        Args:
            nmi: The negotiator-mechanism interface.
            state: The current mechanism state.
            preferences: Optional preferences for this negotiator.
            ufun: Optional utility function (overrides preferences).
            role: The role in the negotiation.

        Returns:
            True if successfully joined, False otherwise.
        """
        # Use GBNegotiator's join (which handles capabilities etc.)
        joined = GBNegotiator.join(
            self, nmi, state, preferences=preferences, ufun=ufun, role=role
        )
        if not joined:
            return False

        if self._share_nmi:
            # Have sub-negotiators join with shared NMI and optionally shared ufun
            sub_ufun = self.ufun if self._share_ufun else None
            sub_prefs = self.preferences if self._share_ufun and not sub_ufun else None
            for neg in self._negotiators:
                neg.join(nmi, state, preferences=sub_prefs, ufun=sub_ufun, role=role)

        return True

    # GB-specific callbacks - delegate to all sub-negotiators

    def on_partner_joined(self, partner: str) -> None:
        """Notify all sub-negotiators that a partner joined."""
        for neg in self._negotiators:
            if hasattr(neg, "on_partner_joined"):
                neg.on_partner_joined(partner)

    def on_partner_left(self, partner: str) -> None:
        """Notify all sub-negotiators that a partner left."""
        for neg in self._negotiators:
            if hasattr(neg, "on_partner_left"):
                neg.on_partner_left(partner)

    def on_partner_ended(self, partner: str) -> None:
        """Notify all sub-negotiators that a partner ended the negotiation."""
        for neg in self._negotiators:
            if hasattr(neg, "on_partner_ended"):
                neg.on_partner_ended(partner)

    def on_partner_proposal(
        self, state: GBState, partner_id: str, offer: Outcome
    ) -> None:
        """Notify all sub-negotiators of a partner's proposal."""
        for neg in self._negotiators:
            if hasattr(neg, "on_partner_proposal"):
                neg.on_partner_proposal(state, partner_id, offer)

    def on_partner_response(
        self, state: GBState, partner_id: str, outcome: Outcome, response: ResponseType
    ) -> None:
        """Notify all sub-negotiators of a partner's response."""
        for neg in self._negotiators:
            if hasattr(neg, "on_partner_response"):
                neg.on_partner_response(state, partner_id, outcome, response)

    def on_partner_refused_to_propose(self, state: GBState, partner_id: str) -> None:
        """Notify all sub-negotiators that a partner refused to propose."""
        for neg in self._negotiators:
            if hasattr(neg, "on_partner_refused_to_propose"):
                neg.on_partner_refused_to_propose(state, partner_id)

    # Lifecycle callbacks - delegate to both parent and sub-negotiators

    def on_negotiation_start(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that negotiation has started."""
        for neg in self._negotiators:
            neg.on_negotiation_start(state)

    def on_round_start(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that a round has started."""
        for neg in self._negotiators:
            neg.on_round_start(state)

    def on_round_end(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that a round has ended."""
        for neg in self._negotiators:
            neg.on_round_end(state)

    def on_leave(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that we're leaving the negotiation."""
        for neg in self._negotiators:
            neg.on_leave(state)
        GBNegotiator.on_leave(self, state)

    def on_negotiation_end(self, state: MechanismState) -> None:
        """Notify all sub-negotiators that negotiation has ended."""
        for neg in self._negotiators:
            neg.on_negotiation_end(state)

    def on_mechanism_error(self, state: MechanismState) -> None:
        """Notify all sub-negotiators of a mechanism error."""
        for neg in self._negotiators:
            neg.on_mechanism_error(state)
