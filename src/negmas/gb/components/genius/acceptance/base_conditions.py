"""acceptance components (split from acceptance.py): base_conditions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from negmas.gb.common import ResponseType

from negmas.gb.components.base import OfferingPolicy
from negmas.gb.components.genius.base import GeniusAcceptancePolicy

if TYPE_CHECKING:
    from negmas.gb import GBState
    from negmas.outcomes import Outcome
    from negmas.outcomes.common import ExtendedOutcome

__all__ = [
    "GACNext",
    "GACConst",
    "GACTime",
    "GACPrevious",
    "GACGap",
    "GACTrue",
    "GACFalse",
    "GACConstDiscounted",
]


@define
class GACNext(GeniusAcceptancePolicy):
    """
    AC_Next acceptance strategy from Genius.

    Accepts an offer if:
        a * u(opponent_offer) + b >= u(my_next_offer)

    where:
        - u(opponent_offer): Utility of the opponent's current offer
        - u(my_next_offer): Utility of the offer we would make next
        - a: Scaling factor (default 1.0)
        - b: Offset factor (default 0.0)

    With default parameters (a=1, b=0), this accepts if the opponent's offer
    is at least as good as what we would offer next.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        a: Scaling factor for opponent's offer utility (default 1.0).
        b: Offset added to scaled opponent utility (default 0.0).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_Next
    """

    offering_policy: OfferingPolicy
    a: float = 1.0
    b: float = 0.0

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if the offer meets the acceptance criterion,
            REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        # Get utility of opponent's offer
        opponent_util = float(self.negotiator.ufun(offer))

        # Get utility of my next offer
        my_next_offer = self.offering_policy(state)
        if my_next_offer is None:
            # If we can't make an offer, accept any rational offer
            if opponent_util >= float(self.negotiator.ufun.reserved_value):
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        # Handle ExtendedOutcome by extracting the outcome
        actual_offer: Outcome | ExtendedOutcome | None
        from negmas.outcomes.common import ExtendedOutcome

        if isinstance(my_next_offer, ExtendedOutcome):
            actual_offer = (
                my_next_offer.best_for(self.negotiator.ufun)
                if self.negotiator.ufun
                else my_next_offer.outcome
            )
        else:
            actual_offer = my_next_offer

        my_next_util = float(self.negotiator.ufun(actual_offer))

        # Accept if: a * opponent_util + b >= my_next_util
        if self.a * opponent_util + self.b >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACConst(GeniusAcceptancePolicy):
    """
    AC_Const acceptance strategy from Genius.

    Accepts an offer if its utility exceeds a constant threshold.

    Args:
        c: Constant threshold. Accept if utility > c (default 0.9).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_Const
    """

    c: float = 0.9

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if utility > c, REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        if opponent_util > self.c:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACTime(GeniusAcceptancePolicy):
    """
    AC_Time acceptance strategy from Genius.

    Accepts any offer after a certain time threshold has passed.

    Args:
        t: Time threshold (0 to 1). Accept any offer when time > t (default 0.99).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_Time
    """

    t: float = 0.99

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if time > t, REJECT_OFFER otherwise.
        """
        if offer is None:
            return ResponseType.REJECT_OFFER

        if state.relative_time > self.t:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACPrevious(GeniusAcceptancePolicy):
    """
    AC_Previous acceptance strategy from Genius.

    Accepts an offer if:
        a * u(opponent_offer) + b >= u(my_previous_offer)

    Similar to AC_Next but compares against our previous offer instead of next.

    Args:
        a: Scaling factor for opponent's offer utility (default 1.0).
        b: Offset added to scaled opponent utility (default 0.0).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_Previous
    """

    a: float = 1.0
    b: float = 0.0

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if a * opponent_util + b >= prev_own_util,
            REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        # Get our own bid history via NMI
        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        # Get utility of opponent's offer
        opponent_util = float(self.negotiator.ufun(offer))

        # Get utility of my previous offer
        my_prev_offer = my_offers[-1]
        if my_prev_offer is None:
            return ResponseType.REJECT_OFFER

        my_prev_util = float(self.negotiator.ufun(my_prev_offer))

        # Accept if: a * opponent_util + b >= my_prev_util
        if self.a * opponent_util + self.b >= my_prev_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACGap(GeniusAcceptancePolicy):
    """
    AC_Gap acceptance strategy from Genius.

    Accepts an offer if:
        u(opponent_offer) + c >= u(my_previous_offer)

    A restricted version of AC_Previous with a=1 and configurable gap.

    Args:
        c: Gap constant added to opponent utility (default 0.01).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_Gap
    """

    c: float = 0.01

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if opponent_util + c >= prev_own_util,
            REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        # Get our own bid history via NMI
        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        # Get utility of opponent's offer
        opponent_util = float(self.negotiator.ufun(offer))

        # Get utility of my previous offer
        my_prev_offer = my_offers[-1]
        if my_prev_offer is None:
            return ResponseType.REJECT_OFFER

        my_prev_util = float(self.negotiator.ufun(my_prev_offer))

        # Accept if: opponent_util + c >= my_prev_util
        if opponent_util + self.c >= my_prev_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACTrue(GeniusAcceptancePolicy):
    """
    AC_True acceptance strategy from Genius.

    Always accepts any offer. Useful for debugging and testing.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_True
    """

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Always accept the offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER always.
        """
        if offer is None:
            return ResponseType.REJECT_OFFER
        return ResponseType.ACCEPT_OFFER


@define
class GACFalse(GeniusAcceptancePolicy):
    """
    AC_False acceptance strategy from Genius.

    Never accepts any offer. Useful for debugging and testing.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_False
    """

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Always reject the offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            REJECT_OFFER always.
        """
        return ResponseType.REJECT_OFFER


@define
class GACConstDiscounted(GeniusAcceptancePolicy):
    """
    AC_ConstDiscounted acceptance strategy from Genius.

    Accepts an offer if its discounted utility exceeds a constant threshold.
    Takes time discount into account.

    Args:
        c: Constant threshold. Accept if discounted utility > c (default 0.9).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_ConstDiscounted
    """

    c: float = 0.9

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if discounted utility > c, REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        # Get raw utility
        raw_util = float(self.negotiator.ufun(offer))

        # Apply time discount if available
        discount = 1.0
        if hasattr(self.negotiator.ufun, "discount_factor"):
            discount_factor = getattr(self.negotiator.ufun, "discount_factor", None)
            if discount_factor is not None and discount_factor < 1.0:
                discount = pow(discount_factor, state.relative_time)

        discounted_util = raw_util * discount

        if discounted_util > self.c:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
