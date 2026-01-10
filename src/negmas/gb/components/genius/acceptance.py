"""Genius acceptance policies.

This module contains Python implementations of Genius acceptance strategies,
transcompiled from the original Java implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from negmas.gb.common import ResponseType

from ..base import OfferingPolicy
from .base import GeniusAcceptancePolicy

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
    "GACCombi",
    "GACCombiMaxInWindow",
    "GACTrue",
    "GACFalse",
    "GACConstDiscounted",
    "GACCombiAvg",
    "GACCombiBestAvg",
    "GACCombiMax",
    "GACCombiV2",
    "GACCombiV3",
    "GACCombiV4",
    "GACCombiBestAvgDiscounted",
    "GACCombiMaxInWindowDiscounted",
    "GACCombiProb",
    "GACCombiProbDiscounted",
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

    Transcompiled from: bilateralexamples.boacomponents.AC_Next
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
            actual_offer = my_next_offer.outcome
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
class GACCombi(GeniusAcceptancePolicy):
    """
    AC_Combi acceptance strategy from Genius.

    Combines AC_Next and AC_Time: accepts if either condition is met.

    Accepts if:
        (a * u(opponent_offer) + b >= u(my_next_offer)) OR (time >= t)

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        a: Scaling factor for opponent's offer utility (default 1.0).
        b: Offset added to scaled opponent utility (default 0.0).
        t: Time threshold for acceptance (default 0.99).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_Combi
    """

    offering_policy: OfferingPolicy
    a: float = 1.0
    b: float = 0.0
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
            ACCEPT_OFFER if either AC_Next or AC_Time condition is met,
            REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        # AC_Time: accept if time >= t
        if state.relative_time >= self.t:
            return ResponseType.ACCEPT_OFFER

        # AC_Next part: check against next offer
        opponent_util = float(self.negotiator.ufun(offer))

        my_next_offer = self.offering_policy(state)
        if my_next_offer is None:
            # If we can't make an offer, accept any rational offer
            if opponent_util >= float(self.negotiator.ufun.reserved_value):
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        # Handle ExtendedOutcome by extracting the outcome
        from negmas.outcomes.common import ExtendedOutcome

        if isinstance(my_next_offer, ExtendedOutcome):
            actual_offer = my_next_offer.outcome
        else:
            actual_offer = my_next_offer

        my_next_util = float(self.negotiator.ufun(actual_offer))

        # Accept if: a * opponent_util + b >= my_next_util
        if self.a * opponent_util + self.b >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiMaxInWindow(GeniusAcceptancePolicy):
    """
    AC_CombiMaxInWindow acceptance strategy from Genius.

    Combines AC_Next with a time-window-based acceptance criterion.

    Before time t: acts like AC_Next only.
    After time t: accepts if opponent's offer is >= best offer seen in remaining time window.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which window-based acceptance kicks in (default 0.98).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiMaxInWindow
    """

    offering_policy: OfferingPolicy
    t: float = 0.98

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if acceptance criterion is met, REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # AC_Next part: always accept if opponent's offer >= our next offer
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer))
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: window-based acceptance
        # Window is the time left
        now = state.relative_time
        window = 1.0 - now

        # Find best offer in the recent window
        # Get partner offers
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        max_util_in_window = 0.0
        if partner_offers:
            # Filter offers in the recent time window
            # We need to estimate times for each offer
            n_offers = len(partner_offers)
            if n_offers > 0:
                # Approximate: distribute offers across time
                for i, past_offer in enumerate(partner_offers):
                    if past_offer is not None:
                        # Estimate time of this offer
                        offer_time = (i + 1) / (n_offers + 1) * now
                        if offer_time >= now - window:
                            past_util = float(self.negotiator.ufun(past_offer))
                            max_util_in_window = max(max_util_in_window, past_util)

        # Accept if opponent's offer >= best in window
        if opponent_util >= max_util_in_window:
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


@define
class GACCombiAvg(GeniusAcceptancePolicy):
    """
    AC_CombiAvg acceptance strategy from Genius.

    Combines AC_Next with average-based acceptance in the end game.

    Before time t: acts like AC_Next.
    After time t: accepts if opponent's offer >= average of opponent's offers in window.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which average-based acceptance kicks in (default 0.98).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiAvg
    """

    offering_policy: OfferingPolicy
    t: float = 0.98

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if acceptance criterion is met, REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # AC_Next part: always accept if opponent's offer >= our next offer
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer))
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: average-based acceptance
        # Get partner offers
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        if partner_offers:
            # Calculate average utility of partner offers
            total_util = sum(
                float(self.negotiator.ufun(o)) for o in partner_offers if o is not None
            )
            avg_util = total_util / len(partner_offers)

            # Accept if opponent's offer >= average
            if opponent_util >= avg_util:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiBestAvg(GeniusAcceptancePolicy):
    """
    AC_CombiBestAvg acceptance strategy from Genius.

    Combines AC_Next with best-average-based acceptance.

    Before time t: acts like AC_Next.
    After time t: accepts if opponent's offer >= average of offers better than current offer.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which best-average acceptance kicks in (default 0.98).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiBestAvg
    """

    offering_policy: OfferingPolicy
    t: float = 0.98

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if acceptance criterion is met, REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # AC_Next part: always accept if opponent's offer >= our next offer
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer))
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: best-average-based acceptance
        # Get partner offers
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        # Filter offers better than current offer and compute their average
        better_offers = [
            o
            for o in partner_offers
            if o is not None and float(self.negotiator.ufun(o)) >= opponent_util
        ]

        if better_offers:
            avg_better_util = sum(
                float(self.negotiator.ufun(o)) for o in better_offers
            ) / len(better_offers)

            # Accept if current offer >= expected utility of waiting for better
            if opponent_util >= avg_better_util:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiMax(GeniusAcceptancePolicy):
    """
    AC_CombiMax acceptance strategy from Genius.

    Combines AC_Next with maximum-based acceptance.

    Before time t: acts like AC_Next.
    After time t: accepts if opponent's offer >= max of all previous opponent offers.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which max-based acceptance kicks in (default 0.98).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiMax
    """

    offering_policy: OfferingPolicy
    t: float = 0.98

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer.

        Args:
            state: Current negotiation state.
            offer: The opponent's offer to evaluate.
            source: Source identifier of the offer.

        Returns:
            ACCEPT_OFFER if acceptance criterion is met, REJECT_OFFER otherwise.
        """
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # AC_Next part: always accept if opponent's offer >= our next offer
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer))
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: max-based acceptance
        # Get partner offers
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        if partner_offers:
            # Find max utility of all partner offers
            max_util = max(
                float(self.negotiator.ufun(o)) for o in partner_offers if o is not None
            )

            # Accept if current offer >= max seen
            if opponent_util >= max_util:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiV2(GeniusAcceptancePolicy):
    """
    AC_CombiV2 acceptance strategy from Genius.

    A variant of AC_Combi that uses a different combination logic.
    Accepts if the opponent's offer utility exceeds a time-dependent threshold
    based on both the next offer utility and a decay factor.

    Before time t: acts like AC_Next.
    After time t: accepts if opponent's offer >= next offer utility * decay.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        a: Scaling factor for opponent's offer utility (default 1.0).
        b: Offset added to scaled opponent utility (default 0.0).
        t: Time threshold (default 0.99).
        decay: Decay factor applied after time threshold (default 0.9).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiV2
    """

    offering_policy: OfferingPolicy
    a: float = 1.0
    b: float = 0.0
    t: float = 0.99
    decay: float = 0.9

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        my_next_offer = self.offering_policy(state)
        if my_next_offer is None:
            if opponent_util >= float(self.negotiator.ufun.reserved_value):
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        from negmas.outcomes.common import ExtendedOutcome

        if isinstance(my_next_offer, ExtendedOutcome):
            actual_offer = my_next_offer.outcome
        else:
            actual_offer = my_next_offer

        my_next_util = float(self.negotiator.ufun(actual_offer))

        # Standard AC_Next condition
        if self.a * opponent_util + self.b >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        # After time t: apply decay factor to threshold
        if state.relative_time >= self.t:
            decayed_threshold = my_next_util * self.decay
            if opponent_util >= decayed_threshold:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiV3(GeniusAcceptancePolicy):
    """
    AC_CombiV3 acceptance strategy from Genius.

    A variant of AC_Combi that uses linear interpolation between
    AC_Next threshold and reserved value based on time.

    The acceptance threshold decreases linearly from next offer utility
    to reserved value as time progresses past threshold t.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        a: Scaling factor for opponent's offer utility (default 1.0).
        b: Offset added to scaled opponent utility (default 0.0).
        t: Time threshold when interpolation begins (default 0.95).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiV3
    """

    offering_policy: OfferingPolicy
    a: float = 1.0
    b: float = 0.0
    t: float = 0.95

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        my_next_offer = self.offering_policy(state)
        if my_next_offer is None:
            if opponent_util >= float(self.negotiator.ufun.reserved_value):
                return ResponseType.ACCEPT_OFFER
            return ResponseType.REJECT_OFFER

        from negmas.outcomes.common import ExtendedOutcome

        if isinstance(my_next_offer, ExtendedOutcome):
            actual_offer = my_next_offer.outcome
        else:
            actual_offer = my_next_offer

        my_next_util = float(self.negotiator.ufun(actual_offer))

        # Standard AC_Next condition
        if self.a * opponent_util + self.b >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        # After time t: linearly interpolate threshold from next_util to reserved_value
        if state.relative_time >= self.t:
            reserved = float(self.negotiator.ufun.reserved_value)
            # Progress from t to 1.0
            progress = (
                (state.relative_time - self.t) / (1.0 - self.t) if self.t < 1.0 else 1.0
            )
            # Interpolate from my_next_util to reserved
            threshold = my_next_util + (reserved - my_next_util) * progress
            if opponent_util >= threshold:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiV4(GeniusAcceptancePolicy):
    """
    AC_CombiV4 acceptance strategy from Genius.

    A variant of AC_Combi that combines AC_Next with a weighted combination
    of max and average opponent offers in the end game.

    Before time t: acts like AC_Next.
    After time t: accepts if opponent's offer >= weighted combo of max and avg.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which combined strategy kicks in (default 0.98).
        w: Weight for max utility (1-w used for avg utility) (default 0.5).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiV4
    """

    offering_policy: OfferingPolicy
    t: float = 0.98
    w: float = 0.5

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # AC_Next part
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer))
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: weighted combination of max and average
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        if partner_offers:
            utils = [
                float(self.negotiator.ufun(o)) for o in partner_offers if o is not None
            ]
            if utils:
                max_util = max(utils)
                avg_util = sum(utils) / len(utils)
                # Weighted combination
                threshold = self.w * max_util + (1 - self.w) * avg_util
                if opponent_util >= threshold:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiBestAvgDiscounted(GeniusAcceptancePolicy):
    """
    AC_CombiBestAvgDiscounted acceptance strategy from Genius.

    Like AC_CombiBestAvg but applies time discount to utilities.

    Before time t: acts like AC_Next.
    After time t: accepts if discounted opponent offer >= discounted avg of better offers.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which best-average acceptance kicks in (default 0.98).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiBestAvgDiscounted
    """

    offering_policy: OfferingPolicy
    t: float = 0.98

    def _get_discount(self, time: float) -> float:
        """Get discount factor at given time."""
        if not self.negotiator or not self.negotiator.ufun:
            return 1.0
        if hasattr(self.negotiator.ufun, "discount_factor"):
            discount_factor = getattr(self.negotiator.ufun, "discount_factor", None)
            if discount_factor is not None and discount_factor < 1.0:
                return pow(discount_factor, time)
        return 1.0

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        discount = self._get_discount(state.relative_time)
        opponent_util = float(self.negotiator.ufun(offer)) * discount

        # AC_Next part
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer)) * discount
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: best-average-based acceptance with discounting
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        # Filter offers better than current offer (discounted)
        better_offers = [
            o
            for o in partner_offers
            if o is not None
            and float(self.negotiator.ufun(o)) * discount >= opponent_util
        ]

        if better_offers:
            avg_better_util = sum(
                float(self.negotiator.ufun(o)) * discount for o in better_offers
            ) / len(better_offers)

            if opponent_util >= avg_better_util:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiMaxInWindowDiscounted(GeniusAcceptancePolicy):
    """
    AC_CombiMaxInWindowDiscounted acceptance strategy from Genius.

    Like AC_CombiMaxInWindow but applies time discount to utilities.

    Before time t: acts like AC_Next.
    After time t: accepts if discounted offer >= discounted best in window.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which window-based acceptance kicks in (default 0.98).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiMaxInWindowDiscounted
    """

    offering_policy: OfferingPolicy
    t: float = 0.98

    def _get_discount(self, time: float) -> float:
        """Get discount factor at given time."""
        if not self.negotiator or not self.negotiator.ufun:
            return 1.0
        if hasattr(self.negotiator.ufun, "discount_factor"):
            discount_factor = getattr(self.negotiator.ufun, "discount_factor", None)
            if discount_factor is not None and discount_factor < 1.0:
                return pow(discount_factor, time)
        return 1.0

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        discount = self._get_discount(state.relative_time)
        opponent_util = float(self.negotiator.ufun(offer)) * discount

        # AC_Next part
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer)) * discount
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: window-based acceptance with discounting
        now = state.relative_time
        window = 1.0 - now

        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        max_util_in_window = 0.0
        if partner_offers:
            n_offers = len(partner_offers)
            if n_offers > 0:
                for i, past_offer in enumerate(partner_offers):
                    if past_offer is not None:
                        offer_time = (i + 1) / (n_offers + 1) * now
                        if offer_time >= now - window:
                            past_util = (
                                float(self.negotiator.ufun(past_offer)) * discount
                            )
                            max_util_in_window = max(max_util_in_window, past_util)

        if opponent_util >= max_util_in_window:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiProb(GeniusAcceptancePolicy):
    """
    AC_CombiProb acceptance strategy from Genius.

    Probability-based acceptance that combines AC_Next with probabilistic
    acceptance based on the expected utility of waiting.

    Before time t: acts like AC_Next.
    After time t: accepts with probability based on how good the offer is
    relative to expected future offers.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which probabilistic acceptance kicks in (default 0.98).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiProb
    """

    offering_policy: OfferingPolicy
    t: float = 0.98

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        import random

        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # AC_Next part
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer))
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: probabilistic acceptance
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        if partner_offers:
            utils = [
                float(self.negotiator.ufun(o)) for o in partner_offers if o is not None
            ]
            if utils:
                avg_util = sum(utils) / len(utils)
                # Probability increases as offer gets better relative to average
                if avg_util > 0:
                    # Scale probability: if offer == avg, prob ~= 0.5
                    # if offer > avg, prob increases toward 1
                    ratio = opponent_util / avg_util
                    prob = min(1.0, ratio / 2.0)  # ratio=2 -> prob=1
                    if random.random() < prob:
                        return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACCombiProbDiscounted(GeniusAcceptancePolicy):
    """
    AC_CombiProbDiscounted acceptance strategy from Genius.

    Like AC_CombiProb but applies time discount to utilities.

    Before time t: acts like AC_Next.
    After time t: probabilistic acceptance with discounted utilities.

    Args:
        offering_policy: The offering strategy used to determine my next offer.
        t: Time threshold after which probabilistic acceptance kicks in (default 0.98).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_CombiProbDiscounted
    """

    offering_policy: OfferingPolicy
    t: float = 0.98

    def _get_discount(self, time: float) -> float:
        """Get discount factor at given time."""
        if not self.negotiator or not self.negotiator.ufun:
            return 1.0
        if hasattr(self.negotiator.ufun, "discount_factor"):
            discount_factor = getattr(self.negotiator.ufun, "discount_factor", None)
            if discount_factor is not None and discount_factor < 1.0:
                return pow(discount_factor, time)
        return 1.0

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        import random

        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        nmi = self.negotiator.nmi
        if nmi is None:
            return ResponseType.REJECT_OFFER

        discount = self._get_discount(state.relative_time)
        opponent_util = float(self.negotiator.ufun(offer)) * discount

        # AC_Next part
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual_offer = my_next_offer.outcome
            else:
                actual_offer = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual_offer)) * discount
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        # Before time t, only AC_Next applies
        if state.relative_time < self.t:
            return ResponseType.REJECT_OFFER

        # After time t: probabilistic acceptance with discounting
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        if partner_offers:
            utils = [
                float(self.negotiator.ufun(o)) * discount
                for o in partner_offers
                if o is not None
            ]
            if utils:
                avg_util = sum(utils) / len(utils)
                if avg_util > 0:
                    ratio = opponent_util / avg_util
                    prob = min(1.0, ratio / 2.0)
                    if random.random() < prob:
                        return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
