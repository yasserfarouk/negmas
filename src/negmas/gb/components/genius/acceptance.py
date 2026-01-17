"""Genius acceptance policies.

This module contains Python implementations of Genius acceptance strategies,
transcompiled from the original Java implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.gb.common import ResponseType

from ..base import OfferingPolicy
from .base import GeniusAcceptancePolicy

if TYPE_CHECKING:
    from negmas.gb import GBState
    from negmas.outcomes import Outcome
    from negmas.outcomes.common import ExtendedOutcome

__all__ = [
    # Base acceptance strategies
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
    # ANAC 2010 acceptance strategies
    "GACABMP",
    "GACAgentK",
    "GACAgentFSEGA",
    "GACIAMCrazyHaggler",
    "GACYushu",
    "GACNozomi",
    "GACIAMHaggler2010",
    "GACAgentSmith",
    # ANAC 2011 acceptance strategies
    "GACHardHeaded",
    "GACAgentK2",
    "GACBRAMAgent",
    "GACGahboninho",
    "GACNiceTitForTat",
    "GACTheNegotiator",
    "GACValueModelAgent",
    "GACIAMHaggler2011",
    # ANAC 2012 acceptance strategies
    "GACCUHKAgent",
    "GACOMACagent",
    "GACAgentLG",
    "GACAgentMR",
    "GACBRAMAgent2",
    "GACIAMHaggler2012",
    "GACTheNegotiatorReloaded",
    # ANAC 2013 acceptance strategies
    "GACTheFawkes",
    "GACInoxAgent",
    "GACInoxAgentOneIssue",
    # Other acceptance strategies
    "GACUncertain",
    "GACMAC",
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


# =============================================================================
# ANAC 2010 Acceptance Strategies
# =============================================================================


@define
class GACABMP(GeniusAcceptancePolicy):
    """
    AC_ABMP acceptance strategy from Genius.

    Accepts an offer if the opponent's utility is within a gap of our last offer.
    Based on the ABMP (Adaptive Bargaining with Multiple Proposals) agent.

    Args:
        utility_gap: The maximum gap between opponent's offer and our last offer (default 0.05).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_ABMP
    """

    utility_gap: float = 0.05

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

        # Get our last bid
        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        my_last_offer = my_offers[-1]
        if my_last_offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        my_last_util = float(self.negotiator.ufun(my_last_offer))

        if opponent_util >= my_last_util - self.utility_gap:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACAgentK(GeniusAcceptancePolicy):
    """
    AC_AgentK acceptance strategy from Genius (ANAC2010).

    Probabilistic acceptance based on time and utility. Calculates an acceptance
    probability and accepts if a random value is below this probability.

    The acceptance probability increases as time progresses and as the opponent's
    offers improve relative to expectations.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_AgentK
    """

    _max_utility: float = field(init=False, default=1.0)
    _min_utility: float = field(init=False, default=0.0)
    _opponent_max_util: float = field(init=False, default=0.0)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        import random

        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # Update opponent max utility
        if opponent_util > self._opponent_max_util:
            self._opponent_max_util = opponent_util

        # Calculate acceptance probability
        p = self._calculate_accept_probability(state, opponent_util)

        if p > random.random():
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _calculate_accept_probability(
        self, state: GBState, opponent_util: float
    ) -> float:
        """Calculate the probability of accepting the current offer."""
        time = state.relative_time

        # Expected utility decreases over time
        expected_util = 1.0 - time * 0.5

        if opponent_util >= expected_util:
            return 1.0

        # Probability based on how close opponent's offer is to expected
        if expected_util > 0:
            ratio = opponent_util / expected_util
            # Increase probability as time runs out
            time_pressure = time**2
            return min(1.0, ratio * (1 + time_pressure))

        return 0.0


@define
class GACAgentFSEGA(GeniusAcceptancePolicy):
    """
    AC_AgentFSEGA acceptance strategy from Genius (ANAC2010).

    Accepts if:
    - opponent_util * multiplier >= my_last_util, OR
    - opponent_util > my_next_util, OR
    - opponent_util == max_utility_in_domain

    Args:
        offering_policy: The offering strategy to determine next bid.
        multiplier: Multiplier for opponent's offer comparison (default 1.03).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_AgentFSEGA
    """

    offering_policy: OfferingPolicy
    multiplier: float = 1.03

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

        # Need both histories to have at least one bid
        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        my_last_util = float(self.negotiator.ufun(my_offers[-1]))

        # Get max utility in domain
        max_util = 1.0  # Assume normalized

        # Get my next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept conditions
        if (
            opponent_util * self.multiplier >= my_last_util
            or opponent_util > my_next_util
            or opponent_util >= max_util * 0.999  # Close to max
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACIAMCrazyHaggler(GeniusAcceptancePolicy):
    """
    AC_IAMcrazyHaggler acceptance strategy from Genius (ANAC2010).

    A high-aspiration strategy that accepts only when the opponent's offer
    is close to our maximum aspiration or our own offers.

    Args:
        offering_policy: The offering strategy to determine next bid.
        maximum_aspiration: Target utility threshold (default 0.85).
        accept_multiplier: Multiplier for acceptance comparison (default 1.02).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_IAMcrazyHaggler
    """

    offering_policy: OfferingPolicy
    maximum_aspiration: float = 0.85
    accept_multiplier: float = 1.02

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

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        my_last_util = float(self.negotiator.ufun(my_offers[-1]))

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept if opponent * multiplier exceeds our last/next/aspiration
        if (
            opponent_util * self.accept_multiplier > my_last_util
            or opponent_util * self.accept_multiplier > my_next_util
            or opponent_util * self.accept_multiplier > self.maximum_aspiration
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACYushu(GeniusAcceptancePolicy):
    """
    AC_Yushu acceptance strategy from Genius (ANAC2010).

    Time-dependent threshold strategy. The target utility decreases from
    a high value (0.95) towards a lower acceptable value (0.7) as time progresses.

    Args:
        initial_target: Initial target utility (default 0.95).
        final_target: Final target utility at deadline (default 0.7).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_Yushu
    """

    initial_target: float = 0.95
    final_target: float = 0.7
    _acceptable_util: float = field(init=False, default=0.8)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        time = state.relative_time

        # Calculate target utility based on time
        target_util = (
            self.initial_target - (self.initial_target - self.final_target) * time
        )

        # Update acceptable utility
        self._acceptable_util = max(self._acceptable_util, self.final_target)

        # Accept if opponent offer meets target or acceptable threshold
        if opponent_util >= target_util or opponent_util >= self._acceptable_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACNozomi(GeniusAcceptancePolicy):
    """
    AC_Nozomi acceptance strategy from Genius (ANAC2010).

    A sophisticated strategy that considers opponent modeling, time pressure,
    and evaluation gap between bids. Accepts based on multiple conditions
    that change with time phases.

    Args:
        max_util_threshold: Threshold relative to max utility (default 0.95).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_Nozomi
    """

    max_util_threshold: float = 0.95
    _max_opponent_util: float = field(init=False, default=0.0)
    _max_compromise_util: float = field(init=False, default=0.0)

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

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        my_last_util = float(self.negotiator.ufun(my_offers[-1])) if my_offers else 1.0

        opponent_util = float(self.negotiator.ufun(offer))
        time = state.relative_time

        # Update max opponent utility
        if opponent_util > self._max_opponent_util:
            self._max_opponent_util = opponent_util

        # Update max compromise utility
        if self._max_opponent_util > self._max_compromise_util:
            self._max_compromise_util = min(
                self._max_opponent_util, self.max_util_threshold
            )

        # Accept if very good offer or better than our last
        if opponent_util > self.max_util_threshold or opponent_util >= my_last_util:
            return ResponseType.ACCEPT_OFFER

        # Time-based acceptance phases
        if time < 0.50:
            if (
                opponent_util > self._max_compromise_util
                and opponent_util >= self._max_opponent_util
            ):
                accept_coeff = -0.1 * time + 1.0
                if opponent_util > my_last_util * accept_coeff:
                    return ResponseType.ACCEPT_OFFER
        elif time < 0.80:
            if opponent_util > self._max_compromise_util * 0.95:
                accept_coeff = -0.16 * (time - 0.50) + 0.95
                if opponent_util > my_last_util * accept_coeff:
                    return ResponseType.ACCEPT_OFFER
        else:
            # Late game - more willing to accept
            if opponent_util > self._max_compromise_util * 0.90:
                threshold = 0.40 if time <= 0.90 else 0.50
                if opponent_util >= my_last_util * (1 - threshold):
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACIAMHaggler2010(GeniusAcceptancePolicy):
    """
    AC_IAMHaggler2010 acceptance strategy from Genius (ANAC2010).

    Similar to IAMCrazyHaggler but with slightly different thresholds.
    Uses concession rate estimation for acceptance.

    Args:
        offering_policy: The offering strategy to determine next bid.
        maximum_aspiration: Target utility threshold (default 0.9).
        accept_multiplier: Multiplier for acceptance comparison (default 1.02).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_IAMHaggler2010
    """

    offering_policy: OfferingPolicy
    maximum_aspiration: float = 0.9
    accept_multiplier: float = 1.02

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

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        my_last_util = float(self.negotiator.ufun(my_offers[-1]))

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept if multiplied opponent util exceeds thresholds
        if (
            opponent_util * self.accept_multiplier >= my_last_util
            or opponent_util * self.accept_multiplier >= my_next_util
            or opponent_util * self.accept_multiplier >= self.maximum_aspiration
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACAgentSmith(GeniusAcceptancePolicy):
    """
    AC_AgentSmith acceptance strategy from Genius (ANAC2010).

    Probabilistic acceptance with a minimum utility threshold.
    Accepts if opponent's offer is above the accept margin or
    if it's better than or equal to our last offer.

    Args:
        accept_margin: Minimum utility to accept unconditionally (default 0.9).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_AgentSmith
    """

    accept_margin: float = 0.9

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

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        my_last_util = float(self.negotiator.ufun(my_offers[-1]))

        # Accept if above margin or better than our last offer
        if opponent_util > self.accept_margin or opponent_util >= my_last_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


# =============================================================================
# ANAC 2011 Acceptance Strategies
# =============================================================================


@define
class GACHardHeaded(GeniusAcceptancePolicy):
    """
    AC_HardHeaded acceptance strategy from Genius (ANAC2011).

    Accepts if the opponent's offer utility is greater than our lowest
    offered utility so far, or if it's at least as good as our next bid.

    Args:
        offering_policy: The offering strategy to determine next bid.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2011.AC_HardHeaded
    """

    offering_policy: OfferingPolicy
    _lowest_own_util: float = field(init=False, default=1.0)

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

        # Track lowest utility we've offered
        my_offers = nmi.negotiator_offers(self.negotiator.id)
        for my_offer in my_offers:
            if my_offer is not None:
                util = float(self.negotiator.ufun(my_offer))
                if util < self._lowest_own_util:
                    self._lowest_own_util = util

        opponent_util = float(self.negotiator.ufun(offer))

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept if opponent offer > our lowest or >= our next bid
        if opponent_util > self._lowest_own_util or my_next_util <= opponent_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACAgentK2(GeniusAcceptancePolicy):
    """
    AC_AgentK2 acceptance strategy from Genius (ANAC2011).

    Enhanced probabilistic acceptance with statistics tracking.
    Similar to AgentK but with improved probability calculations.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2011.AC_AgentK2
    """

    _opponent_utils: list = field(init=False, factory=list)
    _accept_probability: float = field(init=False, default=0.0)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        import random

        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        self._opponent_utils.append(opponent_util)

        # Calculate acceptance probability
        p = self._calculate_accept_probability(state, opponent_util)
        self._accept_probability = p

        if p > random.random():
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _calculate_accept_probability(
        self, state: GBState, opponent_util: float
    ) -> float:
        """Calculate the probability of accepting the current offer."""
        time = state.relative_time

        if not self._opponent_utils:
            return 0.0

        # Statistics from opponent offers
        avg_util = sum(self._opponent_utils) / len(self._opponent_utils)
        max_util = max(self._opponent_utils)

        # Expected utility based on time and opponent behavior
        expected = max_util - (max_util - avg_util) * (1 - time)

        if opponent_util >= expected:
            return min(1.0, 0.5 + time * 0.5)

        # Lower probability for offers below expected
        if expected > 0:
            ratio = opponent_util / expected
            return min(1.0, ratio * time)

        return 0.0


@define
class GACBRAMAgent(GeniusAcceptancePolicy):
    """
    AC_BRAMAgent acceptance strategy from Genius (ANAC2011).

    Best Response Adaptive Model - accepts based on a dynamically
    calculated threshold that accounts for discounting.

    Args:
        offering_policy: The offering strategy to determine next bid.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2011.AC_BRAMAgent
    """

    offering_policy: OfferingPolicy
    _threshold: float = field(init=False, default=0.9)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # Update threshold based on time
        self._threshold = self._calculate_threshold(state)

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept if above threshold or better than our next bid
        if opponent_util >= self._threshold or opponent_util >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _calculate_threshold(self, state: GBState) -> float:
        """Calculate acceptance threshold based on time and discount."""
        time = state.relative_time
        # Threshold decreases over time
        min_threshold = 0.5
        max_threshold = 0.95
        return max_threshold - (max_threshold - min_threshold) * (time**2)


@define
class GACGahboninho(GeniusAcceptancePolicy):
    """
    AC_Gahboninho acceptance strategy from Genius (ANAC2011).

    High threshold strategy that accepts offers above 0.95 utility early,
    or above a minimum acceptable threshold that adapts over time.

    Args:
        high_threshold: Utility threshold for early acceptance (default 0.95).
        min_acceptable: Minimum acceptable utility (default 0.7).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2011.AC_Gahboninho
    """

    high_threshold: float = 0.95
    min_acceptable: float = 0.7
    _min_util_for_acceptance: float = field(init=False, default=0.8)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        time = state.relative_time

        # Update minimum utility for acceptance based on time
        self._min_util_for_acceptance = max(
            self.min_acceptable,
            self.high_threshold - (self.high_threshold - self.min_acceptable) * time,
        )

        # Accept very good offers early
        if time < 0.5 and opponent_util > self.high_threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept if above minimum threshold
        if opponent_util >= self._min_util_for_acceptance:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACNiceTitForTat(GeniusAcceptancePolicy):
    """
    AC_NiceTitForTat acceptance strategy from Genius (ANAC2011).

    Cooperative strategy based on opponent behavior. Uses AC_Next logic
    combined with probabilistic acceptance near the deadline.

    Args:
        offering_policy: The offering strategy to determine next bid.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2011.AC_NiceTitForTat
    """

    offering_policy: OfferingPolicy

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
        time = state.relative_time

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # AC_Next: accept if opponent offer >= our next bid
        if opponent_util >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        # Before 98% of time, only AC_Next applies
        if time < 0.98:
            return ResponseType.REJECT_OFFER

        # Near deadline: probabilistic acceptance
        time_left = 1 - time

        # Get opponent's best offer
        my_id = self.negotiator.id
        partner_offers: list[Outcome] = []
        for nid in nmi.negotiator_ids:
            if nid != my_id:
                partner_offers.extend(nmi.negotiator_offers(nid))

        if not partner_offers:
            return ResponseType.REJECT_OFFER

        best_opponent_util = max(
            float(self.negotiator.ufun(o)) for o in partner_offers if o is not None
        )

        # Don't accept if we expect better offers
        if best_opponent_util > opponent_util and time_left > 0.001:
            return ResponseType.REJECT_OFFER

        # Calculate expected utility of waiting
        better_offers = [
            float(self.negotiator.ufun(o))
            for o in partner_offers
            if o is not None and float(self.negotiator.ufun(o)) > opponent_util
        ]

        if better_offers:
            avg_better = sum(better_offers) / len(better_offers)
            p_at_least_one = 1 - (1 - time_left) ** len(better_offers)
            expected_util = p_at_least_one * avg_better

            if opponent_util > expected_util:
                return ResponseType.ACCEPT_OFFER
        else:
            # No better offers seen, accept current
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACTheNegotiator(GeniusAcceptancePolicy):
    """
    AC_TheNegotiator acceptance strategy from Genius (ANAC2011).

    State machine with phases: hardball, conceding, and desperate.
    Acceptance threshold varies based on the current phase.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2011.AC_TheNegotiator
    """

    _phase: int = field(init=False, default=1)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        time = state.relative_time

        # Calculate current phase
        self._phase = self._calculate_phase(time)

        # Calculate threshold based on phase
        threshold = self._calculate_threshold(time)

        # Calculate moves left estimate
        moves_left = self._estimate_moves_left(state)

        if self._phase < 3:
            if opponent_util >= threshold:
                return ResponseType.ACCEPT_OFFER
        else:
            # Phase 3 - desperate
            if moves_left >= 15:
                if opponent_util >= threshold:
                    return ResponseType.ACCEPT_OFFER
            else:
                # Very few moves left - accept almost anything reasonable
                if moves_left < 15:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _calculate_phase(self, time: float) -> int:
        """Determine negotiation phase based on time."""
        if time < 0.3:
            return 1  # Hardball
        elif time < 0.7:
            return 2  # Conceding
        else:
            return 3  # Desperate

    def _calculate_threshold(self, time: float) -> float:
        """Calculate acceptance threshold based on time."""
        if time < 0.3:
            return 0.95
        elif time < 0.7:
            return 0.85 - (time - 0.3) * 0.25
        else:
            return 0.70 - (time - 0.7) * 0.3

    def _estimate_moves_left(self, state: GBState) -> int:
        """Estimate number of moves left in negotiation."""
        time_left = 1 - state.relative_time
        # Rough estimate based on step
        if state.step > 0:
            time_per_step = state.relative_time / state.step
            if time_per_step > 0:
                return int(time_left / time_per_step)
        return 100  # Default if we can't estimate


@define
class GACValueModelAgent(GeniusAcceptancePolicy):
    """
    AC_ValueModelAgent acceptance strategy from Genius (ANAC2011).

    Value model based acceptance that tracks opponent's maximum utility
    and accepts based on various thresholds that change with time.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2011.AC_ValueModelAgent
    """

    _opponent_max_util: float = field(init=False, default=0.0)
    _lowest_approved: float = field(init=False, default=0.9)
    _planned_threshold: float = field(init=False, default=0.85)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        time = state.relative_time

        # Track opponent's maximum utility
        if opponent_util > self._opponent_max_util:
            self._opponent_max_util = opponent_util

        # Various acceptance conditions based on time
        if time > 0.98 and time <= 0.99:
            if opponent_util >= self._lowest_approved - 0.01:
                return ResponseType.ACCEPT_OFFER

        if time > 0.995 and self._opponent_max_util > 0.55:
            if opponent_util >= self._opponent_max_util * 0.99:
                return ResponseType.ACCEPT_OFFER

        # Accept if opponent settled enough
        if opponent_util > self._lowest_approved and opponent_util > 0.975:
            return ResponseType.ACCEPT_OFFER

        if time > 0.9:
            if opponent_util >= self._planned_threshold - 0.01:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACIAMHaggler2011(GeniusAcceptancePolicy):
    """
    AC_IAMHaggler2011 acceptance strategy from Genius (ANAC2011).

    GP-smoothed estimate based acceptance. Similar to IAMHaggler2010
    with improved estimation.

    Args:
        offering_policy: The offering strategy to determine next bid.
        maximum_aspiration: Target utility threshold (default 0.9).
        accept_multiplier: Multiplier for acceptance comparison (default 1.02).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2011.AC_IAMHaggler2011
    """

    offering_policy: OfferingPolicy
    maximum_aspiration: float = 0.9
    accept_multiplier: float = 1.02

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

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        my_last_util = float(self.negotiator.ufun(my_offers[-1]))

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept conditions
        if (
            opponent_util * self.accept_multiplier > my_last_util
            or opponent_util * self.accept_multiplier > my_next_util
            or opponent_util * self.accept_multiplier > self.maximum_aspiration
        ):
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


# =============================================================================
# ANAC 2012 Acceptance Strategies
# =============================================================================


@define
class GACCUHKAgent(GeniusAcceptancePolicy):
    """
    AC_CUHKAgent acceptance strategy from Genius (ANAC2012).

    Complex acceptance with concede degree calculation. Accepts based on
    threshold that adapts to discounting and opponent behavior.

    Args:
        offering_policy: The offering strategy to determine next bid.
        min_threshold: Minimum utility threshold (default 0.65).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2012.AC_CUHKAgent
    """

    offering_policy: OfferingPolicy
    min_threshold: float = 0.65
    _utility_threshold: float = field(init=False, default=0.9)
    _concede_factor: float = field(init=False, default=0.9)
    _opponent_max_util: float = field(init=False, default=0.0)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        time = state.relative_time

        # Track opponent max utility
        if opponent_util > self._opponent_max_util:
            self._opponent_max_util = opponent_util

        # Update threshold based on time
        self._utility_threshold = self._calculate_threshold(time)

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept if above threshold or better than our next bid
        if opponent_util >= self._utility_threshold or opponent_util >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, more flexible
        if time > 0.9985:
            if opponent_util >= self._opponent_max_util - 0.01:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _calculate_threshold(self, time: float) -> float:
        """Calculate utility threshold based on time."""
        # Threshold decreases over time with concede factor
        base = 0.95
        target = self.min_threshold
        return base - (base - target) * (time**self._concede_factor)


@define
class GACOMACagent(GeniusAcceptancePolicy):
    """
    AC_OMACagent acceptance strategy from Genius (ANAC2012).

    Accepts if we've made this bid before or if opponent's utility
    is at least as good as our planned bid.

    Args:
        offering_policy: The offering strategy to determine next bid.
        discount_threshold: Discount threshold for special behavior (default 0.845).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2012.AC_OMACagent
    """

    offering_policy: OfferingPolicy
    discount_threshold: float = 0.845

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

        time = state.relative_time

        # Check if we've made this bid before
        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if offer in my_offers:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, accept if we've made this bid
        if time > 0.97 and offer in my_offers:
            return ResponseType.ACCEPT_OFFER

        # Get our next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        opponent_util = float(self.negotiator.ufun(offer))

        # Accept if opponent's offer >= our planned offer
        if opponent_util >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACAgentLG(GeniusAcceptancePolicy):
    """
    AC_AgentLG acceptance strategy from Genius (ANAC2012).

    Frequency-based acceptance that tracks opponent bids and accepts
    based on relative utility and time pressure.

    Args:
        accept_ratio: Ratio for acceptance comparison (default 0.99).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2012.AC_AgentLG
    """

    accept_ratio: float = 0.99
    _min_acceptable: float = field(init=False, default=0.8)

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

        time = state.relative_time

        # Need our own bid history
        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        my_last_util = float(self.negotiator.ufun(my_offers[-1]))
        opponent_util = float(self.negotiator.ufun(offer))

        # Update minimum acceptable based on time
        self._min_acceptable = max(0.5, 0.9 - time * 0.3)

        # Accept if opponent's offer is close enough to ours
        if opponent_util >= my_last_util * self.accept_ratio:
            return ResponseType.ACCEPT_OFFER

        # Near deadline, more flexible
        if time > 0.999 and opponent_util >= my_last_util * 0.9:
            return ResponseType.ACCEPT_OFFER

        # Accept if above minimum threshold
        if opponent_util >= self._min_acceptable:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACAgentMR(GeniusAcceptancePolicy):
    """
    AC_AgentMR acceptance strategy from Genius (ANAC2012).

    Time-based concession with sigmoid acceptance probability.
    Tracks opponent offers and adjusts acceptance based on forecasting.

    Args:
        minimum_accept_p: Minimum acceptance probability threshold (default 0.965).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2012.AC_AgentMR
    """

    minimum_accept_p: float = 0.965
    _opponent_utils: list = field(init=False, factory=list)
    _min_bid_util: float = field(init=False, default=0.8)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        time = state.relative_time

        # Track opponent utilities
        self._opponent_utils.append(opponent_util)

        # Update minimum bid utility based on time
        self._update_min_bid_utility(time)

        # Calculate acceptance probability
        p = self._paccept(opponent_util, time)

        # Accept if probability exceeds threshold or utility exceeds minimum
        if p > self.minimum_accept_p or opponent_util > self._min_bid_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _update_min_bid_utility(self, time: float) -> None:
        """Update minimum bid utility based on time."""
        # Sigmoid-based decrease
        sigmoid_gain = -5.0
        percent = 0.70
        self._min_bid_util = 1.0 - percent * (
            1 / (1 + pow(10, sigmoid_gain * (time - 0.5)))
        )

    def _paccept(self, util: float, time: float) -> float:
        """Calculate acceptance probability."""
        t = time**3  # Steeper increase near deadline
        if util < 0 or util > 1.05:
            return 0.0
        if t < 0 or t > 1:
            return 0.0
        if util > 1:
            util = 1.0
        if t == 0.5:
            return util
        return (
            util
            - 2 * util * t
            + 2 * (-1 + t + ((-1 + t) ** 2 + util * (-1 + 2 * t)) ** 0.5)
        ) / (-1 + 2 * t)


@define
class GACBRAMAgent2(GeniusAcceptancePolicy):
    """
    AC_BRAMAgent2 acceptance strategy from Genius (ANAC2012).

    Enhanced BRAM with better threshold adaptation. Similar to BRAMAgent
    but with improved handling of edge cases.

    Args:
        offering_policy: The offering strategy to determine next bid.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2012.AC_BRAMAgent2
    """

    offering_policy: OfferingPolicy
    _threshold: float = field(init=False, default=0.9)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # Update threshold
        self._threshold = self._calculate_threshold(state)

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept if above threshold
        if opponent_util >= self._threshold:
            return ResponseType.ACCEPT_OFFER

        # Accept if better than our next bid
        if opponent_util >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _calculate_threshold(self, state: GBState) -> float:
        """Calculate acceptance threshold."""
        time = state.relative_time
        # Threshold decreases quadratically
        return 0.95 - 0.4 * (time**2)


@define
class GACIAMHaggler2012(GeniusAcceptancePolicy):
    """
    AC_IAMHaggler2012 acceptance strategy from Genius (ANAC2012).

    Adaptive threshold acceptance with multiplier-based comparison.

    Args:
        offering_policy: The offering strategy to determine next bid.
        accept_multiplier: Multiplier for acceptance (default 1.02).
        maximum_aspiration: Maximum target utility (default 0.9).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2012.AC_IAMHaggler2012
    """

    offering_policy: OfferingPolicy
    accept_multiplier: float = 1.02
    maximum_aspiration: float = 0.9

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

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        my_last_util = float(self.negotiator.ufun(my_offers[-1]))

        # Accept based on my last bid
        if opponent_util * self.accept_multiplier >= my_last_util:
            return ResponseType.ACCEPT_OFFER

        # Accept based on aspiration
        if opponent_util * self.accept_multiplier >= self.maximum_aspiration:
            return ResponseType.ACCEPT_OFFER

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

            if opponent_util * self.accept_multiplier >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACTheNegotiatorReloaded(GeniusAcceptancePolicy):
    """
    AC_TheNegotiatorReloaded acceptance strategy from Genius (ANAC2012).

    Phase-based acceptance with domain analysis. Uses AC_Next variant
    combined with AC_MaxInWindow for panic phase.

    Args:
        offering_policy: The offering strategy to determine next bid.
        a_next: Scaling factor for AC_next no discount (default 1.0).
        b_next: Addition factor for AC_next no discount (default 0.0).
        constant: Utility threshold above which to always accept (default 0.98).
        panic_time: Time after which panic phase begins (default 0.99).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2012.AC_TheNegotiatorReloaded
    """

    offering_policy: OfferingPolicy
    a_next: float = 1.0
    b_next: float = 0.0
    constant: float = 0.98
    panic_time: float = 0.99

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

        time = state.relative_time
        opponent_util = float(self.negotiator.ufun(offer))

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # Accept if above constant threshold
        if opponent_util >= self.constant:
            return ResponseType.ACCEPT_OFFER

        # AC_Next with parameters
        if self.a_next * opponent_util + self.b_next >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        # Panic phase: AC_MaxInWindow
        if time > self.panic_time:
            # Get opponent's best recent offer
            my_id = self.negotiator.id
            partner_offers: list[Outcome] = []
            for nid in nmi.negotiator_ids:
                if nid != my_id:
                    partner_offers.extend(nmi.negotiator_offers(nid))

            if partner_offers:
                # Simple window: last portion of offers
                window_size = max(1, len(partner_offers) // 4)
                recent_offers = partner_offers[-window_size:]
                best_recent = max(
                    float(self.negotiator.ufun(o))
                    for o in recent_offers
                    if o is not None
                )

                if opponent_util >= best_recent:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


# =============================================================================
# ANAC 2013 Acceptance Strategies
# =============================================================================


@define
class GACTheFawkes(GeniusAcceptancePolicy):
    """
    AC_TheFawkes acceptance strategy from Genius (ANAC2013).

    ACcombi = ACnext || (ACtime(T) & ACconst(MAXw)).
    Accepts when our bid is worse than opponent's, or near deadline
    when opponent's bid has maximum value in a window.

    Args:
        offering_policy: The offering strategy to determine next bid.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2013.AC_TheFawkes
    """

    offering_policy: OfferingPolicy
    _min_acceptable: float = field(init=False, default=0.5)
    _max_time_diff: float = field(init=False, default=0.01)

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
        time = state.relative_time

        # Calculate minimum acceptable (average utility)
        if state.step == 0:
            self._min_acceptable = 0.5  # Default

        # Reject if below minimum
        if opponent_util < self._min_acceptable:
            return ResponseType.REJECT_OFFER

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        my_next_util = 1.0
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

        # AC_Next: accept if opponent >= our next bid
        if opponent_util >= my_next_util:
            return ResponseType.ACCEPT_OFFER

        # Near deadline: ACtime && ACconst
        if time >= (1 - self._max_time_diff):
            # Get best opponent offer in window
            my_id = self.negotiator.id
            partner_offers: list[Outcome] = []
            for nid in nmi.negotiator_ids:
                if nid != my_id:
                    partner_offers.extend(nmi.negotiator_offers(nid))

            if partner_offers:
                # Window is last portion
                window_size = max(
                    1, int(len(partner_offers) * self._max_time_diff * 10)
                )
                recent = partner_offers[-window_size:]
                best_recent = max(
                    float(self.negotiator.ufun(o)) for o in recent if o is not None
                )

                if opponent_util >= best_recent:
                    return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACInoxAgent(GeniusAcceptancePolicy):
    """
    AC_InoxAgent acceptance strategy from Genius (ANAC2013).

    Scaling threshold acceptance. Breaks when reservation value is better,
    accepts when opponent's offer exceeds a time-dependent threshold.

    Args:
        reservation_value: Minimum acceptable utility (default 0.0).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2013.AC_InoxAgent
    """

    reservation_value: float = 0.0
    _median_util: float = field(init=False, default=0.5)
    _rounds_left: int = field(init=False, default=100)
    _time_diffs: list = field(init=False, factory=list)
    _last_time: float = field(init=False, default=0.0)

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

        time = state.relative_time
        opponent_util = float(self.negotiator.ufun(offer))

        # Update rounds left estimation
        self._update_rounds_left(time)

        # Get our worst bid utility
        my_offers = nmi.negotiator_offers(self.negotiator.id)
        my_worst_util = 1.0
        if my_offers:
            my_worst_util = min(
                float(self.negotiator.ufun(o)) for o in my_offers if o is not None
            )

        # Break if our worst is below reservation
        if my_worst_util < self.reservation_value:
            return ResponseType.END_NEGOTIATION

        # Accept if opponent's offer is close to our worst
        if my_worst_util <= opponent_util + 0.05:
            return ResponseType.ACCEPT_OFFER

        # Accept if above acceptance utility threshold
        accept_util = self._accept_util(time)
        if opponent_util >= accept_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER

    def _update_rounds_left(self, time: float) -> None:
        """Estimate rounds left."""
        self._time_diffs.append(time - self._last_time)
        self._last_time = time

        if len(self._time_diffs) >= 10:
            if len(self._time_diffs) > 10:
                self._time_diffs.pop(0)
            avg_diff = sum(self._time_diffs) / len(self._time_diffs)
            if avg_diff > 0:
                self._rounds_left = int((1 - time) / avg_diff)

    def _accept_util(self, time: float) -> float:
        """Calculate acceptance utility threshold."""
        if self._rounds_left < 8:
            return max(self._median_util, self.reservation_value)

        # Start high, decrease over time
        start_val = 1.0
        final_val = max(self._median_util, self.reservation_value)
        power = 27
        return start_val - (start_val - final_val) * (time**power)


@define
class GACInoxAgentOneIssue(GeniusAcceptancePolicy):
    """
    AC_InoxAgent_OneIssue acceptance strategy from Genius (ANAC2013).

    Simplified InoxAgent for single-issue domains. Accepts when
    opponent's offer exceeds median utility.

    Args:
        reservation_value: Minimum acceptable utility (default 0.0).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2013.AC_InoxAgent_OneIssue
    """

    reservation_value: float = 0.0
    _median_util: float = field(init=False, default=0.5)

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # Accept if above median
        if opponent_util >= self._median_util:
            return ResponseType.ACCEPT_OFFER

        # Break if reservation looks better
        if self.reservation_value >= self._median_util:
            return ResponseType.END_NEGOTIATION

        return ResponseType.REJECT_OFFER


# =============================================================================
# Other Acceptance Strategies
# =============================================================================


@define
class GACUncertain(GeniusAcceptancePolicy):
    """
    AC_Uncertain acceptance strategy from Genius.

    Handles uncertainty profiles. Accepts if offer is in top 10% of bids
    or if utility is at least 90% of our last offer.

    Args:
        top_percentile: Top percentile to accept (default 0.1).
        utility_ratio: Minimum ratio to our last offer (default 0.9).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_Uncertain
    """

    top_percentile: float = 0.1
    utility_ratio: float = 0.9

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

        my_offers = nmi.negotiator_offers(self.negotiator.id)
        if not my_offers:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))
        my_last_util = float(self.negotiator.ufun(my_offers[-1]))

        # Accept if opponent's offer is at least ratio * our last offer
        if opponent_util >= self.utility_ratio * my_last_util:
            return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER


@define
class GACMAC(GeniusAcceptancePolicy):
    """
    AC_MAC acceptance strategy from Genius.

    Multi-acceptance condition testing. Combines multiple AC strategies
    and accepts if any of them would accept.

    This is a simplified version that combines AC_CombiV4 and AC_CombiMaxInWindow
    with default parameters.

    Args:
        offering_policy: The offering strategy to determine next bid.
        constant: Utility threshold (default 0.95).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.other.AC_MAC
    """

    offering_policy: OfferingPolicy
    constant: float = 0.95

    def __call__(
        self, state: GBState, offer: Outcome | None, source: str | None
    ) -> ResponseType:
        """Decide whether to accept the opponent's offer."""
        if not self.negotiator or not self.negotiator.ufun:
            return ResponseType.REJECT_OFFER

        if offer is None:
            return ResponseType.REJECT_OFFER

        opponent_util = float(self.negotiator.ufun(offer))

        # Simple combined condition: accept if above constant
        if opponent_util >= self.constant:
            return ResponseType.ACCEPT_OFFER

        # Get next bid utility
        my_next_offer = self.offering_policy(state)
        if my_next_offer is not None:
            from negmas.outcomes.common import ExtendedOutcome

            if isinstance(my_next_offer, ExtendedOutcome):
                actual = my_next_offer.outcome
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

            # AC_Next variant
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
