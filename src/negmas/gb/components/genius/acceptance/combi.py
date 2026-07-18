"""acceptance components (split from acceptance.py): combi."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from negmas.gb.common import ResponseType

from negmas.gb.components.base import OfferingPolicy
from negmas.gb.components.genius.base import GeniusAcceptancePolicy

if TYPE_CHECKING:
    from negmas.gb import GBState
    from negmas.outcomes import Outcome

__all__ = [
    "GACCombi",
    "GACCombiMaxInWindow",
    "GACCombiAvg",
    "GACCombiBestAvg",
    "GACCombiMax",
]


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
                actual_offer = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual_offer = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual_offer = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual_offer = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
