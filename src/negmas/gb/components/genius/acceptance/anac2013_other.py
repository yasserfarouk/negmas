"""acceptance components (split from acceptance.py): anac2013_other."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field

from negmas.gb.common import ResponseType

from negmas.gb.components.base import OfferingPolicy
from negmas.gb.components.genius.base import GeniusAcceptancePolicy

if TYPE_CHECKING:
    from negmas.gb import GBState
    from negmas.outcomes import Outcome

__all__ = [
    "GACTheFawkes",
    "GACInoxAgent",
    "GACInoxAgentOneIssue",
    "GACUncertain",
    "GACMAC",
]


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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
            else:
                actual = my_next_offer
            my_next_util = float(self.negotiator.ufun(actual))

            # AC_Next variant
            if opponent_util >= my_next_util:
                return ResponseType.ACCEPT_OFFER

        return ResponseType.REJECT_OFFER
