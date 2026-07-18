"""acceptance components (split from acceptance.py): anac2012."""

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
    "GACCUHKAgent",
    "GACOMACagent",
    "GACAgentLG",
    "GACAgentMR",
    "GACBRAMAgent2",
    "GACIAMHaggler2012",
    "GACTheNegotiatorReloaded",
]


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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
