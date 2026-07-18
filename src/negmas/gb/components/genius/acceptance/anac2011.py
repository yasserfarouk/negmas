"""acceptance components (split from acceptance.py): anac2011."""

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
    "GACHardHeaded",
    "GACAgentK2",
    "GACBRAMAgent",
    "GACGahboninho",
    "GACNiceTitForTat",
    "GACTheNegotiator",
    "GACValueModelAgent",
    "GACIAMHaggler2011",
]


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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
