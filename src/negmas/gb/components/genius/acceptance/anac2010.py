"""acceptance components (split from acceptance.py): anac2010."""

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
    "GACABMP",
    "GACAgentK",
    "GACAgentFSEGA",
    "GACIAMCrazyHaggler",
    "GACYushu",
    "GACNozomi",
    "GACIAMHaggler2010",
    "GACAgentSmith",
]


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

    Args:
        expected_utility_decay: How much the expected utility drops over the full
            negotiation (``expected = 1 - decay * t``). Defaults to ``0.5``.
        time_pressure_exponent: Exponent applied to relative time when computing
            time pressure. Defaults to ``2.0``.

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_AgentK
    """

    expected_utility_decay: float = 0.5
    time_pressure_exponent: float = 2.0
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
        expected_util = 1.0 - time * self.expected_utility_decay

        if opponent_util >= expected_util:
            return 1.0

        # Probability based on how close opponent's offer is to expected
        if expected_util > 0:
            ratio = opponent_util / expected_util
            # Increase probability as time runs out
            time_pressure = time**self.time_pressure_exponent
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
        max_utility: The assumed maximum utility in the domain (default 1.0,
            i.e. a normalized ufun).
        max_utility_tolerance: Fraction of ``max_utility`` above which an offer
            counts as "close to max" and is accepted (default 0.999).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_AgentFSEGA
    """

    offering_policy: OfferingPolicy
    multiplier: float = 1.03
    max_utility: float = 1.0
    max_utility_tolerance: float = 0.999

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
        max_util = self.max_utility  # Assume normalized

        # Get my next bid utility
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
            opponent_util * self.multiplier >= my_last_util
            or opponent_util > my_next_util
            or opponent_util >= max_util * self.max_utility_tolerance  # Close to max
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
        phase1_end: Relative time at which the early phase ends (default 0.50).
        phase2_end: Relative time at which the middle phase ends (default 0.80).
        phase1_coeff_slope: Slope of the early-phase acceptance coefficient
            ``slope * t + intercept`` (default -0.1).
        phase1_coeff_intercept: Intercept of the early-phase coefficient
            (default 1.0).
        phase2_coeff_slope: Slope of the middle-phase coefficient
            ``slope * (t - phase1_end) + intercept`` (default -0.16).
        phase2_coeff_intercept: Intercept of the middle-phase coefficient
            (default 0.95).
        phase2_compromise_factor: Fraction of the max compromise utility an
            offer must beat in the middle phase (default 0.95).
        phase3_compromise_factor: Same, for the late phase (default 0.90).
        phase3_split_time: Relative time splitting the late phase into its
            early and final parts (default 0.90).
        phase3_early_discount: Discount applied to our last utility in the early
            part of the late phase (default 0.40).
        phase3_late_discount: Same, for the final part (default 0.50).

    Transcompiled from: negotiator.boaframework.acceptanceconditions.anac2010.AC_Nozomi
    """

    max_util_threshold: float = 0.95
    phase1_end: float = 0.50
    phase2_end: float = 0.80
    phase1_coeff_slope: float = -0.1
    phase1_coeff_intercept: float = 1.0
    phase2_coeff_slope: float = -0.16
    phase2_coeff_intercept: float = 0.95
    phase2_compromise_factor: float = 0.95
    phase3_compromise_factor: float = 0.90
    phase3_split_time: float = 0.90
    phase3_early_discount: float = 0.40
    phase3_late_discount: float = 0.50
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
        if time < self.phase1_end:
            if (
                opponent_util > self._max_compromise_util
                and opponent_util >= self._max_opponent_util
            ):
                accept_coeff = (
                    self.phase1_coeff_slope * time + self.phase1_coeff_intercept
                )
                if opponent_util > my_last_util * accept_coeff:
                    return ResponseType.ACCEPT_OFFER
        elif time < self.phase2_end:
            if (
                opponent_util
                > self._max_compromise_util * self.phase2_compromise_factor
            ):
                accept_coeff = (
                    self.phase2_coeff_slope * (time - self.phase1_end)
                    + self.phase2_coeff_intercept
                )
                if opponent_util > my_last_util * accept_coeff:
                    return ResponseType.ACCEPT_OFFER
        else:
            # Late game - more willing to accept
            if (
                opponent_util
                > self._max_compromise_util * self.phase3_compromise_factor
            ):
                threshold = (
                    self.phase3_early_discount
                    if time <= self.phase3_split_time
                    else self.phase3_late_discount
                )
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
                actual = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
