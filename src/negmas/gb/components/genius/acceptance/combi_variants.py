"""acceptance components (split from acceptance.py): combi_variants."""

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
    "GACCombiV2",
    "GACCombiV3",
    "GACCombiV4",
    "GACCombiBestAvgDiscounted",
    "GACCombiMaxInWindowDiscounted",
    "GACCombiProb",
    "GACCombiProbDiscounted",
]


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
            actual_offer = (
                my_next_offer.best_for(self.negotiator.ufun)
                if self.negotiator.ufun
                else my_next_offer.outcome
            )
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
            actual_offer = (
                my_next_offer.best_for(self.negotiator.ufun)
                if self.negotiator.ufun
                else my_next_offer.outcome
            )
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
                actual_offer = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual_offer = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
                actual_offer = (
                    my_next_offer.best_for(self.negotiator.ufun)
                    if self.negotiator.ufun
                    else my_next_offer.outcome
                )
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
