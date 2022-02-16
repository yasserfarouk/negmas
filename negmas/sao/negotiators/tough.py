from __future__ import annotations

from ..components import AcceptAbove, AcceptBest, AcceptTop, OfferBest, OfferTop
from .modular import MAPNegotiator

__all__ = [
    "ToughNegotiator",
    "TopFractionNegotiator",
]


class ToughNegotiator(MAPNegotiator):
    """
    Accepts and proposes only the top offer (i.e. the one with highest utility).

    Args:
         name: Negotiator name
         parent: Parent controller if any
         can_propose: If `False` the negotiator will never propose but can only accept
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         owner: The `Agent` that owns the negotiator.

    Remarks:
        - If there are multiple outcome with the same maximum utility, only one of them will be used.

    """

    def __init__(
        self,
        can_propose=True,
        **kwargs,
    ):
        acceptance = AcceptBest()
        offering = None if not can_propose else OfferBest()
        super().__init__(acceptance=acceptance, offering=offering, **kwargs)


class TopFractionNegotiator(MAPNegotiator):
    """
    Offers and accepts only one of the top outcomes for the negotiator.

    Args:
        name: Negotiator name
        parent: Parent controller if any
        can_propose: If `False` the negotiator will never propose but can only accept
        preferences: The preferences of the negotiator
        ufun: The ufun of the negotiator (overrides preferences)
        min_utility: The minimum utility to offer or accept
        top_fraction: The fraction of the outcomes (ordered decreasingly by utility) to offer or accept
        best_first: Guarantee offering will non-increasing in terms of utility value
        probabilistic_offering: Offer randomly from the outcomes selected based on `top_fraction` and `min_utility`
        owner: The `Agent` that owns the negotiator.
    """

    def __init__(
        self,
        min_utility=0.95,
        top_fraction=0.05,
        best_first=True,
        can_propose=True,
        **kwargs,
    ):
        acceptance = AcceptAbove(min_utility) and AcceptTop(top_fraction)
        offering = None if not can_propose else OfferTop(top_fraction)
        self._best_first = best_first
        self.__offered = False
        super().__init__(acceptance=acceptance, offering=offering, **kwargs)

    def propose(self, state):
        if not self.ufun:
            return None
        if not self.__offered:
            _, best = self.ufun.extreme_outcomes()
            self.__offered = True
            return best
        return super().propose(state)
