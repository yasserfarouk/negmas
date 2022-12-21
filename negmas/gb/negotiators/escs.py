from __future__ import annotations

from ..components.acceptance import AcceptAnyRational, SCSAcceptancePolicy
from ..components.offering import ESCSOfferingPolicy
from .modular.mapneg import MAPNegotiator

__all__ = [
    "ESCSNegotiator",
    "ESCSARNegotiator",
]


class ESCSNegotiator(MAPNegotiator):
    """
    Exploring Slow Concession Negotiator

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = SCSAcceptancePolicy()
        kwargs["offering"] = ESCSOfferingPolicy()
        super().__init__(*args, **kwargs)


class ESCSARNegotiator(MAPNegotiator):
    """
    Exploring Slow Concession Negotiator (Accepting any rational outcome)

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptAnyRational()
        kwargs["offering"] = ESCSOfferingPolicy()
        super().__init__(*args, **kwargs)
