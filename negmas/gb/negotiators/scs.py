from __future__ import annotations

from ..components.acceptance import AcceptAnyRational, SCSAcceptancePolicy
from ..components.offering import SCSOfferingPolicy
from .modular.mapneg import MAPNegotiator

__all__ = [
    "SCSNegotiator",
    "SCSARNegotiator",
]


class SCSNegotiator(MAPNegotiator):
    """
    Rational Concession Negotiator

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = SCSAcceptancePolicy()
        kwargs["offering"] = SCSOfferingPolicy()
        super().__init__(*args, **kwargs)


class SCSARNegotiator(MAPNegotiator):
    """
    Rational Concession Negotiator

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptAnyRational()
        kwargs["offering"] = SCSOfferingPolicy()
        super().__init__(*args, **kwargs)
