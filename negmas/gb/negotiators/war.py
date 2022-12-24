from __future__ import annotations

from ..components.acceptance import (
    AcceptAnyRational,
    AcceptBetterRational,
    AcceptNotWorseRational,
)
from ..components.offering import WAROfferingPolicy
from .modular.mapneg import MAPNegotiator

__all__ = [
    "WABNegotiator",
    "WARNegotiator",
    "WANNegotiator",
]


class WABNegotiator(MAPNegotiator):
    """
    Wasting Accepting Better (neither complete nor an equilibrium)

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptBetterRational()
        kwargs["offering"] = WAROfferingPolicy()
        super().__init__(*args, **kwargs)


class WARNegotiator(MAPNegotiator):
    """
    Wasting Accepting Any (an equilibrium but not complete)

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptAnyRational()
        kwargs["offering"] = WAROfferingPolicy()
        super().__init__(*args, **kwargs)


class WANNegotiator(MAPNegotiator):
    """
    Wasting Accepting Any (an equilibrium but not complete)

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides preferences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptNotWorseRational()
        kwargs["offering"] = WAROfferingPolicy()
        super().__init__(*args, **kwargs)
