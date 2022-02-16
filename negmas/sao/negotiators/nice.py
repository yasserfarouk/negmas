from __future__ import annotations

from ..components.acceptance import AcceptImmediately
from ..components.offering import RandomOfferingStrategy
from .modular.mapneg import MAPNegotiator

__all__ = [
    "NiceNegotiator",
]


class NiceNegotiator(MAPNegotiator):
    """
    Offers and accepts anything.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptImmediately()
        kwargs["offering"] = RandomOfferingStrategy()
        super().__init__(*args, **kwargs)
