from __future__ import annotations

from ..components.acceptance import MiCROAcceptancePolicy
from ..components.offering import MiCROOfferingPolicy
from .modular.mapneg import MAPNegotiator

__all__ = [
    "MiCRONegotiator",
]


class MiCRONegotiator(MAPNegotiator):
    """
    Rational Concession Negotiator

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, accept_same: bool = True, **kwargs):
        kwargs["offering"] = MiCROOfferingPolicy()
        kwargs["acceptance"] = MiCROAcceptancePolicy(kwargs["offering"], accept_same)
        super().__init__(*args, **kwargs)
