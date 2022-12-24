from __future__ import annotations

from ..components.acceptance import (
    AcceptAnyRational,
    AcceptBetterRational,
    AcceptNotWorseRational,
)
from ..components.offering import CABOfferingPolicy
from .modular.mapneg import MAPNegotiator

__all__ = [
    "CABNegotiator",
    "CARNegotiator",
    "CANNegotiator",
]


class CANNegotiator(MAPNegotiator):
    """
    Conceding Accepting Not Worse Strategy (optimal, complete, but not an equilibirum)

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptNotWorseRational()
        kwargs["offering"] = CABOfferingPolicy()
        super().__init__(*args, **kwargs)


class CABNegotiator(MAPNegotiator):
    """
    Conceding Accepting Better Strategy (optimal, complete, but not an equilibirum)

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptBetterRational()
        kwargs["offering"] = CABOfferingPolicy()
        super().__init__(*args, **kwargs)


class CARNegotiator(MAPNegotiator):
    """
    Conceding Accepting Rational Strategy (neither complete nor an equilibrium)

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        kwargs["acceptance"] = AcceptAnyRational()
        kwargs["offering"] = CABOfferingPolicy()
        super().__init__(*args, **kwargs)
