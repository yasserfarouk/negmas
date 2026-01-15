"""MiCRO (Monotonic Concession with Rational Offers) negotiator implementations."""

from __future__ import annotations

from ..components.acceptance import MiCROAcceptancePolicy
from ..components.offering import MiCROOfferingPolicy, FastMiCROOfferingPolicy
from .modular.mapneg import MAPNegotiator

__all__ = ["MiCRONegotiator", "FastMiCRONegotiator"]


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
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        kwargs["offering"] = MiCROOfferingPolicy()
        kwargs["acceptance"] = MiCROAcceptancePolicy(kwargs["offering"], accept_same)
        super().__init__(*args, **kwargs)


class FastMiCRONegotiator(MAPNegotiator):
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
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        kwargs["offering"] = FastMiCROOfferingPolicy()
        kwargs["acceptance"] = MiCROAcceptancePolicy(kwargs["offering"], accept_same)
        super().__init__(*args, **kwargs)
