"""Hybrid negotiator combining multiple strategies."""

from __future__ import annotations

from ..components.acceptance import ACNext
from ..components.offering import HybridOfferingPolicy
from .modular.mapneg import MAPNegotiator

__all__ = ["HybridNegotiator"]


class HybridNegotiator(MAPNegotiator):
    """
    Rational Concession Negotiator

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, alpha: float = 1.0, beta: float = 0.0, **kwargs):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        kwargs["offering"] = HybridOfferingPolicy()
        kwargs["acceptance"] = ACNext(kwargs["offering"], alpha=alpha, beta=beta)
        super().__init__(*args, **kwargs)
