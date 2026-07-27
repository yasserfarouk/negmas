"""MiCRO (Monotonic Concession with Rational Offers) negotiator implementations."""

from __future__ import annotations

from ..components.acceptance import MiCROAcceptancePolicy
from ..components.offering import FastMiCROOfferingPolicy, MiCROOfferingPolicy
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
         accept_same: Accept an offer equal in utility to our own next offer.
         offering: A ready `MiCROOfferingPolicy` to use instead of the default.
         acceptance: A ready `AcceptancePolicy` to use instead of
             `MiCROAcceptancePolicy`.
    """

    def __init__(self, *args, accept_same: bool = True, **kwargs):
        """Initializes the instance."""
        offering = kwargs.pop("offering", None) or MiCROOfferingPolicy()
        kwargs["offering"] = offering
        kwargs.setdefault("acceptance", MiCROAcceptancePolicy(offering, accept_same))
        super().__init__(*args, **kwargs)


class FastMiCRONegotiator(MAPNegotiator):
    """
    Rational Concession Negotiator that can skip outcomes so as to traverse the
    whole outcome list before the deadline.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.
         accept_same: Accept an offer equal in utility to our own next offer.
         forced_concession_time: Relative time after which concession is allowed
             even if we already sent more offers than we received.
         min_time_before_skipping: Skipping is disabled before this relative time.
         min_offers_before_skipping: Skipping is disabled until this many offers
             have been sent.
         expected_offers_rounding: Added before truncating the estimated number of
             remaining offers (``0.5`` rounds to nearest).
         offering: A ready `FastMiCROOfferingPolicy` to use instead of building
             one from the arguments above (which are then ignored).
         acceptance: A ready `AcceptancePolicy` to use instead of
             `MiCROAcceptancePolicy`.
    """

    def __init__(
        self,
        *args,
        accept_same: bool = True,
        forced_concession_time: float = 0.95,
        min_time_before_skipping: float = 0.1,
        min_offers_before_skipping: int = 5,
        expected_offers_rounding: float = 0.5,
        **kwargs,
    ):
        """Initializes the instance."""
        offering = kwargs.pop("offering", None) or FastMiCROOfferingPolicy(
            forced_concession_time=forced_concession_time,
            min_time_before_skipping=min_time_before_skipping,
            min_offers_before_skipping=min_offers_before_skipping,
            expected_offers_rounding=expected_offers_rounding,
        )
        kwargs["offering"] = offering
        kwargs.setdefault("acceptance", MiCROAcceptancePolicy(offering, accept_same))
        super().__init__(*args, **kwargs)
