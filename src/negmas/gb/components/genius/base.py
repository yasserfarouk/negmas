"""Base classes for Genius BOA components.

This module provides base classes for Genius BOA (Bidding, Opponent modeling,
Acceptance) components transcompiled from the original Java implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.stability import VOLATILE

from ..base import AcceptancePolicy, GBComponent, OfferingPolicy

if TYPE_CHECKING:
    pass

__all__ = ["GeniusOfferingPolicy", "GeniusAcceptancePolicy", "GeniusOpponentModel"]


@define
class GeniusOfferingPolicy(OfferingPolicy):
    """Base class for Genius offering policies.

    Holds the search knobs shared by (almost) every transcompiled Genius
    offering strategy, so they can be tuned uniformly.

    Args:
        utility_band_tolerance: Slack added on each side of the target utility
            when asking the inverter for an outcome, i.e. the policy searches
            ``[target - tol, pmax + tol]``. Defaults to ``0.01``.
    """

    utility_band_tolerance: float = 0.01


class GeniusAcceptancePolicy(AcceptancePolicy):
    """Base class for Genius acceptance policies."""

    pass


class GeniusOpponentModel(GBComponent, BaseUtilityFunction):
    """Base class for Genius opponent models.

    This base class provides helper methods for updating the negotiator's
    private_info with learned opponent utility function estimates.
    """

    def __attrs_post_init__(self) -> None:
        """Initialize parent classes after attrs initialization."""
        BaseUtilityFunction.__init__(self, stability=VOLATILE)

    def on_negotiation_start(self, state) -> None:
        """Publish this model as the negotiator's opponent ufun.

        `_update_private_info` existed on this base but nothing called it, so a
        negotiator using a Genius model still reported `opponent_ufun is None`. Doing it
        here covers all of them at once, and at a callback that means what it says --
        the `UFunModel` family publishes from `on_preferences_changed`, which works only
        because negmas happens to fire that first.
        """
        super().on_negotiation_start(state)
        self._update_private_info()

    def _update_private_info(self, partner_id: str | None = None) -> None:
        """Update the negotiator's private_info with this model.

        For bilateral negotiations, sets private_info["opponent_ufun"].
        For multilateral negotiations, sets private_info["opponent_ufuns"][partner_id].

        Args:
            partner_id: The partner's ID for multilateral negotiations.
                       If None, assumes bilateral and uses "opponent_ufun".
        """
        if not self.negotiator:
            return

        # Ensure private_info exists
        if not hasattr(self.negotiator, "private_info"):
            return

        private_info = self.negotiator.private_info
        if private_info is None:
            return

        # Check if this is a multilateral negotiation
        nmi = self.negotiator.nmi
        is_multilateral = nmi is not None and nmi.n_negotiators > 2

        if is_multilateral and partner_id is not None:
            # Multilateral: store in opponent_ufuns dict
            if "opponent_ufuns" not in private_info:
                private_info["opponent_ufuns"] = {}
            private_info["opponent_ufuns"][partner_id] = self
        else:
            # Bilateral: store directly
            private_info["opponent_ufun"] = self
