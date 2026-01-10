"""Base classes for Genius BOA components.

This module provides base classes for Genius BOA (Bidding, Opponent modeling,
Acceptance) components transcompiled from the original Java implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from negmas.preferences.base_ufun import BaseUtilityFunction
from negmas.preferences.mixins import VolatileUFunMixin

from ..base import AcceptancePolicy, GBComponent, OfferingPolicy

if TYPE_CHECKING:
    pass

__all__ = ["GeniusOfferingPolicy", "GeniusAcceptancePolicy", "GeniusOpponentModel"]


class GeniusOfferingPolicy(OfferingPolicy):
    """Base class for Genius offering policies."""

    pass


class GeniusAcceptancePolicy(AcceptancePolicy):
    """Base class for Genius acceptance policies."""

    pass


class GeniusOpponentModel(VolatileUFunMixin, GBComponent, BaseUtilityFunction):
    """Base class for Genius opponent models.

    This base class provides helper methods for updating the negotiator's
    private_info with learned opponent utility function estimates.
    """

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
