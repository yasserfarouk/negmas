"""Module for unique functionality."""

from __future__ import annotations

from ..common import ThreadState
from .base import LocalOfferingConstraint

__all__ = ["UniqueOffers"]


class UniqueOffers(LocalOfferingConstraint):
    """UniqueOffers implementation."""

    def __call__(self, state: ThreadState, history: list[ThreadState]) -> bool:
        """Check if the current offer is unique (hasn't been made before).

        Args:
            state: Current thread state containing the new offer.
            history: List of previous thread states to check for duplicate offers.

        Returns:
            True if the offer is unique, False if it was already made.
        """
        offer = state.new_offer
        if not offer:
            return False
        return offer not in {_.new_offer for _ in history}
