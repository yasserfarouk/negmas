"""Module for unique functionality."""

from __future__ import annotations

from ..common import ThreadState
from .base import LocalOfferingConstraint

__all__ = ["UniqueOffers"]


class UniqueOffers(LocalOfferingConstraint):
    """UniqueOffers implementation."""

    def __call__(self, state: ThreadState, history: list[ThreadState]) -> bool:
        """Make instance callable.

        Args:
            state: Current state.
            history: History.

        Returns:
            bool: The result.
        """
        offer = state.new_offer
        if not offer:
            return False
        return offer not in {_.new_offer for _ in history}
