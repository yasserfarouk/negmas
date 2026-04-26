"""Module for rfo functionality."""

from __future__ import annotations

import sys

from attrs import define

from ..common import ThreadState
from .base import LocalOfferingConstraint

__all__ = ["RepeatFinalOfferOnly"]


@define
class RepeatFinalOfferOnly(LocalOfferingConstraint):
    """RepeatFinalOfferOnly implementation."""

    n: int = sys.maxsize

    def __call__(self, state: ThreadState, history: list[ThreadState]) -> bool:
        """Check if the repeating free offers constraint is satisfied.

        Args:
            state: Current thread state.
            history: List of previous thread states for detecting repeated offers.

        Returns:
            True if the offer hasn't been repeated too many times, False otherwise.
        """
        offer = state.new_offer
        if not offer:
            return False
        outcomes = [_.new_offer for _ in history]
        if outcomes:
            for a, b in zip(outcomes[:-1], outcomes[1:]):
                if a == offer:
                    break
                if a == b:
                    return False
        past = set(outcomes)
        if history and len(past) >= self.n:
            past = past.difference({history[-1].new_offer})
        return offer not in past
