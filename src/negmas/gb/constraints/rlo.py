"""Module for rlo functionality."""

from __future__ import annotations

import sys

from attrs import define

from ..common import ThreadState
from .base import LocalOfferingConstraint

__all__ = ["RepeatLastOfferOnly"]


@define
class RepeatLastOfferOnly(LocalOfferingConstraint):
    """RepeatLastOfferOnly implementation."""

    n: int = sys.maxsize

    def eval(self, state: ThreadState, history: list[ThreadState]) -> bool:
        """Eval.

        Args:
            state: Current state.
            history: History.

        Returns:
            bool: The result.
        """
        offer = state.new_offer
        if not offer:
            return False
        past = {_.new_offer for _ in history}
        if history and len(past) >= self.n:
            past = past.difference({history[-1].new_offer})
        return offer not in past
