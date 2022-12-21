from __future__ import annotations

from negmas.outcomes import Outcome

from ..common import ThreadState
from .base import LocalOfferingConstraint

__all__ = ["UniqueOffers"]


class UniqueOffers(LocalOfferingConstraint):
    def __call__(
        self,
        state: ThreadState,
        history: list[ThreadState],
    ) -> bool:
        offer = state.new_offer
        if not offer:
            return False
        return offer not in {_.new_offer for _ in history}
