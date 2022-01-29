from __future__ import annotations

from negmas.sao.common import ResponseType

from ...common import MechanismState
from ...outcomes import Outcome
from ..components import RandomProposalMixin
from .base import SAONegotiator

__all__ = [
    "NiceNegotiator",
]


class NiceNegotiator(SAONegotiator, RandomProposalMixin):
    """
    Offers and accepts anything.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         assume_normalized: If true, the negotiator can assume that the ufun is normalized.
         rational_proposal: If `True`, the negotiator will never propose something with a utility value less than its
                            reserved value. If `propose` returned such an outcome, a NO_OFFER will be returned instead.
         owner: The `Agent` that owns the negotiator.

    """

    def __init__(self, *args, **kwargs):
        SAONegotiator.__init__(self, *args, **kwargs)
        self.init_random_proposal()

    def respond_(self, state: MechanismState, offer: Outcome) -> ResponseType:
        return ResponseType.ACCEPT_OFFER

    def propose_(self, state: MechanismState) -> Outcome | None:
        return RandomProposalMixin.propose(self, state)  # type: ignore

    propose = propose_
