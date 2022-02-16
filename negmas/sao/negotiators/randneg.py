from __future__ import annotations

from ..components.acceptance import RandomAcceptanceStrategy
from ..components.offering import RandomOfferingStrategy
from .modular import MAPNegotiator

__all__ = [
    "RandomNegotiator",
]


class RandomNegotiator(MAPNegotiator):
    """
    A negotiation agent that responds randomly in a single negotiation.

    Args:
        p_acceptance: Probability of accepting an offer
        p_rejection:  Probability of rejecting an offer
        p_ending: Probability of ending the negotiation at any round
        can_propose: Whether the agent can propose or not
        **kwargs: Passed to the SAONegotiator

    Remarks:
        - If p_acceptance + p_rejection + p_ending < 1, the rest is the probability of no-response.
    """

    def __init__(
        self,
        p_acceptance=0.15,
        p_rejection=0.75,
        p_ending=0.1,
        can_propose=True,
        **kwargs,
    ) -> None:
        acceptance = RandomAcceptanceStrategy(p_acceptance, p_rejection, p_ending)
        offering = None if not can_propose else RandomOfferingStrategy()
        super().__init__(acceptance=acceptance, offering=offering, **kwargs)
