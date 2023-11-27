from __future__ import annotations

from ..components.acceptance import AcceptAround, RandomAcceptancePolicy
from ..components.offering import RandomOfferingPolicy
from .modular import MAPNegotiator

__all__ = [
    "RandomNegotiator",
    "RandomAlwaysAcceptingNegotiator",
]


class RandomNegotiator(MAPNegotiator):
    """
    A negotiation agent that responds randomly in a single negotiation.

    Args:
        p_acceptance: Probability of accepting an offer
        p_rejection:  Probability of rejecting an offer
        p_ending: Probability of ending the negotiation at any round
        can_propose: Whether the agent can propose or not
        **kwargs: Passed to the GBNegotiator

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
        acceptance = RandomAcceptancePolicy(p_acceptance, p_rejection, p_ending)
        offering = None if not can_propose else RandomOfferingPolicy()
        super().__init__(acceptance=acceptance, offering=offering, **kwargs)


class RandomAlwaysAcceptingNegotiator(MAPNegotiator):
    def __init__(
        self,
        p_acceptance=0.15,
        p_rejection=0.75,
        p_ending=0.1,
        can_propose=True,
        accept_around=1.0,
        eps=1e-3,
        **kwargs,
    ) -> None:
        acceptance = RandomAcceptancePolicy(
            p_acceptance, p_rejection, p_ending
        ) | AcceptAround(accept_around, eps=eps)
        offering = None if not can_propose else RandomOfferingPolicy()
        super().__init__(acceptance=acceptance, offering=offering, **kwargs)
