"""Module for gao functionality."""

from __future__ import annotations

from negmas.common import MechanismState

from ..common import GBResponse, ResponseType, ThreadState
from ..evaluators.base import LocalEvaluationStrategy

__all__ = ["GAOEvaluationStrategy"]


class GAOEvaluationStrategy(LocalEvaluationStrategy):
    """GAOEvaluation strategy."""

    def eval(
        self,
        negotiator_id: str,
        state: ThreadState,
        history: list[ThreadState],
        mechanism_state: MechanismState,
    ) -> GBResponse:
        """Evaluate using the Generalized Accept/Offer protocol.

        Args:
            negotiator_id: ID of the negotiator being evaluated.
            state: Current state of the negotiation thread.
            history: List of previous thread states for context.
            mechanism_state: Overall mechanism state.

        Returns:
            Response indicating whether to accept the offer or continue/end negotiation.
        """
        offer, responses = state.new_offer, state.new_responses
        if offer is None:
            return None
        if any(_ == ResponseType.END_NEGOTIATION for _ in responses.values()):
            return None
        if all(_ == ResponseType.ACCEPT_OFFER for _ in responses.values()):
            return offer
        return "continue"
