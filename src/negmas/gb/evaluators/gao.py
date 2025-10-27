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
        """Eval.

        Args:
            negotiator_id: Negotiator id.
            state: Current state.
            history: History.
            mechanism_state: Mechanism state.

        Returns:
            GBResponse: The result.
        """
        offer, responses = state.new_offer, state.new_responses
        if offer is None:
            return None
        if any(_ == ResponseType.END_NEGOTIATION for _ in responses.values()):
            return None
        if all(_ == ResponseType.ACCEPT_OFFER for _ in responses.values()):
            return offer
        return "continue"
