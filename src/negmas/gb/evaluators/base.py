"""
Base Evaluation Strategies
"""

from __future__ import annotations


from abc import ABC, abstractmethod
from random import choice

from attrs import define

from negmas.common import MechanismState
from negmas.gb.common import GBResponse, GBState, ThreadState
from negmas.outcomes.common import Outcome

__all__ = [
    "EvaluationStrategy",
    "LocalEvaluationStrategy",
    "AnyAcceptEvaluationStrategy",
    "AllAcceptEvaluationStrategy",
    "all_accept",
    "any_accept",
]


@define
class EvaluationStrategy(ABC):
    @abstractmethod
    def __call__(
        self,
        negotiator_ids: list[str],
        state: GBState,
        history: list[GBState],
        active_thread: int | None,
    ) -> GBResponse:
        """Base class for evaluation strategies

        Args:
            negotiator_ids (list[str]): List of negotiator IDs (in the same order as threads)
            state (GBState): Current state of the mechanism
            history (list[GBState]): History of past states
            active_thread (int | None): If integer, the current thread (used for local evaluators). Global evaluators

        Returns:
            GBResponse
        """

        ...

    def __and__(self, other: EvaluationStrategy) -> EvaluationStrategy:
        """Combine two evaluation strategies with logical AND.

        Args:
            other: The evaluation strategy to combine with this one.

        Returns:
            A strategy that accepts only when both strategies accept.
        """
        return AllAcceptEvaluationStrategy([self, other])

    def __or__(self, other: EvaluationStrategy) -> EvaluationStrategy:
        """Combine two evaluation strategies with logical OR.

        Args:
            other: The evaluation strategy to combine with this one.

        Returns:
            A strategy that accepts when either strategy accepts.
        """
        return AnyAcceptEvaluationStrategy([self, other])


@define
class LocalEvaluationStrategy(EvaluationStrategy):
    """LocalEvaluation strategy."""

    def __call__(
        self,
        negotiator_ids: list[str],
        state: GBState,
        history: list[GBState],
        active_thread: int | None,
    ) -> GBResponse:
        """Base class for evaluation strategies

        Args:
            negotiator_ids (list[str]): List of negotiator IDs (in the same order as threads)
            state (GBState): Current state of the mechanism
            history (list[GBState]): History of past states
            active_thread (int): the current thread.

        """
        assert active_thread is not None
        source = negotiator_ids[active_thread]
        return self.eval(
            source,
            state.threads[source],
            state.thread_history(history, source),
            state.base_state,
        )

    @abstractmethod
    def eval(
        self,
        negotiator_id: str,
        state: ThreadState,
        history: list[ThreadState],
        mechanism_state: MechanismState,
    ) -> GBResponse:
        """Evaluate the current state and return a response.

        Args:
            negotiator_id: ID of the negotiator being evaluated.
            state: Current state of the negotiation thread.
            history: List of previous thread states for context.
            mechanism_state: Overall mechanism state.

        Returns:
            Response indicating whether to accept, reject, or continue.
        """
        ...


class AnyAcceptEvaluationStrategy(EvaluationStrategy):
    """AnyAcceptEvaluation strategy."""

    def __init__(self, strategies: list[EvaluationStrategy]):
        """Initializes the instance."""
        self._strategies = strategies

    def __call__(self, *args, **kwargs) -> GBResponse:
        """Make instance callable.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            GBResponse: The result.
        """
        return any_accept([_(*args, **kwargs) for _ in self._strategies])


class AllAcceptEvaluationStrategy(EvaluationStrategy):
    """AllAcceptEvaluation strategy."""

    def __init__(self, strategies: list[EvaluationStrategy]):
        """Initializes the instance."""
        self._strategies = strategies

    def __call__(self, *args, **kwargs) -> GBResponse:
        """Make instance callable.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            GBResponse: The result.
        """
        return all_accept([_(*args, **kwargs) for _ in self._strategies])


def all_accept(responses: list[GBResponse]) -> GBResponse:
    """Combine multiple responses requiring all to agree for acceptance.

    Args:
        responses: List of responses from different evaluation strategies.

    Returns:
        Accepted outcome if all agree, None if any reject, otherwise 'continue'.
    """
    if not responses:
        return "continue"
    if len(set(responses)) == 1:
        return responses[0]
    if any(_ is None for _ in responses):
        return None
    # we either have multiple agreements from multiple strategies or some strategies says continue
    return "continue"


def any_accept(responses: list[GBResponse]) -> GBResponse:
    """Combine multiple responses accepting if any strategy accepts.

    Args:
        responses: List of responses from different evaluation strategies.

    Returns:
        Random accepted outcome if any strategy accepts, None if all reject, otherwise 'continue'.
    """
    if not responses:
        return "continue"
    acceptances = [_ for _ in responses if isinstance(_, Outcome)]
    if acceptances:
        return choice(acceptances)
    if any(_ is None for _ in responses):
        return None
    # we either have multiple agreements from multiple strategies or some strategies says continue
    return "continue"
