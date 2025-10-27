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
        """and  .

        Args:
            other: Other.

        Returns:
            EvaluationStrategy: The result.
        """
        return AllAcceptEvaluationStrategy([self, other])

    def __or__(self, other: EvaluationStrategy) -> EvaluationStrategy:
        """or  .

        Args:
            other: Other.

        Returns:
            EvaluationStrategy: The result.
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
        """Eval.

        Args:
            negotiator_id: Negotiator id.
            state: Current state.
            history: History.
            mechanism_state: Mechanism state.

        Returns:
            GBResponse: The result.
        """
        ...


class AnyAcceptEvaluationStrategy(EvaluationStrategy):
    """AnyAcceptEvaluation strategy."""

    def __init__(self, strategies: list[EvaluationStrategy]):
        """Initialize the instance.

        Args:
            strategies: Strategies.
        """
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
        """Initialize the instance.

        Args:
            strategies: Strategies.
        """
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
    """All accept.

    Args:
        responses: Responses.

    Returns:
        GBResponse: The result.
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
    """Any accept.

    Args:
        responses: Responses.

    Returns:
        GBResponse: The result.
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
