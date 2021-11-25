from __future__ import annotations

from typing import Callable, Mapping, Protocol, Sequence, Union

from negmas.helpers import Distribution, Value
from negmas.outcomes import Outcome

__all__ = [
    "INVALID_UTILITY",
    "Distribution",
    "Value",
    "UtilityValue",
    "OrdinalRankingPreferences",
    "CardinalRankingPreferences",
    "OutcomeUtilityMapping",
]

INVALID_UTILITY = float("-inf")


UtilityValue = Union[Distribution, Value]
"""
Either a utility_function distribution or an exact offerable_outcomes
utility_function value.

`UtilityFunction`s always return a `UtilityValue` which makes it easier to
implement algorithms relying  on probabilistic modeling of utility functions.
"""

OutcomeUtilityMapping = Union[
    Callable[[Union[Outcome, int, str, float]], UtilityValue],
    Mapping[Union[Sequence, Mapping, int, str, float], UtilityValue],
]
"""A mapping from an outcome to its utility value"""


class OrdinalRankingPreferences(Protocol):
    """
    Implements ranking of outcomes
    """

    def rank(
        self, outcomes: tuple[tuple[Outcome | None]], descending=True
    ) -> list[list[int]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:
            A list of lists of integers giving the outcome index in the input. The list is sorted by utlity value

        """


class CardinalRankingPreferences(Protocol):
    """
    Implements ranking of outcomes
    """

    def rank_with_weights(
        self, outcomes: list[Outcome | None], descending=True
    ) -> list[tuple[tuple[int], float]]:
        """
        Ranks the given list of outcomes with weights. None stands for the null outcome.

        Returns:

            - A list of tuples each with two values:
                - an list of integers giving the index in the input array (outcomes) of an outcome (at the given utility level)
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
