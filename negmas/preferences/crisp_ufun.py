from __future__ import annotations

import random
from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from negmas.helpers.prob import Distribution, ScipyDistribution
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.common import check_one_at_most, os_or_none
from negmas.outcomes.protocols import OutcomeSpace

from .base_ufun import BaseUtilityFunction, _General
from .value_fun import MAX_CARINALITY

if TYPE_CHECKING:
    from .complex import WeightedUtilityFunction

__all__ = [
    "UtilityFunction",
]

T = TypeVar("T", bound="UtilityFunction")


class UtilityFunction(_General, BaseUtilityFunction):
    """Base for all crisp ufuns"""

    @abstractmethod
    def eval(self, offer: Outcome) -> float:
        ...

    def __call__(self, offer: Outcome | None) -> float:
        """Calculate the utility_function value for a given outcome.

        Args:
            offer: The offer to be evaluated.


        Remarks:

            - It calls the abstract method `eval` after opationally adjusting the
              outcome type.
            - It is preferred to override eval instead of directly overriding this method
            - You cannot return None from overriden eval() functions but raise an exception (ValueError) if it was
              not possible to calculate the Value.
            - Return a float from your `eval` implementation.
            - Return the reserved value if the offer was None

        Returns:
            The utility of the given outcome
        """
        if offer is None:
            return self.reserved_value
        return self.eval(offer)

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> float:
        return float(super().eval_normalized(offer, above_reserve, expected_limits))

    def to_crisp(self) -> UtilityFunction:
        return self

    def __getitem__(self, offer: Outcome | None) -> float | None:
        """Overrides [] operator to call the ufun allowing it to act as a mapping"""
        return self(offer)

    @classmethod
    def generate_bilateral(
        cls,
        outcomes: int | list[Outcome],
        conflict_level: float = 0.5,
        conflict_delta=0.005,
    ) -> tuple["UtilityFunction", "UtilityFunction"]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate.
                            1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.

        Examples:

            >>> from negmas.preferences import conflict_level
            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=0.0
            ...                                             , conflict_delta=0.0)
            >>> print(conflict_level(u1, u2, outcomes=10))
            0.0

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=1.0
            ...                                             , conflict_delta=0.0)
            >>> print(conflict_level(u1, u2, outcomes=10))
            1.0

            >>> u1, u2 = UtilityFunction.generate_bilateral(outcomes=10, conflict_level=0.5
            ...                                             , conflict_delta=0.0)
            >>> 0.0 < conflict_level(u1, u2, outcomes=10) < 1.0
            True


        """
        from negmas.preferences.crisp.mapping import MappingUtilityFunction

        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        n_outcomes = len(outcomes)
        u1 = np.random.random(n_outcomes)
        rand = np.random.random(n_outcomes)
        if conflict_level > 0.5:
            conflicting = 1.0 - u1 + conflict_delta * np.random.random(n_outcomes)
            u2 = conflicting * conflict_level + rand * (1 - conflict_level)
        elif conflict_level < 0.5:
            same = u1 + conflict_delta * np.random.random(n_outcomes)
            u2 = same * (1 - conflict_level) + rand * conflict_level
        else:
            u2 = rand

        # todo implement win_win correctly. Order the ufun then make outcomes with good outcome even better and vice
        # versa
        # u2 += u2 * win_win
        # u2 += np.random.random(n_outcomes) * conflict_delta
        u1 -= u1.min()
        u2 -= u2.min()
        u1 = u1 / u1.max()
        u2 = u2 / u2.max()
        if random.random() > 0.5:
            u1, u2 = u2, u1
        return (
            MappingUtilityFunction(dict(zip(outcomes, u1))),
            MappingUtilityFunction(dict(zip(outcomes, u2))),
        )

    @classmethod
    def generate_random_bilateral(
        cls, outcomes: int | list[Outcome]
    ) -> tuple["UtilityFunction", "UtilityFunction"]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate. 1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.
            zero_summness: How zero-sum like are the two ufuns.


        """
        from negmas.preferences.crisp.mapping import MappingUtilityFunction

        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        n_outcomes = len(outcomes)
        u1 = np.random.random(n_outcomes)
        u2 = np.random.random(n_outcomes)
        u1 -= u1.min()
        u2 -= u2.min()
        u1 /= u1.max()
        u2 /= u2.max()
        return (
            MappingUtilityFunction(dict(zip(outcomes, u1))),
            MappingUtilityFunction(dict(zip(outcomes, u2))),
        )

    @classmethod
    def generate_random(
        cls, n: int, outcomes: int | list[Outcome], normalized: bool = True
    ) -> list["UtilityFunction"]:
        """Generates N mapping utility functions

        Args:
            n: number of utility functions to generate
            outcomes: number of outcomes to use
            normalized: if true, the resulting ufuns will be normlized between zero and one.


        """
        from negmas.preferences.crisp.mapping import MappingUtilityFunction

        if isinstance(outcomes, int):
            outcomes = [(_,) for _ in range(outcomes)]
        else:
            outcomes = list(outcomes)
        n_outcomes = len(outcomes)
        ufuns = []
        for _ in range(n):
            u1 = np.random.random(n_outcomes)
            if normalized:
                u1 -= u1.min()
                u1 /= u1.max()
            ufuns.append(
                MappingUtilityFunction(dict(zip(outcomes, u1)), outcomes=outcomes)
            )
        return ufuns

    def normalize_for(
        self: T,
        to: tuple[float, float] = (0.0, 1.0),
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        minmax: tuple[float, float] | None = None,
    ) -> T | WeightedUtilityFunction:
        """
        Creates a new utility function that is normalized based on input conditions.

        Args:
            to: The minimum and maximum value to normalize to. If either is None, it is ignored.
                 This means that passing `(None, 1.0)` will normalize the ufun so that the maximum
                 is `1` but will not guarantee any limit for the minimum and so on.
            outcomes: A set of outcomes to limit our attention to. If not given,
                      the whole ufun is normalized
            outcome_space: The outcome-space to focus on when normalizing
            minmax: The current minimum and maximum to use for normalization. Pass if known to avoid
                  calculating them using the outcome-space given or defined for the ufun.
        """
        max_cardinality: int = MAX_CARINALITY
        outcome_space = None
        if minmax is not None:
            mn, mx = minmax
        else:
            check_one_at_most(outcome_space, issues, outcomes)
            outcome_space = os_or_none(outcome_space, issues, outcomes)
            if not outcome_space:
                outcome_space = self.outcome_space
            if not outcome_space:
                raise ValueError(
                    "Cannot find the outcome-space to normalize for. "
                    "You must pass outcome_space, issues or outcomes or have the ufun being constructed with one of them"
                )
            mn, mx = self.minmax(outcome_space, max_cardinality=max_cardinality)

        scale = (to[1] - to[0]) / (mx - mn)

        # u = self.shift_by(-mn, shift_reserved=True)
        u = self.scale_by(scale, scale_reserved=True)
        return u.shift_by(to[0] - scale * mn, shift_reserved=True)

    def is_not_worse(self, first: Outcome, second: Outcome) -> bool:
        return self.difference(first, second) > 0

    def difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return ScipyDistribution(
            loc=self.difference(first, second), scale=0.0, type="uniform"
        )


class CrispAdapter(UtilityFunction):
    """
    Adapts any utility function to act as a crisp utility function (i.e. returning a real number)
    """

    def __init__(self, prob: BaseUtilityFunction):
        self._prob = prob

    def eval(self, offer):
        return float(self._prob.eval(offer))

    def to_stationary(self):
        return CrispAdapter(prob=self._prob.to_stationary())
