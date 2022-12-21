from __future__ import annotations

import math
import random
from abc import abstractmethod
from typing import TypeVar

import numpy as np

from negmas import warnings
from negmas.helpers.prob import Distribution, ScipyDistribution
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.protocols import OutcomeSpace

from .base_ufun import BaseUtilityFunction, _General

__all__ = [
    "UtilityFunction",
]

T = TypeVar("T", bound="UtilityFunction")


class UtilityFunction(_General, BaseUtilityFunction):
    """Base for all crisp ufuns"""

    @abstractmethod
    def eval(self, offer: Outcome) -> float:
        ...

    def to_crisp(self) -> UtilityFunction:
        return self

    @classmethod
    def generate_bilateral(
        cls,
        outcomes: int | list[Outcome],
        conflict_level: float = 0.5,
        conflict_delta=0.005,
    ) -> tuple[UtilityFunction, UtilityFunction]:
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
    ) -> tuple[UtilityFunction, UtilityFunction]:
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
    ) -> list[UtilityFunction]:
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

    def is_not_worse(self, first: Outcome, second: Outcome) -> bool:
        return self.difference(first, second) >= 0

    def difference_prob(self, first: Outcome, second: Outcome) -> Distribution:
        """
        Returns a numeric difference between the utility of the two given outcomes
        """
        return ScipyDistribution(
            loc=self.difference(first, second), scale=0.0, type="uniform"
        )

    def minmax(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | tuple[Issue, ...] | None = None,
        outcomes: list[Outcome] | tuple[Outcome, ...] | None = None,
        max_cardinality=1000,
        above_reserve=False,
    ) -> tuple[float, float]:
        """Finds the range of the given utility function for the given outcomes

        Args:
            self: The utility function
            issues: List of issues (optional)
            outcomes: A collection of outcomes (optional)
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not given)
            above_reserve: If given, the minimum and maximum will be set to reserved value if they were less than it.

        Returns:
            (lowest, highest) utilities in that order

        """
        (worst, best) = self.extreme_outcomes(
            outcome_space,
            tuple(issues) if issues else issues,
            tuple(outcomes) if outcomes else outcomes,
            max_cardinality,
        )
        w, b = self(worst), self(best)
        if above_reserve:
            r = self.reserved_value
            if r is None:
                return w, b
            if b < r:
                b, w = r, r
            elif w < r:
                w = r
        return w, b

    def eval_normalized(
        self,
        offer: Outcome | None,
        above_reserve: bool = True,
        expected_limits: bool = True,
    ) -> float:
        """
        Evaluates the ufun normalizing the result between zero and one

        Args:
            offer (Outcome | None): offer
            above_reserve (bool): If True, zero corresponds to the reserved value not the minimum
            expected_limits (bool): If True, the expectation of the utility limits will be used for normalization instead of the maximum range and minimum lowest limit

        Remarks:
            - If the maximum and the minium are equal, finite and above reserve, will return 1.0.
            - If the maximum and the minium are equal, initinte or below reserve, will return 0.0.
            - For probabilistic ufuns, a distribution will still be returned.
            - The minimum and maximum will be evaluated freshly every time. If they are already caached in the ufun, the cache will be used.

        """
        r = self.reserved_value
        u = self.eval(offer) if offer else r
        mn, mx = self.minmax()
        if above_reserve:
            if mx < r:
                mx = mn = float("-inf")
            elif mn < r:
                mn = r
        d = mx - mn
        if d < 1e-5:
            warnings.warn(
                f"Ufun has equal max and min. The outcome will be normalized to zero if they were finite otherwise 1.0: {mn=}, {mx=}, {r=}, {u=}"
            )
            return 1.0 if math.isfinite(mx) else 0.0
        d = 1 / d
        return (u - mn) * d

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

    def __getitem__(self, offer: Outcome | None) -> float | None:
        """Overrides [] operator to call the ufun allowing it to act as a mapping"""
        return self(offer)


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
