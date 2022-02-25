from __future__ import annotations

import numbers
import random
from abc import abstractmethod

import numpy as np

from negmas.helpers.numeric import get_one_float
from negmas.helpers.prob import Distribution, Real, ScipyDistribution
from negmas.outcomes import Outcome

from .base_ufun import BaseUtilityFunction, _General

__all__ = [
    "ProbUtilityFunction",
]


class ProbUtilityFunction(_General, BaseUtilityFunction):
    """A probablistic utility function. One that returns a probability distribution when called"""

    @abstractmethod
    def eval(self, offer: Outcome) -> Distribution:
        ...

    def to_prob(self) -> ProbUtilityFunction:
        return self

    @classmethod
    def generate_bilateral(
        cls,
        outcomes: int | list[Outcome],
        conflict_level: float = 0.5,
        conflict_delta=0.005,
        scale: float | tuple[float, float] = 0.5,
    ) -> tuple[ProbUtilityFunction, ProbUtilityFunction]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate.
                            1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.

        Examples:

            >>> from negmas.preferences import UtilityFunction
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
        from negmas.preferences.prob.mapping import ProbMappingUtilityFunction

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
            ProbMappingUtilityFunction(
                dict(
                    zip(
                        outcomes,
                        (
                            ScipyDistribution(
                                type="unifomr", loc=_, scale=get_one_float(scale)
                            )
                            for _ in u1
                        ),
                    )
                )
            ),
            ProbMappingUtilityFunction(
                dict(
                    zip(
                        outcomes,
                        (
                            ScipyDistribution(
                                type="unifomr", loc=_, scale=get_one_float(scale)
                            )
                            for _ in u2
                        ),
                    )
                )
            ),
        )

    @classmethod
    def generate_random_bilateral(
        cls, outcomes: int | list[Outcome], scale: float = 0.5
    ) -> tuple[ProbUtilityFunction, ProbUtilityFunction]:
        """Generates a couple of utility functions

        Args:

            n_outcomes (int): number of outcomes to use
            conflict_level: How conflicting are the two ufuns to generate. 1.0 means maximum conflict.
            conflict_delta: How variable is the conflict at different outcomes.
            zero_summness: How zero-sum like are the two ufuns.


        """
        from negmas.preferences.prob.mapping import ProbMappingUtilityFunction

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
            ProbMappingUtilityFunction(
                dict(
                    zip(
                        outcomes,
                        (
                            ScipyDistribution(
                                type="unifomr", loc=_, scale=get_one_float(scale)
                            )
                            for _ in u1
                        ),
                    )
                )
            ),
            ProbMappingUtilityFunction(
                dict(
                    zip(
                        outcomes,
                        (
                            ScipyDistribution(
                                type="unifomr", loc=_, scale=get_one_float(scale)
                            )
                            for _ in u2
                        ),
                    )
                )
            ),
        )

    @classmethod
    def generate_random(
        cls,
        n: int,
        outcomes: int | list[Outcome],
        normalized: bool = True,
        scale: float | tuple[float, float] = 0.5,
    ) -> list[ProbUtilityFunction]:
        """Generates N mapping utility functions

        Args:
            n: number of utility functions to generate
            outcomes: number of outcomes to use
            normalized: if true, the resulting ufuns will be normlized between zero and one.


        """
        from negmas.preferences.prob.mapping import ProbMappingUtilityFunction

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
                ProbMappingUtilityFunction(
                    dict(
                        zip(
                            outcomes,
                            (
                                ScipyDistribution(
                                    type="unifomr", loc=_, scale=get_one_float(scale)
                                )
                                for _ in u1
                            ),
                        )
                    )
                )
            )
        return ufuns

    def __call__(self, offer: Outcome | None) -> Distribution:
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
            return ScipyDistribution("uniform", loc=self.reserved_value, scale=0.0)
        v = self.eval(offer)
        if isinstance(v, float):
            return Real(v)
        return v


class ProbAdapter(ProbUtilityFunction):
    """
    Adapts any utility function to act as a probabilistic utility function (i.e. returning a `Distribution` )
    """

    def __init__(self, ufun: BaseUtilityFunction):
        self._ufun = ufun

    def eval(self, offer):
        v = self._ufun.eval(offer)
        if isinstance(v, numbers.Real):
            return Real(v)
        return v

    def to_stationary(self):
        return ProbAdapter(self._ufun.to_stationary())
