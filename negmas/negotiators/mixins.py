from __future__ import annotations

import itertools
import math
from random import sample
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np

from negmas.outcomes import Issue, sample_issues
from negmas.preferences import Preferences

from .components import PolyAspiration

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.preferences import Preferences, Value


__all__ = [
    "AspirationMixin",
    "EvaluatorMixin",
    "RealComparatorMixin",
    "BinaryComparatorMixin",
    "NLevelsComparatorMixin",
    "RankerMixin",
    "RankerWithWeightsMixin",
    "SorterMixin",
]


class AspirationMixin:
    "A mixin that adds aspiration capability (and utility_at() method)"

    def aspiration_init(
        self,
        max_aspiration: float,
        aspiration_type: Union[str, int, float],
    ):
        """
        Initializes the mixin.

        Args:
            max_aspiration: The aspiration level to start from (usually 1.0)
            aspiration_type: The aspiration type. Can be a string ("boulware", "linear", "conceder") or a number giving the exponent of the aspiration curve.
        """
        self.__asp = PolyAspiration(max_aspiration, aspiration_type)

    def utility_at(self, t: float) -> float:
        """
        The aspiration level

        Args:
            t: relative time (a number between zero and one)

        Returns:
            aspiration level
        """
        return self.__asp.utility_at(t)


class EvaluatorMixin:
    """
    A mixin that can be used to have the negotiator respond to evaluate messages from the server.
    """

    def init(self):
        self.capabilities["evaluate"] = True

    def evaluate(self, outcome: "Outcome") -> Optional["Value"]:
        return self.ufun(outcome)


class RealComparatorMixin:
    def init(self):
        self.capabilities["compare-real"] = True
        self.capabilities["compare-binary"] = True

    def compare_real(self, first: "Outcome", second: "Outcome") -> Optional[float]:
        """
        Compares two offers using the `ufun` returning the difference in their utility

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            "Value": An estimate of the differences between the two outcomes. It can be a real number between -1, 1
            or a probability distribution over the same range.
        """
        return self.preferences.compare_real(first, second)

    def is_better(self, first: "Outcome", second: "Outcome") -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        return self._preferences.is_better(first, second)


class BinaryComparatorMixin:
    def init(self):
        self.capabilities["compare-binary"] = True

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_preferences:
            raise ValueError("Cannot compare outcomes without a ufun")
        return self._preferences.is_better(first, second)


class NLevelsComparatorMixin:
    def init(self):
        self.capabilities["compare-nlevels"] = True
        self.capabilities["compare-binary"] = True
        self.__preferences_thresholds = None

    @classmethod
    def generate_thresholds(
        cls,
        n: int,
        ufun_min: float = 0.0,
        ufun_max: float = 1.0,
        scale: Union[str, Callable[[float], float]] = None,
    ) -> List[float]:
        """
        Generates thresholds for the n given levels assuming the ufun ranges and scale function

        Args:
            n: Number of scale levels (one side)
            ufun_min: minimum value of all utilities
            ufun_max: maximum value of all utilities
            scale: Scales the ufun values. Can be a callable or 'log', 'exp', 'linear'. If None, it is 'linear'

        """
        if scale is not None:
            if isinstance(scale, str):
                scale = dict(
                    linear=lambda x: x,
                    log=math.log,
                    exp=math.exp,
                ).get(scale, None)
                if scale is None:
                    raise ValueError(f"Unknown scale function {scale}")
        thresholds = np.linspace(ufun_min, ufun_max, num=n + 2)[1:-1].tolist()
        if scale is not None:
            thresholds = [scale(_) for _ in thresholds]
        return thresholds

    @classmethod
    def equiprobable_thresholds(
        cls,
        n: int,
        preferences: "Preferences",
        issues: List[Issue],
        n_samples: int = 1000,
    ) -> List[float]:
        """
        Generates thresholds for the n given levels where levels are equally likely approximately

        Args:
            n: Number of scale levels (one side)
            preferences: The utility function to use
            issues: The issues to generate the thresholds for
            n_samples: The number of samples to use during the process

        """
        samples = list(
            sample_issues(
                issues, n_samples, with_replacement=False, fail_if_not_enough=False
            )
        )
        n_samples = len(samples)
        diffs = []
        for i, first in enumerate(samples):
            n_diffs = min(10, n_samples - i - 1)
            for second in sample(samples[i + 1 :], k=n_diffs):
                diffs.append(abs(preferences.compare_real(first, second)))
        diffs = np.array(diffs)
        _, edges = np.histogram(diffs, bins=n + 1)
        return edges[1:-1].tolist()

    @property
    def thresholds(self) -> Optional[List[float]]:
        """Returns the internal thresholds and None if they do  not exist"""
        return self.__preferences_thresholds

    @thresholds.setter
    def thresholds(self, thresholds: List[float]) -> None:
        self.__preferences_thresholds = thresholds

    def compare_nlevels(
        self, first: "Outcome", second: "Outcome", n: int = 2
    ) -> Optional[int]:
        """
        Compares two offers using the `ufun` returning an integer in [-n, n] (i.e. 2n+1 possible values) which defines
        which outcome is better and the strength of the difference (discretized using internal thresholds)

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            n: number of levels to use

        Returns:

            - None if either there is no ufun defined or the number of thresholds required cannot be satisfied
            - 0 iff |u(first) - u(second)| <= thresholds[0]
            - -i if  - thresholds[i-1] < u(first) - u(second) <= -thresholds[i]
            - +i if  thresholds[i-1] > u(first) - u(second) >= thresholds[i]

        Remarks:

            - thresholds is an internal array that can be set using `thresholds` property
            - thresholds[n] is assumed to equal infinity
            - n must be <= the length of the internal thresholds array. If n > that length, a ValueError will be raised.
              If n < the length of the internal thresholds array, the first n values of the array will be used
        """
        if not self.has_preferences:
            return None
        if self.thresholds is None:
            raise ValueError(
                f"Internal thresholds array is not set. Please set the threshold property with an array"
                f" of length >= {n}"
            )
        if len(self.thresholds) < n:
            raise ValueError(
                f"Internal thresholds array is only of length {len(self.thresholds)}. It cannot be used"
                f" to compare outcomes with {n} levels. len(self.thresholds) MUST be >= {n}"
            )
        diff = self._preferences(first) - self._preferences(second)
        sign = 1 if diff > 0.0 else -1
        for i, th in enumerate(self.thresholds):
            if diff < th:
                return sign * i
        return sign * n

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_preferences:
            return None
        return self._preferences.is_better(first, second)


class RankerWithWeightsMixin:
    """Adds the ability to rank outcomes returning the ranks and weights"""

    def init(self):
        self.capabilities["rank-weighted"] = True
        self.capabilities["compare-binary"] = True

    def rank_with_weights(
        self, outcomes: List[Optional["Outcome"]], descending=True
    ) -> List[Tuple[int, float]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome. Outcomes of equal utility
        are ordered arbitrarily.

        Returns:

            - A list of tuples each with two values:
                - an integer giving the index in the input array (outcomes) of an outcome
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
        if not self.has_preferences:
            return None
        return self._preferences.rank_with_weights(outcomes, descending)

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_preferences:
            return None
        return self._preferences.is_better(first, second)


class RankerMixin:
    """Adds the ability to rank outcomes returning the ranks without weights. Outcomes of equal utility are ordered
    arbitrarily. None stands for the null outcome"""

    def init(self):
        self.capabilities["rank"] = True
        self.capabilities["compare-binary"] = True

    def rank(self, outcomes: List[Optional["Outcome"]], descending=True) -> List[int]:
        """Ranks the given list of outcomes. None stands for the null outcome.

        Returns:

            - A list of integers in the specified order of utility values of outcomes

        """
        if not self.has_preferences:
            return None
        return self._preferences.rank(outcomes, descending)

    def is_better(
        self, first: "Outcome", second: "Outcome", epsilon: float = 1e-10
    ) -> Optional[bool]:
        """
        Compares two offers using the `ufun` returning whether the first is better than the second

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared
            epsilon: comparison threshold. If the utility difference within the range [-epsilon, epsilon] the two
                     outcomes are assumed to be compatible

        Returns:
            True if utility(first) > utility(second) + epsilon
            None if |utility(first) - utility(second)| <= epsilon or the utun is not defined
            False if utility(first) < utility(second) - epsilon
        """
        if not self.has_preferences:
            return None
        return self._preferences.is_better(first, second)


class SorterMixin:
    """Adds the ability to sort outcomes according to utility. Outcomes of equal utility are ordered
    arbitrarily. None stands for the null outcome"""

    def init(self):
        self.capabilities["sort"] = True

    def sort(self, outcomes: List[Optional["Outcome"]], descending=True) -> None:
        """Ranks the given list of outcomes. None stands for the null outcome.

        Returns:

            - The outcomes are sorted IN PLACE.
            - There is no way to know if the ufun is not defined from the return value. Use `has_preferences` to check for
              the availability of the ufun

        """
        if not self.has_preferences:
            return None
        ranks = self._preferences.rank(outcomes, descending)
        outcomes = itertools.chain(*tuple(ranks))
