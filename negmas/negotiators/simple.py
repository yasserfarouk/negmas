from __future__ import annotations

import itertools
import math
from random import sample
from typing import TYPE_CHECKING, Callable

import numpy as np

from negmas.negotiators import Negotiator

from ..outcomes.issue_ops import sample_issues
from .negotiator import Negotiator

if TYPE_CHECKING:
    from ..common import Value
    from ..outcomes.base_issue import Issue
    from ..outcomes.common import Outcome
    from ..preferences import Preferences

__all__ = [
    "EvaluatorNegotiator",
    "RealComparatorNegotiator",
    "BinaryComparatorNegotiator",
    "NLevelsComparatorNegotiator",
    "RankerNegotiator",
    "RankerWithWeightsNegotiator",
    "SorterNegotiator",
]


class EvaluatorNegotiator(Negotiator):
    """
    A negotiator that can be asked to evaluate outcomes using its internal ufun.

    Th change the way it evaluates outcomes, override `evaluate`.

    It has the `evaluate` capability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities["evaluate"] = True

    def evaluate(self, outcome: Outcome) -> Value | None:
        if not self.ufun:
            return None
        return self.ufun(outcome)


class RealComparatorNegotiator(Negotiator):
    """
    A negotiator that can be asked to evaluate outcomes using its internal ufun.

    Th change the way it evaluates outcomes, override `compare_real`

    It has the `compare-real` capability
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities["compare-real"] = True
        self.capabilities["compare-binary"] = True

    def difference(self, first: Outcome, second: Outcome) -> float:
        """
        Compares two offers using the `ufun` returning the difference in their utility

        Args:
            first: First outcome to be compared
            second: Second outcome to be compared

        Returns:
            "Value": An estimate of the differences between the two outcomes. It can be a real number between -1, 1
            or a probability distribution over the same range.
        """
        if not self.preferences:
            raise ValueError(f"Cannot compare outcomes. I have no preferences")
        return self.preferences.difference(first, second)

    def is_better(self, first: Outcome, second: Outcome) -> bool | None:
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
        if not self.preferences:
            return None
        return self.preferences.is_better(first, second)


class BinaryComparatorNegotiator(Negotiator):
    """
    A negotiator that can be asked to compare two outcomes using is_better. By default is just consults the ufun.

    To change that behavior, override `is_better`.

    It has the `compare-binary` capability.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities["compare-binary"] = True

    def is_better(
        self, first: Outcome, second: Outcome, epsilon: float = 1e-10
    ) -> bool | None:
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


class NLevelsComparatorNegotiator(Negotiator):
    """
    A negotiator that can be asked to compare two outcomes using compare_nlevels which returns the strength of
    the difference between two outcomes as an integer from [-n, n] in the C compare sense.
    By default is just consults the ufun.

    To change that behavior, override `compare_nlevels`.

    It has the `compare-nlevels` capability.

    """

    def __init__(self, *args, thresholds: list[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.thresholds = thresholds  # type: ignore I am not sure why
        self.capabilities["compare-nlevels"] = True
        self.capabilities["compare-binary"] = True
        self.__preferences_thresholds = None

    @classmethod
    def generate_thresholds(
        cls,
        n: int,
        ufun_min: float = 0.0,
        ufun_max: float = 1.0,
        scale: str | Callable[[float], float] = None,
    ) -> list[float]:
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
        preferences: Preferences,
        issues: list[Issue],
        n_samples: int = 1000,
    ) -> list[float]:
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
    def thresholds(self) -> list[float] | None:
        """Returns the internal thresholds and None if they do  not exist"""
        return self.__preferences_thresholds

    @thresholds.setter
    def thresholds(self, thresholds: list[float]) -> None:
        self.__preferences_thresholds = thresholds

    def compare_nlevels(
        self, first: Outcome, second: Outcome, n: int = 2
    ) -> int | None:
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
        if not self.ufun:
            raise ValueError("Unknown preferences")
        diff = self.ufun(first) - self.ufun(second)
        sign = 1 if diff > 0.0 else -1
        for i, th in enumerate(self.thresholds):
            if diff < th:
                return sign * i
        return sign * n

    def is_better(
        self, first: Outcome, second: Outcome, epsilon: float = 1e-10
    ) -> bool | None:
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


class RankerWithWeightsNegotiator(Negotiator):
    """
    A negotiator that can be asked to rank outcomes returning rank and weight. By default is just consults the ufun.

    To change that behavior, override `rank_with_weights`.

    It has the `rank-weighted` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities["rank-weighted"] = True
        self.capabilities["compare-binary"] = True

    def rank_with_weights(
        self, outcomes: list[Outcome] | None, descending=True
    ) -> list[tuple[int, float]]:
        """Ranks the given list of outcomes with weights. None stands for the null outcome. Outcomes of equal utility
        are ordered arbitrarily.

        Returns:

            - A list of tuples each with two values:
                - an integer giving the index in the input array (outcomes) of an outcome
                - the weight of that outcome
            - The list is sorted by weights descendingly

        """
        if not self.preferences:
            raise ValueError(f"Has no preferences. Cannot rank")
        return self.preferences.rank_with_weights(outcomes, descending)

    def is_better(
        self, first: Outcome, second: Outcome, epsilon: float = 1e-10
    ) -> bool | None:
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


class RankerNegotiator(Negotiator):
    """
    A negotiator that can be asked to rank outcomes. By default is just consults the ufun.

    To change that behavior, override `rank`.

    It has the `rank` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities["rank"] = True
        self.capabilities["compare-binary"] = True

    def rank(self, outcomes: list[Outcome | None], descending=True) -> list[int]:
        """Ranks the given list of outcomes. None stands for the null outcome.

        Returns:

            - A list of integers in the specified order of utility values of outcomes

        """
        if not self.preferences:
            raise ValueError(f"Unknown preferences. Cannot rank")
        return self.preferences.rank(outcomes, descending)

    def is_better(
        self, first: Outcome, second: Outcome, epsilon: float = 1e-10
    ) -> bool | None:
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


class SorterNegotiator(Negotiator):
    """
    A negotiator that can be asked to rank outcomes returning rank without weight.
    By default is just consults the ufun.

    To change that behavior, override `sort`.

    It has the `sort` capability.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capabilities["sort"] = True

    def sort(self, outcomes: list[Outcome | None], descending=True) -> None:
        """Ranks the given list of outcomes. None stands for the null outcome.

        Returns:

            - The outcomes are sorted IN PLACE.
            - There is no way to know if the ufun is not defined from the return value. Use `has_preferences` to check for
              the availability of the ufun

        """
        if not self.has_preferences:
            raise ValueError(f"Cannot sort outcomes. Unknown preferences")
        ranks = self._preferences.rank(outcomes, descending)
        outcomes = itertools.chain(*tuple(ranks))
