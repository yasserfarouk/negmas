from __future__ import annotations

import numbers
import random
import sys

from negmas.outcomes.contiguous_issue import ContiguousIssue
from negmas.outcomes.continuous_issue import ContinuousIssue

__all__ = ["CountableInfiniteIssue", "ContinuousInfiniteIssue", "InfiniteIssue"]

INFINITE_INT = sys.maxsize


class InfiniteIssue:
    """
    Indicates that the issue is infinite (i.e. one or more of its limits is infinity)
    """


class CountableInfiniteIssue(ContiguousIssue, InfiniteIssue):
    """
    An issue that can have all integer values.

    Remarks:
        - Actually, inifinties are replace with +- INFINITE_INT which is a very large number
    """

    def __init__(
        self, values: tuple[int | float, int | float], *args, **kwargs
    ) -> None:
        v: tuple[int, int] = tuple(  # type: ignore
            int(_)
            if isinstance(_, numbers.Integral)
            else INFINITE_INT
            if _ > 0
            else -INFINITE_INT
            for _ in values
        )
        super().__init__(v, *args, **kwargs)
        self._value_type = int
        self._n_values = float("inf")
        self.min_value, self.max_value = values

    def is_countable(self) -> bool:
        return True

    def is_continuous(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return True

    @property
    def cardinality(self) -> float:
        return float("inf")

    def rand_invalid(self):
        if self._values[0] == -INFINITE_INT and self._values[1] == INFINITE_INT:
            return None
        if self._values[0] == -INFINITE_INT:
            return random.randint(self._values[1] + 1, INFINITE_INT)
        return random.randint(-INFINITE_INT, self._values[0] - 1)


class ContinuousInfiniteIssue(ContinuousIssue, InfiniteIssue):
    """
    An issue that can all real
    """
