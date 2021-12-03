from __future__ import annotations

import numbers

from negmas.helpers.numeric import sample
from negmas.outcomes.cardinal_issue import CardinalIssue

__all__ = ["RangeIssue"]


class RangeIssue(CardinalIssue):
    """
    An issue representing a range of values (can be continuous or discrete)
    """

    def __init__(self, values, name=None, id=None) -> None:
        super().__init__(values, name, id)
        self._value_type = type(values[0])
        self.min_value, self.max_value = values[0], values[1]

    @property
    def cardinality(self) -> int | float:
        if not issubclass(self._value_type, numbers.Integral):
            return float("inf")
        return self.max_value - self.min_value + 1

    def is_valid(self, v):
        return self.min_value <= v <= self.max_value
