from __future__ import annotations

from negmas.outcomes.cardinal_issue import CardinalIssue

__all__ = ["RangeIssue"]


class RangeIssue(CardinalIssue):
    """
    An issue representing a range of values (can be continuous or discrete)
    """

    def __init__(self, values, name=None) -> None:
        if values[1] < values[0]:
            raise ValueError(
                f"Trying to create a range issue with min {values[0]} and max {values[1]}!!!"
            )
        super().__init__(values, name)
        self._value_type = type(values[0])
        self.min_value, self.max_value = values[0], values[1]

    def is_valid(self, v):
        return self.min_value <= v <= self.max_value
