from __future__ import annotations

import math
import numbers
import random
from typing import Generator

import numpy as np

from negmas.helpers.numeric import sample
from negmas.outcomes.base_issue import DiscreteIssue, Issue
from negmas.outcomes.range_issue import RangeIssue

__all__ = ["ContiguousIssue"]


class ContiguousIssue(RangeIssue, DiscreteIssue):
    """
    A `RangeIssue` (also a `DiscreteIssue`) representing a contiguous range of integers.
    """

    def __init__(
        self,
        values: int | tuple[int, int],
        name: str | None = None,
    ) -> None:
        if isinstance(values, numbers.Integral):
            values = (0, int(values) - 1)
        super().__init__(values, name)
        self._n_values = values[1] - values[0] + 1  # type: ignore

    def _to_xml_str(self, indx):
        # return (
        #     f'    <issue etype="integer" index="{indx + 1}" name="{self.name}" type="integer" vtype="integer"'
        #     f' lowerbound="{self._values[0]}" upperbound="{self._values[1]}" />\n'
        # )

        output = f'    <issue etype="discrete" index="{indx + 1}" name="{self.name}" type="discrete" vtype="integer">\n'
        for i, v in enumerate(range(self._values[0], self._values[1] + 1)):
            output += f'        <item index="{i + 1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
        output += "    </issue>\n"
        return output

    @property
    def all(self) -> Generator[int, None, None]:
        yield from range(self._values[0], self._values[1] + 1)

    @property
    def cardinality(self) -> int:
        return self.max_value - self.min_value + 1

    def ordered_value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[int, None, None]:
        m = self.cardinality
        n = m if n is None or not math.isfinite(n) else int(n)
        for i in range(n):
            yield self._values[0] + (i % m)

    def value_generator(
        self, n: int | float | None = 10, grid=True, compact=False, endpoints=True
    ) -> Generator[int, None, None]:
        yield from (
            _ + self._values[0]
            for _ in sample(
                self.cardinality, n, grid=grid, compact=compact, endpoints=endpoints
            )
        )

    def to_discrete(
        self, n: int | None, grid=True, compact=False, endpoints=True
    ) -> DiscreteIssue:
        if n is None or self.cardinality < n:
            return self
        if not compact:
            return super().to_discrete(
                n, grid=grid, compact=compact, endpoints=endpoints
            )

        beg = (self.cardinality - n) // 2
        return ContiguousIssue((int(beg), int(beg + n)), name=self.name + f"{n}")

    def rand(self) -> int:
        """Picks a random valid value."""
        return random.randint(*self._values)

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list[int]:
        """Picks a random valid value."""

        if n > self._n_values and not with_replacement:
            if fail_if_not_enough:
                raise ValueError(
                    f"Cannot sample {n} outcomes out of {self._values} without replacement"
                )
            return [_ for _ in range(self._values[0], self._values[1] + 1)]

        if with_replacement:
            return np.random.randint(
                low=self._values[0], high=self._values[1] + 1, size=n
            ).tolist()
        vals = [_ for _ in range(self._values[0], self._values[1] + 1)]
        random.shuffle(vals)
        return vals[:n]

    def rand_invalid(self):
        """Pick a random *invalid* value"""

        return random.randint(self.max_value + 1, 2 * self.max_value)

    def is_continuous(self) -> bool:
        return False

    def value_at(self, index: int):
        if index < 0 or index > self.cardinality - 1:
            raise IndexError(index)
        return self.min_value + index

    def contains(self, issue: Issue) -> bool:
        """Checks weather this issue contains the input issue (i.e. every value in the input issue is in this issue)"""
        return (
            issubclass(issue.value_type, numbers.Integral)
            and issue.min_value >= self.min_value
            and issue.max_value <= self.max_value
        )
