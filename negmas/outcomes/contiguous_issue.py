from __future__ import annotations

import numbers
import random
from typing import Generator

import numpy as np

from negmas.java import PYTHON_CLASS_IDENTIFIER
from negmas.outcomes.base_issue import RangeIssue
from negmas.outcomes.ordinal_issue import OrdinalIssue

__all__ = ["ContiguousIssue"]


class ContiguousIssue(RangeIssue, OrdinalIssue):
    def __init__(
        self,
        values: int | tuple[int, int],
        name: str | None = None,
        id=None,
    ) -> None:
        if isinstance(values, numbers.Integral):
            values = (0, int(values) - 1)
        super().__init__(values, name, id)
        self._n_values = values[1] - values[0] + 1

    def _to_xml_str(self, indx, enumerate_integer=False):
        if not enumerate_integer:
            return (
                f'    <issue etype="integer" index="{indx + 1}" name="{self.name}" type="integer" vtype="integer"'
                f' lowerbound="{self._values[0]}" upperbound="{self._values[1]}" />\n'
            )

        output = f'    <issue etype="discrete" index="{indx + 1}" name="{self.name}" type="discrete" vtype="integer">\n'
        for i, v in enumerate(range(self._values[0], self._values[1] + 1)):
            output += f'        <item index="{i + 1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
        output += "    </issue>\n"
        return output

    @property
    def all(self) -> Generator:
        yield from range(self._values[0], self._values[1] + 1)

    def alli(self, n: int | None = 10) -> Generator:
        yield from range(self._values[0], self._values[1] + 1)

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

    def to_java(self):
        if self._values is None:
            return None

        return {
            "name": self.name,
            "min": int(self._values[0]),
            "max": int(self._values[0]),
            PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.IntRangeIssue",
        }
