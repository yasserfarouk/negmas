from __future__ import annotations

import random
from typing import Generator

import numpy as np

from negmas.outcomes.range_issue import RangeIssue

from .common import DEFAULT_LEVELS

__all__ = ["ContinuousIssue"]


class ContinuousIssue(RangeIssue):
    def __init__(self, values, name=None, id=None, n_levels=DEFAULT_LEVELS) -> None:
        super().__init__(values, name=name, id=id)
        self._n_levels = n_levels
        self.delta = (self.max_value - self.min_value) / self._n_levels

    def _to_xml_str(self, indx, enumerate_integer=True):
        return (
            f'    <issue etype="real" index="{indx + 1}" name="{self.name}" type="real" vtype="real">\n'
            f'        <range lowerbound="{self._values[0]}" upperbound="{self._values[1]}"></range>\n    </issue>\n'
        )

    def to_dict(self):
        d = super().to_dict()
        d["n_levels"] = self._n_levels
        return d

    @property
    def cardinality(self) -> int | float:
        return float("inf")

    @property
    def type(self) -> str:
        return "continuous"

    def is_continuous(self) -> bool:
        return True

    def is_uncountable(self) -> bool:
        return True

    def rand(self) -> float:
        """Picks a random valid value."""

        return (
            random.random() * (self._values[1] - self._values[0]) + self._values[0]
        )  # type: ignore

    def value_generator(
        self, n: int | None = DEFAULT_LEVELS, grid=True, compact=False, endpoints=True
    ) -> Generator:
        yield from self.ordered_value_generator(
            n, grid=grid, compact=compact, endpoints=endpoints
        )

    def ordered_value_generator(
        self, n: int = DEFAULT_LEVELS, grid=True, compact=False, endpoints=True
    ) -> Generator:
        if n is None:
            raise ValueError("Real valued issue with no discretization value")
        if grid:
            yield from np.linspace(
                self._values[0], self._values[1], num=n, endpoint=endpoints
            ).tolist()
            return
        if endpoints:
            yield from [self._values[0]] + (
                (self._values[1] - self._values[0]) * np.random.rand(n - 2)
                + self._values[0]
            ).tolist() + [self._values[1]]
        yield from (
            (self._values[1] - self._values[0]) * np.random.rand(n) + self._values[0]
        ).tolist()

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list[float]:
        if with_replacement:
            return (
                np.random.rand(n) * (self._values[1] - self._values[0])
                + self._values[0]
            ).tolist()
        return np.linspace(
            self._values[0], self._values[1], num=n, endpoint=True
        ).tolist()

    def rand_invalid(self):
        """Pick a random *invalid* value"""

        return (
            random.random() * (self.max_value - self.min_value) + self.max_value * 1.1
        )

    @property
    def all(self):
        raise ValueError(f"Cannot enumerate all values of a continuous issue")

    def value_at(self, index: int):
        v = self.min_value + self.delta * index
        if v > self.max_value:
            return IndexError(index)
        return v