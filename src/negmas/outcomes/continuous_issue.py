"""Continuous issue implementation for real-valued ranges with finite limits."""

from __future__ import annotations

import math
import numbers
import random
from typing import Generator

import numpy as np

from negmas.outcomes.base_issue import Issue
from negmas.outcomes.range_issue import RangeIssue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER

from .common import DEFAULT_LEVELS

__all__ = ["ContinuousIssue"]


class ContinuousIssue(RangeIssue):
    """
    A `RangeIssue` representing a continous range of real numbers with finite limits.
    """

    def __init__(self, values, name=None, n_levels=DEFAULT_LEVELS) -> None:
        """Initializes the instance."""
        super().__init__(values, name=name)
        self._n_levels = n_levels
        self.delta = (self.max_value - self.min_value) / self._n_levels

    def _to_xml_str(self, indx):
        return (
            f'    <issue etype="real" index="{indx + 1}" name="{self.name}" type="real" vtype="real">\n'
            f'        <range lowerbound="{self._values[0]}" upperbound="{self._values[1]}"></range>\n    </issue>\n'
        )

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Serialize this issue to a dictionary for reconstruction.

        Args:
            python_class_identifier: Key used to store the class type info.
        """
        d = super().to_dict(python_class_identifier=python_class_identifier)
        d["n_levels"] = self._n_levels
        return d

    @property
    def cardinality(self) -> int | float:
        """Number of possible values (infinite for continuous issues).

        Returns:
            int | float: Always returns infinity for continuous issues
        """
        return float("inf")

    @property
    def type(self) -> str:
        """Type of issue (continuous, discrete, categorical, etc.).

        Returns:
            str: Always returns 'continuous' for this issue type
        """
        return "continuous"

    def is_continuous(self) -> bool:
        """Check if this issue has continuous values.

        Returns:
            bool: Always True for continuous issues
        """
        return True

    def is_uncountable(self) -> bool:
        """Check if the issue has uncountably infinite values.

        Returns:
            bool: Always True for continuous issues (uncountably infinite)
        """
        return True

    def rand(self) -> float:
        """Picks a random valid value."""

        return random.random() * (self._values[1] - self._values[0]) + self._values[0]  # type: ignore

    def ordered_value_generator(
        self,
        n: int | float | None = DEFAULT_LEVELS,
        grid=True,
        compact=False,
        endpoints=True,
    ) -> Generator[float, None, None]:
        """Generate float values in ascending order from this range.

        Args:
            n: Number of values to generate (required, must be finite).
            grid: If True, use evenly spaced values; otherwise sample randomly.
            compact: If True, concentrate values around the center of the range.
            endpoints: If True, include min and max values in the output.

        Returns:
            Generator yielding float values in ascending order.

        Raises:
            ValueError: If n is None or infinite.
        """
        if n is None or not math.isfinite(n):
            raise ValueError(f"Cannot generate {n} values from issue: {self}")
        n = int(n)
        if grid:
            yield from np.linspace(
                self._values[0], self._values[1], num=n, endpoint=endpoints
            ).tolist()
            return
        if endpoints:
            yield from (
                [self._values[0]]
                + (
                    (self._values[1] - self._values[0]) * np.random.rand(n - 2)
                    + self._values[0]
                ).tolist()
                + [self._values[1]]
            )
        yield from (
            (self._values[1] - self._values[0]) * np.random.rand(n) + self._values[0]
        ).tolist()

    def value_generator(
        self,
        n: int | float | None = DEFAULT_LEVELS * 10,
        grid=True,
        compact=False,
        endpoints=True,
    ) -> Generator[float, None, None]:
        """Generate a sample of float values from this continuous range.

        Args:
            n: Number of values to generate (required, must be finite).
            grid: If True, use evenly spaced values; otherwise sample randomly.
            compact: If True, concentrate values around the center of the range.
            endpoints: If True, include min and max values in the output.

        Returns:
            Generator yielding sampled float values.
        """
        return self.ordered_value_generator(n, grid, compact, endpoints)

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list[float]:
        """Generate n random float values from this continuous range.

        Args:
            n: Number of random values to generate.
            with_replacement: If True, sample randomly; otherwise use linspace.
            fail_if_not_enough: Ignored for continuous issues (always enough values).

        Returns:
            A list of n float values from the range.
        """
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
        """Cannot enumerate all values of a continuous issue.

        Raises:
            ValueError: Always raised since continuous issues are uncountable.
        """
        raise ValueError("Cannot enumerate all values of a continuous issue")

    def value_at(self, index: int):
        """Return the value at the given discretization level.

        Args:
            index: Zero-based index into discretized levels.

        Raises:
            IndexError: If the computed value exceeds max_value.
        """
        v = self.min_value + self.delta * index
        if v > self.max_value:
            return IndexError(index)
        return v

    def contains(self, issue: Issue) -> bool:
        """Checks weather this issue contains the input issue (i.e. every value in the input issue is in this issue)"""
        return (
            issubclass(issue.value_type, numbers.Real)
            and issue.min_value >= self.min_value
            and issue.max_value <= self.max_value
        )
