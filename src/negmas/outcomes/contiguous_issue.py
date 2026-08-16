"""Contiguous integer issue implementation for discrete integer ranges."""

from __future__ import annotations

import math
import numbers
import random
from typing import Generator, Iterable

import numpy as np

from negmas.helpers.numeric import sample
from negmas.outcomes.base_issue import DiscreteIssue, Issue
from negmas.outcomes.range_issue import RangeIssue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize

__all__ = ["ContiguousIssue"]


class ContiguousIssue(RangeIssue, DiscreteIssue):
    """
    A `RangeIssue` (also a `DiscreteIssue`) representing a contiguous range of integers.
    """

    def __init__(
        self,
        values: int | tuple[int, int] | tuple[int, int, int],
        name: str | None = None,
        *,
        step: int = 1,
    ) -> None:
        """Initializes the instance.

        Args:
            values: Either the number of values (starting at zero), a `(min, max)`
                    tuple or a `(min, max, step)` tuple.
            name: Name of the issue
            step: The stride between consecutive values (defaults to 1). Cannot be
                  passed if `values` is a three-valued tuple that already defines it.
        """
        vt: tuple[int, int]
        vt = values  # type: ignore
        if isinstance(vt, numbers.Integral):
            vt = (0, int(vt) - 1)
        if isinstance(vt, Iterable):
            vt = tuple(vt)
        if len(vt) == 3:
            if step != 1:
                raise ValueError(
                    f"{self.__class__.__name__} received a step both in `values` "
                    f"({values=}) and as an explicit parameter ({step=})"
                )
            vt, step = (vt[0], vt[1]), vt[2]  # type: ignore
        if len(vt) != 2:
            raise ValueError(
                f"{self.__class__.__name__} should receive one, two or three values for "
                f"the minimum, maximum limits and step but received {values=}"
            )
        if not isinstance(vt[0], numbers.Integral) or not isinstance(
            vt[1], numbers.Integral
        ):
            raise ValueError(
                f"{self.__class__.__name__} should receive one or two integers for"
                f" the minimum and maximum limits but received {values=}"
            )
        if not isinstance(step, numbers.Integral):
            raise ValueError(
                f"{self.__class__.__name__} should receive an integer step but received {step=}"
            )
        step = int(step)
        if step < 1:
            raise ValueError(
                f"{self.__class__.__name__} should receive a positive step but received {step=}"
            )
        vt = tuple(vt)
        self._step = step
        if step != 1:
            # clamp the maximum to the largest attainable value so that all the
            # bookkeeping (cardinality, `all`, `value_at`, ...) stays consistent.
            vt = (vt[0], vt[0] + step * ((vt[1] - vt[0]) // step))  # type: ignore
        super().__init__(vt, name)
        self._n_values = (vt[1] - vt[0]) // step + 1  # type: ignore

    @property
    def step(self) -> int:
        """The stride between consecutive values of this issue."""
        return self._step

    def _to_xml_str(self, indx):
        if self._step != 1:
            # Genius integer issues cannot represent a stride so we enumerate the
            # values as a discrete issue. Note that (as for any discrete issue) the
            # values are read back as strings when the XML is parsed again.
            output = f'    <issue etype="discrete" index="{indx + 1}" name="{self.name}" type="discrete" vtype="discrete">\n'
            for i, v in enumerate(self.all):
                output += f'        <item index="{i + 1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
            output += "    </issue>\n"
            return output
        return (
            f'    <issue etype="integer" index="{indx + 1}" name="{self.name}" type="integer" vtype="integer"'
            f' lowerbound="{self._values[0]}" upperbound="{self._values[1]}" />\n'
        )

    @property
    def values(self):
        """The raw values representation: `(min, max)` or `(min, max, step)` if a step is used."""
        if self._step == 1:
            return self._values
        return (self._values[0], self._values[1], self._step)

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Constructs a `ContiguousIssue` from a dict generated using `to_dict()`."""
        if isinstance(d, cls):
            return d
        d = dict(d)
        d.pop(python_class_identifier, None)
        values = deserialize(
            d.get("values", None), python_class_identifier=python_class_identifier
        )
        return cls(
            values=values,  # type: ignore
            name=d.get("name", None),
            **({"step": d["step"]} if "step" in d else {}),
        )

    def __copy__(self):
        """Returns a shallow copy of the issue (preserving the step)."""
        return ContiguousIssue(self._values, name=self.name, step=self._step)

    def __deepcopy__(self, memodict={}):
        """Returns a deep copy of the issue (preserving the step)."""
        _ = memodict
        return ContiguousIssue(self._values, name=self.name, step=self._step)

    def __repr__(self):
        """Returns a detailed string representation for debugging."""
        if self._step == 1:
            return super().__repr__()
        return (
            f"{self.__class__.__name__}({self._values}, {self.name}, step={self._step})"
        )

    def __str__(self):
        """Returns a human-readable string showing the issue name and values."""
        if self._step == 1:
            return super().__str__()
        return f"{self.name}: {self._values} step {self._step}"

    def is_valid(self, v):
        """Checks that the value is within the limits *and* on the grid defined by `step`."""
        if not self._values[0] <= v <= self._values[1]:
            return False
        return (v - self._values[0]) % self._step == 0

    @property
    def all(self) -> Generator[int, None, None]:
        """Generator yielding all integer values in the range [min, max] stepping by `step`."""
        yield from range(self._values[0], self._values[1] + 1, self._step)

    @property
    def cardinality(self) -> int:
        """Number of distinct integer values in this issue's range."""
        return (self._values[1] - self._values[0]) // self._step + 1

    def ordered_value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[int, None, None]:
        """Generate integer values in ascending order from this range.

        Args:
            n: Maximum number of values to generate, or None for all.
            grid: If True, use evenly spaced values; otherwise sample randomly.
            compact: If True, concentrate values around the center of the range.
            endpoints: If True, include min and max values in the output.

        Returns:
            Generator yielding integers from the range in order.
        """
        m = self.cardinality
        n = m if n is None or not math.isfinite(n) else int(n)
        for i in range(n):
            yield self._values[0] + self._step * (i % m)

    def value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[int, None, None]:
        """Generate a sample of integer values from this range.

        Args:
            n: Maximum number of values to generate, or None for all.
            grid: If True, use evenly spaced values; otherwise sample randomly.
            compact: If True, concentrate values around the center of the range.
            endpoints: If True, include min and max values in the output.

        Returns:
            Generator yielding sampled integer values from the range.
        """
        yield from (
            self._values[0] + self._step * _
            for _ in sample(
                self.cardinality, n, grid=grid, compact=compact, endpoints=endpoints
            )
        )

    def to_discrete(
        self, n: int | None, grid=True, compact=False, endpoints=True
    ) -> DiscreteIssue:
        """Convert to a discrete issue with at most n values.

        Args:
            n: Maximum number of discrete values, or None to keep all.
            grid: If True, use evenly spaced values; otherwise sample randomly.
            compact: If True, select a contiguous subrange from the center.
            endpoints: If True, include min and max values in the discretization.

        Returns:
            A ContiguousIssue (if compact) or general DiscreteIssue with sampled values.
        """
        if n is None or self.cardinality < n:
            return self
        if not compact:
            return super().to_discrete(
                n, grid=grid, compact=compact, endpoints=endpoints
            )

        beg = (self.cardinality - n) // 2
        if self._step == 1:
            return ContiguousIssue((int(beg), int(beg + n)), name=self.name + f"{n}")
        return ContiguousIssue(
            (
                int(self.min_value + self._step * beg),
                int(self.min_value + self._step * (beg + n - 1)),
            ),
            name=self.name + f"{n}",
            step=self._step,
        )

    def rand(self) -> int:
        """Picks a random valid value."""
        if self._step == 1:
            return random.randint(*self._values)
        return self._values[0] + self._step * random.randint(0, self.cardinality - 1)

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list[int]:
        """Picks a random valid value."""

        if n > self._n_values and not with_replacement:
            if fail_if_not_enough:
                raise ValueError(
                    f"Cannot sample {n} outcomes out of {self._values} without replacement"
                )
            return list(self.all)

        if with_replacement:
            if self._step == 1:
                return np.random.randint(
                    low=self._values[0], high=self._values[1] + 1, size=n
                ).tolist()
            return (
                self._values[0]
                + self._step * np.random.randint(low=0, high=self.cardinality, size=n)
            ).tolist()
        vals = list(self.all)
        random.shuffle(vals)
        return vals[:n]

    def rand_invalid(self):
        """Pick a random *invalid* value"""

        return random.randint(self.max_value + 1, 2 * self.max_value)

    def is_continuous(self) -> bool:
        """Check if this issue has continuous values (always False for integers)."""
        return False

    def value_at(self, index: int):
        """Return the integer value at the given index position.

        Args:
            index: Zero-based position in the range.

        Raises:
            IndexError: If index is out of bounds.
        """
        if index < 0 or index > self.cardinality - 1:
            raise IndexError(index)
        return self.min_value + self._step * index

    def contains(self, issue: Issue) -> bool:
        """Checks weather this issue contains the input issue (i.e. every value in the input issue is in this issue)"""
        if not (
            issubclass(issue.value_type, numbers.Integral)
            and issue.min_value >= self.min_value
            and issue.max_value <= self.max_value
        ):
            return False
        if self._step == 1:
            return True
        if isinstance(issue, ContiguousIssue):
            return (
                issue.step % self._step == 0
                and (issue.min_value - self.min_value) % self._step == 0
            )
        return super().contains(issue)
