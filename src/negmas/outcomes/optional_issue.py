"""Optional issue wrapper that allows None as a valid value."""

from __future__ import annotations

import numbers
from random import random
from typing import Any, Generator

from negmas.helpers.types import get_full_type_name
from negmas.outcomes.base_issue import Issue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

__all__ = ["OptionalIssue"]


class OptionalIssue(Issue):
    """
    Base class of an issues that is explicitly marked to be optional.
    Protocols can use that as they see fit. The main effect of defining an issue as optional
    is that the value `None` is allowed and returned first when enumerated (it is also counted in
    the cardinality)
    """

    def __init__(self, base: Issue, name: str | None = None) -> None:
        """Initializes the instance."""
        self.base = base
        self._n_values = self.base._n_values + 1
        super().__init__(values=base.values, name=name)

    @property
    def value_type(self):
        """
        Returns the type of values in this issue
        """
        return self.base._value_type

    @property
    def values(self):
        """
        Returns the raw values representation of the issue. Only use if you know what you are doing. To get all the values that can be assigned to this issue use `all` or `generate_values`
        """
        return self.base._values

    def has_limits(self) -> bool:
        """
        Checks whether the minimum and maximum values of the issue are known
        """
        return self.min_value is not None and self.max_value is not None

    def is_numeric(self) -> bool:
        """Check if the issue values are numeric types."""
        return issubclass(self.base._value_type, numbers.Number)

    def is_integer(self) -> bool:
        """Check if the issue values are integer types."""
        return issubclass(self.base.value_type, numbers.Integral)

    def is_float(self) -> bool:
        """Check if the issue values are floating-point types."""
        return issubclass(self._value_type, numbers.Real) and not issubclass(
            self._value_type, numbers.Integral
        )

    def is_continuous(self) -> bool:
        """Check if the issue has continuous (real-valued) domain."""
        return self.base.is_continuous()

    def is_discrete(self) -> bool:
        """Check if the issue has a discrete (countable) domain."""
        return not self.is_continuous()

    @property
    def cardinality(self) -> int | float:
        """Adds one to the base cardinality to handle None"""
        return self.base.cardinality + 1

    def rand(self) -> int | float | str | None:  # type: ignore
        """Picks a random valid value."""
        if self.is_continuous():
            return self.base.rand()
        p = 1.0 - self.base.cardinality / self.cardinality
        if random() < p:
            return None
        return self.base.rand()

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """
        Constructs an issue from a dict generated using `to_dict()`
        """
        if isinstance(d, cls):
            return d
        d.pop(python_class_identifier, None)
        d["base"] = deserialize(
            d["base"], python_class_identifier=python_class_identifier
        )
        return cls(base=d.get("base", None), name=d.get("name", None))

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """
        Converts the issue to a dictionary from which it can be constructed again using `Issue.from_dict()`
        """
        d = {python_class_identifier: get_full_type_name(type(self))}
        return dict(
            **d,
            base=serialize(self.base, python_class_identifier=python_class_identifier),
            name=self.name,
            n_values=self.cardinality + 1,
        )

    def is_valid(self, v) -> bool:
        """Checks whether the given value is valid for this issue"""
        return v is None or self.base.is_valid(v)

    def contains(self, issue: Issue) -> bool:
        """
        Checks weather this issue contains the input issue (i.e. every value in the input issue is in this issue)
        """
        return self.base.contains(issue)

    @property
    def type(self) -> str:
        """
        Returns a nice name for the issue type
        """
        return "optional_" + self.base.__class__.__name__.lower().replace("issue", "")

    def ordered_value_generator(  # type: ignore
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[int | None, None, None]:
        """Generate values in ascending order, starting with None.

        Args:
            n: Maximum number of values to generate, or None for all.
            grid: If True, use evenly spaced values; otherwise sample randomly.
            compact: If True, concentrate values around the center of the range.
            endpoints: If True, include min and max values in the output.

        Returns:
            Generator yielding None first, then values from the base issue.
        """
        yield None
        yield from self.base.ordered_value_generator(n, grid, compact, endpoints)

    def value_generator(
        self, n: int | float | None = 10, grid=True, compact=True, endpoints=True
    ) -> Generator[Any, None, None]:
        """Generate a sample of values from this issue, starting with None.

        Args:
            n: Maximum number of values to generate, or None for all.
            grid: If True, use evenly spaced values; otherwise sample randomly.
            compact: If True, concentrate values around the center of the range.
            endpoints: If True, include min and max values in the output.

        Returns:
            Generator yielding None first, then sampled values from the base issue.
        """
        yield None
        yield from self.base.value_generator(n, grid, compact, endpoints)

    def to_discrete(  # type: ignore
        self, n: int | float | None = 10, grid=True, compact=True, endpoints=True
    ) -> OptionalIssue:
        """Convert to a discrete optional issue with at most n values.

        Args:
            n: Maximum number of discrete values in the resulting issue.
            grid: If True, use evenly spaced values; otherwise sample randomly.
            compact: If True, concentrate values around the center of the range.
            endpoints: If True, include min and max values in the discretization.

        Returns:
            A new OptionalIssue wrapping the discretized base issue.
        """
        return OptionalIssue(self.base.to_discrete(n, grid, compact, endpoints))

    def _to_xml_str(self, indx: int) -> str:
        # TODO: For now, we do not mark the issue as optional when saving it
        return self.base._to_xml_str(indx)

    def value_at(self, index: int):
        """
        None is assumed to be first
        """
        if index == 0:
            return None
        return self.base.value_at(index - 1)

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list:
        """Generate n random values from this issue.

        Args:
            n: Number of random values to generate.
            with_replacement: If True, allow duplicate values in the result.
            fail_if_not_enough: If True, raise error when n exceeds cardinality.

        Returns:
            A list of n randomly selected values from the base issue.
        """
        return self.base.rand_outcomes(n, with_replacement, fail_if_not_enough)

    def rand_invalid(self):
        """Pick a random *invalid* value"""
        return self.base.rand_invalid()

    @property
    def all(self) -> Generator[Any, None, None]:
        """
        A generator that generates all possible values.
        """
        yield None
        yield from self.base.all

    def __eq__(self, other):
        """Check equality based on the wrapped base issue."""
        if not isinstance(other, OptionalIssue):
            return False
        return self.base == other.base

    def __repr__(self):
        """Return detailed string representation for debugging."""
        return f"{self.__class__.__name__}({self.base}, {self.name})"

    def __str__(self):
        """Return human-readable string showing name and base issue."""
        return f"{self.name}: {self.base}"
