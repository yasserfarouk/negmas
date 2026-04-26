"""Singleton issue implementation for issues with a single value."""

from __future__ import annotations

from typing import Any, Generator

from negmas.helpers import unique_name
from negmas.outcomes.base_issue import DiscreteIssue, Issue

__all__ = ["SingletonIssue"]


class SingletonIssue(DiscreteIssue):
    """
    An `Issue` type representing a single fixed value.

    This is useful for representing an outcome as an outcome space
    with a single outcome.
    """

    def __init__(self, value: Any, name: str | None = None) -> None:
        """
        Initializes the singleton issue with a single value.

        Args:
            value: The single value this issue can take
            name: The name of the issue (auto-generated if None)
        """
        super().__init__([value], name)
        self._n_values = 1
        self._value_type = type(value) if value is not None else object
        self._single_value = value
        if self.is_numeric():
            self.min_value = self.max_value = value

    @property
    def value(self) -> Any:
        """Returns the single value of this issue."""
        return self._single_value

    @property
    def type(self) -> str:
        """Type of issue.

        Returns:
            str: Always returns 'singleton' for this issue type
        """
        return "singleton"

    def _to_xml_str(self, indx):
        output = f'    <issue etype="discrete" index="{indx + 1}" name="{self.name}" type="discrete" vtype="discrete">\n'
        output += f'        <item index="1" value="{self._single_value}" cost="0" description="{self._single_value}">\n        </item>\n'
        output += "    </issue>\n"
        return output

    def is_continuous(self) -> bool:
        """Check if this issue has continuous values.

        Returns:
            bool: Always False for singleton issues
        """
        return False

    def is_uncountable(self) -> bool:
        """Check if the issue has uncountably infinite values.

        Returns:
            bool: Always False for singleton issues
        """
        return False

    @property
    def all(self) -> Generator[Any, None, None]:
        """Generate all possible values for this singleton issue.

        Returns:
            Generator[Any, None, None]: Generator yielding the single value
        """
        yield self._single_value

    def rand(self) -> Any:
        """Picks a random valid value (always the single value)."""
        return self._single_value

    def rand_invalid(self) -> Any:
        """Pick a random *invalid* value"""
        if self.is_float():
            return self._single_value + 1.0

        if self.is_integer():
            return self._single_value + 1

        if isinstance(self._single_value, str):
            return unique_name("") + str(self._single_value) + unique_name("")

        if isinstance(self._single_value, tuple):
            return (unique_name(""),) + self._single_value

        return unique_name("invalid_")

    def is_valid(self, v) -> bool:
        """Checks whether the given value equals the singleton value."""
        return v == self._single_value

    def contains(self, issue: Issue) -> bool:
        """
        Checks whether this issue contains the input issue.

        A singleton issue only contains another issue if that issue
        has exactly the same single value.
        """
        if isinstance(issue, SingletonIssue):
            return self._single_value == issue._single_value
        if isinstance(issue, DiscreteIssue):
            return issue.cardinality == 1 and self.is_valid(next(iter(issue.all)))
        return False

    def __eq__(self, other):
        """Checks equality based on value and name."""
        if isinstance(other, SingletonIssue):
            return self._single_value == other._single_value and self.name == other.name
        return False

    def __hash__(self):
        """Returns a hash based on the string representation."""
        return hash(str(self))

    def __repr__(self):
        """Returns a detailed string representation for debugging."""
        return f"SingletonIssue({self._single_value!r}, {self.name!r})"

    def __str__(self):
        """Returns a human-readable string showing the issue name and value."""
        return f"{self.name}: {self._single_value}"
