"""Categorical issue implementation for discrete unordered values."""

from __future__ import annotations

import random
from typing import Any, Generator

from negmas.helpers import unique_name
from negmas.outcomes.base_issue import DiscreteIssue

__all__ = ["CategoricalIssue"]


class CategoricalIssue(DiscreteIssue):
    """
    An `Issue` type representing discrete values that have no ordering or difference defined.
    """

    def __init__(self, values, name=None) -> None:
        """Initialize the instance.

        Args:
            values: Values.
            name: Name.
        """
        super().__init__(values, name)
        values = list(values)
        self._n_values = len(values)
        self._value_type = (
            type(values[0])
            if len({type(_) for _ in values if _ is not None}) == 1
            else object
        )
        if self.is_numeric():
            self.min_value, self.max_value = min(values), max(values)

    @property
    def type(self) -> str:
        """Type of issue (continuous, discrete, categorical, etc.).

        Returns:
            str: Always returns 'categorical' for this issue type
        """
        return "categorical"

    def _to_xml_str(self, indx):
        output = f'    <issue etype="discrete" index="{indx + 1}" name="{self.name}" type="discrete" vtype="discrete">\n'

        for i, v in enumerate(self._values):
            output += f'        <item index="{i + 1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
        output += "    </issue>\n"
        return output

    def is_continuous(self) -> bool:
        """Check if this issue has continuous values.

        Returns:
            bool: Always False for categorical issues (discrete values only)
        """
        return False

    def is_uncountable(self) -> bool:
        """Check if the issue has uncountably infinite values.

        Returns:
            bool: Always False for categorical issues (finite countable values)
        """
        return False

    @property
    def all(self) -> Generator[Any, None, None]:
        """Generate all possible values for this categorical issue.

        Returns:
            Generator[Any, None, None]: Generator yielding each valid categorical value
        """
        yield from self._values

    def rand_invalid(self):  # type: ignore
        """Pick a random *invalid* value"""

        if self.is_float():
            return random.random() * self.max_value + self.max_value * 1.1

        if self.is_integer():
            return random.randint(self.max_value + 1, self.max_value * 2)

        if issubclass(self._value_type, tuple):
            return (random.randint(100, 200), unique_name(""))

        return unique_name("") + str(random.choice(self._values)) + unique_name("")
