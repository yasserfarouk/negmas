from __future__ import annotations

import random
from abc import abstractmethod
from typing import Any, Generator

from negmas import warnings
from negmas.helpers import unique_name
from negmas.helpers.numeric import is_float_type, is_int_type, sample
from negmas.outcomes.base_issue import DiscreteIssue, Issue

__all__ = ["OrdinalIssue", "DiscreteOrdinalIssue"]


def generate_values(n: int) -> list[str]:
    if n > 1000_000:
        warnings.warn(
            f"You are creating an OrdinalIssue with {n} items. This is too large. Consider using something like ContiguousIssue if possible",
            warnings.NegmasMemoryWarning,
        )
    width = len(str(n))
    return list(f"{_:0{width}d}" for _ in range(n))


class OrdinalIssue(Issue):
    """
    An `Issue` that have some defined ordering of outcomes but not necessarily a meaningful difference function between its values.
    """

    @abstractmethod
    def ordered_value_generator(
        self, n: int = 10, grid=True, compact=False, endpoints=True
    ) -> Generator[Any, None, None]:
        ...


class DiscreteOrdinalIssue(DiscreteIssue, OrdinalIssue):
    """
    A `DiscreteIssue` that have some defined ordering of outcomes but not necessarily a meaningful difference function between its values.
    """

    def __init__(self, values, name=None) -> None:
        """
        `values` can be an integer and in this case, values will be strings
        """
        super().__init__(values, name)
        if isinstance(values, int):
            values = generate_values(values)
        else:
            values = list(values)
        types = {type(_) for _ in values}
        if len(types) == 1:
            type_ = list(types)[0]
        elif all(is_int_type(_) for _ in types):
            type_ = int
        elif all(is_float_type(_) for _ in types):
            type_ = float
        else:
            raise TypeError(
                f"Found the following types in the list of values for an "
                f"ordinal issue ({types}). Can only have one type. Try "
                f"CategoricalIssue"
            )
        self._value_type = type_
        self._n_values = len(values)
        self.min_value, self.max_value = min(values), max(values)

    def _to_xml_str(self, indx):
        output = f'    <issue etype="discrete" index="{indx + 1}" name="{self.name}" type="discrete" vtype="discrete">\n'

        for i, v in enumerate(self._values):
            output += f'        <item index="{i + 1}" value="{v}" cost="0" description="{v}">\n        </item>\n'
        output += "    </issue>\n"
        return output

    @property
    def all(self) -> Generator[Any, None, None]:
        yield from self._values  # type: ignore

    def rand_invalid(self):
        """Pick a random *invalid* value"""

        if self.is_float():
            return random.random() * self.max_value + self.max_value * 1.1

        if self.is_integer():
            return random.randint(self.max_value + 1, self.max_value * 2)

        return unique_name("") + str(random.choice(self._values)) + unique_name("")

    def ordered_value_generator(
        self, n: int = 10, grid=True, compact=False, endpoints=True
    ) -> Generator[Any, None, None]:
        """
        A generator that generates at most `n` values (in order)

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:

            >>> from negmas.outcomes import make_issue
            >>> list(make_issue(5).ordered_value_generator())
            [0, 1, 2, 3, 4]
            >>> list(int(10 * _) for _ in make_issue((0.0, 1.0)).ordered_value_generator(11))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        """
        yield from (
            self._values[_]
            for _ in sample(
                self.cardinality, n, grid=grid, compact=compact, endpoints=endpoints
            )
        )
