from __future__ import annotations

import numbers
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Sequence, Union

import numpy as np

from negmas.java import PYTHON_CLASS_IDENTIFIER
from negmas.types import NamedObject

__all__ = ["Issue", "DiscreteIssue", "RangeIssue"]


class Issue(NamedObject, ABC):
    """Encodes an Issue.

    Args:
            values: Possible values for the issue
            name: Name of the issue. If not given, a random name will be generated
            id: The ID of the issue. Should be unique in the whole system.

    Remarks:

        - Issues can be initialized by either an iterable of strings, an integer or a tuple of two values with
          the following meanings:

          - ``list of anything`` : This is an issue that can any value within the given set of values
            (strings, ints, floats, etc)
          - ``int`` : This is an issue that takes any value from 0 to the given value -1 (int)
          - Tuple[ ``float`` , ``float`` ] : This is an issue that can take any real value between the given limits (min, max)
          - Tuple[ ``int`` , ``int`` ] : This is an issue that can take any integer value between the given limits (min, max)
          - ``Callable`` : The callable should take no parameters and should act as a generator of issue values. This
             type of issue is always assumed to be neither countable nor continuous and are called uncountable. For
             example, you can use this type to make an issue that generates all integers from 0 to infinity.
        - If a list is given, min, max must be callable on it.
    """

    def __new__(cls, values=None, *args, **kwargs):
        from negmas.outcomes.callable_issue import CallableIssue
        from negmas.outcomes.categorical_issue import CategoricalIssue
        from negmas.outcomes.contiguous_issue import ContiguousIssue
        from negmas.outcomes.continuous_issue import ContinuousIssue
        from negmas.outcomes.ordinal_issue import OrdinalIssue

        if isinstance(values, numbers.Integral):
            cls = ContiguousIssue
        elif isinstance(values, tuple):
            if len(values) != 2:
                raise ValueError(
                    f"Passing {values} is illegal. Issues with ranges need 2-values tuples"
                )
            if isinstance(values[0], numbers.Integral) and isinstance(
                values[0], numbers.Integral
            ):
                cls = ContiguousIssue
            elif isinstance(values[0], numbers.Real) and isinstance(
                values[0], numbers.Real
            ):
                cls = ContinuousIssue
            else:
                raise ValueError(
                    f"Passing {values} with mixed types. Both values must be either integers or reals"
                )
        elif isinstance(values, Callable):
            cls = CallableIssue
        elif isinstance(values, Sequence) and all(
            isinstance(_, numbers.Integral) for _ in values
        ):
            cls = OrdinalIssue
        else:
            cls = CategoricalIssue

        obj = super().__new__(cls)
        return obj

    def __init__(
        self,
        values,
        name: Optional[str] = None,
        id=None,
    ) -> None:
        super().__init__(name, id=id)
        self._value_type = object
        self._values = values
        self._n_values = float("inf")
        self.min_value, self.max_value = None, None

    @property
    def value_type(self):
        return self._value_type

    @property
    def values(self):
        return self._values

    @abstractmethod
    def _to_xml_str(self, indx: int, enumerate_integer=True):
        ...

    @property
    @abstractmethod
    def type(self) -> str:
        ...

    @abstractmethod
    def is_continuous(self) -> bool:
        ...

    def is_numeric(self) -> bool:
        return issubclass(self._value_type, numbers.Number)

    def is_integer(self) -> bool:
        return issubclass(self._value_type, numbers.Integral)

    def is_float(self) -> bool:
        return issubclass(self._value_type, numbers.Real) and not issubclass(
            self._value_type, numbers.Integral
        )

    def is_uncountable(self) -> bool:
        return False

    def is_countable(self) -> bool:
        return not self.is_continuous() and not self.is_uncountable()

    def is_discrete(self) -> bool:
        return self.is_countable()

    @abstractmethod
    def alli(self, n: int | None = 10) -> Generator:
        """
        A generator that generates all possible values or samples n values for real Issues.

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:

            >>> from negmas.outcomes import Issue
            >>> list(Issue(5).alli())
            [0, 1, 2, 3, 4]
            >>> list(int(10 * _) for _ in Issue((0.0, 1.0)).alli(11))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        """

    @property
    def cardinality(self) -> Union[int, float]:
        """The number of possible outcomes for the issue. Returns infinity for continuous and uncountable spaces"""
        return self._n_values

    @abstractmethod
    def rand(self) -> Union[int, float, str]:
        """Picks a random valid value."""

    @abstractmethod
    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list:
        """
        Picks n random valid value (at most).

        Args:
            n: The number of outcome values to sample
            with_replacement: If true, sampling is done with replacement (i.e.repetition is allowed)
            fail_if_not_enough: If true, raises an exception if it is not possible to sample exactly
                                `n` values. If false, will sample as many values as possible up to `n`
        Returns:
            A list of sampled values
        """

    def rand_valid(self):
        return self.rand()

    @abstractmethod
    def rand_invalid(self):
        """Pick a random *invalid* value"""

    @classmethod
    def from_java(cls, d: Dict[str, Any], class_name: str) -> "Issue":
        if class_name.endswith("ListIssue"):
            return Issue(name=d.get("name", None), values=d["values"])

        if class_name.endswith("RangeIssue"):
            return Issue(name=d.get("name", None), values=(d["min"], d["max"]))
        raise ValueError(
            f"Unknown issue type: {class_name} with dict {d} received from Java"
        )

    @classmethod
    def from_dict(cls, d):
        return cls(
            values=d.get("values", None),
            name=d.get("name", None),
            id=d.get("id", None),
        )

    def to_dict(self):
        return dict(
            values=self._values,
            n_values=self._n_values,
            name=self.name,
            id=self.id,
        )

    @abstractmethod
    def to_java(self):
        ...

    def __str__(self):
        return f"{self.name}: {self._values}"

    __repr__ = __str__

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self._values == other.values and self.name == other.name

    def __copy__(self):
        return Issue(name=self.name, values=self._values)

    def __deepcopy__(self, memodict={}):
        if isinstance(self._values, list):
            return Issue(name=self.name, values=[_ for _ in self._values])

        return Issue(name=self.name, values=self._values)

    @abstractmethod
    def is_valid(self, v):
        """Checks whether the given value is valid for this issue"""
        ...


class DiscreteIssue(Issue):
    @property
    def cardinality(self) -> int:
        """The number of possible outcomes for the issue. Guaranteed to  be fininte"""
        return self._n_values

    @property
    def type(self) -> str:
        return "discrete"

    def is_continuous(self) -> bool:
        return False

    @property
    @abstractmethod
    def all(self) -> Generator:
        """A generator that generates all possible values.

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:

            >>> from negmas.outcomes import Issue
            >>> list(Issue(5).all)
            [0, 1, 2, 3, 4]

        """

    def alli(self, n: int | None = 10) -> Generator:
        yield from self._values

    def rand(self):
        """Picks a random valid value."""
        return random.choice(self._values)  # type: ignore

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> Iterable["Outcome"]:
        """Picks a random valid value."""

        if n > len(self._values) and not with_replacement:
            if fail_if_not_enough:
                raise ValueError(
                    f"Cannot sample {n} outcomes out of {self._values} without replacement"
                )
            else:
                return self._values

        return np.random.choice(
            np.asarray(self._values, dtype=self._value_type),
            size=n,
            replace=with_replacement,
        ).tolist()

    def to_java(self):
        if self._values is None:
            return None

        if self.is_integer():
            return {
                "name": self.name,
                "values": [int(_) for _ in self._values],
                PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.IntListIssue",
            }

        if self.is_float():
            return {
                "name": self.name,
                "values": [float(_) for _ in self._values],
                PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.DoubleListIssue",
            }

        return {
            "name": self.name,
            "values": [str(_) for _ in self._values],
            PYTHON_CLASS_IDENTIFIER: "negmas.outcomes.StringListIssue",
        }

    def is_valid(self, v):
        return v in self._values


class RangeIssue(Issue):
    def __init__(self, values, name=None, id=None) -> None:
        super().__init__(values, name, id)
        self._value_type = type(values[0])
        self.min_value, self.max_value = values[0], values[1]

    @property
    def cardinality(self) -> Union[int, float]:
        if not issubclass(self._value_type, numbers.Integral):
            return float("inf")
        return self.max_value - self.min_value + 1

    def is_valid(self, v):
        return self.min_value <= int(v) <= self.max_value
