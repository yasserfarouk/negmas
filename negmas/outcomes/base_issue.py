from __future__ import annotations

import math
import numbers
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable

import numpy as np

from negmas import warnings
from negmas.helpers.numeric import sample
from negmas.helpers.strings import unique_name
from negmas.helpers.types import get_full_type_name
from negmas.protocols import HasMinMax
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

if TYPE_CHECKING:
    from .common import Outcome

__all__ = ["make_issue", "Issue", "DiscreteIssue"]


def make_issue(values, *args, **kwargs):
    """
    A factory for creating issues based on `values` type as well as the base class of all issues

    Args:
            values: Possible values for the issue
            name: Name of the issue. If not given, a random name will be generated
            id: The ID of the issue. Should be unique in the whole system.

    Remarks:

        - Issues can be initialized by either an iterable of strings, an integer or a tuple of two values with
          the following meanings:

          - ``list of anything`` : This is an issue that can any value within the given set of values
            (strings, ints, floats, etc). Depending on the types in the list, a different issue type will be created:

                - integers -> CardinalIssue
                - a type that supports subtraction -> OrdinalIssue (with defined order and defined difference between values)
                - otherwise -> CategoricalIssue (without defined order or difference between values)

          - ``int`` : This is a ContiguousIssue that takes any value from 0 to the given value -1 (int)
          - Tuple[ ``int`` , ``int`` ] : This is a ContiguousIssue that can take any integer value in the given limits (min, max)
          - Tuple[ ``float`` , ``float`` ] : This is a ContinuousIssue that can take any real value in the given limits (min, max)
          - Tuple[ ``int`` , ``inf`` ] : This is a CountableInfiniteIssue that can take any integer value in the given limits
          - Tuple[ ``-inf`` , ``int`` ] : This is a CountableInfiniteIssue that can take any integer value in the given limits
          - Tuple[ ``float`` , ``inf`` ] : This is a ContinuousInfiniteIssue that can take any real value in the given limits
          - Tuple[ ``-inf`` , ``float`` ] : This is a ContinuousInfiniteIssue that can take any real value in the given limits
          - ``Callable`` : The callable should take no parameters and should act as a generator of issue values. This
             type of issue is always assumed to be neither countable nor continuous and are called uncountable. For
             example, you can use this type to make an issue that generates all integers from 0 to infinity. Most operations are not
             supported on this issue type.
        - If a list is given, min, max must be callable on it.
    """
    from negmas.outcomes.callable_issue import CallableIssue
    from negmas.outcomes.cardinal_issue import DiscreteCardinalIssue
    from negmas.outcomes.categorical_issue import CategoricalIssue
    from negmas.outcomes.contiguous_issue import ContiguousIssue
    from negmas.outcomes.continuous_issue import ContinuousIssue
    from negmas.outcomes.infinite import ContinuousInfiniteIssue, CountableInfiniteIssue
    from negmas.outcomes.ordinal_issue import DiscreteOrdinalIssue

    if isinstance(values, numbers.Integral):
        return ContiguousIssue(int(values), *args, **kwargs)
    if isinstance(values, tuple):
        if len(values) != 2:
            raise ValueError(
                f"Passing {values} is illegal. Issues with ranges need 2-values tuples"
            )
        if isinstance(values[0], numbers.Integral) and isinstance(
            values[1], numbers.Integral
        ):
            return ContiguousIssue(values, *args, **kwargs)  # type: ignore (we know that the types are OK here)
        if (
            isinstance(values[0], numbers.Integral)
            and values[1] == float("inf")
            or isinstance(values[1], numbers.Integral)
            and values[0] == float("-inf")
        ):
            return CountableInfiniteIssue(values, *args, **kwargs)  # type: ignore (we know that the types are OK here)
        if (
            isinstance(values[0], numbers.Real)
            and values[1] == float("inf")
            or isinstance(values[1], numbers.Real)
            and values[0] == float("-inf")
        ):
            return ContinuousInfiniteIssue(values, *args, **kwargs)
        if isinstance(values[0], numbers.Real) and isinstance(values[1], numbers.Real):
            return ContinuousIssue(values, *args, **kwargs)
        raise ValueError(
            f"Passing {values} with mixed types. Both values must be either integers or reals"
        )
    if isinstance(values, Callable):
        return CallableIssue(values, *args, **kwargs)  # type: ignore
    if isinstance(values, Iterable) and all(
        isinstance(_, numbers.Integral) for _ in values
    ):
        return DiscreteCardinalIssue(values, *args, **kwargs)
    if isinstance(values, Iterable):
        values = list(values)
        try:
            for a, b in zip(values[1:], values[:-1]):
                a - b
        except:
            return CategoricalIssue(values, *args, **kwargs)
        return DiscreteOrdinalIssue(values, *args, **kwargs)
    return CategoricalIssue(values, *args, **kwargs)


class Issue(HasMinMax, Iterable, ABC):
    """
    Base class of all issues in NegMAS
    """

    def __init__(
        self,
        values,
        name: str | None = None,
    ) -> None:
        self.name = name if name else unique_name("issue", add_time=False, sep="")
        self._value_type = object
        self._values = values
        self._n_values = float("inf")
        self.min_value, self.max_value = None, None

    def __copy__(self):
        return make_issue(name=self.name, values=self._values)

    def __deepcopy__(self, memodict={}):
        return make_issue(name=self.name, values=self._values)

    @property
    def value_type(self):
        """
        Returns the type of values in this issue
        """
        return self._value_type

    @property
    def values(self):
        """
        Returns the raw values representation of the issue. Only use if you know what you are doing. To get all the values that can be assigned to this issue use `all` or `generate_values`
        """
        return self._values

    def has_limits(self) -> bool:
        """
        Checks whether the minimum and maximum values of the issue are known
        """
        return self.min_value is not None and self.max_value is not None

    def is_numeric(self) -> bool:
        """
        Checks that each value of this issue is a number
        """
        return issubclass(self._value_type, numbers.Number)

    def has_finite_limits(self) -> bool:
        """
        Checks whether the minimum and maximum values of the issue are known and are finite
        """
        return (
            self.has_limits()
            and self.is_numeric()
            and math.isfinite(self.min_value)
            and math.isfinite(self.max_value)
        )

    def is_integer(self) -> bool:
        """
        Checks that each value of this issue is an integer
        """
        return issubclass(self._value_type, numbers.Integral)

    def is_float(self) -> bool:
        """
        Checks that each value of this issue is a real number
        """
        return issubclass(self._value_type, numbers.Real) and not issubclass(
            self._value_type, numbers.Integral
        )

    @abstractmethod
    def is_continuous(self) -> bool:
        """
        The issue has a continuous set of values. Note that this is different from having values that are real (which is tested using `is_float` )
        """
        ...

    def is_discrete(self) -> bool:
        """
        Checks whether the issue has a discrete set of values. This is different from `is_integer` which checks that the values themselves are integers and `is_discrete_valued` which checks that they are discrete.
        """
        return not self.is_continuous()

    def is_finite(self) -> bool:
        """
        Checks whether the issue has a discrete set of values
        """
        return self.is_discrete()

    def is_discrete_valued(self) -> bool:
        """
        Checks that each value of this issue is not a real number
        """
        return not self.is_float()

    @property
    def cardinality(self) -> int | float:
        """The number of possible outcomes for the issue. Returns infinity for continuous and uncountable spaces"""
        return self._n_values

    @abstractmethod
    def rand(self) -> int | float | str:
        """Picks a random valid value."""

    def rand_valid(self):
        """
        Generates a random  valid value for this issue
        """
        return self.rand()

    @classmethod
    def from_dict(cls, d):
        """
        Constructs an issue from a dict generated using `to_dict()`
        """
        if isinstance(d, cls):
            return d
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        d["values"] = deserialize(d["values"])
        return cls(
            values=d.get("values", None),
            name=d.get("name", None),
        )

    def to_dict(self):
        """
        Converts the issue to a dictionary from which it can be constructed again using `Issue.from_dict()`
        """
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        return dict(
            **d,
            values=serialize(self.values),
            name=self.name,
            n_values=self._n_values,
        )

    @abstractmethod
    def is_valid(self, v):
        """Checks whether the given value is valid for this issue"""
        ...

    def contains(self, issue: Issue) -> bool:
        """
        Checks weather this issue contains the input issue (i.e. every value in the input issue is in this issue)
        """
        if isinstance(issue, DiscreteIssue):
            return all(self.is_valid(_) for _ in issue.all)
        return False

    @property
    def type(self) -> str:
        """
        Returns a nice name for the issue type
        """
        return self.__class__.__name__.lower().replace("issue", "")

    @abstractmethod
    def ordered_value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[int, None, None]:
        """
        A generator that generates at most `n` values (in a stable order)


        Args:
            n: The number of samples. If inf or None, all values will be generated but when the issue is infinite, it will just fail
            grid: Sample on a grid (equally distanced as much as possible)
            compact: If True, the samples will be choosen near each other (see endpoints though)
            endpoints: If given, the first and last index are guaranteed to be in the samples

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If the order is not defined for this issue, this generator will still generate values in the same order every time it is called.
            - If you need a list then use something like:

            >>> from negmas.outcomes import make_issue
            >>> list(make_issue(5).value_generator())
            [0, 1, 2, 3, 4]
            >>> list(int(10 * _) for _ in make_issue((0.0, 1.0)).value_generator(11))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        """

    @abstractmethod
    def value_generator(
        self, n: int | float | None = 10, grid=True, compact=True, endpoints=True
    ) -> Generator[Any, None, None]:
        """
        A generator that generates at most `n` values (in any order)


        Args:
            grid: Sample on a grid (equally distanced as much as possible)
            compact: If True, the samples will be choosen near each other (see endpoints though)
            endpoints: If given, the first and last index are guaranteed to be in the samples

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:

            >>> from negmas.outcomes import make_issue
            >>> list(make_issue(5).value_generator())
            [0, 1, 2, 3, 4]
            >>> list(int(10 * _) for _ in make_issue((0.0, 1.0)).value_generator(11))
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        """

    def to_discrete(
        self, n: int | float | None = 10, grid=True, compact=True, endpoints=True
    ) -> DiscreteIssue:
        """
        Converts the issue to a discrete issue by samling from it.

        If the issue is already discret it will just return itself. This method cannot be used to reduce the cardinality of a discrete issue.

        Args:
            n (int | float | None): Number of values in the resulting discrete issue. This will be ignored if the issue is already discrete. The only allowed float value is `float("inf")`. If any other float is passed, it will be silently cast to an int
            grid (bool): Sample on a grid
            compact (bool): Sample around the center
            endpoints (bool): Always incllude minimum and maximum  values

        """
        from negmas.outcomes.categorical_issue import CategoricalIssue

        if isinstance(self, DiscreteIssue):
            return self

        return CategoricalIssue(
            list(self.value_generator(n, grid, compact, endpoints)),
            name=self.name,
        )

    @abstractmethod
    def _to_xml_str(self, indx: int) -> str:
        ...

    @abstractmethod
    def value_at(self, index: int):
        """
        Returns the value at the given indes of the issue. The same index  will have the same values always indepdendent of whether the values of the issue have defined ordering.
        """
        ...

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

    @abstractmethod
    def rand_invalid(self):
        """Pick a random *invalid* value"""

    @property
    def all(self) -> Generator[Any, None, None]:
        """
        A generator that generates all possible values.
        """
        if self.is_discrete():
            return self.value_generator()
        raise ValueError(
            f"The issue ({self}) is not discrete and `all` cannot be called on it"
        )

    def __getitem__(self, indx):
        return self.value_at(indx)

    def __iter__(self):
        return self.value_generator().__iter__()

    def __contains__(self, item):
        try:
            if isinstance(item, Issue):
                return self.contains(item)
            return self.is_valid(item)
        except Exception as e:
            warnings.warn(
                f"Testing whether {item} is contained  in {self} threw an exception: {e}. continuing as if it is not",
                warnings.NegmasCaughtExceptionWarning,
            )
        return False

    def __len__(self):
        return self.cardinality

    def __eq__(self, other):
        return self._values == other.values and self.name == other.name

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"Issue({self._values}, {self.name})"

    def __str__(self):
        return f"{self.name}: {self._values}"


class DiscreteIssue(Issue):
    """
    An `Issue` with a discrete set of values.
    """

    @property
    def cardinality(self) -> int:
        """The number of possible outcomes for the issue. Guaranteed to  be fininte"""
        return self._n_values  # type: ignore

    def is_continuous(self) -> bool:
        return False

    @property
    @abstractmethod
    def all(self) -> Generator[Any, None, None]:
        """
        A generator that generates all possible values.

        Remarks:
            - This function returns a generator for the case when the number of values is very large.
            - If you need a list then use something like:

            >>> from negmas.outcomes import make_issue
            >>> list(make_issue(5).all)
            [0, 1, 2, 3, 4]

        """

    def ordered_value_generator(
        self, n: int | float | None = 10, grid=True, compact=True, endpoints=True
    ) -> Generator[Any, None, None]:
        m = self.cardinality
        n = m if n is None or not math.isfinite(n) else int(n)

        for i in range(n):
            yield self._values[i % m]

    def value_generator(
        self, n: int | float | None = 10, grid=True, compact=True, endpoints=True
    ) -> Generator[Any, None, None]:
        m = self.cardinality
        n = m if n is None or not math.isfinite(n) else int(n)

        yield from (
            self._values[_]
            for _ in sample(m, n, grid=grid, compact=compact, endpoints=endpoints)
        )

    def value_at(self, index: int):
        if index < 0 or index > self.cardinality - 1:
            raise IndexError(index)
        return self._values[index]

    def rand(self):
        """Picks a random valid value."""
        return random.choice(self._values)  # type: ignore

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> Iterable[Outcome]:
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

    def is_valid(self, v):
        return v in self._values

    def __getitem__(self, indx):
        return self.values(indx)
