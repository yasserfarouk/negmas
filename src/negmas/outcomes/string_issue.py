"""String issues: any string within length bounds.

Examples:

    >>> from negmas.outcomes import StrIssue
    >>> s = StrIssue("comment", min_length=1, max_length=20)
    >>> s.is_valid("hello"), s.is_valid(""), s.is_valid(3)
    (True, False, False)
    >>> s.cardinality
    inf
    >>> 1 <= len(s.rand()) <= 20
    True
"""

from __future__ import annotations

import math
import random
import string
from typing import Any, Generator

from negmas.helpers.types import get_full_type_name
from negmas.outcomes.base_issue import Issue
from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER

__all__ = ["StrIssue"]

DEFAULT_ALPHABET = string.ascii_letters + string.digits
# Samples drawn by `value_generator()` when `n` is None.
DEFAULT_STR_SAMPLES = 1000


class StrIssue(Issue):
    """Any string with `min_length` to `max_length` characters.

    Args:
        name: The issue name.
        min_length: The minimum length.
        max_length: The maximum length (None: unbounded).
        alphabet: Characters `rand()` draws from (validity accepts any).
        typical_length: Mean length drawn by `rand()` (default ``min_length + 8``).

    Remarks:
        The cardinality is reported as infinite (it is astronomically large
        even when `max_length` is given), so the issue is sampled, never
        enumerated. Score it with a callable value function (e.g. `LambdaFun`).
    """

    def __init__(
        self,
        name: str | None = None,
        min_length: int = 0,
        max_length: int | None = None,
        alphabet: str = DEFAULT_ALPHABET,
        typical_length: float | None = None,
    ) -> None:
        if int(min_length) != min_length or min_length < 0:
            raise ValueError(f"min_length must be a non-negative integer: {min_length}")
        if max_length is not None and int(max_length) != max_length:
            raise ValueError(f"max_length must be an integer: {max_length}")
        if max_length is not None and max_length < min_length:
            raise ValueError(f"max_length ({max_length}) < min_length ({min_length})")
        if not alphabet:
            raise ValueError("alphabet must not be empty")
        super().__init__((min_length, max_length), name=name)
        self.min_length = int(min_length)
        self.max_length = None if max_length is None else int(max_length)
        self.alphabet = alphabet
        self.typical_length = typical_length
        self._value_type = str
        self._n_values = float("inf")

    @property
    def type(self) -> str:
        return "string"

    def is_valid(self, v) -> bool:
        if not isinstance(v, str):
            return False
        n = len(v)
        return n >= self.min_length and (
            self.max_length is None or n <= self.max_length
        )

    def is_continuous(self) -> bool:
        return False

    def is_discrete(self) -> bool:
        """False: too many values to enumerate (sampled instead)."""
        return False

    def is_finite(self) -> bool:
        return False

    def is_numeric(self) -> bool:
        return False

    @property
    def cardinality(self) -> float:
        return float("inf")

    def _rand_length(self) -> int:
        lo, hi = self.min_length, self.max_length
        mean = self.typical_length if self.typical_length is not None else lo + 8
        extra = max(float(mean) - lo, 0.0)
        n = lo
        if extra > 0:
            p = 1.0 / (extra + 1.0)
            n = lo + (
                int(math.log(1.0 - random.random()) / math.log(1.0 - p))
                if p < 1.0
                else 0
            )
        return n if hi is None else min(n, hi)

    def rand(self) -> str:
        """A random string from `alphabet` with a geometric length."""
        return "".join(random.choices(self.alphabet, k=self._rand_length()))

    def mutate(self, value: str) -> str:
        """A neighbour of `value`: one character inserted, removed or replaced."""
        ops = ["change"] if value else []
        if self.max_length is None or len(value) < self.max_length:
            ops.append("add")
        if len(value) > self.min_length:
            ops.append("remove")
        if not ops:
            return value
        op = random.choice(ops)
        c = random.choice(self.alphabet)
        if op == "add":
            i = random.randint(0, len(value))
            return value[:i] + c + value[i:]
        i = random.randrange(len(value))
        if op == "remove":
            return value[:i] + value[i + 1 :]
        return value[:i] + c + value[i + 1 :]

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list:
        if with_replacement:
            return [self.rand() for _ in range(n)]
        seen: dict[str, None] = {}
        for _ in range(50 * n):
            if len(seen) >= n:
                break
            seen.setdefault(self.rand(), None)
        if fail_if_not_enough and len(seen) < n:
            raise ValueError(
                f"Could only sample {len(seen)} distinct strings out of {n}"
            )
        return list(seen)

    def rand_invalid(self):
        """A string of an invalid length, or None if every length is valid."""
        if self.max_length is not None:
            return "".join(random.choices(self.alphabet, k=self.max_length + 1))
        if self.min_length > 0:
            return "".join(random.choices(self.alphabet, k=self.min_length - 1))
        return None

    def ordered_value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[str, None, None]:
        """`n` (default 1000) distinct sampled strings, sorted."""
        if n is None or not math.isfinite(n):
            n = DEFAULT_STR_SAMPLES
        yield from sorted(self.rand_outcomes(int(n)))

    def value_generator(
        self, n: int | float | None = None, grid=True, compact=True, endpoints=True
    ) -> Generator[str, None, None]:
        yield from self.ordered_value_generator(n)

    def to_discrete(
        self, n: int | float | None = 10, grid=True, compact=True, endpoints=True
    ):
        """A `CategoricalIssue` of `n` sampled strings."""
        return CategoricalIssue(
            list(self.ordered_value_generator(n or 10)), name=self.name
        )

    @property
    def all(self):
        raise ValueError(f"The string issue {self.name} cannot be enumerated")

    def value_at(self, index: int):
        raise ValueError(f"Cannot index the string issue {self.name}")

    def contains(self, issue: Issue) -> bool:
        if isinstance(issue, StrIssue):
            mine = math.inf if self.max_length is None else self.max_length
            theirs = math.inf if issue.max_length is None else issue.max_length
            return issue.min_length >= self.min_length and theirs <= mine
        if isinstance(issue, CategoricalIssue):
            return all(self.is_valid(_) for _ in issue.all)
        return False

    def _to_xml_str(self, indx: int) -> str:
        raise NotImplementedError(
            f"Genius has no string issues: cannot export {self.name}"
        )

    def to_dict(
        self, python_class_identifier=PYTHON_CLASS_IDENTIFIER
    ) -> dict[str, Any]:
        return {
            python_class_identifier: get_full_type_name(type(self)),
            "name": self.name,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "alphabet": self.alphabet,
            "typical_length": self.typical_length,
        }

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        if isinstance(d, cls):
            return d
        d = dict(d)
        d.pop(python_class_identifier, None)
        d.pop("n_values", None)
        return cls(**d)

    def _clone(self) -> StrIssue:
        return StrIssue(
            self.name,
            self.min_length,
            self.max_length,
            self.alphabet,
            self.typical_length,
        )

    def __copy__(self):
        return self._clone()

    def __deepcopy__(self, memodict={}):
        return self._clone()

    def __eq__(self, other):
        return (
            isinstance(other, StrIssue)
            and self.name == other.name
            and self.min_length == other.min_length
            and self.max_length == other.max_length
        )

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"StrIssue({self.name!r}, {self.min_length}, {self.max_length})"

    def __str__(self):
        mx = "*" if self.max_length is None else self.max_length
        return f"{self.name}: str[{self.min_length}..{mx}]"
