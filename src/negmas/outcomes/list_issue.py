"""List issues: one issue whose value is a variable-length list of records.

A `ListIssue` models a *repeated group of issues* (e.g. the deliveries of a
delivery plan, each an ``(order_id, quantity, date)`` record) without making
the number of issues variable. The outcome keeps a fixed number of issues; only
the value of the list issue varies in length.

Examples:

    A delivery plan: any number of ``(order_id, quantity, date)`` records, with
    no two deliveries of the same order on the same date:

    >>> from negmas.outcomes import ListIssue, RecordIssue, make_issue
    >>> delivery = RecordIssue(
    ...     [
    ...         make_issue(["PO-1", "PO-7"], "order_id"),
    ...         make_issue((1, 100), "quantity"),
    ...         make_issue((0, 30), "date"),
    ...     ]
    ... )
    >>> plan = ListIssue("deliveries", delivery, unique_by=("order_id", "date"))
    >>> plan.effective_max_count  # 2 orders x 31 dates: at most 62 deliveries
    62
    >>> ListIssue("lots", delivery).cardinality  # no max_count and no key cap
    inf
    >>> plan.is_valid([("PO-1", 30, 12), ("PO-7", 50, 12)])
    True
    >>> plan.is_valid([("PO-1", 30, 12), ("PO-1", 20, 12)])  # same order and date
    False

    Unordered lists are multisets, canonicalized by sorting:

    >>> plan.canonicalize([["PO-7", 50, 12], ["PO-1", 30, 12]])
    (('PO-1', 30, 12), ('PO-7', 50, 12))
    >>> plan.is_valid(plan.rand())
    True
"""

from __future__ import annotations

import math
import random
from functools import reduce
from itertools import combinations, combinations_with_replacement, islice, product
from operator import mul
from typing import Any, Generator, Iterable, Sequence

from negmas.helpers.types import get_full_type_name
from negmas.outcomes.base_issue import DiscreteIssue, Issue
from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

__all__ = ["RecordIssue", "ListIssue", "DiscreteListValuesIssue"]

# Maximum number of attempts made by rejection-sampling loops (per value).
_MAX_TRIALS = 50
# When the count range is wider than this, `rand()` draws the number of
# elements from a geometric distribution instead of uniformly.
_UNIFORM_COUNT_RANGE = 10
# Default mean number of elements above `min_count` that `rand()` draws when
# the count range is wide (and no `typical_count` is given).
_DEFAULT_EXTRA_COUNT = 5


def _canonical(issue: Issue, value: Any) -> Any:
    """Canonicalizes `value` for `issue` if the issue defines a canonical form."""
    canon = getattr(issue, "canonicalize", None)
    return canon(value) if canon is not None else value


def _issue_index_value(issue: Issue, index: int) -> Any:
    """Returns the value at `index` of a finite issue (in enumeration order)."""
    try:
        return issue.value_at(index)
    except Exception:
        return next(islice(issue.all, index, None))


def _sorted(values: Iterable[Any]) -> tuple:
    """Sorts values into a canonical order, falling back to `repr` order."""
    values = list(values)
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=repr))


class RecordIssue(Issue):
    """An ordered tuple of named field issues. Its value is a tuple of field values.

    `RecordIssue` is the element type of a `ListIssue` with several fields (it
    can also be used as a stand-alone issue). Field names must be unique.

    Args:
        fields: The field issues, in order.
        name: The name of the record (optional).

    Examples:
        >>> from negmas.outcomes import RecordIssue, make_issue
        >>> r = RecordIssue([make_issue(["a", "b"], "x"), make_issue((0, 2), "y")])
        >>> r.field_names
        ('x', 'y')
        >>> r.cardinality
        6
        >>> r.is_valid(("a", 2)), r.is_valid(["b", 1]), r.is_valid(("c", 0))
        (True, True, False)
        >>> list(r.all)[:3]
        [('a', 0), ('a', 1), ('a', 2)]
    """

    def __init__(self, fields: Sequence[Issue], name: str | None = None) -> None:
        fields = tuple(fields)
        if not fields:
            raise ValueError("A RecordIssue needs at least one field")
        for f in fields:
            if not isinstance(f, Issue):
                raise TypeError(f"Record fields must be issues but got {f!r}")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError(f"Record field names must be unique but got {names}")
        super().__init__(fields, name=name)
        self._value_type = tuple
        self._n_values = reduce(mul, [f.cardinality for f in fields], 1)

    # ------------------------------------------------------------------
    # structure
    # ------------------------------------------------------------------
    @property
    def fields(self) -> tuple[Issue, ...]:
        """The field issues, in order."""
        return self._values

    @property
    def field_names(self) -> tuple[str, ...]:
        """The names of the fields, in order."""
        return tuple(f.name for f in self._values)

    def field_index(self, name: str) -> int:
        """Returns the position of the field called `name`."""
        try:
            return self.field_names.index(name)
        except ValueError:
            raise ValueError(
                f"{self.name} has no field called {name!r}. Fields: {self.field_names}"
            ) from None

    def field(self, name: str) -> Issue:
        """Returns the field issue called `name`."""
        return self._values[self.field_index(name)]

    @property
    def type(self) -> str:
        """Returns a nice name for the issue type."""
        return "record"

    # ------------------------------------------------------------------
    # values
    # ------------------------------------------------------------------
    def canonicalize(self, value: Any) -> tuple:
        """Returns `value` as a tuple of canonical field values (lists become tuples)."""
        return tuple(_canonical(f, v) for f, v in zip(self._values, value))

    def is_valid(self, v) -> bool:
        """Checks that `v` has one valid value per field."""
        if not isinstance(v, (tuple, list)) or len(v) != len(self._values):
            return False
        return all(f.is_valid(_) for f, _ in zip(self._values, v))

    def is_continuous(self) -> bool:
        """A record is continuous if any of its fields is."""
        return any(f.is_continuous() for f in self._values)

    def is_discrete(self) -> bool:
        """A record is discrete (finite and enumerable) if all its fields are."""
        return all(f.is_discrete() for f in self._values)

    def is_finite(self) -> bool:
        """A record is finite if all its fields are."""
        return self.is_discrete()

    def is_numeric(self) -> bool:
        """Records are tuples, not numbers."""
        return False

    @property
    def cardinality(self) -> int | float:
        """The product of the field cardinalities."""
        return self._n_values

    def rand(self) -> tuple:  # type: ignore[override]
        """A random record (each field sampled independently)."""
        return tuple(f.rand() for f in self._values)

    def rand_invalid(self):
        """A record with one invalid field, or None if no field has invalid values."""
        for i in random.sample(range(len(self._values)), len(self._values)):
            try:
                bad = self._values[i].rand_invalid()
            except Exception:
                continue
            if bad is None:
                continue
            v = list(self.rand())
            v[i] = bad
            return tuple(v)
        return None

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list:
        """Samples `n` records (distinct unless `with_replacement`)."""
        return _rand_values(self, n, with_replacement, fail_if_not_enough)

    @property
    def all(self) -> Generator[tuple, None, None]:
        """All records (only for finite records)."""
        if not self.is_discrete():
            raise ValueError(f"The record {self.name} is not finite")
        yield from product(*(list(f.all) for f in self._values))

    def ordered_value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[tuple, None, None]:
        """Records in enumeration order (finite) or `n` samples (infinite)."""
        yield from _value_generator(self, n)

    def value_generator(
        self, n: int | float | None = None, grid=True, compact=True, endpoints=True
    ) -> Generator[tuple, None, None]:
        """All records if finite and few enough, otherwise `n` distinct samples."""
        yield from _value_generator(self, n)

    def value_at(self, index: int):
        """The record at `index` in enumeration order (finite records only)."""
        if not self.is_discrete():
            raise ValueError(f"Cannot index the infinite record {self.name}")
        if index < 0 or index >= self.cardinality:
            raise IndexError(index)
        vals = []
        for f in reversed(self._values):
            index, r = divmod(index, int(f.cardinality))
            vals.append(_issue_index_value(f, r))
        return tuple(reversed(vals))

    def to_discrete(  # type: ignore[override]
        self, n: int | float | None = 10, grid=True, compact=True, endpoints=True
    ) -> RecordIssue:
        """A finite record with every non-finite field discretized to `n` levels."""
        if self.is_discrete():
            return self
        levels = 10 if n is None or not math.isfinite(n) else int(n)
        return RecordIssue(
            [
                f
                if f.is_discrete()
                else f.to_discrete(levels, grid=grid, compact=False, endpoints=True)
                for f in self._values
            ],
            name=self.name,
        )

    def contains(self, issue: Issue) -> bool:
        """Checks whether every record of `issue` is a record of this issue."""
        if not isinstance(issue, RecordIssue) or len(issue.fields) != len(self.fields):
            return False
        return all(a.contains(b) for a, b in zip(self.fields, issue.fields))

    def _to_xml_str(self, indx: int) -> str:
        raise NotImplementedError(
            f"Genius has no record issues: cannot export {self.name} to XML"
        )

    # ------------------------------------------------------------------
    # serialization / identity
    # ------------------------------------------------------------------
    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Converts the issue to a dict (see `from_dict`)."""
        return {
            python_class_identifier: get_full_type_name(type(self)),
            "fields": [
                serialize(f, python_class_identifier=python_class_identifier)
                for f in self._values
            ],
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Constructs the issue from a dict made by `to_dict`."""
        if isinstance(d, cls):
            return d
        d = dict(d)
        d.pop(python_class_identifier, None)
        fields = [
            deserialize(f, python_class_identifier=python_class_identifier)
            for f in d["fields"]
        ]
        return cls(fields, name=d.get("name", None))

    def __copy__(self):
        return RecordIssue(self._values, name=self.name)

    def __deepcopy__(self, memodict={}):
        import copy

        return RecordIssue(
            [copy.deepcopy(f, memodict) for f in self._values], name=self.name
        )

    def __eq__(self, other):
        return (
            isinstance(other, RecordIssue)
            and self.name == other.name
            and self.fields == other.fields
        )

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f"RecordIssue({list(self._values)!r}, {self.name!r})"

    def __str__(self):
        return f"{self.name}: {{{', '.join(str(f) for f in self._values)}}}"


class ListIssue(Issue):
    """An issue whose value is a variable-length list of element values.

    The value is a tuple of element values (records for a `RecordIssue`
    element). The number of elements is between `min_count` and `max_count`
    (unbounded when `max_count` is None, which makes the issue infinite).

    Args:
        name: The issue name.
        element: The element issue: a `RecordIssue`, any `Issue`, or a sequence
            of issues (made into a `RecordIssue`).
        min_count: The minimum number of elements.
        max_count: The maximum number of elements, or None for no limit.
        ordered: If True the value is a sequence (order matters). If False it
            is a multiset: values are canonicalized by sorting, so equality and
            hashing do not depend on the order of the elements.
        unique_by: Field names whose combination must not repeat across the
            elements, e.g. ``("order_id", "date")``. For a non-record element,
            pass the element's name (or True) to make the element values
            themselves unique. True on a record means whole records are unique.
        typical_count: The mean number of elements drawn by `rand()` when the
            count range is wide (defaults to ``min_count + 5``, capped at
            `max_count`).

    Remarks:
        - `is_valid` accepts the elements in any order and as lists or tuples
          (e.g. JSON); use `canonicalize` to get the canonical value.
        - Never enumerate an infinite list issue: sample with `rand`,
          `rand_outcomes` or `value_generator(n)`, search with `mutate`, or get
          a finite view with `to_discrete`.
    """

    def __init__(
        self,
        name: str | None,
        element: RecordIssue | Issue | Sequence[Issue],
        min_count: int = 0,
        max_count: int | None = None,
        ordered: bool = False,
        unique_by: Sequence[str] | str | bool | None = None,
        *,
        typical_count: int | float | None = None,
    ) -> None:
        if not isinstance(element, Issue):
            element = RecordIssue(list(element))
        if int(min_count) != min_count or min_count < 0:
            raise ValueError(f"min_count must be a non-negative integer ({min_count=})")
        if max_count is not None and (
            int(max_count) != max_count or max_count < min_count
        ):
            raise ValueError(
                f"max_count must be None or an integer >= min_count "
                f"({min_count=}, {max_count=})"
            )
        super().__init__(element, name=name)
        self.min_count = int(min_count)
        self.max_count = None if max_count is None else int(max_count)
        self.ordered = bool(ordered)
        self.typical_count = typical_count
        self.unique_by: tuple[str, ...] | None = self._resolve_unique_by(unique_by)
        self._key_idx: tuple[int, ...] | None = self._key_indices()
        self._value_type = tuple
        self._n_values = self._compute_cardinality()
        if self._effective_max_count() < self.min_count:
            raise ValueError(
                f"{self.name}: unique_by={self.unique_by} allows at most "
                f"{self._n_keys()} elements but min_count={self.min_count}"
            )

    # ------------------------------------------------------------------
    # structure
    # ------------------------------------------------------------------
    @property
    def element(self) -> Issue:
        """The element issue."""
        return self._values

    @property
    def is_record(self) -> bool:
        """True if the elements are records."""
        return isinstance(self._values, RecordIssue)

    @property
    def field_names(self) -> tuple[str, ...]:
        """The element's field names (the element name for a non-record element)."""
        if self.is_record:
            return self._values.field_names  # type: ignore[attr-defined]
        return (self._values.name,)

    @property
    def type(self) -> str:
        """Returns a nice name for the issue type."""
        return "list"

    def _resolve_unique_by(self, unique_by) -> tuple[str, ...] | None:
        if unique_by is None or unique_by is False:
            return None
        if unique_by is True:
            return self.field_names
        if isinstance(unique_by, str):
            unique_by = (unique_by,)
        unique_by = tuple(unique_by)
        if not unique_by:
            return None
        for n in unique_by:
            if n not in self.field_names:
                raise ValueError(
                    f"unique_by field {n!r} is not a field of {self.name}. "
                    f"Fields: {self.field_names}"
                )
        if len(set(unique_by)) != len(unique_by):
            raise ValueError(f"unique_by has repeated fields: {unique_by}")
        return unique_by

    def _key_indices(self) -> tuple[int, ...] | None:
        if self.unique_by is None:
            return None
        if not self.is_record:
            return ()  # the whole element value is the key
        return tuple(self.field_names.index(_) for _ in self.unique_by)

    def key_of(self, element_value: Any) -> Any:
        """The `unique_by` key of an element value (None if there is no `unique_by`)."""
        if self._key_idx is None:
            return None
        if not self._key_idx:
            return element_value
        return tuple(element_value[i] for i in self._key_idx)

    def _key_fields(self) -> tuple[Issue, ...]:
        if self._key_idx is None:
            return ()
        if not self._key_idx:
            return (self._values,)
        return tuple(self._values.fields[i] for i in self._key_idx)  # type: ignore[attr-defined]

    def _n_keys(self) -> int | float:
        """The number of distinct `unique_by` keys (inf if unbounded or no key)."""
        if self._key_idx is None:
            return float("inf")
        fields = self._key_fields()
        if not all(f.is_discrete() for f in fields):
            return float("inf")
        return reduce(mul, [f.cardinality for f in fields], 1)

    def _effective_max_count(self) -> int | float:
        """The largest attainable number of elements (inf if unbounded)."""
        mx = float("inf") if self.max_count is None else self.max_count
        return min(mx, self._n_keys())

    @property
    def effective_max_count(self) -> int | float:
        """The largest number of elements a valid value can have (inf if unbounded).

        This is `max_count` capped by the number of distinct `unique_by` keys.
        """
        return self._effective_max_count()

    def _compute_cardinality(self) -> int | float:
        lo, hi = self.min_count, self._effective_max_count()
        if hi == 0:
            return 1
        m = self._values.cardinality
        if not math.isfinite(hi) or not self._values.is_discrete():
            return float("inf")
        hi = int(hi)
        m = int(m)
        total = 0
        if self._key_idx is None:
            for n in range(lo, hi + 1):
                total += m**n if self.ordered else math.comb(m + n - 1, n)
            return total
        k = int(self._n_keys())
        g = m // k  # element values per key
        for n in range(lo, hi + 1):
            keys = math.perm(k, n) if self.ordered else math.comb(k, n)
            total += keys * g**n
        return total

    # ------------------------------------------------------------------
    # values
    # ------------------------------------------------------------------
    def canonicalize(self, value: Any) -> tuple:
        """The canonical form of `value`: a tuple of canonical elements.

        Lists (e.g. from JSON) become tuples. For unordered issues the elements
        are sorted, so two orderings of the same multiset have the same value.
        """
        elements = tuple(_canonical(self._values, _) for _ in value)
        if self.ordered:
            return elements
        return _sorted(elements)

    def is_valid(self, v) -> bool:
        """Checks the count, every element and the `unique_by` constraint."""
        if not isinstance(v, (tuple, list)):
            return False
        n = len(v)
        if n < self.min_count or (self.max_count is not None and n > self.max_count):
            return False
        if not all(self._values.is_valid(_) for _ in v):
            return False
        if self._key_idx is not None:
            try:
                v = [_canonical(self._values, _) for _ in v]
                keys = [self.key_of(_) for _ in v]
                if len(set(keys)) != len(keys):
                    return False
            except TypeError:
                return False
        return True

    def is_continuous(self) -> bool:
        """List issues are never continuous ranges (see `is_discrete`)."""
        return False

    def is_discrete(self) -> bool:
        """True only if the issue is finite (and hence can be enumerated)."""
        return math.isfinite(self._n_values)

    def is_finite(self) -> bool:
        """True if the issue has finitely many values."""
        return self.is_discrete()

    def is_numeric(self) -> bool:
        """List values are tuples, not numbers."""
        return False

    def is_integer(self) -> bool:
        """List values are tuples, not integers."""
        return False

    def is_float(self) -> bool:
        """List values are tuples, not reals."""
        return False

    @property
    def cardinality(self) -> int | float:
        """The number of valid values (inf if `max_count` is None or the element is infinite)."""
        return self._n_values

    def _rand_count(self) -> int:
        lo, hi = self.min_count, self._effective_max_count()
        if math.isfinite(hi) and hi - lo <= _UNIFORM_COUNT_RANGE:
            return random.randint(lo, int(hi))
        mean = self.typical_count
        if mean is None:
            mean = lo + _DEFAULT_EXTRA_COUNT
        extra = max(float(mean) - lo, 0.0)
        if extra <= 0:
            return lo
        # geometric number of extra elements with the requested mean
        p = 1.0 / (extra + 1.0)
        n = lo + int(math.log(1.0 - random.random()) / math.log(1.0 - p))
        return int(min(n, hi))

    def _rand_elements(self, n: int) -> list:
        """`n` random element values satisfying `unique_by` (fewer if impossible)."""
        element = self._values
        if self._key_idx is None:
            return [_canonical(element, element.rand()) for _ in range(n)]
        key_fields = self._key_fields()
        if all(f.is_discrete() for f in key_fields):
            # sample distinct keys directly, then fill the other fields
            k = int(self._n_keys())
            n = min(n, k)
            sizes = [int(f.cardinality) for f in key_fields]
            out = []
            for idx in random.sample(range(k), n):
                key = []
                for s, f in zip(reversed(sizes), reversed(key_fields)):
                    idx, r = divmod(idx, s)
                    key.append(_issue_index_value(f, r))
                key.reverse()
                out.append(self._with_key(tuple(key)))
            return out
        out, keys = [], set()
        for _ in range(n * _MAX_TRIALS):
            if len(out) >= n:
                break
            e = _canonical(element, element.rand())
            k = self.key_of(e)
            if k in keys:
                continue
            keys.add(k)
            out.append(e)
        return out

    def _with_key(self, key: tuple) -> Any:
        """A random element value with the given `unique_by` key."""
        element = self._values
        if not self._key_idx:
            return _canonical(element, key[0])
        v = list(element.rand())
        for i, kv in zip(self._key_idx, key):
            v[i] = kv
        return _canonical(element, tuple(v))

    def rand(self) -> tuple:  # type: ignore[override]
        """A random valid value: samples a number of elements, then the elements."""
        for _ in range(_MAX_TRIALS):
            n = self._rand_count()
            els = self._rand_elements(n)
            if len(els) >= self.min_count:
                return self.canonicalize(els)
        raise ValueError(
            f"Could not sample a valid value for {self.name} after {_MAX_TRIALS} trials"
        )

    def mutate(self, value: Any, n_trials: int = _MAX_TRIALS) -> tuple:
        """A random neighbour of `value`: one element added, removed or changed.

        Used by search-based inverters and negotiators. The result is always a
        valid canonical value; `value` itself is returned if it has no valid
        neighbour.
        """
        value = list(self.canonicalize(value))
        n = len(value)
        hi = self._effective_max_count()
        element = self._values
        for _ in range(n_trials):
            ops = []
            if n < hi:
                ops.append("add")
            if n > self.min_count:
                ops.append("remove")
            if n > 0:
                ops.append("change")
            if self.ordered and n > 1:
                ops.append("swap")
            if not ops:
                break
            op = random.choice(ops)
            new = list(value)
            if op == "add":
                new.insert(random.randint(0, n), _canonical(element, element.rand()))
            elif op == "remove":
                new.pop(random.randrange(n))
            elif op == "swap":
                i, j = random.sample(range(n), 2)
                new[i], new[j] = new[j], new[i]
            else:
                i = random.randrange(n)
                if self.is_record and random.random() < 0.8:
                    f = random.randrange(len(element.fields))  # type: ignore[attr-defined]
                    e = list(new[i])
                    e[f] = _canonical(
                        element.fields[f],  # type: ignore[attr-defined]
                        element.fields[f].rand(),  # type: ignore[attr-defined]
                    )
                    new[i] = tuple(e)
                else:
                    new[i] = _canonical(element, element.rand())
            if self.is_valid(new):
                new_value = self.canonicalize(new)
                if new_value != tuple(value) or op == "swap":
                    return new_value
        return self.canonicalize(value)

    def rand_invalid(self):
        """An invalid value (too short/long, an invalid element, or a repeated key)."""
        options = []
        if self.min_count > 0:
            options.append(lambda: tuple(self.rand()[: self.min_count - 1]))
        if self.max_count is not None:
            options.append(
                lambda: tuple(
                    _canonical(self._values, self._values.rand())
                    for _ in range(self.max_count + 1)  # type: ignore[operator]
                )
            )
        try:
            bad = self._values.rand_invalid()
        except Exception:
            bad = None
        if bad is not None:
            options.append(lambda: (bad,) * max(1, self.min_count))
        if self._key_idx is not None and self._effective_max_count() >= 1:

            def _repeated():
                e = _canonical(self._values, self._values.rand())
                return (e, e) + tuple(
                    _canonical(self._values, self._values.rand())
                    for _ in range(max(0, self.min_count - 2))
                )

            options.append(_repeated)
        if not options:
            return None
        return random.choice(options)()

    def rand_outcomes(
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list:
        """Samples `n` values (distinct unless `with_replacement`)."""
        return _rand_values(self, n, with_replacement, fail_if_not_enough)

    @property
    def all(self) -> Generator[tuple, None, None]:
        """All values, in a stable order. Only for finite list issues."""
        if not self.is_discrete():
            raise ValueError(
                f"The list issue {self.name} is infinite and cannot be enumerated. "
                f"Sample it (rand, value_generator(n)) or use to_discrete()."
            )
        element_values = list(self._values.all)
        lo, hi = self.min_count, int(self._effective_max_count())
        for n in range(lo, hi + 1):
            if self.ordered:
                candidates: Iterable = product(element_values, repeat=n)
            elif self._key_idx is not None:
                candidates = combinations(element_values, n)
            else:
                candidates = combinations_with_replacement(element_values, n)
            for c in candidates:
                c = [_canonical(self._values, _) for _ in c]
                if self._key_idx is not None:
                    keys = [self.key_of(_) for _ in c]
                    if len(set(keys)) != len(keys):
                        continue
                yield self.canonicalize(c)

    def ordered_value_generator(
        self, n: int | float | None = None, grid=True, compact=False, endpoints=True
    ) -> Generator[tuple, None, None]:
        """All values in a stable order (finite), or `n` samples (infinite)."""
        yield from _value_generator(self, n)

    def value_generator(
        self, n: int | float | None = None, grid=True, compact=True, endpoints=True
    ) -> Generator[tuple, None, None]:
        """All values if finite and at most `n`, otherwise `n` distinct samples."""
        yield from _value_generator(self, n)

    def value_at(self, index: int):
        """The value at `index` in enumeration order (finite issues only)."""
        if not self.is_discrete():
            raise ValueError(f"Cannot index the infinite list issue {self.name}")
        if index < 0 or index >= self.cardinality:
            raise IndexError(index)
        return next(islice(self.all, index, None))

    def to_discrete(  # type: ignore[override]
        self,
        n: int | float | None = 10,
        grid=True,
        compact=True,
        endpoints=True,
        *,
        levels: int | None = None,
        max_count_cap: int | None = None,
    ) -> Issue:
        """A finite view of this issue (only when asked; nothing here enumerates).

        Two forms:

        - **Structural** (when `levels` or `max_count_cap` is given): a finite
          `ListIssue` with every non-finite element field discretized to
          `levels` values and the count capped at `max_count_cap`.
        - **Sampled** (otherwise, the form the outcome-space machinery uses):
          a `DiscreteListValuesIssue` holding `n` distinct sampled values.

        A finite issue is returned unchanged by the sampled form if it has at
        most `n` values.
        """
        if levels is not None or max_count_cap is not None:
            element = self._values
            if not element.is_discrete():
                element = element.to_discrete(levels if levels is not None else 10)
            mx = self.max_count
            if max_count_cap is not None:
                mx = max_count_cap if mx is None else min(mx, max_count_cap)
            if mx is None:
                raise ValueError(
                    f"{self.name} has no max_count: pass max_count_cap to get a "
                    f"finite view"
                )
            mx = max(mx, self.min_count)
            return ListIssue(
                self.name,
                element,
                min_count=self.min_count,
                max_count=mx,
                ordered=self.ordered,
                unique_by=self.unique_by,
                typical_count=self.typical_count,
            )
        if n is None or not math.isfinite(n):
            if self.is_discrete():
                return self
            raise ValueError(
                f"Cannot discretize the infinite list issue {self.name} without a "
                f"finite number of values"
            )
        if self.is_discrete() and self.cardinality <= n:
            return self
        return DiscreteListValuesIssue(
            list(_value_generator(self, int(n))), name=self.name
        )

    def contains(self, issue: Issue) -> bool:
        """Checks whether every value of `issue` is a value of this issue."""
        if isinstance(issue, ListIssue):
            mine = float("inf") if self.max_count is None else self.max_count
            theirs = float("inf") if issue.max_count is None else issue.max_count
            return (
                self.ordered == issue.ordered
                and issue.min_count >= self.min_count
                and theirs <= mine
                and set(self.unique_by or ()) <= set(issue.unique_by or ())
                and self.element.contains(issue.element)
            )
        if isinstance(issue, DiscreteIssue):
            return all(self.is_valid(_) for _ in issue.all)
        return False

    def _to_xml_str(self, indx: int) -> str:
        raise NotImplementedError(
            f"Genius has no list (repeated) issues: cannot export {self.name} to XML"
        )

    # ------------------------------------------------------------------
    # serialization / identity
    # ------------------------------------------------------------------
    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Converts the issue to a dict (see `from_dict`)."""
        return {
            python_class_identifier: get_full_type_name(type(self)),
            "name": self.name,
            "element": serialize(
                self._values, python_class_identifier=python_class_identifier
            ),
            "min_count": self.min_count,
            "max_count": self.max_count,
            "ordered": self.ordered,
            "unique_by": list(self.unique_by) if self.unique_by else None,
            "typical_count": self.typical_count,
        }

    @classmethod
    def from_dict(cls, d, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        """Constructs the issue from a dict made by `to_dict`."""
        if isinstance(d, cls):
            return d
        d = dict(d)
        d.pop(python_class_identifier, None)
        return cls(
            d.get("name", None),
            deserialize(d["element"], python_class_identifier=python_class_identifier),
            min_count=d.get("min_count", 0),
            max_count=d.get("max_count", None),
            ordered=d.get("ordered", False),
            unique_by=d.get("unique_by", None),
            typical_count=d.get("typical_count", None),
        )

    def _clone(self, element: Issue) -> ListIssue:
        return ListIssue(
            self.name,
            element,
            min_count=self.min_count,
            max_count=self.max_count,
            ordered=self.ordered,
            unique_by=self.unique_by,
            typical_count=self.typical_count,
        )

    def __copy__(self):
        return self._clone(self._values)

    def __deepcopy__(self, memodict={}):
        import copy

        return self._clone(copy.deepcopy(self._values, memodict))

    def __eq__(self, other):
        return (
            isinstance(other, ListIssue)
            and self.name == other.name
            and self.element == other.element
            and self.min_count == other.min_count
            and self.max_count == other.max_count
            and self.ordered == other.ordered
            and self.unique_by == other.unique_by
        )

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return (
            f"ListIssue({self.name!r}, {self._values!r}, min_count={self.min_count}, "
            f"max_count={self.max_count}, ordered={self.ordered}, "
            f"unique_by={self.unique_by})"
        )

    def __str__(self):
        mx = "*" if self.max_count is None else self.max_count
        kind = "seq" if self.ordered else "bag"
        u = f" unique_by={list(self.unique_by)}" if self.unique_by else ""
        return f"{self.name}: {kind}[{self.min_count}..{mx}] of {self._values}{u}"


class DiscreteListValuesIssue(CategoricalIssue):
    """A finite set of explicit list (or record) values: the sampled view of a `ListIssue`.

    Behaves like a `CategoricalIssue` but samples values without converting the
    tuples to numpy arrays (which would reshape equally-long tuples).
    """

    def rand_outcomes(  # type: ignore[override]
        self, n: int, with_replacement=False, fail_if_not_enough=False
    ) -> list:
        vals = list(self._values)
        if with_replacement:
            return random.choices(vals, k=n)
        if n > len(vals):
            if fail_if_not_enough:
                raise ValueError(
                    f"Cannot sample {n} values out of {len(vals)} without replacement"
                )
            return vals
        return random.sample(vals, n)

    def _to_xml_str(self, indx: int) -> str:
        raise NotImplementedError(
            f"Genius has no list (repeated) issues: cannot export {self.name} to XML"
        )


def _value_generator(issue: Issue, n: int | float | None) -> Generator:
    """All values of a finite issue (at most `n`), or `n` distinct samples."""
    if issue.is_discrete() and (
        n is None or not math.isfinite(n) or issue.cardinality <= n
    ):
        yield from issue.all
        return
    if n is None or not math.isfinite(n):
        raise ValueError(
            f"Cannot generate all values of the infinite issue {issue.name}: pass n"
        )
    yield from _rand_values(issue, int(n), False, False)


def _rand_values(
    issue: Issue, n: int, with_replacement: bool, fail_if_not_enough: bool
) -> list:
    """`n` random values of `issue` (distinct unless `with_replacement`)."""
    if with_replacement:
        return [issue.rand() for _ in range(n)]
    if issue.is_discrete() and issue.cardinality <= n:
        if fail_if_not_enough and issue.cardinality < n:
            raise ValueError(
                f"Cannot sample {n} values out of {issue.cardinality} without "
                f"replacement"
            )
        vals = list(issue.all)
        random.shuffle(vals)
        return vals
    seen: dict = {}
    for _ in range(n * _MAX_TRIALS):
        if len(seen) >= n:
            break
        v = issue.rand()
        seen.setdefault(v, None)
    if fail_if_not_enough and len(seen) < n:
        raise ValueError(f"Could only sample {len(seen)} distinct values out of {n}")
    return list(seen)
