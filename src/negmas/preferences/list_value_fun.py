"""Value functions for list issues.

A list value is scored by *lifting* an element scorer: score every element,
aggregate the scores and optionally add a function of the count.

- `RecordAdditiveFun`: a record scored as a weighted sum of per-field funs.
- `ListAggregationFun`: lifts an element scorer to a single list issue (a value
  fun for `LinearAdditiveUtilityFunction` or a one-issue `GLAUtilityFunction`
  factor).
- `ListAggregationMultiFun`: lifts a multi-issue function ``g(..., element, ...)``
  to a `GLAUtilityFunction` factor over a list issue and other issues.
- `lift_element_ufun`: converts a ufun over plain issues plus one copy of the
  element's fields into a ufun over the plain issues plus the list issue.

An element scorer can be a `BaseFun` (plain elements), a `BaseMultiFun`, a
`RecordAdditiveFun`, any ufun over the record's fields, or a callable.

Examples:

    >>> from negmas.outcomes import ListIssue, make_issue, make_os
    >>> from negmas.preferences import LinearAdditiveUtilityFunction
    >>> from negmas.preferences.value_fun import AffineFun, TableFun
    >>> lots = ListIssue(
    ...     "lots",
    ...     [make_issue((1, 10), "price"), make_issue((1, 5), "quantity")],
    ...     max_count=4,
    ... )
    >>> lot = RecordAdditiveFun((AffineFun(-1.0, bias=10.0), AffineFun(1.0)))
    >>> f = ListAggregationFun(lot, aggregate="sum")
    >>> f(((2, 3), (4, 1)))
    18.0
    >>> f.minmax(lots)
    (0.0, 56.0)
    >>> os = make_os([make_issue(["fast", "slow"], "shipping"), lots])
    >>> u = LinearAdditiveUtilityFunction(
    ...     {"shipping": TableFun({"fast": 1.0, "slow": 0.0}), "lots": f},
    ...     outcome_space=os,
    ... )
    >>> u.minmax()
    (0.0, 57.0)
"""

from __future__ import annotations

import math
import random
from itertools import product
from typing import Any, Callable, Iterable, Mapping, Sequence

from attrs import define, field

from negmas import warnings
from negmas.helpers.types import get_full_type_name
from negmas.outcomes.base_issue import Issue, make_issue
from negmas.outcomes.list_issue import ListIssue, RecordIssue
from negmas.preferences.value_fun import (
    AffineFun,
    BaseFun,
    BaseMultiFun,
    ConstFun,
    IdentityFun,
    LambdaFun,
    LinearFun,
)
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

__all__ = [
    "RecordAdditiveFun",
    "ListAggregationFun",
    "ListAggregationMultiFun",
    "UnboundedListUtilityError",
    "lift_element_ufun",
    "list_issue_candidates",
]

# Issues (or field combinations) with at most this many values are enumerated.
_MAX_ENUMERATION = 100_000
# Samples drawn from issues too large to enumerate.
_N_SAMPLES = 2_000
# Candidates per field tried by coordinate ascent.
_MAX_FIELD_CANDIDATES = 200
# Most `unique_by` keys ranked exactly (additive / other element scorers).
_MAX_KEYS_ADDITIVE = 100_000
_MAX_KEYS_GENERIC = 10_000
# Most context combinations tried by `ListAggregationMultiFun`.
_MAX_CONTEXTS = 1_000
# Count ranges wider than this are not scanned one count at a time.
_MAX_COUNT_SCAN = 1_000_000
_MAX_CUSTOM_COUNT_SCAN = 10_000

_AGGREGATES = ("sum", "mean", "min", "max")


class UnboundedListUtilityError(ValueError):
    """A list value function has no finite minimum or maximum."""


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _is_ufun(f: Any) -> bool:
    from negmas.preferences.base_ufun import BaseUtilityFunction

    return isinstance(f, BaseUtilityFunction)


def _as_scorer(f: Any) -> Any:
    """Keeps value funs, multi funs and ufuns; wraps plain callables."""
    if isinstance(f, (BaseFun, BaseMultiFun)) or _is_ufun(f):
        return f
    if callable(f):
        return LambdaFun(f)
    raise TypeError(f"Expected a value function, a ufun or a callable: {f!r}")


def _as_fun(f: Any) -> BaseFun | None:
    if f is None or isinstance(f, BaseFun):
        return f
    if callable(f):
        return LambdaFun(f)
    raise TypeError(f"Expected a value function or a callable: {f!r}")


def _candidates(issue: Issue, limit: int = _N_SAMPLES) -> list:
    """All values of a small finite issue, else samples plus finite endpoints."""
    if issue.is_discrete() and issue.cardinality <= _MAX_ENUMERATION:
        return list(issue.all)
    vals = list(issue.value_generator(limit))
    if issue.is_numeric():
        for v in (issue.min_value, issue.max_value):
            try:
                if v is not None and math.isfinite(v) and issue.is_valid(v):
                    vals.append(v)
            except TypeError:
                pass
    return vals


def _arg_extremes(f: Callable[[Any], float], values: Iterable) -> tuple[Any, Any]:
    """(argmin, argmax) of `f` over `values`."""
    best = worst = None
    fb, fw = float("-inf"), float("inf")
    for v in values:
        x = float(f(v))
        if x > fb:
            best, fb = v, x
        if x < fw:
            worst, fw = v, x
    return worst, best


def _builtin_aggregate(name: str) -> Callable[[list[float]], float]:
    if name == "sum":
        return lambda v: float(sum(v))
    if name == "mean":
        return lambda v: float(sum(v)) / len(v)
    if name == "min":
        return lambda v: float(min(v))
    if name == "max":
        return lambda v: float(max(v))
    raise ValueError(f"Unknown aggregate {name!r}: use one of {_AGGREGATES}")


def _count_slope(f: BaseFun) -> float | None:
    """The slope of `f` if it is affine in the count, else None."""
    if isinstance(f, ConstFun):
        return 0.0
    if isinstance(f, (AffineFun, LinearFun)):
        return float(f.slope)
    if isinstance(f, IdentityFun):
        return 1.0
    return None


# ----------------------------------------------------------------------------
# RecordAdditiveFun
# ----------------------------------------------------------------------------


@define(frozen=True)
class RecordAdditiveFun(BaseFun):
    r"""A record scored as :math:`b + \sum_i w_i f_i(r_i)`.

    Args:
        field_funs: One fun (or callable) per field, in field order, or a
            mapping from field name to fun (missing fields score 0).
        weights: Field weights (sequence or mapping by name). Default: 1.
        fields: The record's field names, in order (needed with mappings
            that do not list every field in order).
        bias: A constant added to the score.

    Examples:
        >>> from negmas.preferences.value_fun import AffineFun, TableFun
        >>> f = RecordAdditiveFun(
        ...     {"kind": TableFun({"a": 1.0, "b": 2.0}), "quantity": AffineFun(0.1)},
        ...     fields=("kind", "quantity", "day"),
        ... )
        >>> f(("b", 30, 12))
        5.0
    """

    field_funs: Sequence[Any] | Mapping[str, Any]
    weights: Sequence[float] | Mapping[str, float] | None = None
    fields: Sequence[str] | None = None
    bias: float = 0.0
    _funs: tuple[BaseFun, ...] = field(init=False, repr=False, eq=False)
    _weights: tuple[float, ...] = field(init=False, repr=False, eq=False)

    def __attrs_post_init__(self):
        names = self.fields
        if names is None and isinstance(self.field_funs, Mapping):
            names = tuple(self.field_funs.keys())
        if names is not None:
            names = tuple(names)
            object.__setattr__(self, "fields", names)
        if isinstance(self.field_funs, Mapping):
            unknown = set(self.field_funs) - set(names or ())
            if unknown:
                raise ValueError(f"Functions given for unknown fields {unknown}")
            funs = tuple(
                _as_fun(self.field_funs.get(n)) or ConstFun(0.0)
                for n in names  # type: ignore[union-attr]
            )
        else:
            funs = tuple(_as_fun(_) or ConstFun(0.0) for _ in self.field_funs)
            if names is not None and len(names) != len(funs):
                raise ValueError(f"Got {len(funs)} funs for {len(names)} fields")
        if self.weights is None:
            weights = tuple(1.0 for _ in funs)
        elif isinstance(self.weights, Mapping):
            if names is None:
                raise ValueError("Pass `fields` when `weights` is a mapping")
            unknown = set(self.weights) - set(names)
            if unknown:
                raise ValueError(f"Weights given for unknown fields {unknown}")
            weights = tuple(float(self.weights.get(n, 1.0)) for n in names)
        else:
            weights = tuple(float(_) for _ in self.weights)
        if len(weights) != len(funs):
            raise ValueError(f"Got {len(funs)} funs but {len(weights)} weights")
        object.__setattr__(self, "_funs", funs)
        object.__setattr__(self, "_weights", weights)
        # store the normalized form so equality survives round-trips
        object.__setattr__(self, "field_funs", funs)
        object.__setattr__(self, "weights", weights)

    @property
    def funs(self) -> tuple[BaseFun, ...]:
        """The per-field funs, in field order."""
        return self._funs

    @property
    def field_weights(self) -> tuple[float, ...]:
        """The per-field weights, in field order."""
        return self._weights

    def check(self, issue: Issue) -> RecordIssue:
        """Raises unless `issue` is a record matching this function's fields."""
        if not isinstance(issue, RecordIssue):
            raise TypeError(f"RecordAdditiveFun needs a RecordIssue, got {issue}")
        if len(issue.fields) != len(self._funs):
            raise ValueError(
                f"{len(self._funs)} field funs but {issue.name} has "
                f"{len(issue.fields)} fields"
            )
        if self.fields is not None and tuple(self.fields) != issue.field_names:
            raise ValueError(
                f"Defined for fields {tuple(self.fields)} but the record has "
                f"{issue.field_names}"
            )
        return issue

    def field_value(self, i: int, v: Any) -> float:
        """The weighted score of value `v` of field `i`."""
        return self._weights[i] * float(self._funs[i](v))

    def minmax(self, input: Issue) -> tuple[float, float]:
        """Exact: the sum of the per-field ranges."""
        issue = self.check(input)
        mn = mx = self.bias
        for w, f, fi in zip(self._weights, self._funs, issue.fields):
            a, b = f.minmax(fi)
            a, b = w * float(a), w * float(b)
            mn, mx = mn + min(a, b), mx + max(a, b)
        return mn, mx

    def __call__(self, x) -> float:
        return self.bias + sum(
            w * float(f(v)) for w, f, v in zip(self._weights, self._funs, x)
        )

    def shift_by(self, offset: float) -> RecordAdditiveFun:
        return RecordAdditiveFun(
            self._funs, self._weights, fields=self.fields, bias=self.bias + offset
        )

    def scale_by(self, scale: float) -> RecordAdditiveFun:
        return RecordAdditiveFun(
            self._funs,
            tuple(scale * w for w in self._weights),
            fields=self.fields,
            bias=self.bias * scale,
        )

    def to_ufun(self, record: RecordIssue):
        """The equivalent `LinearAdditiveUtilityFunction` over the record's fields."""
        from negmas.outcomes import make_os
        from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

        record = self.check(record)
        return LinearAdditiveUtilityFunction(
            list(self._funs),
            list(self._weights),
            bias=self.bias,
            outcome_space=make_os(record.fields),
        )

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        raise NotImplementedError("Genius has no record issues")

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER) -> dict:
        return {
            python_class_identifier: get_full_type_name(type(self)),
            "field_funs": _ser(list(self._funs), python_class_identifier),
            "weights": list(self._weights),
            "fields": list(self.fields) if self.fields is not None else None,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        if isinstance(d, cls):
            return d
        return cls(**_deserialize_fields(d, python_class_identifier))


def _ser(x: Any, python_class_identifier) -> Any:
    """Serializes `x`, keeping the type of value funs (via their `to_dict`)."""
    if x is None or isinstance(x, (str, int, float)):
        return x
    if isinstance(x, (list, tuple)):
        return [_ser(_, python_class_identifier) for _ in x]
    to_dict = getattr(x, "to_dict", None)
    if to_dict is not None:
        try:
            return to_dict(python_class_identifier=python_class_identifier)
        except TypeError:
            pass
    return serialize(x, python_class_identifier=python_class_identifier)


def _deserialize_fields(d: dict, python_class_identifier) -> dict:
    d = dict(d)
    d.pop(python_class_identifier, None)
    return deserialize(
        d,
        deep=True,
        remove_type_field=True,
        python_class_identifier=python_class_identifier,
    )  # type: ignore


# ----------------------------------------------------------------------------
# element scorers: the best (worst) element, optionally with fixed fields
# ----------------------------------------------------------------------------


def _additive_parts(f: Any, element: Issue):
    """(weights, funs) if `f` scores the record additively, else None."""
    if not isinstance(element, RecordIssue):
        return None
    if isinstance(f, RecordAdditiveFun):
        f.check(element)
        return f.field_weights, f.funs
    from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

    if isinstance(f, LinearAdditiveUtilityFunction) and not f._constraints:
        if len(f.values) == len(element.fields):
            return tuple(f.weights), tuple(f.values)
    return None


class _ElementScorer:
    """Scores elements of a list issue and finds the best ones."""

    def __init__(self, f: Any, element: Issue):
        self.f = f
        self.element = element
        self.is_record = isinstance(element, RecordIssue)
        self.additive = _additive_parts(f, element)
        self.ufun = _is_ufun(f)
        self.exact = True  # False once a search had to sample
        if self.ufun and self.is_record:
            os_ = getattr(f, "outcome_space", None)
            names = tuple(_.name for _ in getattr(os_, "issues", ()))
            if names and names != element.field_names:  # type: ignore[attr-defined]
                raise ValueError(
                    f"The element ufun is over {names} but the record has fields "
                    f"{element.field_names}"  # type: ignore[attr-defined]
                )

    def score(self, e: Any) -> float:
        if self.ufun:
            return float(self.f(tuple(e) if self.is_record else (e,)))
        return float(self.f(e))

    def best(self, maximize: bool, fixed: dict[int, Any] | None = None) -> Any:
        """The best (or worst) element whose fields in `fixed` are fixed."""
        fixed = fixed or {}
        element = self.element
        sign = 1.0 if maximize else -1.0
        if not self.is_record:
            if fixed:
                return fixed[0]
            if not (element.is_discrete() and element.cardinality <= _MAX_ENUMERATION):
                self.exact = isinstance(self.f, (AffineFun, LinearFun, IdentityFun))
            w, b = _arg_extremes(self.score, _candidates(element))
            return b if maximize else w
        fields = element.fields  # type: ignore[attr-defined]
        free = [i for i in range(len(fields)) if i not in fixed]
        if self.additive is not None:
            weights, funs = self.additive
            v = [fixed.get(i) for i in range(len(fields))]
            for i in free:
                fi = fields[i]
                if not (fi.is_discrete() and fi.cardinality <= _MAX_ENUMERATION):
                    self.exact = self.exact and isinstance(
                        funs[i], (AffineFun, LinearFun, IdentityFun, ConstFun)
                    )
                w, b = _arg_extremes(
                    lambda x, i=i: weights[i] * float(funs[i](x)), _candidates(fi)
                )
                v[i] = b if maximize else w
            return element.canonicalize(tuple(v))  # type: ignore[attr-defined]
        if self.ufun:
            from negmas.outcomes import make_os

            issues = [
                make_issue([fixed[i]], f.name) if i in fixed else f
                for i, f in enumerate(fields)
            ]
            worst, best = self.f.extreme_outcomes(outcome_space=make_os(issues))
            return element.canonicalize(best if maximize else worst)  # type: ignore[attr-defined]
        # any other scorer: enumerate the free fields if few, else search
        n = math.prod(fields[i].cardinality for i in free) if free else 1
        if n <= _MAX_ENUMERATION:
            grids = [
                list(fields[i].all) if i in free else [fixed[i]]
                for i in range(len(fields))
            ]
            w, b = _arg_extremes(self.score, product(*grids))
            return element.canonicalize(b if maximize else w)  # type: ignore[attr-defined]
        self.exact = False
        return element.canonicalize(  # type: ignore[attr-defined]
            self._search(fields, fixed, free, sign)
        )

    def _search(self, fields, fixed, free, sign) -> tuple:
        """Best of random starts, then coordinate ascent on the free fields."""

        def rand():
            return tuple(
                fixed[i] if i in fixed else f.rand() for i, f in enumerate(fields)
            )

        v = max(
            (rand() for _ in range(_MAX_FIELD_CANDIDATES)),
            key=lambda e: sign * self.score(e),
        )
        best = sign * self.score(v)
        cands = {i: _candidates(fields[i], _MAX_FIELD_CANDIDATES) for i in free}
        for _ in range(3):
            improved = False
            for i in free:
                for c in cands[i]:
                    w = list(v)
                    w[i] = c
                    s = sign * self.score(tuple(w))
                    if s > best:
                        best, v, improved = s, tuple(w), True
            if not improved:
                break
        return v


# ----------------------------------------------------------------------------
# ListAggregationFun
# ----------------------------------------------------------------------------


@define(frozen=True)
class ListAggregationFun(BaseFun):
    r"""Scores a list as :math:`b + s(\operatorname{agg}_{e} g(e) + c(|L|))`.

    Args:
        element_fun: Scores one element: a `BaseFun` (plain elements), a
            `BaseMultiFun` or `RecordAdditiveFun` (records), any ufun over the
            record's fields, or a callable.
        aggregate: "sum", "mean", "min", "max" or a callable over the list of
            element scores (assumed non-decreasing in every score).
        empty: The aggregate of an empty list.
        count_fun: Optional fun (or callable) of the number of elements.
        scale: Multiplies the aggregate plus count term.
        bias: Added to the result.

    Remarks:
        `minmax` is exact for mean/min/max. For sum it is exact with a bounded
        count (`max_count` or `unique_by` keys) or when extra elements cannot
        help; otherwise it raises `UnboundedListUtilityError`. Elements with
        large continuous fields are sampled (approximate) unless their funs
        are affine.
    """

    element_fun: Any
    aggregate: str | Callable[[list[float]], float] = "sum"
    empty: float = 0.0
    count_fun: Any = None
    scale: float = 1.0
    bias: float = 0.0

    def __attrs_post_init__(self):
        object.__setattr__(self, "element_fun", _as_scorer(self.element_fun))
        object.__setattr__(self, "count_fun", _as_fun(self.count_fun))
        if isinstance(self.aggregate, str):
            _builtin_aggregate(self.aggregate)
        elif not callable(self.aggregate):
            raise TypeError(f"aggregate must be a name or callable: {self.aggregate!r}")

    def _agg(self, scores: list[float]) -> float:
        if not scores:
            return float(self.empty)
        if isinstance(self.aggregate, str):
            return _builtin_aggregate(self.aggregate)(scores)
        return float(self.aggregate(scores))

    def _count(self, n: int) -> float:
        return 0.0 if self.count_fun is None else float(self.count_fun(n))

    def _lifted(self, scores: list[float]) -> float:
        return self.bias + self.scale * (self._agg(scores) + self._count(len(scores)))

    def __call__(self, x) -> float:
        if x is None:
            return float("nan")
        f = self.element_fun
        if _is_ufun(f):
            scores = [
                float(f(tuple(e) if isinstance(e, (tuple, list)) else (e,))) for e in x
            ]
        else:
            scores = [float(f(e)) for e in x]
        return self._lifted(scores)

    def _with(self, **kw) -> ListAggregationFun:
        d = dict(
            element_fun=self.element_fun,
            aggregate=self.aggregate,
            empty=self.empty,
            count_fun=self.count_fun,
            scale=self.scale,
            bias=self.bias,
        )
        d.update(kw)
        return ListAggregationFun(**d)

    def shift_by(self, offset: float) -> ListAggregationFun:
        return self._with(bias=self.bias + offset)

    def scale_by(self, scale: float) -> ListAggregationFun:
        return self._with(scale=self.scale * scale, bias=self.bias * scale)

    def xml(self, indx: int, issue: Issue, bias=0.0) -> str:
        raise NotImplementedError("Genius has no list issues")

    def minmax(self, input: Issue) -> tuple[float, float]:
        """The exact range over a `ListIssue` (raises if unbounded)."""
        (mn, _), (mx, _) = self.arg_extremes(input)
        return mn, mx

    def arg_extremes(self, issue: Issue) -> tuple[tuple[float, Any], tuple[float, Any]]:
        """((min, a worst list), (max, a best list)) over a `ListIssue`.

        Greedy: the best feasible elements (one per `unique_by` key) for the
        best count. Optimal because elements are scored independently.
        """
        if not isinstance(issue, ListIssue):
            if not issue.is_discrete():
                raise TypeError(f"ListAggregationFun needs a ListIssue, got {issue}")
            # an explicit finite set of lists (e.g. a discretized view)
            scored = [(self(v), v) for v in issue.all]
            return min(scored, key=lambda _: _[0]), max(scored, key=lambda _: _[0])
        return _list_extremes(
            self, issue, _ElementScorer(self.element_fun, issue.element)
        )

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER) -> dict:
        return _agg_to_dict(self, python_class_identifier)

    @classmethod
    def from_dict(cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        if isinstance(d, cls):
            return d
        return cls(**_deserialize_fields(d, python_class_identifier))


def _agg_to_dict(f, python_class_identifier) -> dict:
    d = {
        python_class_identifier: get_full_type_name(type(f)),
        "element_fun": _ser(f.element_fun, python_class_identifier),
        "aggregate": f.aggregate
        if isinstance(f.aggregate, str)
        else _ser(f.aggregate, python_class_identifier),
        "empty": f.empty,
        "count_fun": _ser(f.count_fun, python_class_identifier),
        "scale": f.scale,
        "bias": f.bias,
    }
    for extra in ("list_index", "flatten"):
        if hasattr(f, extra):
            d[extra] = getattr(f, extra)
    return d


def _list_extremes(f, issue: ListIssue, scorer: _ElementScorer):
    """Shared by the lifted funs: ((min, worst list), (max, best list))."""
    lo_raw, lo_list = _raw_extreme(f, issue, scorer, maximize=False)
    hi_raw, hi_list = _raw_extreme(f, issue, scorer, maximize=True)
    lo, hi = f.bias + f.scale * lo_raw, f.bias + f.scale * hi_raw
    if f.scale < 0:
        return (hi, hi_list), (lo, lo_list)
    return (lo, lo_list), (hi, hi_list)


def _ranked(issue: ListIssue, scorer: _ElementScorer, maximize: bool):
    """Element candidates best first as (signed score, element), and whether
    each can be used once (one per `unique_by` key) or repeated."""
    sign = 1.0 if maximize else -1.0
    if issue.unique_by is None:
        e = scorer.best(maximize)
        return [(sign * scorer.score(e), e)], False
    n_keys = issue._n_keys()
    limit = _MAX_KEYS_ADDITIVE if scorer.additive is not None else _MAX_KEYS_GENERIC
    if math.isfinite(n_keys) and n_keys <= limit:
        key_idx = issue._key_idx or (0,)
        key_fields = issue._key_fields()
        ranked = []
        for key in product(*(list(k.all) for k in key_fields)):
            e = scorer.best(maximize, dict(zip(key_idx, key)))
            ranked.append((sign * scorer.score(e), e))
        ranked.sort(key=lambda _: -_[0])
        return ranked, True
    if math.isfinite(n_keys):
        warnings.warn(
            f"{issue.name}: {n_keys} unique_by keys are too many to rank; "
            f"extremes are approximate",
            warnings.NegmasSpeedWarning,
        )
    scorer.exact = False
    e = scorer.best(maximize)
    return [(sign * scorer.score(e), e)], False


def _raw_extreme(f, issue: ListIssue, scorer: _ElementScorer, maximize: bool):
    """The best (or worst) value before scale/bias, and a list achieving it."""
    sign = 1.0 if maximize else -1.0
    ranked, distinct = _ranked(issue, scorer, maximize)
    agg = f.aggregate
    if isinstance(agg, str):
        if not maximize and agg in ("min", "max"):
            agg = "max" if agg == "min" else "min"
        signed_agg = _builtin_aggregate(agg)
    elif maximize:
        signed_agg = lambda v: float(agg(v))  # noqa: E731
    else:
        signed_agg = lambda v: -float(agg([-_ for _ in v]))  # noqa: E731

    builtin = agg if isinstance(agg, str) else None
    top = ranked[0][0]
    prefix = [0.0]
    if distinct and builtin in ("sum", "mean"):
        for s_, _ in ranked:
            prefix.append(prefix[-1] + s_)

    def part_of(n: int) -> float:
        # O(1) for built-in aggregates (the scores are sorted best first)
        if builtin is None:
            vals = [s for s, _ in ranked[:n]] if distinct else [top] * n
            return signed_agg(vals)
        if not distinct:
            return n * top if builtin == "sum" else top
        if builtin == "sum":
            return prefix[n]
        if builtin == "mean":
            return prefix[n] / n
        return ranked[n - 1][0] if builtin == "min" else top

    def value(n: int) -> float:
        part = sign * float(f.empty) if n == 0 else part_of(n)
        return part + sign * f._count(n)

    lo, hi = issue.min_count, issue.effective_max_count
    if distinct:
        hi = min(hi, len(ranked))
    # custom aggregates rebuild each list, so they scan fewer counts
    scan = _MAX_COUNT_SCAN if builtin is not None else _MAX_CUSTOM_COUNT_SCAN
    if math.isfinite(hi) and hi - lo <= scan:
        counts: Iterable[int] = range(lo, int(hi) + 1)
    elif math.isfinite(hi):
        counts = _wide_counts(f, issue, lo, int(hi))
    else:
        counts = _unbounded_counts(f, issue, ranked[0][0], lo, sign)
    best_n = max(counts, key=value)
    raw = sign * value(best_n)
    if distinct:
        elements = [e for _, e in ranked[:best_n]]
    else:
        elements = _repeat(issue, ranked[0][1], best_n)
        if issue.unique_by is not None and best_n > 1:
            # nudged copies need not score like the best element: report the
            # value of the list actually returned
            scores = [scorer.score(e) for e in elements]
            raw = f._agg(scores) + f._count(len(scores))
    return raw, issue.canonicalize(elements)


def _wide_counts(f, issue: ListIssue, lo: int, hi: int) -> list[int]:
    """Counts to try for a bounded but very wide count range.

    With a built-in aggregate and an affine count_fun the value is linear in
    the count (for repeated elements), so the ends suffice.
    """
    affine = f.count_fun is None or _count_slope(f.count_fun) is not None
    if not isinstance(f.aggregate, str) or not affine:
        raise ValueError(
            f"{issue.name}: {hi - lo + 1} possible counts are too many to scan with "
            f"a custom aggregate or a non-affine count_fun; lower max_count"
        )
    return sorted({lo, max(lo, 1), hi})


def _unbounded_counts(f, issue: ListIssue, top: float, lo: int, sign: float):
    """Counts to try when the count is unbounded; raises if no extremum exists."""
    what = "maximum" if sign > 0 else "minimum"
    if not isinstance(f.aggregate, str):
        raise UnboundedListUtilityError(
            f"{issue.name}: a custom aggregate over unboundedly many elements has "
            f"no known {what}; set max_count"
        )
    slope = 0.0 if f.count_fun is None else _count_slope(f.count_fun)
    if slope is None:
        raise UnboundedListUtilityError(
            f"{issue.name}: a non-affine count_fun over unboundedly many elements "
            f"has no known {what}; set max_count"
        )
    growth = sign * slope + (top if f.aggregate == "sum" else 0.0)
    if growth > 0:
        hint = (
            "set max_count or use aggregate='mean'"
            if f.aggregate == "sum"
            else "set max_count"
        )
        raise UnboundedListUtilityError(
            f"{issue.name}: the {what} is unbounded (more elements always change "
            f"it by {sign * growth:+g} each); {hint}"
        )
    return sorted({lo, max(lo, 1)})


def _repeat(issue: ListIssue, e: Any, n: int) -> list:
    """`n` copies of `e`, with nudged keys if `unique_by` requires distinct ones."""
    if issue.unique_by is None or n <= 1:
        return [e] * n
    out, keys = [e], {issue.key_of(e)}
    for j in range(1, n * 50):
        if len(out) >= n:
            break
        c = _nudge(issue, e, j)
        k = issue.key_of(c)
        if k in keys or not issue.element.is_valid(c):
            continue
        keys.add(k)
        out.append(c)
    return out


def _nudge(issue: ListIssue, e: Any, j: int) -> Any:
    """A copy of `e` with its key fields moved a little (the j-th copy)."""
    element = issue.element
    is_record = isinstance(element, RecordIssue)
    fields = element.fields if is_record else (element,)  # type: ignore[attr-defined]
    v = list(e) if is_record else [e]
    for i in issue._key_idx or (0,):
        fi, x = fields[i], v[i]
        if fi.is_numeric() and isinstance(x, (int, float)):
            lo = fi.min_value if fi.min_value is not None else float("-inf")
            hi = fi.max_value if fi.max_value is not None else float("inf")
            step = 1 if fi.is_integer() else max(abs(x), 1.0) * 1e-9
            up, down = x + j * step, x - j * step
            v[i] = up if up <= hi else down if down >= lo else fi.rand()
        else:
            v[i] = fi.rand()
    return element.canonicalize(tuple(v)) if is_record else v[0]  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
# ListAggregationMultiFun
# ----------------------------------------------------------------------------


@define(frozen=True)
class ListAggregationMultiFun(BaseMultiFun):
    r"""A multi-issue factor over a list issue and other issues.

    Scores :math:`b + s(\operatorname{agg}_{e \in L} g(x_{e}) + c(|L|))` where
    :math:`x_e` is the factor's values with the list replaced by element `e`.

    Args:
        element_fun: A `BaseMultiFun` or callable over the factor's values with
            one element in place of the list.
        list_index: The position of the list issue among the factor's issues.
        aggregate, empty, count_fun, scale, bias: As in `ListAggregationFun`.
        flatten: Splice a record element's fields into the tuple instead of
            passing the record as one value.

    Examples:
        >>> from negmas.outcomes import ListIssue, make_issue
        >>> from negmas.preferences.value_fun import LinearMultiFun
        >>> issues = (
        ...     make_issue((0, 2), "discount"),
        ...     ListIssue("q", make_issue((1, 3), "q"), max_count=2),
        ... )
        >>> f = ListAggregationMultiFun(LinearMultiFun((1.0, 1.0)), list_index=1)
        >>> f((1, (2, 3)))  # (1 + 2) + (1 + 3)
        7.0
        >>> f.minmax(issues)
        (0.0, 10.0)
    """

    element_fun: Any
    list_index: int = -1
    aggregate: str | Callable[[list[float]], float] = "sum"
    empty: float = 0.0
    count_fun: Any = None
    scale: float = 1.0
    bias: float = 0.0
    flatten: bool = False

    def __attrs_post_init__(self):
        object.__setattr__(self, "element_fun", _as_scorer(self.element_fun))
        object.__setattr__(self, "count_fun", _as_fun(self.count_fun))
        if isinstance(self.aggregate, str):
            _builtin_aggregate(self.aggregate)
        elif not callable(self.aggregate):
            raise TypeError(f"aggregate must be a name or callable: {self.aggregate!r}")

    @property
    def dim(self) -> int:
        raise ValueError("ListAggregationMultiFun has no fixed dimension")

    def _args(self, x: Sequence, e: Any) -> tuple:
        i = self.list_index % len(x)
        e = tuple(e) if self.flatten and isinstance(e, (tuple, list)) else (e,)
        return tuple(x[:i]) + e + tuple(x[i + 1 :])

    def _element_scorer(self, x: Sequence) -> Callable[[Any], float]:
        g = self.element_fun
        return lambda e: float(g(self._args(x, e)))

    _agg = ListAggregationFun._agg
    _count = ListAggregationFun._count
    _lifted = ListAggregationFun._lifted

    def __call__(self, x: tuple) -> float:
        i = self.list_index % len(x)
        score = self._element_scorer(x)
        return self._lifted([score(e) for e in x[i]])

    def _with(self, **kw) -> ListAggregationMultiFun:
        d = {
            k: getattr(self, k)
            for k in (
                "element_fun",
                "list_index",
                "aggregate",
                "empty",
                "count_fun",
                "scale",
                "bias",
                "flatten",
            )
        }
        d.update(kw)
        return ListAggregationMultiFun(**d)

    def shift_by(self, offset: float) -> ListAggregationMultiFun:
        return self._with(bias=self.bias + offset)

    def scale_by(self, scale: float) -> ListAggregationMultiFun:
        return self._with(scale=self.scale * scale, bias=self.bias * scale)

    def xml(self, indx, issues, bias: float = 0) -> str:
        raise NotImplementedError("Genius has no list issues")

    def minmax(self, input: Iterable[Issue]) -> tuple[float, float]:
        (mn, _), (mx, _) = self.arg_extremes(input)
        return mn, mx

    def arg_extremes(self, issues: Iterable[Issue]):
        """((min, worst values), (max, best values)) in factor order.

        Exact when the other issues are small and finite (enumerated);
        sampled otherwise (warns).
        """
        issues = tuple(issues)
        i = self.list_index % len(issues)
        list_issue = issues[i]
        if not isinstance(list_issue, ListIssue):
            raise TypeError(f"Issue {i} of the factor is not a ListIssue: {list_issue}")
        context = [_ for j, _ in enumerate(issues) if j != i]
        if any(isinstance(_, ListIssue) for _ in context):
            raise ValueError(
                "A factor over two list issues is not supported; use a ufun over "
                "the whole outcome"
            )
        combos = _context_combinations(context)
        worst = (float("inf"), None)
        best = (float("-inf"), None)
        for ctx in combos:
            x = list(ctx)
            x.insert(i, ())
            inner = ListAggregationFun(
                self._element_scorer(tuple(x)),
                self.aggregate,
                self.empty,
                self.count_fun,
                scale=self.scale,
                bias=self.bias,
            )
            (mn, wl), (mx, bl) = inner.arg_extremes(list_issue)
            if mn < worst[0]:
                worst = (mn, tuple(x[:i]) + (wl,) + tuple(x[i + 1 :]))
            if mx > best[0]:
                best = (mx, tuple(x[:i]) + (bl,) + tuple(x[i + 1 :]))
        return worst, best

    def to_dict(self, python_class_identifier=PYTHON_CLASS_IDENTIFIER) -> dict:
        return _agg_to_dict(self, python_class_identifier)

    @classmethod
    def from_dict(cls, d: dict, python_class_identifier=PYTHON_CLASS_IDENTIFIER):
        if isinstance(d, cls):
            return d
        return cls(**_deserialize_fields(d, python_class_identifier))


def _context_combinations(context: Sequence[Issue]) -> list[tuple]:
    """All combinations of the context issues' values, or a sample of them."""
    cands = [_candidates(_, _MAX_CONTEXTS) for _ in context]
    exact = all(_.is_discrete() and _.cardinality <= _MAX_ENUMERATION for _ in context)
    n = math.prod(len(_) for _ in cands)
    if exact and n <= _MAX_ENUMERATION:
        return list(product(*cands))
    warnings.warn(
        "ListAggregationMultiFun: the other issues are sampled; extremes are "
        "approximate",
        warnings.NegmasSpeedWarning,
    )
    if n <= _MAX_CONTEXTS:
        return list(product(*cands))
    return [tuple(random.choice(c) for c in cands) for _ in range(_MAX_CONTEXTS)]


# ----------------------------------------------------------------------------
# lifting a one-element ufun
# ----------------------------------------------------------------------------


class _RecordFactor:
    """Picks a factor's values from (context..., record): a picklable closure."""

    def __init__(self, fun, slots: Sequence[tuple[bool, int]], weight: float):
        self.fun, self.slots, self.weight = fun, tuple(slots), weight

    def __call__(self, ctx: Sequence, record: Sequence) -> float:
        vals = [record[j] if from_record else ctx[j] for from_record, j in self.slots]
        return self.weight * float(self.fun(vals[0] if len(vals) == 1 else tuple(vals)))


class _SumOfRecordFactors:
    """Element score of a lifted ufun: the sum of its record-touching factors."""

    def __init__(self, factors: Sequence[_RecordFactor], n_context: int):
        self.factors, self.n_context = tuple(factors), n_context

    def __call__(self, x: Sequence) -> float:
        ctx, record = x[: self.n_context], x[self.n_context]
        if not isinstance(record, (tuple, list)):
            record = (record,)
        return sum(f(ctx, record) for f in self.factors)


class _ElementOnly:
    """Adapts an element scorer over (record,) to one over the record."""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, e) -> float:
        return self.inner((e,))


def lift_element_ufun(
    ufun,
    list_issue: ListIssue,
    aggregate: str | Callable[[list[float]], float] = "sum",
    empty: float = 0.0,
    count_fun: Any = None,
):
    """Lifts a ufun over plain issues plus one copy of the element's fields.

    The fields are matched to the ufun's issues by name (the element's name
    for a plain element). Factors over plain issues only are kept; all factors
    touching a field become one lifted factor, so any aggregate keeps its
    meaning.

    Args:
        ufun: A `LinearAdditiveUtilityFunction` or `GLAUtilityFunction`.
        list_issue: The list issue that replaces the fields.
        aggregate, empty, count_fun: As in `ListAggregationFun`.

    Returns:
        A `GLAUtilityFunction` over the plain issues followed by `list_issue`.

    Examples:
        >>> from negmas.outcomes import ListIssue, make_issue, make_os
        >>> from negmas.preferences import LinearAdditiveUtilityFunction
        >>> from negmas.preferences.value_fun import AffineFun
        >>> flat = make_os([make_issue((0, 3), "speed"), make_issue((1, 5), "q")])
        >>> one = LinearAdditiveUtilityFunction(
        ...     [AffineFun(1.0), AffineFun(2.0)], outcome_space=flat
        ... )
        >>> u = lift_element_ufun(
        ...     one, ListIssue("lots", make_issue((1, 5), "q"), max_count=3)
        ... )
        >>> u((2, (1, 5)))  # 2 + 2*1 + 2*5
        14.0
    """
    from negmas.outcomes import make_os
    from negmas.preferences.crisp.gla import GLAUtilityFunction
    from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction

    issues = list(ufun.outcome_space.issues)
    names = [_.name for _ in issues]
    field_names = list(list_issue.field_names)
    missing = [n for n in field_names if n not in names]
    if missing:
        raise ValueError(f"The ufun has no issues named {missing}")
    plain = [i for i, n in enumerate(names) if n not in field_names]
    plain_pos = {i: k for k, i in enumerate(plain)}
    field_pos = {names.index(n): k for k, n in enumerate(field_names)}

    if isinstance(ufun, LinearAdditiveUtilityFunction):
        factors = [
            ((i,), f, w) for i, (f, w) in enumerate(zip(ufun.values, ufun.weights))
        ]
        bias = ufun._bias
    elif isinstance(ufun, GLAUtilityFunction):
        factors = [(idx, f, w) for (idx, f), w in zip(ufun.factors, ufun.weights)]
        bias = ufun.bias
    else:
        raise TypeError(f"Cannot lift a {type(ufun).__name__}")

    kept, lifted, context = [], [], []
    for idx, f, w in factors:
        if all(i in plain_pos for i in idx):
            kept.append((tuple(plain_pos[i] for i in idx), f, w))
            continue
        for i in idx:
            if i in plain_pos and i not in context:
                context.append(i)
        lifted.append((idx, f, w))
    ctx_pos = {i: k for k, i in enumerate(context)}
    element_fun = _SumOfRecordFactors(
        [
            _RecordFactor(
                f,
                [
                    (True, field_pos[i]) if i in field_pos else (False, ctx_pos[i])
                    for i in idx
                ],
                w,
            )
            for idx, f, w in lifted
        ],
        len(context),
    )
    list_pos = len(plain)
    if not lifted:
        # no factor reads the fields: the list still scores its count/empty
        kept.append(
            (
                (list_pos,),
                ListAggregationFun(ConstFun(0.0), aggregate, empty, count_fun),
                1.0,
            )
        )
    elif not context:
        kept.append(
            (
                (list_pos,),
                ListAggregationFun(
                    _ElementOnly(element_fun), aggregate, empty, count_fun
                ),
                1.0,
            )
        )
    elif lifted:
        lifted_fun = ListAggregationMultiFun(
            element_fun,
            list_index=len(context),
            aggregate=aggregate,
            empty=empty,
            count_fun=count_fun,
        )
        kept.append(
            (tuple(plain_pos[i] for i in context) + (list_pos,), lifted_fun, 1.0)
        )
    os = make_os([issues[i] for i in plain] + [list_issue])
    return GLAUtilityFunction(
        factors=kept,
        bias=bias,
        outcome_space=os,
        reserved_value=ufun.reserved_value,
        name=ufun.name,
    )


def list_issue_candidates(issue: Issue, funs: Iterable[Any] = (), n: int = 200) -> list:
    """Values of a list issue for per-issue searches: `n` samples plus the
    greedy extremes of every fun in `funs` that has `arg_extremes`."""
    vals = list(issue.value_generator(n))
    for f in funs:
        ext = getattr(f, "arg_extremes", None)
        if ext is None:
            continue
        try:
            (_, worst), (_, best) = ext(issue)
        except ValueError:
            continue
        vals += [worst, best]
    return list(dict.fromkeys(vals))
