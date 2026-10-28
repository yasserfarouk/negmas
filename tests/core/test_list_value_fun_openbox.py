"""Open-box tests for list value functions and their minmax (chunk 2).

Written from the code of `negmas.preferences.list_value_fun` and the hooks in
`crisp/linear.py` and `crisp/gla.py`. Tests marked
``xfail(strict=True, reason="DEFECT: ...")`` document defects.
"""

from __future__ import annotations

import json
import math
import pickle
import warnings

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from negmas.outcomes import ListIssue, RecordIssue, make_issue, make_os
from negmas.outcomes.list_issue import DiscreteListValuesIssue
from negmas.preferences import (
    GLAUtilityFunction,
    LinearAdditiveUtilityFunction,
    ListAggregationFun,
    ListAggregationMultiFun,
    RecordAdditiveFun,
    UnboundedListUtilityError,
    lift_element_ufun,
)
from negmas.preferences.value_fun import (
    AffineFun,
    ConstFun,
    IdentityFun,
    LambdaFun,
    LambdaMultiFun,
    LinearFun,
    LinearMultiFun,
    QuadraticFun,
    TableFun,
)

TOL = 1e-9


def close(a, b, tol=TOL):
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


def brute(f, issue):
    vals = [f(v) for v in issue.all]
    return min(vals), max(vals)


def brute_ufun(u, os):
    vals = [u(o) for o in os.enumerate()]
    return min(vals), max(vals)


def rec2():
    return RecordIssue([make_issue(["a", "b"], "k"), make_issue((0, 2), "q")])


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------- boundaries


def test_empty_only_issue_uses_empty_and_count_zero():
    """With max_count=0 the only list is () and its value is empty + c(0)."""
    f = ListAggregationFun(IdentityFun(), empty=2.0, count_fun=AffineFun(1.0, bias=1))
    (mn, worst), (mx, best) = f.arg_extremes(ListIssue("L", make_issue(4, "x"), 0, 0))
    assert worst == best == () and mn == mx == 3.0 == f(())


def test_fixed_count_with_unique_elements():
    """min_count == max_count forces exactly that many distinct elements."""
    issue = ListIssue("L", make_issue((0, 3), "x"), 2, 2, unique_by=True)
    f = ListAggregationFun(IdentityFun())
    (mn, worst), (mx, best) = f.arg_extremes(issue)
    assert (mn, mx) == brute(f, issue) == (1.0, 5.0)
    assert issue.is_valid(worst) and issue.is_valid(best)


def test_none_list_scores_nan():
    """A missing list value (None) scores NaN rather than raising."""
    assert math.isnan(ListAggregationFun(IdentityFun())(None))


def test_minmax_on_a_non_list_issue_raises_type_error():
    """arg_extremes refuses anything but a ListIssue with a TypeError."""
    with pytest.raises(TypeError):
        ListAggregationFun(IdentityFun()).minmax(make_issue(3, "x"))


def test_non_callable_element_fun_raises():
    """element_fun must be a fun, a ufun or a callable."""
    with pytest.raises(TypeError):
        ListAggregationFun(3)


def test_multi_fun_rejects_a_non_callable_aggregate():
    """ListAggregationMultiFun validates aggregate like ListAggregationFun."""
    with pytest.raises(TypeError):
        ListAggregationMultiFun(IdentityFun(), aggregate=3)


# ---------------------------------------------------------------- greedy optimality


def _monotone_custom(v):
    return sum(v) + max(v)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    w=st.lists(st.floats(-3, 3), min_size=2, max_size=2),
    table=st.lists(st.floats(-2, 2), min_size=2, max_size=2),
    bias=st.floats(-2, 2),
    aggregate=st.sampled_from(["sum", "mean", "min", "max", "custom"]),
    min_count=st.integers(0, 2),
    extra=st.integers(0, 2),
    unique=st.sampled_from([None, ("k",), ("k", "q"), ("q",)]),
    ordered=st.booleans(),
    count=st.sampled_from(["none", "linear", "quadratic"]),
    scale=st.sampled_from([1.0, -0.5, 2.0]),
)
def test_property_additive_greedy_matches_brute_force(
    w, table, bias, aggregate, min_count, extra, unique, ordered, count, scale
):
    """Greedy arg_extremes equal brute force, and their lists attain the values."""
    try:
        issue = ListIssue(
            "L",
            rec2(),
            min_count=min_count,
            max_count=min_count + extra,
            unique_by=unique,
            ordered=ordered,
        )
    except ValueError:
        assume(False)
    count_fun = {
        "none": None,
        "linear": LinearFun(-0.7),
        "quadratic": QuadraticFun(a2=-0.6, a1=1.1),
    }[count]
    f = ListAggregationFun(
        RecordAdditiveFun(
            (TableFun(dict(zip("ab", table))), AffineFun(1.0)), weights=w, bias=bias
        ),
        aggregate=_monotone_custom if aggregate == "custom" else aggregate,
        count_fun=count_fun,
        empty=0.25,
    ).scale_by(scale)
    (mn, worst), (mx, best) = f.arg_extremes(issue)
    assert close((mn, mx), brute(f, issue))
    assert issue.is_valid(worst) and issue.is_valid(best)
    assert math.isclose(f(worst), mn, abs_tol=TOL)
    assert math.isclose(f(best), mx, abs_tol=TOL)


def test_unique_whole_records():
    """unique_by=True on a record keeps whole records distinct and is exact."""
    r = RecordIssue([make_issue((0, 1), "a"), make_issue((0, 1), "b")])
    issue = ListIssue("L", r, unique_by=True)
    f = ListAggregationFun(RecordAdditiveFun((AffineFun(1.0), AffineFun(-2.0))))
    assert close(f.minmax(issue), brute(f, issue))


def test_continuous_field_affine_fun_is_exact():
    """Affine funs on continuous fields reach the field endpoints exactly."""
    r = RecordIssue([make_issue((0.0, 2.0), "p"), make_issue(["a", "b"], "k")])
    issue = ListIssue("L", r, max_count=2, unique_by=("k",))
    f = ListAggregationFun(
        RecordAdditiveFun((AffineFun(-1.0), TableFun({"a": 1.0, "b": 3.0})))
    )
    assert close(f.minmax(issue), (-1.0, 4.0))


def test_continuous_unique_key_lists_are_valid_and_near_the_supremum():
    """With a continuous unique key, the best list is valid and near-optimal."""
    r = RecordIssue([make_issue((0.0, 2.0), "p"), make_issue(["a", "b"], "k")])
    issue = ListIssue("L", r, max_count=3, unique_by=("p",))
    f = ListAggregationFun(
        RecordAdditiveFun((AffineFun(-1.0), TableFun({"a": 1.0, "b": 3.0})))
    )
    (mn, worst), (mx, best) = f.arg_extremes(issue)
    assert issue.is_valid(worst) and issue.is_valid(best)
    assert math.isclose(f(best), mx, abs_tol=1e-6)
    assert math.isclose(f(worst), mn, abs_tol=1e-6)


def test_too_many_keys_reports_an_attainable_maximum():
    """arg_extremes' max is attained by its list even when keys are sampled."""
    el = RecordIssue([make_issue((0, 10**6), "k"), make_issue((0, 3), "q")])
    issue = ListIssue("L", el, max_count=3, unique_by=("k",))
    f = ListAggregationFun(RecordAdditiveFun((AffineFun(-1.0), AffineFun(1.0))))
    _, (mx, best) = f.arg_extremes(issue)
    assert math.isclose(f(best), mx, abs_tol=1e-6)


# ---------------------------------------------------------------- bounded counts


def test_large_finite_max_count_is_bounded():
    """sum over up to 2e6 elements of [1, 5] has max 1e7, not 'unbounded'."""
    issue = ListIssue("L", make_issue((1, 5), "x"), max_count=2_000_000)
    assert ListAggregationFun(IdentityFun()).minmax(issue) == (0.0, 10_000_000.0)


def test_large_unique_key_cap_is_bounded():
    """A count capped by 1.5e6+1 unique keys is bounded for sum."""
    issue = ListIssue("L", make_issue((0, 1_500_000), "k"), unique_by=True)
    assert ListAggregationFun(ConstFun(1.0)).minmax(issue) == (0.0, 1_500_001.0)


# ---------------------------------------------------------------- unbounded rules


def test_unbounded_min_max_aggregates_are_exact():
    """min/max aggregates are exact without max_count."""
    x = make_issue((0, 3), "x")
    for agg in ("min", "max"):
        f = ListAggregationFun(IdentityFun(), aggregate=agg, empty=-1.0)
        assert f.minmax(ListIssue("L", x)) == (-1.0, 3.0)
        assert f.minmax(ListIssue("L", x, min_count=2)) == (0.0, 3.0)


def test_unbounded_sum_of_zero_elements_is_exact():
    """Elements that add nothing make an unbounded sum exact."""
    f = ListAggregationFun(ConstFun(0.0))
    assert f.minmax(ListIssue("L", make_issue(3, "x"))) == (0.0, 0.0)


def test_unbounded_sum_of_mixed_sign_elements_raises():
    """An unbounded sum with positive and negative elements has no extrema."""
    with pytest.raises(UnboundedListUtilityError):
        ListAggregationFun(AffineFun(1.0, bias=-1.0)).minmax(
            ListIssue("L", make_issue((0, 3), "x"))
        )


def test_identity_and_linear_count_funs_are_treated_as_affine():
    """IdentityFun / LinearFun count terms are exact when they balance the sum."""
    x = make_issue((0, 3), "x")
    f = ListAggregationFun(ConstFun(-1.0), count_fun=IdentityFun())
    assert f.minmax(ListIssue("L", x)) == (0.0, 0.0)
    g = ListAggregationFun(ConstFun(2.0), count_fun=LinearFun(-2.0))
    assert g.minmax(ListIssue("L", x)) == (0.0, 0.0)


# ---------------------------------------------------------------- element scorers


def test_ufun_element_scores_json_list_records():
    """A ufun element scorer gives the same value for list and tuple records."""
    r = RecordIssue([make_issue((0, 3), "a"), make_issue((0, 3), "b")])
    uf = LinearAdditiveUtilityFunction(
        [AffineFun(1.0), AffineFun(2.0)], outcome_space=make_os(r.fields)
    )
    f = ListAggregationFun(uf)
    assert f([[1, 2], [3, 0]]) == f(((1, 2), (3, 0))) == 8.0


def test_record_additive_element_scores_json_list_records():
    """RecordAdditiveFun accepts list-form records (the JSON shape)."""
    f = ListAggregationFun(RecordAdditiveFun((AffineFun(1.0), AffineFun(2.0))))
    assert f([[1, 2], [3, 0]]) == 8.0


def test_record_additive_fun_mapping_defaults():
    """Mapped field funs default to 0 and mapped weights to 1."""
    f = RecordAdditiveFun(
        {"q": AffineFun(1.0)}, weights={"k": 5.0}, fields=("k", "q"), bias=1.0
    )
    assert f(("a", 2)) == 3.0
    assert f.minmax(rec2()) == (1.0, 3.0)


def test_record_additive_fun_rejects_unknown_names():
    """Unknown field names in funs or weights raise ValueError."""
    with pytest.raises(ValueError):
        RecordAdditiveFun({"zz": AffineFun(1.0)}, fields=("k", "q"))
    with pytest.raises(ValueError):
        RecordAdditiveFun((AffineFun(1.0), AffineFun(1.0)), weights={"k": 1.0})
    with pytest.raises(ValueError):
        RecordAdditiveFun((AffineFun(1.0),), weights=(1.0, 2.0))


def test_record_additive_shift_and_scale():
    """shift_by adds to every value; scale_by multiplies every value."""
    f = RecordAdditiveFun((TableFun({"a": 1.0, "b": 2.0}), AffineFun(3.0)), bias=1.0)
    for v in rec2().all:
        assert math.isclose(f.shift_by(2.0)(v), f(v) + 2.0)
        assert math.isclose(f.scale_by(-3.0)(v), -3.0 * f(v))


def test_plain_element_ufun_scorer_is_exact():
    """A ufun over a plain element's one issue scores (e,) and is exact."""
    x = make_issue((0, 4), "x")
    uf = LinearAdditiveUtilityFunction(
        [AffineFun(-1.0, bias=2.0)], outcome_space=make_os([x])
    )
    issue = ListIssue("L", x, max_count=3, unique_by=True)
    f = ListAggregationFun(uf)
    assert close(f.minmax(issue), brute(f, issue))


def test_callable_element_on_a_large_record_is_valid_and_bounded():
    """A callable over a record too big to enumerate gives valid, bounded lists."""
    r = RecordIssue([make_issue((0, 999), "a"), make_issue((0, 999), "b")])
    issue = ListIssue("L", r, max_count=2)
    f = ListAggregationFun(lambda e: -((e[0] - 500) ** 2) - (e[1] - 10) ** 2)
    (mn, worst), (mx, best) = f.arg_extremes(issue)
    assert issue.is_valid(worst) and issue.is_valid(best)
    assert math.isclose(f(best), mx) and math.isclose(f(worst), mn)
    assert mx <= 0.0 and mn <= f(((0, 0),))


# ---------------------------------------------------------------- ufun hooks


def small_list_os(max_count=2):
    lots = ListIssue("lots", make_issue((0, 3), "x"), max_count=max_count)
    return make_os([make_issue((0, 1), "s"), lots])


@pytest.mark.parametrize("w", [1.0, -1.0])
def test_linear_additive_negative_list_weight(w):
    """extreme_outcomes honours the sign of the list issue's weight."""
    os = small_list_os()
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0), ListAggregationFun(IdentityFun())],
        weights=[1.0, w],
        outcome_space=os,
    )
    worst, best = u.extreme_outcomes()
    assert close((u(worst), u(best)), brute_ufun(u, os))
    assert close(u.minmax(), brute_ufun(u, os))


@pytest.mark.parametrize("w", [1.0, -1.0])
def test_gla_negative_list_weight(w):
    """GLA's separable extreme_outcomes honours negative factor weights."""
    os = small_list_os()
    u = GLAUtilityFunction(
        factors=[((0,), AffineFun(1.0)), ((1,), ListAggregationFun(IdentityFun()))],
        weights=[1.0, w],
        outcome_space=os,
    )
    worst, best = u.extreme_outcomes()
    assert close((u(worst), u(best)), brute_ufun(u, os))


def test_gla_separable_with_uncovered_issue_and_plain_multi_factor():
    """Issues no factor uses get any value; plain multi-issue factors are searched."""
    lots = ListIssue("lots", make_issue((0, 3), "x"), max_count=1)
    os = make_os(
        [make_issue((0, 2), "a"), make_issue((0, 2), "b"), lots, make_issue(3, "z")]
    )
    u = GLAUtilityFunction(
        factors=[
            ((0, 1), LambdaMultiFun(lambda v: -((v[0] - 1) ** 2) - (v[1] - 1) ** 2)),
            ((2,), ListAggregationFun(IdentityFun(), aggregate="mean")),
        ],
        outcome_space=os,
    )
    worst, best = u.extreme_outcomes()
    assert os.is_valid(worst) and os.is_valid(best)
    assert close((u(worst), u(best)), brute_ufun(u, os))


def test_gla_overlapping_factors_fall_back_and_stay_valid():
    """Non-disjoint factors use the generic search: valid, within the true range."""
    os = small_list_os(max_count=2)
    u = GLAUtilityFunction(
        factors=[
            ((0,), AffineFun(1.0)),
            ((0, 1), ListAggregationMultiFun(LinearMultiFun((1.0, 1.0)), list_index=1)),
        ],
        outcome_space=os,
    )
    lo, hi = brute_ufun(u, os)
    worst, best = u.extreme_outcomes()
    assert os.is_valid(worst) and os.is_valid(best)
    assert lo - TOL <= u(worst) <= u(best) <= hi + TOL


@settings(max_examples=40, deadline=None)
@given(
    a=st.floats(-2, 2),
    b=st.floats(-2, 2),
    aggregate=st.sampled_from(["sum", "mean", "min", "max"]),
    flatten=st.booleans(),
    count=st.booleans(),
)
def test_property_multi_fun_in_gla_matches_brute_force(a, b, aggregate, flatten, count):
    """A ListAggregationMultiFun factor's GLA minmax equals brute force."""
    r = RecordIssue([make_issue((0, 2), "q"), make_issue(["x", "y"], "t")])
    lots = ListIssue("lots", r, max_count=2, unique_by=("t",))
    os = make_os([make_issue((0, 2), "d"), lots])
    if flatten:
        g = LambdaMultiFun(lambda v: a * v[0] * v[1] + b * (v[2] == "x"))
    else:
        g = LambdaMultiFun(lambda v: a * v[0] * v[1][0] + b * (v[1][1] == "x"))
    f = ListAggregationMultiFun(
        g,
        list_index=1,
        aggregate=aggregate,
        flatten=flatten,
        count_fun=AffineFun(-0.3) if count else None,
    )
    u = GLAUtilityFunction(factors=[((0, 1), f)], outcome_space=os)
    assert close(u.minmax(), brute_ufun(u, os))
    worst, best = u.extreme_outcomes()
    assert close((u(worst), u(best)), brute_ufun(u, os))


@pytest.mark.parametrize("kind", ["linear", "gla"])
def test_normalize_maps_the_true_range_to_unit(kind):
    """normalize() maps the brute-force range of a list ufun to [0, 1]."""
    r = rec2()
    lots = ListIssue("lots", r, max_count=3, unique_by=("k",))
    os = make_os([make_issue((0, 2), "s"), lots])
    lf = ListAggregationFun(
        RecordAdditiveFun((TableFun({"a": -1.0, "b": 2.0}), AffineFun(1.5))),
        count_fun=AffineFun(-0.5),
    )
    if kind == "linear":
        u = LinearAdditiveUtilityFunction(
            [AffineFun(2.0), lf], weights=[1.0, 0.5], outcome_space=os
        )
    else:
        u = GLAUtilityFunction(
            factors=[((0,), AffineFun(2.0)), ((1,), lf)],
            weights=[1.0, 0.5],
            outcome_space=os,
        )
    n = u.normalize()
    assert close(brute_ufun(n, os), (0.0, 1.0), tol=1e-7)


def test_linear_additive_extremes_on_a_discretized_space():
    """extreme_outcomes works on os.to_discrete(), as for other issues."""
    os = make_os([make_issue((0, 1), "s"), ListIssue("free", make_issue((0, 3), "x"))])
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0), ListAggregationFun(IdentityFun(), aggregate="mean")],
        outcome_space=os,
    )
    dos = os.to_discrete(20)
    assert isinstance(dos.issues[1], DiscreteListValuesIssue)
    worst, best = u.extreme_outcomes(outcome_space=dos)
    assert close((u(worst), u(best)), brute_ufun(u, dos))


def test_linear_additive_normalize_for_a_discretized_space():
    """normalize_for(os.to_discrete()) maps that space's range to [0, 1]."""
    os = make_os([make_issue((0, 1), "s"), ListIssue("free", make_issue((0, 3), "x"))])
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0), ListAggregationFun(IdentityFun(), aggregate="mean")],
        outcome_space=os,
    )
    dos = os.to_discrete(20)
    n = u.normalize_for(outcome_space=dos)
    assert close(brute_ufun(n, dos), (0.0, 1.0), tol=1e-7)


def test_gla_extremes_on_a_discretized_space():
    """GLA (no ListIssue left) falls back to the generic search and works."""
    os = make_os([make_issue((0, 1), "s"), ListIssue("free", make_issue((0, 3), "x"))])
    u = GLAUtilityFunction(
        factors=[
            ((0,), AffineFun(1.0)),
            ((1,), ListAggregationFun(IdentityFun(), aggregate="mean")),
        ],
        outcome_space=os,
    )
    dos = os.to_discrete(20)
    worst, best = u.extreme_outcomes(outcome_space=dos)
    assert close((u(worst), u(best)), brute_ufun(u, dos))


# ---------------------------------------------------------------- no enumeration


@pytest.fixture
def no_list_enumeration(monkeypatch):
    """Makes enumerating any infinite ListIssue fail loudly."""

    original = ListIssue.all

    def guarded(self):
        if not self.is_finite():
            raise AssertionError(f"enumerated the infinite list issue {self.name}")
        return original.fget(self)

    monkeypatch.setattr(ListIssue, "all", property(guarded))
    original_gen = ListIssue.ordered_value_generator

    def guarded_gen(self, *args, **kwargs):
        if not self.is_finite():
            raise AssertionError(f"enumerated the infinite list issue {self.name}")
        return original_gen(self, *args, **kwargs)

    monkeypatch.setattr(ListIssue, "ordered_value_generator", guarded_gen)


def infinite_os():
    return make_os(
        [make_issue((0, 2), "s"), ListIssue("free", rec2(), typical_count=2)]
    )


def mean_fun():
    return ListAggregationFun(
        RecordAdditiveFun((TableFun({"a": 0.0, "b": 1.0}), AffineFun(1.0))),
        aggregate="mean",
    )


def test_linear_on_infinite_list_never_enumerates(no_list_enumeration):
    """Linear minmax / extreme_outcomes / normalize never enumerate the list."""
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0), mean_fun()], outcome_space=infinite_os()
    )
    assert close(u.minmax(), (0.0, 5.0))
    worst, best = u.extreme_outcomes()
    assert close((u(worst), u(best)), (0.0, 5.0))
    u.normalize()


def test_gla_on_infinite_list_never_enumerates(no_list_enumeration):
    """GLA minmax / extreme_outcomes / normalize never enumerate the list."""
    u = GLAUtilityFunction(
        factors=[((0,), AffineFun(1.0)), ((1,), mean_fun())],
        outcome_space=infinite_os(),
    )
    assert close(u.minmax(), (0.0, 5.0))
    u.normalize()


def test_gla_overlapping_factors_on_infinite_list_never_enumerate(no_list_enumeration):
    """The non-separable GLA fallback samples an infinite list, not enumerates."""
    g = LambdaMultiFun(lambda v: v[0] * v[1][1])
    u = GLAUtilityFunction(
        factors=[
            ((0,), AffineFun(1.0)),
            ((0, 1), ListAggregationMultiFun(g, list_index=1, aggregate="mean")),
        ],
        outcome_space=infinite_os(),
    )
    worst, best = u.extreme_outcomes()
    assert u.outcome_space.is_valid(worst) and u.outcome_space.is_valid(best)


def test_multi_fun_with_continuous_context_never_enumerates(no_list_enumeration):
    """A continuous context issue is sampled; the result stays within bounds."""
    issues = (make_issue((0.0, 2.0), "d"), ListIssue("free", make_issue((1, 3), "q")))
    f = ListAggregationMultiFun(
        LinearMultiFun((1.0, 1.0)), list_index=1, aggregate="mean"
    )
    mn, mx = f.minmax(issues)
    assert mn == 0.0 and 4.9 <= mx <= 5.0 + 1e-6  # mn: the empty list


def test_level4_whole_list_fun_on_infinite_list_does_not_raise(no_list_enumeration):
    """A LambdaFun over the whole (infinite) list is sampled, never enumerated."""
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0), LambdaFun(lambda v: -abs(len(v) - 2))],
        outcome_space=infinite_os(),
    )
    mn, mx = u.minmax()
    assert mx == 2.0 and mn <= 0.0


# ---------------------------------------------------------------- lifting


def _one_element_ufun(kind, ws):
    flat = make_os(
        [make_issue((0, 2), "s"), make_issue(["a", "b"], "k"), make_issue((0, 2), "q")]
    )
    if kind == "linear":
        return flat, LinearAdditiveUtilityFunction(
            [AffineFun(ws[0]), TableFun({"a": ws[1], "b": -ws[1]}), AffineFun(ws[2])],
            bias=0.5,
            outcome_space=flat,
        )
    return flat, GLAUtilityFunction(
        factors=[
            ((0,), AffineFun(ws[0])),
            ((0, 2), LinearMultiFun((ws[1], -ws[1]))),
            ((1,), TableFun({"a": ws[2], "b": 0.0})),
        ],
        bias=0.5,
        outcome_space=flat,
    )


@settings(max_examples=40, deadline=None)
@given(
    kind=st.sampled_from(["linear", "gla"]),
    ws=st.lists(st.floats(-3, 3), min_size=3, max_size=3),
    n=st.integers(1, 3),
    data=st.data(),
)
def test_property_lift_is_sum_over_elements(kind, ws, n, data):
    """lifted(s, L) == sum_e one(s, e) - (n - 1) * (bias + plain part of s)."""
    flat, one = _one_element_ufun(kind, ws)
    lots = ListIssue("lots", RecordIssue([flat.issues[1], flat.issues[2]]), max_count=3)
    u = lift_element_ufun(one, lots)
    s = data.draw(st.integers(0, 2))
    elements = tuple(
        data.draw(st.tuples(st.sampled_from("ab"), st.integers(0, 2))) for _ in range(n)
    )
    plain = 0.5 + (ws[0] * s)
    expected = sum(one((s,) + e) for e in elements) - (n - 1) * plain
    assert math.isclose(u((s, elements)), expected, abs_tol=1e-9)
    if n == 1:
        assert math.isclose(u((s, elements)), one((s,) + elements[0]), abs_tol=1e-9)


@pytest.mark.parametrize("kind", ["linear", "gla"])
def test_lifted_ufun_minmax_matches_brute_force(kind):
    """The lifted ufun's minmax and extreme_outcomes equal brute force."""
    flat, one = _one_element_ufun(kind, [1.0, -2.0, 0.5])
    lots = ListIssue("lots", RecordIssue([flat.issues[1], flat.issues[2]]), max_count=2)
    u = lift_element_ufun(one, lots)
    assert close(u.minmax(), brute_ufun(u, u.outcome_space))
    worst, best = u.extreme_outcomes()
    assert close((u(worst), u(best)), brute_ufun(u, u.outcome_space))


def test_lifted_ufun_survives_json_and_pickle():
    """A lifted GLA ufun round-trips through to_dict/JSON and pickle."""
    flat, one = _one_element_ufun("gla", [1.0, -2.0, 0.5])
    lots = ListIssue("lots", RecordIssue([flat.issues[1], flat.issues[2]]), max_count=2)
    u = lift_element_ufun(one, lots)
    back = GLAUtilityFunction.from_dict(json.loads(json.dumps(u.to_dict())))
    pick = pickle.loads(pickle.dumps(u))
    for o in u.outcome_space.enumerate():
        assert math.isclose(back(o), u(o)) and math.isclose(pick(o), u(o))


def test_lift_rejects_other_ufun_types():
    """Only LinearAdditive and GLA ufuns can be lifted."""
    from negmas.preferences import MappingUtilityFunction

    os = make_os([make_issue(2, "q")])
    m = MappingUtilityFunction({(0,): 0.0, (1,): 1.0}, outcome_space=os)
    with pytest.raises(TypeError):
        lift_element_ufun(m, ListIssue("L", make_issue(2, "q"), max_count=2))


def test_lift_keeps_count_fun_without_field_factors():
    """count_fun still scores the count when no factor uses the fields."""
    flat = make_os([make_issue((0, 3), "s"), make_issue((0, 3), "q")])
    one = GLAUtilityFunction(factors=[((0,), AffineFun(1.0))], outcome_space=flat)
    u = lift_element_ufun(
        one,
        ListIssue("L", make_issue((0, 3), "q"), max_count=3),
        count_fun=AffineFun(-1.0),
    )
    assert u((0, (1, 2))) == -2.0


# ---------------------------------------------------------------- serialization


def test_list_fun_json_roundtrip_keeps_behaviour():
    """ListAggregationFun with mapped fields and count_fun survives JSON."""
    ra = RecordAdditiveFun(
        {"k": TableFun({"a": 1.0, "b": 2.0}), "q": AffineFun(0.5)}, weights={"q": 2.0}
    )
    f = ListAggregationFun(
        ra, aggregate="max", count_fun=AffineFun(-0.5), scale=2.0, bias=1
    )
    back = ListAggregationFun.from_dict(json.loads(json.dumps(f.to_dict())))
    issue = ListIssue("L", rec2(), max_count=2)
    for v in issue.all:
        assert math.isclose(back(v), f(v))
    assert close(back.minmax(issue), f.minmax(issue))


def test_multi_fun_json_roundtrip_keeps_behaviour():
    """ListAggregationMultiFun keeps list_index and flatten through JSON."""
    f = ListAggregationMultiFun(
        LinearMultiFun((1.0, 2.0, 3.0)), list_index=0, flatten=True, aggregate="min"
    )
    back = ListAggregationMultiFun.from_dict(json.loads(json.dumps(f.to_dict())))
    assert back.list_index == 0 and back.flatten is True
    x = (((1, 2), (0, 1)), 2)
    assert back(x) == f(x)


def test_record_additive_fun_roundtrip_is_equal():
    """from_dict(to_dict(f)) == f for RecordAdditiveFun and ListAggregationFun."""
    ra = RecordAdditiveFun((AffineFun(1.0), AffineFun(2.0)))
    assert RecordAdditiveFun.from_dict(ra.to_dict()) == ra
    la = ListAggregationFun(ra)
    assert ListAggregationFun.from_dict(la.to_dict()) == la


# ---------------------------------------------------------------- chunk-1 re-checks


def test_unnamed_record_gets_a_deterministic_name():
    """An unnamed RecordIssue is named record(<fields>)."""
    r = RecordIssue([make_issue(2, "a"), make_issue(2, "b")])
    assert r.name == "record(a,b)"
    assert RecordIssue([make_issue(2, "a"), make_issue(2, "b")]).name == r.name


def test_huge_finite_list_reports_infinite_cardinality():
    """A finite list with more than 1e10 values reports inf and not discrete."""
    issue = ListIssue("L", make_issue(1000, "x"), max_count=5)
    assert issue.cardinality == float("inf") and not issue.is_discrete()
    assert issue.is_valid(issue.rand())


def test_discretized_view_accepts_json_and_rejects_others():
    """The sampled view accepts its values as JSON lists and rejects others."""
    issue = ListIssue("L", rec2(), max_count=3, ordered=False)
    d = issue.to_discrete(5)
    v = next(iter(d.all))
    assert d.is_valid(json.loads(json.dumps(v)))
    assert not d.is_valid(42) and not d.is_valid([["zz", 9]])


def test_discrete_list_values_issue_equal_implies_same_hash():
    """Equal DiscreteListValuesIssues hash equally."""
    src = ListIssue("L", make_issue((0, 3), "x"), max_count=2)
    a = DiscreteListValuesIssue([(0,), (1,), (2,)], name="d", source=src)
    b = DiscreteListValuesIssue([(2,), (1,), (0,)], name="d", source=src)
    assert a == b and hash(a) == hash(b)


@pytest.mark.timeout(3)
def test_single_valued_element_with_large_max_count_is_fast():
    """Building a list of a one-value element with max_count=1e8 is quick."""
    issue = ListIssue("L", make_issue(["a"], "x"), max_count=10**8)
    assert issue.cardinality == 10**8 + 1
