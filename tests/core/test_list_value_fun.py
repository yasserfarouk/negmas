"""Tests for list value functions (RecordAdditiveFun, ListAggregationFun, ...)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from negmas.outcomes import ListIssue, RecordIssue, make_issue, make_os
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
    LambdaMultiFun,
    LinearFun,
    LinearMultiFun,
    QuadraticFun,
    TableFun,
    TableMultiFun,
)
from negmas.serialization import deserialize, serialize


def record():
    return RecordIssue(
        [make_issue(["a", "b", "c"], "kind"), make_issue((0, 3), "q")], name="r"
    )


def kind_q_fun():
    return RecordAdditiveFun(
        {"kind": TableFun({"a": 1.0, "b": -2.0, "c": 0.5}), "q": AffineFun(-1.0, 1.0)},
        fields=("kind", "q"),
    )


def brute(f, issue):
    vals = [f(v) for v in issue.all]
    return min(vals), max(vals)


def close(a, b):
    return all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b))


# ---------------------------------------------------------------- RecordAdditiveFun


def test_record_additive_fun_scores_fields():
    f = kind_q_fun()
    assert f(("a", 0)) == 2.0
    assert f(["b", 3]) == -4.0


def test_record_additive_fun_minmax_is_exact():
    f = kind_q_fun()
    assert f.minmax(record()) == brute(f, record())


def test_record_additive_fun_positional_and_weights():
    f = RecordAdditiveFun((AffineFun(1.0), AffineFun(2.0)), weights=(2.0, -1.0))
    assert f((3, 1)) == 2 * 3 - 2


def test_record_additive_fun_rejects_field_mismatch():
    f = RecordAdditiveFun(
        {"q": AffineFun(1.0), "kind": TableFun({})}, fields=("q", "kind")
    )
    with pytest.raises(ValueError):
        f.minmax(record())
    with pytest.raises(ValueError):
        RecordAdditiveFun({"nope": AffineFun(1.0)}, fields=("kind", "q"))


def test_record_additive_fun_to_ufun_is_equivalent():
    f = kind_q_fun()
    u = f.to_ufun(record())
    assert all(u(v) == f(v) for v in record().all)


def test_record_additive_fun_roundtrips():
    f = kind_q_fun()
    g = RecordAdditiveFun.from_dict(f.to_dict())
    assert all(g(v) == f(v) for v in record().all)


# ---------------------------------------------------------------- ListAggregationFun


@pytest.mark.parametrize(
    "aggregate,expected", [("sum", 3.0), ("mean", 1.0), ("min", -1.0), ("max", 3.0)]
)
def test_aggregates(aggregate, expected):
    f = ListAggregationFun(lambda e: e, aggregate=aggregate)
    assert f((3, 1, -1)) == expected


def test_empty_and_count_fun():
    f = ListAggregationFun(lambda e: e, empty=-5.0, count_fun=LinearFun(-0.5))
    assert f(()) == -5.0
    assert f((2, 2)) == 4.0 - 1.0


def test_custom_aggregate():
    f = ListAggregationFun(
        lambda e: e, aggregate=lambda v: sorted(v)[-2] if len(v) > 1 else v[0]
    )
    assert f((5, 1, 3)) == 3


def test_shift_and_scale():
    f = ListAggregationFun(lambda e: e).scale_by(2.0).shift_by(1.0)
    assert f((1, 2)) == 7.0


def test_bad_aggregate_raises():
    with pytest.raises(ValueError):
        ListAggregationFun(lambda e: e, aggregate="median")


ELEMENT_FUNS = {
    "record_additive": kind_q_fun,
    "linear_ufun": lambda: kind_q_fun().to_ufun(record()),
    "multi_fun": lambda: TableMultiFun(
        {(k, q): (ord(k) - 98) * (q - 1.5) for k in "abc" for q in range(4)}
    ),
    "gla_ufun": lambda: GLAUtilityFunction(
        factors=[(("kind", "q"), LambdaMultiFun(lambda x: (x[0] == "b") * x[1] - 1.0))],
        outcome_space=make_os(record().fields),
    ),
    "callable": lambda: (lambda r: math.sin(r[1] + len(r[0]) * 7) + (r[0] == "c")),
}


@pytest.mark.parametrize("kind", list(ELEMENT_FUNS))
@pytest.mark.parametrize("aggregate", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize(
    "counts", [dict(min_count=0, max_count=2), dict(min_count=1, max_count=3)]
)
@pytest.mark.parametrize("unique", [None, "kind"])
def test_minmax_matches_brute_force(kind, aggregate, counts, unique):
    issue = ListIssue("L", record(), unique_by=unique, **counts)
    f = ListAggregationFun(ELEMENT_FUNS[kind](), aggregate=aggregate)
    assert close(f.minmax(issue), brute(f, issue))
    (mn, worst), (mx, best) = f.arg_extremes(issue)
    assert issue.is_valid(worst) and issue.is_valid(best)
    assert math.isclose(f(worst), mn, abs_tol=1e-9)
    assert math.isclose(f(best), mx, abs_tol=1e-9)


@pytest.mark.parametrize("ordered", [False, True])
def test_minmax_with_count_fun_matches_brute_force(ordered):
    issue = ListIssue("L", record(), max_count=3, ordered=ordered)
    f = ListAggregationFun(
        kind_q_fun(), count_fun=QuadraticFun(a2=-0.7, a1=1.0), empty=0.3
    )
    assert close(f.minmax(issue), brute(f, issue))


def test_negative_scale_swaps_extremes():
    issue = ListIssue("L", record(), max_count=2)
    f = ListAggregationFun(kind_q_fun()).scale_by(-1.0)
    assert close(f.minmax(issue), brute(f, issue))


def test_plain_element_list():
    issue = ListIssue("L", make_issue((0, 4), "v"), max_count=3, unique_by=True)
    f = ListAggregationFun(AffineFun(1.0, -2.0))
    assert close(f.minmax(issue), brute(f, issue))


@settings(max_examples=60, deadline=None)
@given(
    table=st.lists(st.floats(-5, 5), min_size=6, max_size=6),
    aggregate=st.sampled_from(["sum", "mean", "min", "max"]),
    min_count=st.integers(0, 2),
    extra=st.integers(0, 2),
    unique=st.sampled_from([None, "x", True]),
    ordered=st.booleans(),
    slope=st.floats(-2, 2),
)
def test_property_greedy_is_optimal(
    table, aggregate, min_count, extra, unique, ordered, slope
):
    r = RecordIssue([make_issue(["a", "b"], "x"), make_issue((0, 2), "y")])
    mapping = dict(zip(r.all, table))
    try:
        issue = ListIssue(
            "L",
            r,
            min_count=min_count,
            max_count=min_count + extra,
            unique_by=unique,
            ordered=ordered,
        )
    except ValueError:
        return
    f = ListAggregationFun(
        TableMultiFun(mapping), aggregate=aggregate, count_fun=LinearFun(slope)
    )
    assert close(f.minmax(issue), brute(f, issue))


# ---------------------------------------------------------------- unbounded counts


def test_unbounded_sum_of_positive_elements_raises():
    issue = ListIssue("L", make_issue((1, 3), "v"))
    with pytest.raises(UnboundedListUtilityError, match="max_count"):
        ListAggregationFun(AffineFun(1.0)).minmax(issue)


def test_unbounded_sum_of_negative_elements_has_no_minimum():
    issue = ListIssue("L", make_issue((1, 3), "v"), min_count=1)
    with pytest.raises(UnboundedListUtilityError):
        ListAggregationFun(AffineFun(-1.0)).minmax(issue)


def test_unbounded_sum_with_a_count_cost_raises_when_still_unbounded():
    issue = ListIssue("L", make_issue((0, 3), "v"))
    # max is bounded (each element adds <= 3 but costs 4); min is not
    with pytest.raises(UnboundedListUtilityError, match="minimum"):
        ListAggregationFun(AffineFun(1.0), count_fun=LinearFun(-4.0)).minmax(issue)
    with pytest.raises(UnboundedListUtilityError, match="maximum"):
        ListAggregationFun(AffineFun(1.0, bias=4.0), count_fun=LinearFun(-4.0)).minmax(
            issue
        )


def test_unbounded_sum_balanced_by_count_fun_is_exact():
    issue = ListIssue("L", make_issue((0, 3), "v"), min_count=1)
    f = ListAggregationFun(AffineFun(0.0, bias=2.0), count_fun=AffineFun(-2.0, 1.0))
    assert f.minmax(issue) == (1.0, 1.0)


def test_unbounded_mean_is_exact():
    issue = ListIssue("L", make_issue((0, 3), "v"))
    f = ListAggregationFun(AffineFun(1.0), aggregate="mean", empty=-1.0)
    assert f.minmax(issue) == (-1.0, 3.0)
    capped = ListIssue("L", make_issue((0, 3), "v"), max_count=4)
    assert f.minmax(capped) == brute(f, capped)


def test_unbounded_mean_with_growing_count_fun_raises():
    issue = ListIssue("L", make_issue((0, 3), "v"))
    f = ListAggregationFun(AffineFun(1.0), aggregate="mean", count_fun=LinearFun(0.1))
    with pytest.raises(UnboundedListUtilityError):
        f.minmax(issue)


def test_unbounded_with_non_affine_count_fun_or_custom_aggregate_raises():
    issue = ListIssue("L", make_issue((0, 3), "v"))
    with pytest.raises(UnboundedListUtilityError):
        ListAggregationFun(
            AffineFun(1.0), aggregate="mean", count_fun=lambda n: -n * n
        ).minmax(issue)
    with pytest.raises(UnboundedListUtilityError):
        ListAggregationFun(AffineFun(1.0), aggregate=lambda v: max(v)).minmax(issue)


def test_unbounded_with_finite_unique_keys_is_exact():
    issue = ListIssue("L", record(), unique_by="kind")  # at most 3 elements
    f = ListAggregationFun(kind_q_fun())
    assert close(f.minmax(issue), brute(f, issue))


def test_error_is_a_value_error():
    assert issubclass(UnboundedListUtilityError, ValueError)


# ---------------------------------------------------------------- ufun integration


def scenario(max_count=3):
    lots = ListIssue("lots", record(), max_count=max_count, unique_by="kind")
    os = make_os([make_issue(["fast", "slow"], "ship"), lots])
    u = LinearAdditiveUtilityFunction(
        {
            "ship": TableFun({"fast": 1.0, "slow": 0.0}),
            "lots": ListAggregationFun(kind_q_fun()),
        },
        weights={"ship": 2.0, "lots": 0.5},
        outcome_space=os,
    )
    return os, u


def brute_ufun(u, os):
    vals = [u(o) for o in os.enumerate()]
    return min(vals), max(vals)


def test_linear_additive_minmax_and_extreme_outcomes():
    os, u = scenario()
    assert close(u.minmax(), brute_ufun(u, os))
    worst, best = u.extreme_outcomes()
    assert os.is_valid(worst) and os.is_valid(best)
    assert close((u(worst), u(best)), brute_ufun(u, os))


def test_linear_additive_normalize():
    os, u = scenario()
    n = u.normalize()
    mn, mx = brute_ufun(n, os)
    assert math.isclose(mn, 0.0, abs_tol=1e-9) and math.isclose(mx, 1.0, abs_tol=1e-9)
    o = next(iter(os.enumerate()))
    lo, hi = brute_ufun(u, os)
    assert math.isclose(n(o), (u(o) - lo) / (hi - lo), abs_tol=1e-9)


def test_infinite_space_ufun_minmax_does_not_enumerate():
    lots = ListIssue("lots", record(), unique_by="kind")
    os = make_os(
        [
            make_issue((0.0, 1.0), "p"),
            ListIssue("free", record(), max_count=None, typical_count=2),
        ]
    )
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0), ListAggregationFun(kind_q_fun(), aggregate="mean")],
        outcome_space=os,
    )
    mn, mx = u.minmax()
    assert math.isclose(mx, 1.0 + 2.0) and math.isclose(mn, 0.0 - 4.0)
    assert lots.is_finite()


def test_unbounded_ufun_normalize_raises_clearly():
    os = make_os([make_issue(2, "p"), ListIssue("free", record())])
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0), ListAggregationFun(kind_q_fun())], outcome_space=os
    )
    with pytest.raises(UnboundedListUtilityError):
        u.normalize()


def test_list_fun_serialization_roundtrips():
    os, u = scenario()
    f = u.values[1]
    g = ListAggregationFun.from_dict(f.to_dict())
    issue = os.issues[1]
    assert all(g(v) == f(v) for v in issue.all)
    v = deserialize(serialize(u))
    assert all(v(o) == u(o) for o in os.enumerate())


def test_xml_export_raises():
    os, u = scenario()
    with pytest.raises(NotImplementedError):
        u.xml()


# ---------------------------------------------------------------- multi-issue factors


def multi_scenario():
    disc = make_issue((0, 2), "discount")
    lots = ListIssue("lots", record(), max_count=2)
    return disc, lots


def test_multi_fun_scores_each_element_in_context():
    disc, lots = multi_scenario()
    g = LambdaMultiFun(lambda x: x[0] * x[1][1] - (x[1][0] == "b"))
    f = ListAggregationMultiFun(g, list_index=1)
    assert f((2, (("a", 1), ("b", 3)))) == 2 * 1 + 2 * 3 - 1


def test_multi_fun_flatten_passes_fields():
    f = ListAggregationMultiFun(
        LambdaMultiFun(lambda x: x[0] + 2 * x[2]), list_index=1, flatten=True
    )
    assert f((1, (("a", 2),))) == 1 + 4


@pytest.mark.parametrize("aggregate", ["sum", "mean", "min", "max"])
def test_gla_with_multi_fun_minmax_matches_brute_force(aggregate):
    disc, lots = multi_scenario()
    os = make_os([disc, make_issue(["x", "y"], "other"), lots])
    g = LambdaMultiFun(lambda x: (x[0] - 1) * x[1][1] + (x[1][0] == "c"))
    u = GLAUtilityFunction(
        factors=[
            (
                ("discount", "lots"),
                ListAggregationMultiFun(g, list_index=1, aggregate=aggregate),
            ),
            (("other",), TableFun({"x": 0.0, "y": 2.0}), -1.0),
        ],
        outcome_space=os,
    )
    assert close(u.minmax(), brute_ufun(u, os))
    worst, best = u.extreme_outcomes()
    assert close((u(worst), u(best)), brute_ufun(u, os))


def test_multi_fun_over_two_lists_raises():
    _, lots = multi_scenario()
    f = ListAggregationMultiFun(LambdaMultiFun(lambda x: 0.0), list_index=0)
    with pytest.raises(ValueError):
        f.minmax((lots, ListIssue("other", record(), max_count=1)))


def test_multi_fun_roundtrips():
    f = ListAggregationMultiFun(
        LinearMultiFun((1.0, 0.0, 2.0)), list_index=1, flatten=True
    )
    g = ListAggregationMultiFun.from_dict(f.to_dict())
    assert g((1, ((3, 2),))) == f((1, ((3, 2),))) == 1 + 4
    assert g.flatten and g.list_index == 1


# ---------------------------------------------------------------- lifting


def flat_scenario():
    speed = make_issue((0, 3), "speed")
    kind, q = record().fields
    return make_os([speed, kind, q])


def test_lift_linear_additive_ufun():
    flat = flat_scenario()
    one = LinearAdditiveUtilityFunction(
        [
            AffineFun(1.0),
            TableFun({"a": 1.0, "b": -2.0, "c": 0.5}),
            AffineFun(-1.0, 1.0),
        ],
        weights=[2.0, 1.0, 3.0],
        bias=0.25,
        outcome_space=flat,
    )
    lots = ListIssue("lots", record(), max_count=2)
    u = lift_element_ufun(one, lots)
    assert [_.name for _ in u.outcome_space.issues] == ["speed", "lots"]
    for s in range(4):
        for r in record().all:
            assert math.isclose(u((s, (r,))), one((s, *r)))
    # sum: the plain part once, the element part per element
    r1, r2 = ("a", 0), ("c", 3)
    part = lambda r: one((0, *r)) - one((0, "a", 0)) + (1.0 + 3.0)  # noqa: E731
    assert math.isclose(u((1, (r1, r2))), 0.25 + 2.0 + part(r1) + part(r2))
    assert close(u.minmax(), brute_ufun(u, u.outcome_space))


def test_lift_gla_ufun_with_cross_factor():
    flat = flat_scenario()
    one = GLAUtilityFunction(
        factors=[
            (("speed",), AffineFun(1.0)),
            (("speed", "q"), LambdaMultiFun(lambda x: x[0] * x[1])),
            (("kind",), TableFun({"a": 1.0, "b": 0.0, "c": -1.0}), 2.0),
        ],
        outcome_space=flat,
    )
    lots = ListIssue("lots", record(), max_count=2, unique_by="kind")
    u = lift_element_ufun(one, lots, aggregate="max")
    for s in range(4):
        for r in record().all:
            assert math.isclose(u((s, (r,))), one((s, *r)))
    assert close(u.minmax(), brute_ufun(u, u.outcome_space))


def test_lift_requires_matching_names():
    flat = flat_scenario()
    one = LinearAdditiveUtilityFunction([AffineFun(1.0)] * 3, outcome_space=flat)
    with pytest.raises(ValueError):
        lift_element_ufun(one, ListIssue("lots", make_issue(3, "nope")))
