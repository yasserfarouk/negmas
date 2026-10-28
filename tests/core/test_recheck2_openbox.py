"""Second re-verification open-box pass for the fixes of test_recheck_openbox.

1. O(1)-per-count extremes of `ListAggregationFun` (`_raw_extreme`).
2. Zero-weight unbounded issues in `LinearAdditiveUtilityFunction`.
3. `LinearAdditiveUtilityFunction.random(normalized=True)` on unbounded issues.
"""

from __future__ import annotations

import math
import random
import warnings

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from negmas.outcomes import IntIssue, ListIssue, RealIssue, make_issue, make_os
from negmas.outcomes.list_issue import RecordIssue
from negmas.outcomes.string_issue import StrIssue
from negmas.preferences import (
    LinearAdditiveUtilityFunction,
    ListAggregationFun,
    RecordAdditiveFun,
)
from negmas.preferences.value_fun import AffineFun, IdentityFun, LambdaFun, TableFun

# ---------------------------------------------------------------------------
# 1. list extremes against brute force
# ---------------------------------------------------------------------------


def _custom_agg(v):
    return float(sum(v) + max(v))  # non-decreasing in every score


AGGS = ["sum", "mean", "min", "max", "custom"]
COUNT_FUNS = {
    "none": lambda: None,
    "affine_up": lambda: AffineFun(0.7, 0.2),
    "affine_down": lambda: AffineFun(-1.3, 0.5),
    "nonaffine": lambda: LambdaFun(lambda n: 0.5 * (n - 2) ** 2),
}
MODES = ["repeated", "unique_plain", "unique_record", "unique_two_keys"]

score = st.floats(-3, 3, allow_nan=False, allow_infinity=False)


@st.composite
def _case(draw, mode):
    lo = draw(st.integers(0, 2))
    hi = draw(st.integers(lo, 4))
    ordered = draw(st.booleans())
    if mode == "repeated":
        k = draw(st.integers(1, 4))
        issue = ListIssue("l", make_issue(k, "e"), lo, hi, ordered=ordered)
        ef = TableFun({i: draw(score) for i in range(k)})
    elif mode == "unique_plain":
        k = draw(st.integers(max(1, lo), 4))
        issue = ListIssue("l", make_issue(k, "e"), lo, hi, ordered, unique_by=True)
        ef = TableFun({i: draw(score) for i in range(k)})
    elif mode == "unique_record":
        k, m = draw(st.integers(max(1, lo), 4)), draw(st.integers(1, 3))
        rec = RecordIssue([make_issue(k, "key"), make_issue(m, "v")])
        issue = ListIssue("l", rec, lo, hi, ordered, unique_by="key")
        if draw(st.booleans()):
            ef = RecordAdditiveFun(
                [
                    TableFun({i: draw(score) for i in range(k)}),
                    TableFun({i: draw(score) for i in range(m)}),
                ]
            )
        else:  # a generic (non-additive) record scorer
            table = {(i, j): draw(score) for i in range(k) for j in range(m)}
            ef = lambda e, t=table: t[tuple(e)]  # noqa: E731
    else:
        a = draw(st.integers(1, 2))
        b = draw(st.integers(max(1, -(-lo // a)), 2))
        m = draw(st.integers(1, 2))
        rec = RecordIssue(
            [make_issue(a, "k1"), make_issue(b, "k2"), make_issue(m, "v")]
        )
        issue = ListIssue("l", rec, lo, hi, ordered, unique_by=["k1", "k2"])
        ef = RecordAdditiveFun(
            [TableFun({i: draw(score) for i in range(n)}) for n in (a, b, m)]
        )
    return issue, ef


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("count", list(COUNT_FUNS))
@pytest.mark.parametrize("agg", AGGS)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_list_arg_extremes_match_brute_force(agg, count, mode, data):
    issue, ef = data.draw(_case(mode))
    f = ListAggregationFun(
        ef,
        aggregate=_custom_agg if agg == "custom" else agg,
        empty=data.draw(score),
        count_fun=COUNT_FUNS[count](),
        scale=data.draw(st.sampled_from([1.0, 0.5, -2.0])),
        bias=data.draw(score),
    )
    values = [f(v) for v in issue.all]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (mn, worst), (mx, best) = f.arg_extremes(issue)
    assert mn == pytest.approx(min(values), abs=1e-9)
    assert mx == pytest.approx(max(values), abs=1e-9)
    assert issue.is_valid(worst) and issue.is_valid(best)
    assert f(worst) == pytest.approx(mn, abs=1e-9)
    assert f(best) == pytest.approx(mx, abs=1e-9)
    assert f.minmax(issue) == pytest.approx((mn, mx), abs=1e-9)


def _brute_minmax_repeated(scores, count_fun, agg, n_max):
    """Exact range over multisets of `scores` with 0..n_max elements."""
    lo, hi = math.inf, -math.inf
    for n in range(n_max + 1):
        if n == 0:
            vals = [0.0]
        elif agg == "sum":
            vals = [n * min(scores), n * max(scores)]
        else:  # mean / min / max of n copies is extreme at constant lists
            vals = [min(scores), max(scores)]
        c = 0.0 if count_fun is None else count_fun(n)
        lo, hi = min(lo, min(vals) + c), max(hi, max(vals) + c)
    return lo, hi


def _brute_minmax_distinct(keys_scores, count_fun, agg):
    """Exact range over subsets of distinct keys (best element per key)."""
    asc = sorted(keys_scores)
    desc = asc[::-1]
    lo_pre, hi_pre = [0.0], [0.0]
    for a, d in zip(asc, desc):
        lo_pre.append(lo_pre[-1] + a)
        hi_pre.append(hi_pre[-1] + d)
    lo, hi = math.inf, -math.inf
    for n in range(len(asc) + 1):
        c = 0.0 if count_fun is None else count_fun(n)
        if n == 0:
            vlo = vhi = 0.0
        elif agg == "sum":
            vlo, vhi = lo_pre[n], hi_pre[n]
        elif agg == "mean":
            vlo, vhi = lo_pre[n] / n, hi_pre[n] / n
        elif agg == "min":
            vlo, vhi = asc[0], desc[n - 1]
        else:
            vlo, vhi = asc[n - 1], desc[0]
        lo, hi = min(lo, vlo + c), max(hi, vhi + c)
    return lo, hi


SPEED_COUNTS = {
    "none": None,
    "affine": AffineFun(-0.01),
    "nonaffine": LambdaFun(lambda n: -((n - 50) ** 2)),
}


@pytest.mark.timeout(20)
@pytest.mark.parametrize("count", list(SPEED_COUNTS))
@pytest.mark.parametrize("agg", ["sum", "mean", "min", "max"])
def test_list_minmax_large_counts_fast_and_exact(agg, count):
    cf = SPEED_COUNTS[count]
    n_max = 100_000
    L = ListIssue("l", make_issue(10, "e"), max_count=n_max)
    f = ListAggregationFun(AffineFun(-1.0, 0.5), aggregate=agg, count_fun=cf)
    scores = [-e + 0.5 for e in range(10)]
    if count == "nonaffine":
        # the non-affine count term dominates; check the obvious extremes
        mn, mx = f.minmax(L)
        assert mx == pytest.approx(
            max((0.5 * n if agg == "sum" else 0.5) - (n - 50) ** 2 for n in (49, 50))
        )
        assert math.isfinite(mn)
        return
    assert f.minmax(L) == pytest.approx(
        _brute_minmax_repeated(scores, cf, agg, n_max), rel=1e-9
    )


@pytest.mark.timeout(20)
@pytest.mark.parametrize("count", list(SPEED_COUNTS))
@pytest.mark.parametrize("agg", ["sum", "mean", "min", "max"])
def test_list_minmax_many_unique_keys_fast_and_exact(agg, count):
    cf = SPEED_COUNTS[count]
    k = 20_000
    rec = RecordIssue([make_issue(k, "key"), make_issue(3, "v")])
    L = ListIssue("l", rec, unique_by="key")
    f = ListAggregationFun(
        RecordAdditiveFun([AffineFun(-1.0), AffineFun(1.0)]),
        aggregate=agg,
        count_fun=cf,
    )
    mn, mx = f.minmax(L)
    if count == "nonaffine":
        assert math.isfinite(mn) and math.isfinite(mx)
        return
    # best element per key: v=2 (max), v=0 (min)
    hi_scores = [-key + 2.0 for key in range(k)]
    lo_scores = [-key + 0.0 for key in range(k)]
    exp_hi = _brute_minmax_distinct(hi_scores, cf, agg)[1]
    exp_lo = _brute_minmax_distinct(lo_scores, cf, agg)[0]
    assert (mn, mx) == pytest.approx((exp_lo, exp_hi), rel=1e-9)


# ---------------------------------------------------------------------------
# 2. zero-weight unbounded issues
# ---------------------------------------------------------------------------

UNBOUNDED = [IntIssue("q", min=0), IntIssue("q"), RealIssue("r"), RealIssue("r", max=0)]
WEIGHTS = [[0.0, 1.0], [0.0, -2.0], [-0.0, 1.0], [0.0, 3.0]]


def _ufun(q, ws):
    os_ = make_os([q, make_issue(5, "x")])
    return LinearAdditiveUtilityFunction(
        values=[IdentityFun(), IdentityFun()], weights=ws, outcome_space=os_
    )


@pytest.mark.parametrize("ws", WEIGHTS, ids=str)
@pytest.mark.parametrize("q", UNBOUNDED, ids=str)
def test_zero_weight_unbounded_issue_minmax_and_extremes(q, ws):
    u = _ufun(q, ws)
    expected = sorted((0.0, 4.0 * ws[1]))
    assert u.minmax() == pytest.approx(tuple(expected))
    worst, best = u.extreme_outcomes()
    assert float(u(worst)) == pytest.approx(expected[0])
    assert float(u(best)) == pytest.approx(expected[1])
    # the zero-weight issue contributes nothing anywhere
    for x in range(5):
        vals = {float(u((v, x))) for v in (0, 1, 10**9, q.rand())}
        assert len(vals) == 1


@pytest.mark.parametrize("method", ["normalize", "normalize_for"])
@pytest.mark.parametrize("ws", WEIGHTS, ids=str)
@pytest.mark.parametrize("q", UNBOUNDED, ids=str)
def test_zero_weight_unbounded_issue_normalizes_to_unit_range(q, ws, method):
    u = _ufun(q, ws)
    n = getattr(u, method)()
    mn, mx = n.minmax()
    assert (mn, mx) == pytest.approx((0.0, 1.0))
    assert all(math.isfinite(w) for w in n.weights)
    for x in range(5):
        vals = [float(n((v, x))) for v in (0, 10**9, q.rand())]
        assert not any(math.isnan(v) for v in vals)
        assert max(vals) - min(vals) == pytest.approx(0.0)
        assert 0.0 - 1e-9 <= vals[0] <= 1.0 + 1e-9
    worst, best = n.extreme_outcomes()
    assert float(n(worst)) == pytest.approx(0.0)
    assert float(n(best)) == pytest.approx(1.0)


def test_zero_weight_mixed_with_negative_weights_normalizes():
    os_ = make_os([IntIssue("q", min=0), make_issue(5, "x"), make_issue(3, "y")])
    u = LinearAdditiveUtilityFunction(
        values=[IdentityFun(), IdentityFun(), TableFun({0: 1.0, 1: 5.0, 2: -2.0})],
        weights=[0.0, 0.5, -1.0],
        outcome_space=os_,
    )
    assert u.minmax() == pytest.approx((-5.0, 4.0))
    for m in ("normalize", "normalize_for"):
        n = getattr(u, m)()
        assert n.minmax() == pytest.approx((0.0, 1.0))
        assert n((0, 4, 2)) == pytest.approx(1.0)
        assert n((10**9, 4, 2)) == pytest.approx(1.0)
        assert n((7, 0, 1)) == pytest.approx(0.0)


@pytest.mark.parametrize("q", UNBOUNDED, ids=str)
def test_all_zero_weights_give_a_constant_finite_ufun(q):
    u = _ufun(q, [0.0, 0.0])
    assert u.minmax() == pytest.approx((0.0, 0.0))
    for m in ("normalize", "normalize_for"):
        n = getattr(u, m)()
        mn, mx = n.minmax()
        assert math.isfinite(mn) and mn == pytest.approx(mx)
        assert not any(math.isnan(float(n((v, 1)))) for v in (0, 10**9))


@pytest.mark.parametrize("q", UNBOUNDED, ids=str)
def test_nonzero_weight_unbounded_issue_still_raises(q):
    u = _ufun(q, [1.0, 1.0])
    with pytest.raises(ValueError):
        u.minmax()
    with pytest.raises(ValueError):
        u.normalize_for()


def test_zero_weight_unbounded_string_issue():
    os_ = make_os([StrIssue("s"), make_issue(5, "x")])
    u = LinearAdditiveUtilityFunction(
        values=[LambdaFun(lambda s: float(len(s))), IdentityFun()],
        weights=[0.0, 1.0],
        outcome_space=os_,
    )
    assert u.minmax() == pytest.approx((0.0, 4.0))
    assert u.normalize_for().minmax() == pytest.approx((0.0, 1.0))


def test_zero_weight_unbounded_list_issue_does_not_make_the_ufun_unbounded():
    os_ = make_os([ListIssue("l", make_issue(5, "e")), make_issue(5, "x")])
    u = LinearAdditiveUtilityFunction(
        values=[ListAggregationFun(IdentityFun()), IdentityFun()],
        weights=[0.0, 1.0],
        outcome_space=os_,
    )
    assert u.minmax() == pytest.approx((0.0, 4.0))
    assert u.normalize_for().minmax() == pytest.approx((0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. random(normalized=True) on unbounded numeric issues
# ---------------------------------------------------------------------------

ALL_UNBOUNDED = UNBOUNDED + [IntIssue("q", max=3), RealIssue("r", min=0)]


@pytest.mark.parametrize("issue", ALL_UNBOUNDED, ids=str)
def test_random_normalized_on_unbounded_numeric_issue_raises(issue):
    random.seed(0)
    with pytest.raises(ValueError, match="unbounded"):
        LinearAdditiveUtilityFunction.random(
            issues=[issue, make_issue(3, "w")], normalized=True
        )


@pytest.mark.parametrize("issue", ALL_UNBOUNDED, ids=str)
def test_random_unnormalized_on_unbounded_numeric_issue_works(issue):
    random.seed(0)
    u = LinearAdditiveUtilityFunction.random(
        issues=[issue, make_issue(3, "w")], normalized=False
    )
    vals = [float(u((issue.rand(), w))) for _ in range(10) for w in range(3)]
    assert all(math.isfinite(v) for v in vals)


@pytest.mark.parametrize(
    "issue", [make_issue((0.0, 1.0), "f"), make_issue((0, 10), "i"), make_issue(4, "d")]
)
@pytest.mark.parametrize("normalized", [True, False])
def test_random_on_bounded_issues_still_works(issue, normalized):
    random.seed(0)
    u = LinearAdditiveUtilityFunction.random(
        issues=[issue, make_issue(3, "w")], normalized=normalized
    )
    vals = [float(u((issue.rand(), w))) for _ in range(10) for w in range(3)]
    assert all(math.isfinite(v) for v in vals)
