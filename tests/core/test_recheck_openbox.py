"""Re-verification open-box tests for the list-issue / infinite-issue fixes.

Covers the branches added by the fixes for the earlier open-box passes
(chunks 2-4) and the defects found while re-checking them.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import random

import pytest

from negmas.outcomes import IntIssue, ListIssue, RealIssue, make_issue, make_os
from negmas.outcomes.infinite import CountableInfiniteIssue
from negmas.outcomes.list_issue import RecordIssue
from negmas.outcomes.optional_issue import OptionalIssue
from negmas.outcomes.string_issue import StrIssue
from negmas.preferences import (
    GLAUtilityFunction,
    LinearAdditiveUtilityFunction,
    ListAggregationFun,
    ListAggregationMultiFun,
    MappingUtilityFunction,
    RecordAdditiveFun,
)
from negmas.preferences.inv_ufun import (
    AdaptiveInverseUtilityFunction,
    BIDSInverseUtilityFunction,
    BruteForceInverseUtilityFunction,
    MCTSInverseUtilityFunction,
    PresortingInverseUtilityFunction,
    PresortingInverseUtilityFunctionBruteForce,
    PresortingLegacyInverseUtilityFunction,
    SearchInverseUtilityFunction,
)
from negmas.preferences.inv_ufun.search import needs_search
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearMultiFun
from negmas.serialization import deserialize, serialize

INVERTERS = [
    BruteForceInverseUtilityFunction,
    PresortingLegacyInverseUtilityFunction,
    PresortingInverseUtilityFunctionBruteForce,
    PresortingInverseUtilityFunction,
    SearchInverseUtilityFunction,
    AdaptiveInverseUtilityFunction,
    BIDSInverseUtilityFunction,
    MCTSInverseUtilityFunction,
]

# ---------------------------------------------------------------------------
# chunk 4: infinite issues (fixed branches)
# ---------------------------------------------------------------------------


def test_intersect_of_half_bounded_int_issues():
    a, b, c = IntIssue("a", min=0), IntIssue("a", max=10), IntIssue("a", min=20)
    ab = a.intersect(b)
    assert (ab.min_value, ab.max_value) == (0, 10) and ab.is_discrete()
    ac = a.intersect(c)
    assert isinstance(ac, CountableInfiniteIssue)
    assert (ac.min_value, ac.max_value) == (20, math.inf)


def test_intersect_of_real_issues_with_a_finite_one():
    x = RealIssue("r", min=0).intersect(make_issue((-5.0, 5.0), "r"))
    y = make_issue((-5.0, 5.0), "r").intersect(RealIssue("r", min=0))
    for z in (x, y):
        assert (z.min_value, z.max_value) == (0.0, 5.0)
        assert z.is_continuous()


def test_contains_on_infinite_and_optional_issues():
    oi = OptionalIssue(IntIssue("x", min=0))
    assert not IntIssue("x").contains(oi)  # None is not an integer
    assert oi.contains(OptionalIssue(IntIssue("x", min=1)))
    assert not oi.contains(IntIssue("x", min=-1))
    assert not make_issue((0, 10), "x").contains(IntIssue("x", min=0))
    assert IntIssue("x", min=0).contains(make_issue((0, 10), "x"))
    assert not IntIssue("x").contains(RealIssue("x", min=0, max=1))
    assert IntIssue("x", min=0).contains(make_issue([1, 2, 3], "x"))
    assert not IntIssue("x", min=0).contains(make_issue([-1, 2], "x"))
    assert not make_issue((0.0, 10.0), "r").contains(RealIssue("r", min=0))
    assert not make_issue([3], "q").contains(IntIssue("q", min=3))


def test_optional_infinite_issue_is_not_discrete():
    oi = OptionalIssue(IntIssue("x", min=0))
    assert not oi.is_discrete()
    assert not make_os([oi]).is_discrete()
    assert oi.is_valid(None) and oi.is_valid(3) and not oi.is_valid(-1)


@pytest.mark.parametrize(
    "issue",
    [
        IntIssue("a", min=2**70),
        IntIssue("b"),
        IntIssue("c", max=-5, typical=(-50, -10)),
        RealIssue("d", min=0.5),
        RealIssue("e"),
    ],
    ids=str,
)
def test_infinite_issue_roundtrips_keep_bounds_typical_and_type(issue):
    for r in (
        deserialize(json.loads(json.dumps(serialize(issue)))),
        copy.deepcopy(issue),
        copy.copy(issue),
        pickle.loads(pickle.dumps(issue)),
    ):
        assert type(r) is type(issue)
        assert (r.min_value, r.max_value) == (issue.min_value, issue.max_value)
        assert tuple(r.typical) == tuple(issue.typical)


def test_bounds_above_maxsize_stay_valid():
    x = IntIssue("a", min=2**70)
    assert x.is_valid(2**70) and x.is_valid(2**70 + 5) and not x.is_valid(2**70 - 1)
    vals = x.rand_outcomes(300)
    assert len(set(vals)) == 300 and min(vals) >= 2**70


@pytest.mark.parametrize(
    "issue",
    [
        IntIssue("y", min=0, typical=(0, 3)),
        IntIssue("z", max=0, typical=(-2, 0)),
        IntIssue("w", min=0, typical=(5, 5)),
    ],
    ids=str,
)
def test_rand_outcomes_fill_is_distinct_and_valid(issue):
    vals = issue.rand_outcomes(25)
    assert len(vals) == 25 and len(set(vals)) == 25
    assert all(issue.is_valid(v) for v in vals)


def test_fractional_bounds_round_inward():
    i = CountableInfiniteIssue((0.5, math.inf), name="f")
    assert i.min_value == 1
    j = IntIssue("q", min=0.5, max=3.5)
    assert (j.min_value, j.max_value) == (1, 3)


def test_int_issue_validity():
    import numpy as np

    q = IntIssue("q")
    assert q.is_valid(3.0) and not q.is_valid(3.5)
    assert not q.is_valid(math.inf) and not q.is_valid(True)
    assert IntIssue("q", min=0).is_valid(np.int64(3))
    assert not RealIssue("r").is_valid(math.inf)


def test_spaces_with_infinite_issues_sample_without_enumerating():
    os_ = make_os([IntIssue("q", min=0), RealIssue("r")])
    assert not os_.is_discrete() and os_.cardinality == math.inf
    assert os_.is_valid(os_.random_outcome())
    assert os_.to_discrete(levels=5).cardinality == 25
    os2 = make_os([IntIssue("q", min=0)])
    sample = list(os2.enumerate_or_sample(max_cardinality=20))
    assert 0 < len(sample) <= 20 and all(os2.is_valid(_) for _ in sample)


def test_str_issue_edges():
    s = StrIssue("s", min_length=2, max_length=2, alphabet="a")
    assert s.rand_outcomes(5) == ["aa"]
    assert not s.is_valid(s.rand_invalid())
    s2 = StrIssue("s", min_length=3)
    assert not s2.is_valid(s2.rand_invalid())
    assert not make_issue(["a", "b"], "s").contains(
        StrIssue("s", max_length=1, alphabet="ab")
    )


# ---------------------------------------------------------------------------
# chunk 3: inverters on non-enumerable spaces (fixed branches)
# ---------------------------------------------------------------------------


def _spaces():
    return {
        "inflist": make_os(
            [ListIssue("l", make_issue(10, "e"), max_count=None), make_issue(5, "x")]
        ),
        "hugelist": make_os(
            [
                ListIssue("l", make_issue(10, "e"), max_count=30, ordered=True),
                make_issue(5, "x"),
            ]
        ),
        "str": make_os(
            [StrIssue("s", max_length=5, alphabet="ab"), make_issue(5, "x")]
        ),
    }


@pytest.mark.parametrize("space", ["inflist", "hugelist", "str"])
@pytest.mark.parametrize("cls", INVERTERS, ids=lambda c: c.__name__)
def test_every_inverter_finds_extremes_without_enumerating(space, cls):
    os_ = _spaces()[space]
    i0 = os_.issues[0]
    if isinstance(i0, ListIssue):
        f = ListAggregationFun(AffineFun(1.0, -4.5), aggregate="mean")
        expected = (-4.5, 8.5)
    else:
        from negmas.preferences.value_fun import LambdaFun

        f = LambdaFun(lambda s: float(s.count("a")))
        expected = (0.0, 9.0)
    u = LinearAdditiveUtilityFunction(
        values=[f, IdentityFun()], weights=[1, 1], outcome_space=os_
    )
    assert u.minmax() == pytest.approx(expected)
    inv = cls(u)
    inv.init()
    assert float(u(inv.best())) == pytest.approx(expected[1])
    assert float(u(inv.worst())) == pytest.approx(expected[0])
    m = inv.one_in((0.4, 0.6), normalized=True)
    assert m is not None and os_.is_valid(m)


@pytest.mark.parametrize(
    "cls",
    [
        MCTSInverseUtilityFunction,
        AdaptiveInverseUtilityFunction,
        PresortingInverseUtilityFunction,
        BruteForceInverseUtilityFunction,
    ],
    ids=lambda c: c.__name__,
)
@pytest.mark.parametrize("kind", ["gla", "mapping"])
def test_non_linear_ufuns_on_infinite_list_space(cls, kind):
    os_ = make_os(
        [make_issue(3, "d"), ListIssue("l", make_issue(10, "e"), max_count=None)]
    )
    if kind == "gla":
        u = GLAUtilityFunction(
            factors=[
                (
                    (0, 1),
                    ListAggregationMultiFun(
                        LinearMultiFun((1.0, -1.0)), list_index=1, aggregate="mean"
                    ),
                    1.0,
                )
            ],
            outcome_space=os_,
        )
    else:
        u = MappingUtilityFunction(
            lambda o: (o[0] - sum(o[1]) / max(len(o[1]), 1)) if o else 0.0,
            outcome_space=os_,
        )
    assert u.minmax() == pytest.approx((-9.0, 2.0))
    inv = cls(u)
    inv.init()
    assert float(u(inv.best())) == pytest.approx(2.0)
    assert float(u(inv.worst())) == pytest.approx(-9.0)


def test_needs_search_small_space_rule():
    small = make_os([ListIssue("l", make_issue(10, "e"), max_count=6)])
    assert small.cardinality <= 100_000 and not needs_search(small)
    assert needs_search(make_os([ListIssue("l", make_issue(10, "e"))]))
    assert needs_search(make_os([IntIssue("q", min=0)]))
    assert not needs_search(make_os([RealIssue("r")]))


# ---------------------------------------------------------------------------
# defects found by this pass
# ---------------------------------------------------------------------------


@pytest.mark.timeout(10)
def test_minmax_with_large_max_count_is_not_quadratic():
    L = ListIssue("l", make_issue(10, "e"), max_count=100_000)
    f = ListAggregationFun(AffineFun(-1.0, 0.0))
    assert f.minmax(L) == pytest.approx((-900_000.0, 0.0))


@pytest.mark.timeout(3)
def test_minmax_with_many_unique_keys_is_not_quadratic():
    k = 20_000
    rec = RecordIssue([make_issue(k, "key"), make_issue(3, "v")])
    L = ListIssue("l", rec, unique_by="key")
    f = ListAggregationFun(RecordAdditiveFun([AffineFun(-1.0), AffineFun(1.0)]))
    mn, mx = f.minmax(L)
    assert mx == pytest.approx(3.0)
    assert mn == pytest.approx(-sum(range(k)))


def test_zero_weight_unbounded_issue_does_not_make_the_ufun_unbounded():
    os_ = make_os([IntIssue("q", min=0), make_issue(5, "x")])
    u = LinearAdditiveUtilityFunction(
        values=[IdentityFun(), IdentityFun()], weights=[0.0, 1.0], outcome_space=os_
    )
    assert u.minmax() == pytest.approx((0.0, 4.0))


@pytest.mark.parametrize(
    "issue", [IntIssue("q"), RealIssue("r"), RealIssue("r", max=0)], ids=str
)
def test_random_normalized_ufun_on_infinite_issue_is_not_nan(issue):
    random.seed(0)
    try:
        u = LinearAdditiveUtilityFunction.random(
            issues=[issue, make_issue(3, "w")], normalized=True
        )
    except ValueError:
        return  # a clear error is acceptable
    assert not any(math.isnan(float(u((issue.rand(), 1)))) for _ in range(20))
