"""Tests for IntIssue, RealIssue, StrIssue and sampling of infinite issues."""

from __future__ import annotations

import copy
import math
import pickle

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import negmas.preferences as P
from negmas.outcomes import (
    ContiguousIssue,
    ContinuousInfiniteIssue,
    ContinuousIssue,
    CountableInfiniteIssue,
    IntIssue,
    RealIssue,
    StrIssue,
    make_issue,
    make_os,
)
from negmas.outcomes.outcome_space import (
    CartesianOutcomeSpace,
    DiscreteCartesianOutcomeSpace,
)
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import AffineFun, LambdaFun, TableFun
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import BoulwareTBNegotiator
from negmas.serialization import deserialize, serialize


@pytest.fixture
def no_enumeration(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("enumerated")

    monkeypatch.setattr(DiscreteCartesianOutcomeSpace, "enumerate", boom)
    monkeypatch.setattr(CountableInfiniteIssue, "all", property(boom))
    monkeypatch.setattr(StrIssue, "all", property(boom))


# ---------------------------------------------------------------- factories


def test_int_issue_factory():
    assert isinstance(IntIssue("q"), CountableInfiniteIssue)
    pos = IntIssue("q", min=1)
    assert pos.min_value == 1 and pos.max_value == float("inf")
    assert pos.is_valid(10**15) and not pos.is_valid(0)
    neg = IntIssue("q", max=-1)
    assert neg.is_valid(-(10**15)) and not neg.is_valid(0)
    assert isinstance(IntIssue("q", min=0, max=5), ContiguousIssue)


def test_real_issue_factory():
    assert isinstance(RealIssue("p"), ContinuousInfiniteIssue)
    pos = RealIssue("p", min=0.0)
    assert pos.is_valid(1e300) and not pos.is_valid(-1e-9)
    assert not pos.is_valid(float("nan"))
    assert isinstance(RealIssue("p", min=0, max=1), ContinuousIssue)


def test_make_issue_still_routes_infinite_tuples():
    assert isinstance(make_issue((0, float("inf")), "q"), CountableInfiniteIssue)
    assert isinstance(make_issue((0.0, float("inf")), "p"), ContinuousInfiniteIssue)
    assert isinstance(make_issue(str, "s", max_length=4), StrIssue)


def test_int_issue_validity_rejects_non_integers():
    q = IntIssue("q", min=0)
    assert q.is_valid(3) and q.is_valid(3.0)
    assert not q.is_valid(3.5) and not q.is_valid("3") and not q.is_valid(True)


# ---------------------------------------------------------------- sampling audit


@pytest.mark.parametrize(
    "issue",
    [
        IntIssue("q"),
        IntIssue("q", min=0),
        IntIssue("q", max=10),
        RealIssue("p"),
        RealIssue("p", min=0.0),
        RealIssue("p", max=-5.0),
    ],
    ids=str,
)
def test_rand_is_useful_not_uniform_over_maxsize(issue):
    xs = [issue.rand() for _ in range(2000)]
    assert all(issue.is_valid(x) and math.isfinite(x) for x in xs)
    a, b = issue.typical
    inside = sum(a <= x <= b for x in xs) / len(xs)
    assert inside > 0.8
    assert max(abs(x) for x in xs) < 1e5


def test_typical_range_is_configurable():
    q = IntIssue("q", min=0, typical=(1000, 1010))
    assert sum(1000 <= q.rand() <= 1010 for _ in range(500)) > 400
    with pytest.raises(ValueError):
        IntIssue("q", min=0, typical=(-10, -5))


@pytest.mark.parametrize("issue", [IntIssue("q", min=0), RealIssue("p")], ids=str)
def test_generators_and_to_discrete_are_finite(issue):
    vals = list(issue.value_generator(7))
    assert 0 < len(vals) <= 7 and all(issue.is_valid(_) for _ in vals)
    assert all(math.isfinite(_) for _ in vals)
    d = issue.to_discrete(5)
    assert d.is_discrete() and 0 < d.cardinality <= 5
    assert len(issue.rand_outcomes(20, with_replacement=False)) == 20


def test_countable_infinite_is_not_discrete_and_never_enumerated():
    q = IntIssue("q", min=0)
    assert not q.is_discrete() and not q.is_finite() and not q.is_continuous()
    with pytest.raises(ValueError):
        list(q.all)
    os = make_os([q, make_issue(3, "k")])
    assert isinstance(os, CartesianOutcomeSpace)
    assert not isinstance(os, DiscreteCartesianOutcomeSpace)
    assert os.to_discrete(levels=4).cardinality == 12


def test_rand_invalid():
    assert not IntIssue("q", min=0).is_valid(IntIssue("q", min=0).rand_invalid())
    assert IntIssue("q").rand_invalid() is None
    assert not RealIssue("p", max=1.0).is_valid(RealIssue("p", max=1.0).rand_invalid())
    assert RealIssue("p").rand_invalid() is None


@pytest.mark.parametrize(
    "issue",
    [
        IntIssue("q", min=1, typical=(1, 9)),
        RealIssue("p", max=2.0, typical=(0.0, 1.0)),
        StrIssue("s", min_length=2, max_length=5, alphabet="ab", typical_length=3),
    ],
    ids=str,
)
def test_roundtrips_keep_everything(issue):
    for other in (
        deserialize(serialize(issue)),
        type(issue).from_dict(issue.to_dict()),
        copy.copy(issue),
        copy.deepcopy(issue),
        pickle.loads(pickle.dumps(issue)),
    ):
        assert type(other) is type(issue) and other == issue
        for attr in ("typical", "min_value", "max_value", "alphabet", "typical_length"):
            assert getattr(other, attr, None) == getattr(issue, attr, None)


# ---------------------------------------------------------------- StrIssue


def test_str_issue_validity_and_rand():
    s = StrIssue("s", min_length=2, max_length=6, alphabet="xyz")
    assert s.is_valid("ab") and not s.is_valid("a") and not s.is_valid("abcdefg")
    assert not s.is_valid(None) and not s.is_valid(12)
    for _ in range(200):
        v = s.rand()
        assert s.is_valid(v) and set(v) <= set("xyz")
    assert s.cardinality == float("inf") and not s.is_discrete()


@settings(max_examples=100, deadline=None)
@given(
    lo=st.integers(0, 4),
    extra=st.one_of(st.none(), st.integers(0, 4)),
    value=st.text(alphabet="abc", max_size=9),
)
def test_property_str_mutate_keeps_validity(lo, extra, value):
    s = StrIssue("s", min_length=lo, max_length=None if extra is None else lo + extra)
    if not s.is_valid(value):
        value = s.rand()
    for _ in range(10):
        value = s.mutate(value)
        assert s.is_valid(value)


def test_str_issue_samplers_and_views():
    s = StrIssue("s", max_length=8)
    vals = list(s.value_generator(30))
    assert len(vals) == len(set(vals)) == 30
    assert len(list(s.value_generator())) > 100
    d = s.to_discrete(6)
    assert d.cardinality == 6 and all(s.is_valid(_) for _ in d.all)
    assert s.contains(StrIssue("t", min_length=1, max_length=3))
    assert not StrIssue("t", max_length=3).contains(s)
    with pytest.raises(ValueError):
        list(s.all)
    with pytest.raises(NotImplementedError, match="Genius"):
        make_os([s]).to_xml_str()


def test_str_issue_bad_arguments():
    with pytest.raises(ValueError):
        StrIssue("s", min_length=3, max_length=2)
    with pytest.raises(ValueError):
        StrIssue("s", alphabet="")


# ---------------------------------------------------------------- ufuns, inverters, negotiation


def mixed():
    issues = [
        IntIssue("q", min=1, typical=(1, 50)),
        RealIssue("p", min=0.0, typical=(10.0, 20.0)),
        StrIssue("note", max_length=10),
        make_issue(["a", "b"], "k"),
    ]
    os = make_os(issues)

    def ufun(target, sign):
        return LinearAdditiveUtilityFunction(
            [
                LambdaFun(lambda x: -abs(x - target), min_value=-1e9, max_value=0.0),
                LambdaFun(
                    lambda p: sign * min(p, 100.0) / 100, min_value=-1.0, max_value=1.0
                ),
                LambdaFun(lambda s: sign * len(s) / 10, min_value=-1.0, max_value=1.0),
                TableFun({"a": float(sign > 0), "b": float(sign < 0)}),
            ],
            outcome_space=os,
        )

    return os, ufun(20, 1), ufun(30, -1)


INVERTERS = [
    "AdaptiveInverseUtilityFunction",
    "SearchInverseUtilityFunction",
    "PresortingInverseUtilityFunction",
    "SamplingInverseUtilityFunction",
    "BIDSInverseUtilityFunction",
    "MCTSInverseUtilityFunction",
    "AttributePlanningInverseUtilityFunction",
    "BruteForceInverseUtilityFunction",
]


@pytest.mark.parametrize("name", INVERTERS)
def test_inverters_on_infinite_basic_issues(name, no_enumeration):
    os, u, _ = mixed()
    inv = getattr(P, name)(u)
    inv.init()
    o = inv.one_in((0.4, 0.6), normalized=True)
    assert o is not None and os.is_valid(o)


def test_boulware_agrees_on_infinite_basic_issues(no_enumeration):
    os, b, s = mixed()
    m = SAOMechanism(outcome_space=os, n_steps=100)
    m.add(BoulwareTBNegotiator(name="b"), ufun=b)
    m.add(BoulwareTBNegotiator(name="s"), ufun=s)
    m.run()
    assert m.agreement is not None and os.is_valid(m.agreement)


def test_normalizing_an_unbounded_value_function_raises_clearly():
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0)], outcome_space=make_os([IntIssue("q", min=0)])
    )
    with pytest.raises(ValueError, match="unbounded"):
        u.normalize()


def test_normalizing_a_bounded_value_function_on_infinite_issues_works():
    os = make_os([IntIssue("q", min=0), StrIssue("s")])
    u = LinearAdditiveUtilityFunction(
        [
            LambdaFun(lambda x: 1.0 / (1 + x), min_value=0.0, max_value=1.0),
            LambdaFun(lambda s: float("z" in s), min_value=0.0, max_value=1.0),
        ],
        outcome_space=os,
    )
    n = u.normalize()
    assert n((0, "z")) == pytest.approx(1.0)
