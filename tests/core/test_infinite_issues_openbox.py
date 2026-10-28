"""Open-box tests for IntIssue, RealIssue, StrIssue and infinite-issue sampling."""

from __future__ import annotations

import copy
import json
import math
import pathlib
import random
import re

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import negmas.preferences as P
from negmas.outcomes import (
    CategoricalIssue,
    ContiguousIssue,
    ContinuousInfiniteIssue,
    CountableInfiniteIssue,
    IntIssue,
    RealIssue,
    StrIssue,
    make_issue,
    make_os,
)
from negmas.outcomes.base_issue import DiscreteIssue
from negmas.outcomes.common import ensure_os
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.pareto_sampler import (
    AdaptiveParetoSampler,
    IPSParetoSampler,
    MOBANOSParetoSampler,
    NB3ParetoSampler,
)
from negmas.preferences.value_fun import AffineFun, LambdaFun, TableFun
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import BoulwareTBNegotiator
from negmas.serialization import deserialize, serialize

INF = float("inf")
ROOT = pathlib.Path(__file__).resolve().parents[2]


def json_roundtrip(issue):
    return deserialize(json.loads(json.dumps(serialize(issue))))


# ---------------------------------------------------------------- factories


def test_int_issue_accepts_inf_as_no_bound():
    """An explicit infinite bound means 'no bound'."""
    i = IntIssue("q", min=0, max=INF)
    assert isinstance(i, CountableInfiniteIssue) and i.is_valid(10**12)


def test_real_issue_accepts_inf_as_no_bound():
    """RealIssue(min=0, max=inf) samples finite values like RealIssue(min=0)."""
    r = RealIssue("p", min=0.0, max=INF)
    assert isinstance(r, ContinuousInfiniteIssue)
    assert all(math.isfinite(r.rand()) for _ in range(20))


def test_int_issue_huge_bound():
    """Bounds beyond sys.maxsize still work (Python ints are unbounded)."""
    i = IntIssue("q", min=10**30)
    assert i.is_valid(10**30) and not i.is_valid(10**30 - 1)
    assert i.rand() >= 10**30


def test_countable_infinite_rejects_or_keeps_fractional_bound():
    """CountableInfiniteIssue((0.5, inf)) must not move its minimum to maxsize."""
    try:
        i = CountableInfiniteIssue((0.5, INF))
    except ValueError:
        return
    assert i.min_value <= 1 and i.is_valid(1)


def test_factories_with_both_bounds_are_finite():
    """Both bounds give the finite ContiguousIssue/ContinuousIssue."""
    assert type(IntIssue("q", min=0, max=5)) is ContiguousIssue
    r = RealIssue("p", min=0.0, max=1.0)
    assert r.is_continuous() and not isinstance(r, ContinuousInfiniteIssue)


def test_make_issue_routes_str_and_typical():
    """make_issue(str) gives a StrIssue; typical passes through to infinite tuples."""
    s = make_issue(str, "note", max_length=4)
    assert isinstance(s, StrIssue) and s.max_length == 4 and s.name == "note"
    assert make_issue((0, INF), "q", typical=(5, 10)).typical == (5, 10)
    assert make_issue((0.0, INF), "p", typical=(5, 10)).typical == (5, 10)


def test_make_issue_str_rejects_step():
    """A step is only valid for integer ranges (as for every other issue type)."""
    with pytest.raises(ValueError):
        make_issue(str, "x", step=2)


# ---------------------------------------------------------------- typical range


def test_typical_is_clipped_and_validated():
    """typical is clipped to the bounds; reversed, outside or integer-free raises."""
    assert IntIssue("p", min=0, typical=(-10, 5)).typical == (0, 5)
    for bad in [(5, 1), (-10, -5)]:
        with pytest.raises(ValueError):
            IntIssue("p", min=0, typical=bad)
    with pytest.raises(ValueError):
        IntIssue("p", typical=(3.2, 3.8))
    assert IntIssue("p", typical=(1.5, 3.5)).typical == (2, 3)


def test_default_typical_next_to_bound():
    """Default typical: 100 wide next to the finite bound, or around 0."""
    assert IntIssue("a", min=7).typical == (7, 107)
    assert IntIssue("b", max=-3).typical == (-103, -3)
    assert IntIssue("c").typical == (-50, 50)
    assert RealIssue("d", min=1.0).typical == (1.0, 101.0)


@pytest.mark.parametrize(
    "issue",
    [IntIssue("q", min=0), IntIssue("q", max=0), RealIssue("p"), RealIssue("p", min=0)],
    ids=["int_lo", "int_hi", "real", "real_lo"],
)
def test_rand_mostly_in_typical(issue):
    """About 90% of rand() falls in the typical range; the rest stays valid."""
    random.seed(3)
    xs = [issue.rand() for _ in range(5000)]
    a, b = issue.typical
    frac = sum(a <= x <= b for x in xs) / len(xs)
    assert 0.85 <= frac <= 0.95
    assert all(issue.is_valid(x) and math.isfinite(x) for x in xs)
    assert max(abs(x) for x in xs) < 10**6


@settings(max_examples=60, deadline=None)
@given(
    lo=st.one_of(st.none(), st.integers(-(10**6), 10**6)),
    width=st.one_of(st.none(), st.integers(0, 10**6)),
    t0=st.integers(-(10**6), 10**6),
    tw=st.integers(0, 500),
    use_typical=st.booleans(),
)
def test_property_int_rand_valid(lo, width, t0, tw, use_typical):
    """rand(), rand_outcomes() and to_discrete() values are always valid."""
    hi = None if width is None or lo is None else lo + width
    if lo is None and width is not None:
        hi = width
    typ = (t0, t0 + tw) if use_typical else None
    try:
        i = IntIssue("q", min=lo, max=hi, typical=typ)
    except ValueError:
        return  # typical outside bounds
    for _ in range(30):
        assert i.is_valid(i.rand())
    assert all(i.is_valid(v) for v in i.rand_outcomes(20))
    assert all(i.is_valid(v) for v in i.to_discrete(7).all)
    if isinstance(i, CountableInfiniteIssue):
        assert i.min_value <= i.typical[0] <= i.typical[1] <= i.max_value
        v = i.rand_invalid()
        assert v is None or not i.is_valid(v)


@settings(max_examples=60, deadline=None)
@given(
    lo=st.one_of(st.none(), st.floats(-1e6, 1e6)),
    t0=st.floats(-1e6, 1e6),
    tw=st.floats(0, 1e3),
    use_typical=st.booleans(),
    upper=st.booleans(),
)
def test_property_real_rand_valid(lo, t0, tw, use_typical, upper):
    """Real rand() is finite and valid; rand_invalid() is invalid or None."""
    typ = (t0, t0 + tw) if use_typical else None
    kw = {"max": lo} if upper else {"min": lo}
    try:
        r = RealIssue("p", typical=typ, **kw)
    except ValueError:
        return
    for _ in range(30):
        x = r.rand()
        assert math.isfinite(x) and r.is_valid(x)
    assert all(r.is_valid(v) for v in r.rand_outcomes(5))
    v = r.rand_invalid()
    assert v is None or not r.is_valid(v)


def test_int_rand_outcomes_distinct_unbounded():
    """rand_outcomes without replacement gives n distinct values."""
    out = IntIssue("q").rand_outcomes(2000)
    assert len(out) == 2000 == len(set(out))


def test_int_rand_outcomes_distinct_upper_bounded():
    """An infinite issue always has n distinct values to give."""
    out = IntIssue("q", max=0).rand_outcomes(2000)
    assert len(set(out)) == 2000 and all(x <= 0 for x in out)


# ---------------------------------------------------------------- validity


def test_int_is_valid_matrix():
    """Ints (incl. numpy and integral floats) valid; bools, nan, str, fractions not."""
    i = IntIssue("q", min=0)
    assert i.is_valid(np.int64(3)) and i.is_valid(np.float64(3.0)) and i.is_valid(3)
    for v in [3.5, True, float("nan"), "3", -1, INF]:
        assert not i.is_valid(v)


def test_real_is_valid_matrix():
    """Reals and ints valid; bools, nan, strings not."""
    r = RealIssue("p")
    assert r.is_valid(3) and r.is_valid(-1e300)
    for v in [True, float("nan"), "1"]:
        assert not r.is_valid(v)


def test_real_rejects_infinity_value():
    """inf is a bound marker, not a value an agent may offer."""
    assert not RealIssue("p").is_valid(INF)
    assert not RealIssue("p", min=0.0).is_valid(INF)


# ---------------------------------------------------------------- serialization


ISSUES = [
    IntIssue("a", min=1, typical=(5, 9)),
    IntIssue("b"),
    IntIssue("c", max=-3),
    RealIssue("d", min=0.0, typical=(1.0, 2.0)),
    RealIssue("e"),
    StrIssue("f", 1, 5, alphabet="xy", typical_length=3),
    StrIssue("g"),
]


@pytest.mark.parametrize("issue", ISSUES, ids=lambda i: i.name)
def test_json_roundtrip_keeps_everything(issue):
    """JSON round trip (tuples become lists) keeps type, bounds and parameters."""
    r = json_roundtrip(issue)
    assert type(r) is type(issue) and r == issue
    for attr in ("typical", "alphabet", "typical_length", "min_value", "max_value"):
        assert getattr(r, attr, None) == getattr(issue, attr, None)
    r2 = type(issue).from_dict(json.loads(json.dumps(issue.to_dict())))
    assert getattr(r2, "typical", None) == getattr(issue, "typical", None)


@pytest.mark.parametrize("issue", ISSUES, ids=lambda i: i.name)
def test_copies_keep_type_and_typical(issue):
    """copy/deepcopy keep the type, the typical range and equality/hash."""
    for c in (copy.copy(issue), copy.deepcopy(issue)):
        assert type(c) is type(issue) and c == issue and hash(c) == hash(issue)
        assert getattr(c, "typical", None) == getattr(issue, "typical", None)


@pytest.mark.parametrize("issue", ISSUES, ids=lambda i: i.name)
def test_views_finite_and_valid(issue):
    """value_generator/to_discrete are finite and valid; all and value_at raise."""
    assert all(issue.is_valid(v) for v in issue.value_generator(5))
    d = issue.to_discrete(5)
    assert d.is_discrete() and d.cardinality <= 5
    assert all(issue.is_valid(v) for v in d.all)
    with pytest.raises(ValueError):
        issue.all
    with pytest.raises(ValueError):
        issue.value_at(0)
    assert issue.cardinality == INF and not issue.is_finite()


@settings(max_examples=50, deadline=None)
@given(
    lo=st.integers(0, 6),
    extra=st.one_of(st.none(), st.integers(0, 6)),
    typ=st.one_of(st.none(), st.floats(0, 20)),
    alphabet=st.text(min_size=1, max_size=5),
)
def test_property_str_roundtrip_and_rand(lo, extra, typ, alphabet):
    """StrIssue: rand valid, from_dict(to_dict) equal, alphabet respected."""
    s = StrIssue("s", lo, None if extra is None else lo + extra, alphabet, typ)
    r = json_roundtrip(s)
    assert r == s and r.alphabet == s.alphabet and r.typical_length == typ
    for _ in range(20):
        v = s.rand()
        assert s.is_valid(v) and set(v) <= set(alphabet)
    v = s.rand_invalid()
    assert v is None or not s.is_valid(v)


def _edit_distance_one(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


@settings(max_examples=100, deadline=None)
@given(lo=st.integers(0, 4), extra=st.integers(0, 4), data=st.data())
def test_property_str_mutate_one_edit(lo, extra, data):
    """mutate keeps validity and changes at most one character."""
    s = StrIssue("s", lo, lo + extra, alphabet="abc")
    v = data.draw(st.text(alphabet="abc", min_size=lo, max_size=lo + extra))
    m = s.mutate(v)
    assert s.is_valid(m) and _edit_distance_one(v, m)


def test_str_fixed_length():
    """min_length == max_length: rand and mutate keep the length; few distinct."""
    s = StrIssue("s", 2, 2, alphabet="ab")
    assert {len(s.rand()) for _ in range(50)} == {2}
    assert {len(s.mutate("ab")) for _ in range(50)} == {2}
    assert len(s.rand_outcomes(10)) == 4
    with pytest.raises(ValueError):
        s.rand_outcomes(10, fail_if_not_enough=True)


def test_str_typical_length_mean_and_clamp():
    """rand() length has mean ~typical_length and never exceeds max_length."""
    random.seed(1)
    s = StrIssue("s", typical_length=6)
    mean = sum(len(s.rand()) for _ in range(5000)) / 5000
    assert 5.4 <= mean <= 6.6
    c = StrIssue("s", 0, 5, typical_length=50)
    assert all(len(c.rand()) <= 5 for _ in range(200))


def test_str_contains_and_genius():
    """contains compares length bounds; Genius export is refused."""
    s = StrIssue("s", 0, 3)
    assert s.contains(StrIssue("t", 1, 2)) and not s.contains(StrIssue("t", 1))
    assert s.contains(CategoricalIssue(["ab", "a"], "x"))
    assert not s.contains(CategoricalIssue(["abcd"], "x"))
    with pytest.raises(NotImplementedError):
        s._to_xml_str(0)


# ---------------------------------------------------------------- is_discrete change


def test_countable_infinite_is_never_enumerated_despite_its_class():
    """Decision: CountableInfiniteIssue keeps its DiscreteIssue base class
    (compatibility) but reports is_discrete() False and refuses enumeration."""
    i = IntIssue("x")
    assert isinstance(i, DiscreteIssue) and not i.is_discrete()
    with pytest.raises(ValueError):
        list(i.all)


def test_finite_issue_contains_infinite_is_false():
    """A finite issue never contains an infinite one."""
    assert not make_issue([1, 2], "a").contains(IntIssue("x"))
    assert not make_issue([3], "a").contains(IntIssue("x"))


def test_intersect_of_infinite_int_issues_stays_infinite():
    """The intersection of two half-infinite int issues is still infinite."""
    r = IntIssue("x").intersect(IntIssue("y", min=0))
    assert isinstance(r, CountableInfiniteIssue) and r.cardinality == INF


def test_intersect_infinite_with_finite():
    """Intersecting with a finite range gives that finite range."""
    assert IntIssue("x").intersect(make_issue((0, 10), "y")).cardinality == 11


@pytest.mark.parametrize(
    "issue",
    [make_issue(str, "n", optional=True), make_issue((0, INF), "n", optional=True)],
    ids=["str", "int"],
)
def test_optional_infinite_issue_is_not_discrete(issue):
    """Optional infinite issues are not discrete and give a CartesianOutcomeSpace."""
    assert not issue.is_discrete() and not issue.is_finite()
    os = make_os([issue, make_issue(["a", "b"], "k")])
    assert type(os) is CartesianOutcomeSpace


def test_optional_str_issue_negotiation_offers():
    """Boulware on an optional-string space makes offers and agrees (same ufun)."""
    os = make_os(
        [make_issue(str, "n", optional=True, max_length=5), make_issue(["a", "b"], "k")]
    )

    def u():
        return LinearAdditiveUtilityFunction(
            [
                LambdaFun(
                    lambda s: 0.0 if s is None else min(len(s), 5) / 5,
                    min_value=0.0,
                    max_value=1.0,
                ),
                TableFun({"a": 1.0, "b": 0.0}),
            ],
            outcome_space=os,
        )

    m = SAOMechanism(outcome_space=os, n_steps=20)
    m.add(BoulwareTBNegotiator(name="a"), ufun=u())
    m.add(BoulwareTBNegotiator(name="b"), ufun=u())
    m.run()
    assert m.extended_trace and m.agreement is not None


# ---------------------------------------------------------------- spaces


def test_space_of_infinite_issues():
    """ensure_os/make_os give a CartesianOutcomeSpace that samples and discretizes."""
    issues = [IntIssue("q", min=1), RealIssue("p", min=0.0), StrIssue("n", 0, 10)]
    os = ensure_os(None, issues, None)
    assert type(os) is CartesianOutcomeSpace
    assert os.cardinality == INF and not os.is_discrete()
    assert all(os.is_valid(os.random_outcome()) for _ in range(20))
    assert os.cardinality_if_discretized(5) == 125
    d = os.to_discrete(levels=3)
    assert d.cardinality == 27 and all(os.is_valid(o) for o in d.enumerate())
    assert len(list(os.enumerate_or_sample(max_cardinality=50))) == 50
    assert json_roundtrip(os) == os


# ---------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    "issue,slope",
    [(IntIssue("q", min=0), 1.0), (IntIssue("q", max=0), -1.0), (RealIssue("p"), 1.0)],
    ids=["int_lo", "int_hi", "real"],
)
def test_normalize_unbounded_affine_raises(issue, slope):
    """An affine value fun over a half-infinite issue cannot be normalized."""
    u = LinearAdditiveUtilityFunction(
        [AffineFun(slope)], outcome_space=make_os([issue])
    )
    with pytest.raises(ValueError, match="unbounded"):
        u.normalize()


def test_normalize_bounded_away_from_infinity_works():
    """Affine over the bounded direction of a half-infinite issue is fine to evaluate."""
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0)], outcome_space=make_os([IntIssue("q", max=0)])
    )
    assert u((0,)) == 0.0 and u((-5,)) == -5.0


def test_minmax_consistent_with_normalize_on_unbounded():
    """minmax() of an unbounded ufun is infinite (or raises), not the typical range."""
    u = LinearAdditiveUtilityFunction(
        [AffineFun(1.0)], outcome_space=make_os([RealIssue("p", min=0.0)])
    )
    try:
        mn, mx = u.minmax()
    except ValueError:
        return
    assert mx == INF


def test_random_ufun_over_str_issue_is_clear():
    """random() over a StrIssue works or raises a clear ValueError."""
    try:
        u = LinearAdditiveUtilityFunction.random(issues=[StrIssue("s")])
    except ValueError:
        return
    u(("ab",))


# ---------------------------------------------------------------- inverters, samplers


def _mixed():
    os = make_os(
        [
            IntIssue("q", min=1, typical=(1, 50)),
            RealIssue("p", min=0.0, typical=(10.0, 20.0)),
            StrIssue("n", max_length=10),
            make_issue(["a", "b"], "k"),
        ]
    )

    def mk(target, sign):
        return LinearAdditiveUtilityFunction(
            [
                LambdaFun(lambda x: -abs(x - target), min_value=-1e9, max_value=0.0),
                # bounded: an unbounded fun has no extremes (and now raises)
                LambdaFun(
                    lambda p: sign * min(p, 100.0) / 100, min_value=-1.0, max_value=1.0
                ),
                LambdaFun(lambda s: sign * len(s) / 10, min_value=-1.0, max_value=1.0),
                TableFun({"a": float(sign > 0), "b": float(sign < 0)}),
            ],
            outcome_space=os,
        )

    return os, mk(20, 1), mk(30, -1)


@pytest.mark.parametrize(
    "name",
    [
        "PresortingLegacyInverseUtilityFunction",
        "PresortingInverseUtilityFunctionBruteForce",
    ],
)
def test_legacy_inverters_on_infinite_issues(name):
    """The legacy presorting inverters also search infinite spaces."""
    from negmas.preferences import inv_ufun as I

    os, u, _ = _mixed()
    inv = getattr(P, name, None) or getattr(I, name)
    inv = inv(u)
    inv.init()
    o = inv.one_in((0.4, 0.6), normalized=True)
    assert o is not None and os.is_valid(o)


def test_inverter_real_and_int_only_space():
    """Real + int (no strings) space: the adaptive inverter returns valid outcomes."""
    os = make_os([RealIssue("p"), IntIssue("q", min=0)])
    u = LinearAdditiveUtilityFunction(
        [
            LambdaFun(lambda x: -abs(x), min_value=-1e9, max_value=0.0),
            LambdaFun(lambda x: -abs(x - 3), min_value=-1e9, max_value=0.0),
        ],
        outcome_space=os,
    )
    inv = P.AdaptiveInverseUtilityFunction(u)
    inv.init()
    for rng in [(0.0, 0.3), (0.4, 0.6), (0.9, 1.0)]:
        o = inv.one_in(rng, normalized=True)
        assert o is not None and os.is_valid(o)


def test_adaptive_pareto_sampler_on_infinite_issues():
    """The default Pareto sampler handles infinite basic issues (via fallback)."""
    os, a, b = _mixed()
    s = AdaptiveParetoSampler()
    s.init(a, b)
    out = list(s.pareto_outcomes())
    assert s.initialized and out and all(os.is_valid(o) for o in out)


@pytest.mark.parametrize(
    "cls", [IPSParetoSampler, NB3ParetoSampler, MOBANOSParetoSampler]
)
def test_scalable_pareto_samplers_on_infinite_issues(cls):
    """Scalable Pareto samplers work on infinite int/str issues without
    enumerating (continuous issues stay unsupported by these samplers)."""
    os = make_os(
        [
            IntIssue("q", min=1, typical=(1, 50)),
            StrIssue("n", max_length=10),
            make_issue(["a", "b"], "k"),
        ]
    )

    def mk(target, sign):
        return LinearAdditiveUtilityFunction(
            [
                LambdaFun(lambda x: -abs(x - target), min_value=-1e9, max_value=0.0),
                LambdaFun(lambda s: sign * len(s) / 10, min_value=-1.0, max_value=1.0),
                TableFun({"a": float(sign > 0), "b": float(sign < 0)}),
            ],
            outcome_space=os,
        )

    a, b = mk(20, 1), mk(30, -1)
    s = cls()
    s.init(a, b)
    out = list(s.pareto_outcomes())
    assert out and all(os.is_valid(o) for o in out)


# ---------------------------------------------------------------- docs


FORBIDDEN = re.compile(r"\b(ando|koshodojo|kosho|scm)\b", re.IGNORECASE)


def _unreleased_history() -> str:
    text = (ROOT / "HISTORY.rst").read_text()
    start = text.index("Unreleased")
    return text[start : text.index("Release ", start)]


@pytest.mark.parametrize(
    "path",
    [
        "docs/list_issues.rst",
        "src/negmas/outcomes/infinite.py",
        "src/negmas/outcomes/string_issue.py",
        "src/negmas/preferences/inv_ufun/search.py",
    ],
)
def test_no_client_names_in_docs_and_code(path):
    """New docs and modules name no client project."""
    assert not FORBIDDEN.search((ROOT / path).read_text())


def test_no_client_names_in_history_and_value_functions():
    """The HISTORY entry and the value-functions section name no client project."""
    assert not FORBIDDEN.search(_unreleased_history())
    vf = (ROOT / "docs/value_functions.rst").read_text()
    sec = vf[vf.index("List-issue functions") : vf.index("Regenerating the figures")]
    assert not FORBIDDEN.search(sec)


def test_docs_index_links_list_issues():
    """docs/index.rst links the new tutorial page."""
    assert "list_issues" in (ROOT / "docs/index.rst").read_text()
