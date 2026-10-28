"""Open-box tests for chunk 3: optimization, inverters and negotiation over list issues."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import negmas.outcomes.list_issue as list_issue_module
import negmas.preferences as P
from negmas.gb.negotiators import WARNegotiator
from negmas.outcomes import ListIssue, RecordIssue, make_issue, make_os
from negmas.outcomes.outcome_space import (
    CartesianOutcomeSpace,
    DiscreteCartesianOutcomeSpace,
)
from negmas.preferences import (
    LinearAdditiveUtilityFunction,
    ListAggregationFun,
    MappingUtilityFunction,
    RecordAdditiveFun,
    SearchInverseUtilityFunction,
)
from negmas.preferences.inv_ufun.search import mutate_outcome, needs_search, search_pool
from negmas.preferences.value_fun import AffineFun, LinearFun, TableFun
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import BoulwareTBNegotiator

INVERTERS = [
    "AdaptiveInverseUtilityFunction",
    "SearchInverseUtilityFunction",
    "PresortingInverseUtilityFunction",
    "PresortingLegacyInverseUtilityFunction",
    "SamplingInverseUtilityFunction",
    "BIDSInverseUtilityFunction",
    "MCTSInverseUtilityFunction",
    "AttributePlanningInverseUtilityFunction",
    "BruteForceInverseUtilityFunction",
]
TREE_INVERTERS = [
    "BIDSInverseUtilityFunction",
    "MCTSInverseUtilityFunction",
    "AttributePlanningInverseUtilityFunction",
]
PARETO = [
    "BruteForceParetoSampler",
    "IPSParetoSampler",
    "MOBANOSParetoSampler",
    "NB3ParetoSampler",
]


def seed(n: int = 0) -> None:
    random.seed(n)
    np.random.seed(n)


@pytest.fixture
def no_enumeration(monkeypatch):
    """Fails the test if a list issue or an outcome space is enumerated."""

    def boom(*args, **kwargs):
        raise AssertionError("enumerated")

    monkeypatch.setattr(list_issue_module.ListIssue, "all", property(boom))
    monkeypatch.setattr(DiscreteCartesianOutcomeSpace, "enumerate", boom)
    monkeypatch.setattr(CartesianOutcomeSpace, "enumerate", boom, raising=False)


def scenario(max_count=None, aggregate="mean", ordered=False):
    lot = RecordIssue([make_issue((1, 20), "price"), make_issue((1, 10), "quantity")])
    lots = ListIssue("lots", lot, min_count=1, max_count=max_count, ordered=ordered)
    os = make_os([make_issue(["fast", "slow"], "ship"), lots])
    buyer = LinearAdditiveUtilityFunction(
        [
            TableFun({"fast": 1.0, "slow": 0.0}),
            ListAggregationFun(
                RecordAdditiveFun((AffineFun(-1.0, 20.0), AffineFun(1.0))),
                aggregate=aggregate,
            ),
        ],
        outcome_space=os,
    ).normalize()
    seller = LinearAdditiveUtilityFunction(
        [
            TableFun({"fast": 0.0, "slow": 1.0}),
            ListAggregationFun(
                RecordAdditiveFun((AffineFun(1.0), AffineFun(0.5))), aggregate=aggregate
            ),
        ],
        outcome_space=os,
    ).normalize()
    return os, buyer, seller


def small_space():
    """A 111-outcome space: x in {0,1,2} and up to 2 lots keyed by k."""
    lot = RecordIssue([make_issue(["a", "b", "c"], "k"), make_issue((0, 2), "q")])
    lots = ListIssue("L", lot, max_count=2, unique_by="k")
    return make_os([make_issue(3, "x"), lots])


def small_pair(os):
    u1 = LinearAdditiveUtilityFunction(
        [
            LinearFun(1.0),
            ListAggregationFun(
                RecordAdditiveFun(
                    (TableFun({"a": 1.0, "b": -1.0, "c": 0.5}), AffineFun(-1.0, 1.0))
                ),
                aggregate="sum",
            ),
        ],
        outcome_space=os,
    )
    u2 = LinearAdditiveUtilityFunction(
        [
            LinearFun(-1.0),
            ListAggregationFun(
                RecordAdditiveFun(
                    (TableFun({"a": -1.0, "b": 1.0, "c": 0.2}), AffineFun(1.0))
                ),
                aggregate="sum",
            ),
        ],
        outcome_space=os,
    )
    return u1, u2


# ---------------------------------------------------------------------------
# needs_search / mutate_outcome / search_pool
# ---------------------------------------------------------------------------


def test_needs_search_truth_table():
    """needs_search is True only for large or infinite spaces with a list (or
    non-range infinite) issue; small finite spaces are enumerated exactly."""
    assert not needs_search(small_space())
    assert needs_search(scenario()[0])
    assert not needs_search(make_os([make_issue(3, "x"), make_issue((0.0, 1.0), "y")]))
    assert not needs_search(make_os([make_issue(["a", "b"], "c")]))
    assert not needs_search(None)


def test_mutate_outcome_changes_at_most_one_issue_and_stays_valid():
    """mutate_outcome returns a valid outcome differing in at most one issue."""
    seed()
    os, _, _ = scenario(4)
    o = os.random_outcome()
    for _ in range(200):
        c = mutate_outcome(os, o)
        assert os.is_valid(c)
        assert sum(a != b for a, b in zip(o, c)) <= 1
        o = c


def test_search_pool_valid_unique_and_contains_extremes():
    """search_pool returns distinct valid outcomes that include the given extremes."""
    seed()
    os, u, _ = scenario()
    worst, best = u.extreme_outcomes()
    pool = search_pool(u, os, (worst, best), n_samples=100, n_targets=10, n_steps=5)
    assert len(pool) == len(set(pool))
    assert all(os.is_valid(_) for _ in pool)
    assert worst in pool and best in pool


def test_search_pool_respects_outcome_space_constraints():
    """search_pool never returns an outcome violating the space's constraints."""
    seed()
    base, u, _ = scenario()
    os = CartesianOutcomeSpace(base.issues, constraints=[lambda o: len(o[1]) <= 2])
    pool = search_pool(u, os, u.extreme_outcomes(), n_samples=200)
    assert pool and all(len(o[1]) <= 2 for o in pool)


def test_search_pool_empty_without_samples_or_extremes():
    """With no samples and no extremes the pool is empty (not an error)."""
    os, u, _ = scenario()
    assert search_pool(u, os, (None, None), n_samples=0) == []


# ---------------------------------------------------------------------------
# SearchInverseUtilityFunction
# ---------------------------------------------------------------------------


def test_search_inverter_without_samples_still_has_extremes(no_enumeration):
    """n_samples=0 still yields the exact best and worst (from extreme_outcomes)."""
    seed()
    _, u, _ = scenario()
    inv = SearchInverseUtilityFunction(u, n_samples=0)
    inv.init()
    assert u(inv.best()) == pytest.approx(1.0)
    assert u(inv.worst()) == pytest.approx(0.0)


def test_search_inverter_is_not_persisted():
    """The random pool is never persisted."""
    _, u, _ = scenario()
    assert SearchInverseUtilityFunction(u).persistable_config() is None


def test_search_inverter_rational_only_never_returns_irrational(no_enumeration):
    """rational_only: worst/one_in never go below the reserved value."""
    seed()
    _, u, _ = scenario()
    u.reserved_value = 0.5
    inv = SearchInverseUtilityFunction(u, rational_only=True)
    inv.init()
    assert u(inv.worst()) >= 0.5 - 1e-9
    o = inv.one_in((0.0, 0.3), normalized=True)
    assert o is None or u(o) >= 0.5 - 1e-9


def test_search_inverter_respects_outcome_space_constraints(no_enumeration):
    """Everything the search inverter returns satisfies the space's constraints."""
    seed()
    base, b, _ = scenario()
    os = CartesianOutcomeSpace(base.issues, constraints=[lambda o: o[0] == "slow"])
    u = LinearAdditiveUtilityFunction(b.values, b.weights, outcome_space=os)
    inv = SearchInverseUtilityFunction(u)
    inv.init()
    assert all(o[0] == "slow" for o in inv.outcomes)


def test_search_inverter_best_in_worst_in_exact_over_pool():
    """best_in/worst_in return the max/min pool utility inside the range."""
    seed()
    os = small_space()
    u, _ = small_pair(os)
    inv = SearchInverseUtilityFunction(u)
    inv.init()
    pool = [float(u(o)) for o in inv.outcomes]
    brute = [float(u(o)) for o in os.enumerate()]
    for lo, hi in ((-1.0, 0.5), (0.0, 2.0), (1.5, 3.5), (-5.0, 10.0)):
        inside = [x for x in pool if lo <= x <= hi]
        best = float(u(inv.best_in((lo, hi), normalized=False)))
        worst = float(u(inv.worst_in((lo, hi), normalized=False)))
        assert best == pytest.approx(max(inside))
        assert worst == pytest.approx(min(inside))
        assert best <= max(x for x in brute if lo <= x <= hi) + 1e-9


def test_adaptive_forwards_rational_only_to_search_delegate():
    """Adaptive passes rational_only/eps to the search delegate."""
    _, u, _ = scenario()
    inv = P.AdaptiveInverseUtilityFunction(u, rational_only=True, eps=1e-3)
    inv.init()
    assert isinstance(inv.delegate, SearchInverseUtilityFunction)
    assert inv.delegate.rational_only and inv.delegate._eps == 1e-3


# ---------------------------------------------------------------------------
# All inverters: greedy optimality vs brute force (DESIGN Part A 4 / 6)
# ---------------------------------------------------------------------------


@st.composite
def finite_instances(draw):
    keys = ["a", "b", "c"][: draw(st.integers(2, 3))]
    scores = {k: draw(st.floats(-2, 2, allow_nan=False)) for k in keys}
    slope = draw(st.floats(-1, 1, allow_nan=False))
    lo = draw(st.integers(0, 1))
    hi = draw(st.integers(max(lo, 1), 2))
    unique = draw(st.booleans())
    ordered = draw(st.booleans())
    agg = draw(st.sampled_from(["sum", "mean", "min", "max"]))
    wsign = draw(st.sampled_from([1.0, -1.0]))
    lot = RecordIssue([make_issue(keys, "k"), make_issue((0, 1), "q")])
    lots = ListIssue(
        "L",
        lot,
        min_count=lo,
        max_count=hi,
        unique_by="k" if unique else None,
        ordered=ordered,
    )
    os = make_os([make_issue(2, "x"), lots])
    u = LinearAdditiveUtilityFunction(
        [
            LinearFun(0.5),
            ListAggregationFun(
                RecordAdditiveFun((TableFun(scores), AffineFun(slope))), aggregate=agg
            ),
        ],
        weights=[1.0, wsign],
        outcome_space=os,
    )
    return os, u


@pytest.mark.parametrize("name", INVERTERS)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(inst=finite_instances())
def test_every_inverter_finds_exact_extremes_on_small_list_spaces(name, inst):
    """best()/worst() of every inverter equal the brute-force max/min."""
    seed()
    os, u = inst
    utils = [float(u(o)) for o in os.enumerate()]
    inv = getattr(P, name)(u)
    inv.init()
    assert float(u(inv.best())) == pytest.approx(max(utils), abs=1e-9)
    assert float(u(inv.worst())) == pytest.approx(min(utils), abs=1e-9)


@pytest.mark.parametrize("name", INVERTERS)
def test_inverters_on_mixed_space_with_continuous_and_two_lists(name, no_enumeration):
    """A continuous issue plus two list issues (one ordered, one infinite) works."""
    seed()
    lot = RecordIssue([make_issue((1, 20), "p"), make_issue((0.0, 1.0), "r")])
    os = make_os(
        [
            make_issue((0.0, 1.0), "c"),
            ListIssue("A", lot, min_count=1, max_count=3),
            ListIssue("B", make_issue(5, "e"), ordered=True),
        ]
    )
    u = LinearAdditiveUtilityFunction(
        [
            LinearFun(1.0),
            ListAggregationFun(
                RecordAdditiveFun((AffineFun(-1.0, 20.0), LinearFun(5.0))),
                aggregate="mean",
            ),
            ListAggregationFun(LinearFun(1.0), aggregate="max"),
        ],
        outcome_space=os,
    ).normalize()
    inv = getattr(P, name)(u)
    inv.init()
    assert u(inv.best()) == pytest.approx(1.0)
    assert u(inv.worst()) == pytest.approx(0.0)
    assert os.is_valid(inv.one_in((0.4, 0.6), normalized=True))


@pytest.mark.parametrize("name", INVERTERS)
def test_inverters_on_degenerate_empty_only_list(name):
    """A list issue whose only value is the empty list does not break inverters."""
    os = make_os([ListIssue("E", make_issue(3, "e"), max_count=0)])
    u = LinearAdditiveUtilityFunction(
        [ListAggregationFun(LinearFun(1.0))], outcome_space=os
    )
    inv = getattr(P, name)(u)
    inv.init()
    assert inv.best() == ((),) and inv.worst() == ((),)
    assert inv.one_in((0.0, 1.0), normalized=True) == ((),)


# ---------------------------------------------------------------------------
# Tree/DP inverters (BIDS, MCTS, AttributePlanning) use _issue_values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "BIDSInverseUtilityFunction",
        "MCTSInverseUtilityFunction",
        "AttributePlanningInverseUtilityFunction",
    ],
)
def test_tree_inverter_list_candidates_include_greedy_extremes(name):
    """The per-issue values of BIDS/MCTS/AP contain the list issue's greedy extremes."""
    seed()
    _, u, _ = scenario()
    inv = getattr(P, name)(u)
    inv.init()
    (_, worst), (_, best) = u.values[1].arg_extremes(u.outcome_space.issues[1])
    assert worst in inv._val_list[1] and best in inv._val_list[1]


@pytest.mark.parametrize(
    "name", ["BIDSInverseUtilityFunction", "AttributePlanningInverseUtilityFunction"]
)
def test_tree_inverter_bounds_match_true_utility_range(name):
    """The bounds used for normalized queries equal the ufun's true minmax."""
    seed()
    _, u, _ = scenario()
    inv = getattr(P, name)(u)
    inv.init()
    lo, hi = u.minmax()
    assert inv._u_min == pytest.approx(lo, abs=1e-6)
    assert inv._u_max == pytest.approx(hi, abs=1e-6)


@pytest.mark.parametrize("name", TREE_INVERTERS)
def test_tree_inverter_one_in_hits_mid_range_on_list_space(name):
    """one_in on a mid-range target returns an outcome near that target (seeds 0..7)."""
    for s in range(8):
        seed(s)
        _, u, _ = scenario()
        inv = getattr(P, name)(u)
        inv.init()
        o = inv.one_in((0.25, 0.35), normalized=True)
        assert abs(float(u(o)) - 0.3) <= 0.1, (s, float(u(o)))


# ---------------------------------------------------------------------------
# No enumeration through extreme_outcomes' fallbacks (DESIGN Part A 4 / 6)
# ---------------------------------------------------------------------------


def constrained(os, b):
    return LinearAdditiveUtilityFunction(
        b.values, b.weights, outcome_space=os, constraints=[lambda o: len(o[1]) <= 3]
    )


def mapping(os, b):
    return MappingUtilityFunction(lambda o: float(b(o)), outcome_space=os)


@pytest.mark.parametrize("make", [constrained, mapping])
def test_search_inverter_without_greedy_hook_works_on_infinite_list_space(
    make, no_enumeration
):
    """Ufuns without arg_extremes (constraints, generic) are sampled on infinite spaces."""
    seed()
    os, b, _ = scenario()
    inv = SearchInverseUtilityFunction(make(os, b))
    inv.init()
    assert os.is_valid(inv.best())


@pytest.mark.parametrize("make", [constrained, mapping])
def test_search_inverter_without_greedy_hook_never_enumerates_huge_finite_list_space(
    make, no_enumeration
):
    """Ufuns without arg_extremes must not enumerate a finite-but-huge list space."""
    seed()
    os, b, _ = scenario(4)
    inv = SearchInverseUtilityFunction(make(os, b))
    inv.init()
    assert os.is_valid(inv.best())


def test_war_on_huge_finite_list_space_does_not_enumerate(no_enumeration):
    """WAR on a finite (1.4e8) list space runs without enumerating it."""
    seed()
    os, b, s = scenario(4)
    m = SAOMechanism(outcome_space=os, n_steps=30)
    m.add(WARNegotiator(name="war"), ufun=b)
    m.add(BoulwareTBNegotiator(name="b"), ufun=s)
    m.run()
    assert m.agreement is None or os.is_valid(m.agreement)


# ---------------------------------------------------------------------------
# Pareto samplers: list encoding agrees with an equivalent categorical encoding
# ---------------------------------------------------------------------------


def categorical_copy(os, u1, u2):
    lst = os.issues[1]
    vals = list(lst.all)
    names = [str(v) for v in vals]
    cos = make_os([os.issues[0], make_issue(names, "C")])

    def copy(u):
        f = u.values[1]
        return LinearAdditiveUtilityFunction(
            [u.values[0], TableFun({n: float(f(v)) for n, v in zip(names, vals)})],
            weights=u.weights,
            outcome_space=cos,
        )

    return copy(u1), copy(u2)


@pytest.mark.parametrize("name", PARETO)
def test_pareto_front_on_small_list_space_equals_categorical_encoding(name):
    """On a small finite list space each sampler finds the same front as on a
    categorical encoding of the same values (so list handling adds no loss)."""
    seed()
    os = small_space()
    u1, u2 = small_pair(os)
    c1, c2 = categorical_copy(os, u1, u2)

    def pts(ps, a, b):
        ps.init(a, b)
        return {
            (round(float(a(o)), 9), round(float(b(o)), 9)) for o in ps.pareto_outcomes()
        }

    assert pts(getattr(P, name)(), u1, u2) == pts(getattr(P, name)(), c1, c2)


# ---------------------------------------------------------------------------
# SAO negotiation (DESIGN Part A 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize("max_count", [None, 4])
def test_boulware_vs_boulware_offers_valid_and_agreement_rational(
    max_count, ordered, no_enumeration
):
    """Across seeds: every offer is valid and the agreement is rational for both."""
    for s in range(3):
        seed(s)
        os, b, sel = scenario(max_count, ordered=ordered)
        b.reserved_value, sel.reserved_value = 0.2, 0.2
        m = SAOMechanism(outcome_space=os, n_steps=100)
        m.add(BoulwareTBNegotiator(name="buyer"), ufun=b)
        m.add(BoulwareTBNegotiator(name="seller"), ufun=sel)
        m.run()
        assert m.agreement is not None, s
        assert os.is_valid(m.agreement)
        assert b(m.agreement) >= 0.2 and sel(m.agreement) >= 0.2
        offers = [o for _, _, o in m.extended_trace if o is not None]
        assert offers and all(os.is_valid(o) for o in offers)
        assert all(math.isfinite(float(b(o))) for o in offers)
