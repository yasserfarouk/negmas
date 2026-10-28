"""Inverters, Pareto samplers and negotiators on outcome spaces with list issues."""

from __future__ import annotations

import pytest

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
    RecordAdditiveFun,
    SearchInverseUtilityFunction,
)
from negmas.preferences.value_fun import AffineFun, LinearFun, TableFun
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import (
    AspirationNegotiator,
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
    NaiveTitForTatNegotiator,
)

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
PARETO = [
    "AdaptiveParetoSampler",
    "BruteForceParetoSampler",
    "IPSParetoSampler",
    "MOBANOSParetoSampler",
    "NB3ParetoSampler",
]
# None: infinite; 4: finite but far too large to enumerate (~7e7 lists)
COUNTS = [None, 4]


@pytest.fixture
def no_enumeration(monkeypatch):
    """Fails the test if a list issue or an outcome space is enumerated."""

    def boom(*args, **kwargs):
        raise AssertionError("enumerated")

    monkeypatch.setattr(list_issue_module.ListIssue, "all", property(boom))
    monkeypatch.setattr(DiscreteCartesianOutcomeSpace, "enumerate", boom)
    monkeypatch.setattr(CartesianOutcomeSpace, "enumerate", boom, raising=False)


def scenario(max_count=None, unique_by=None, aggregate="mean"):
    lot = RecordIssue([make_issue((1, 20), "price"), make_issue((1, 10), "quantity")])
    lots = ListIssue("lots", lot, min_count=1, max_count=max_count, unique_by=unique_by)
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


@pytest.mark.parametrize("max_count", COUNTS)
@pytest.mark.parametrize("name", INVERTERS)
def test_inverters_never_enumerate_and_find_extremes(name, max_count, no_enumeration):
    os, u, _ = scenario(max_count)
    inv = getattr(P, name)(u)
    inv.init()
    assert u(inv.best()) == pytest.approx(1.0)
    assert u(inv.worst()) == pytest.approx(0.0)
    o = inv.one_in((0.3, 0.7), normalized=True)
    assert o is not None and os.is_valid(o)


def test_search_inverter_covers_the_utility_range(no_enumeration):
    _, u, _ = scenario()
    inv = SearchInverseUtilityFunction(u)
    inv.init()
    for t in (0.1, 0.3, 0.5, 0.7, 0.9):
        o = inv.one_in((t - 0.05, t + 0.05), normalized=True)
        assert abs(u(o) - t) <= 0.05 + 1e-9


def test_adaptive_selects_search_for_list_spaces():
    _, u, _ = scenario(4)
    inv = P.AdaptiveInverseUtilityFunction(u)
    inv.init()
    assert isinstance(inv.delegate, SearchInverseUtilityFunction)


def test_search_inverter_on_small_finite_space_finds_exact_extremes():
    lot = RecordIssue([make_issue(["a", "b"], "k"), make_issue((0, 2), "q")])
    os = make_os([make_issue(2, "x"), ListIssue("L", lot, max_count=2, unique_by="k")])
    u = LinearAdditiveUtilityFunction(
        [
            LinearFun(1.0),
            ListAggregationFun(
                RecordAdditiveFun(
                    (TableFun({"a": 1.0, "b": -1.0}), AffineFun(-1.0, 1.0))
                )
            ),
        ],
        outcome_space=os,
    )
    utils = sorted(u(o) for o in os.enumerate())
    inv = SearchInverseUtilityFunction(u)
    inv.init()
    assert u(inv.best()) == utils[-1] and u(inv.worst()) == utils[0]


@pytest.mark.parametrize("max_count", COUNTS)
@pytest.mark.parametrize("name", PARETO)
def test_pareto_samplers_never_enumerate(name, max_count, no_enumeration):
    os, b, s = scenario(max_count)
    ps = getattr(P, name)()
    ps.init(b, s)
    front = list(ps.pareto_outcomes())
    assert front and all(os.is_valid(_) for _ in front)
    # the two ideal outcomes bound the frontier
    assert max(b(_) for _ in front) == pytest.approx(1.0, abs=0.05)
    assert max(s(_) for _ in front) == pytest.approx(1.0, abs=0.05)


def run(a, b, max_count=None, unique_by=None, aggregate="mean", n_steps=100):
    os, buyer, seller = scenario(max_count, unique_by, aggregate)
    m = SAOMechanism(outcome_space=os, n_steps=n_steps)
    m.add(a(name="buyer"), ufun=buyer)
    m.add(b(name="seller"), ufun=seller)
    m.run()
    return m, os, buyer, seller


@pytest.mark.parametrize("max_count", COUNTS)
def test_boulware_vs_boulware_reaches_agreement(max_count, no_enumeration):
    m, os, buyer, seller = run(BoulwareTBNegotiator, BoulwareTBNegotiator, max_count)
    assert m.agreement is not None
    assert os.is_valid(m.agreement)
    assert buyer(m.agreement) > buyer.reserved_value
    assert seller(m.agreement) > seller.reserved_value


def test_boulware_with_unique_by_and_sum(no_enumeration):
    m, os, *_ = run(
        BoulwareTBNegotiator,
        BoulwareTBNegotiator,
        max_count=6,
        aggregate="sum",
        unique_by="price",
    )
    assert m.agreement is not None and os.is_valid(m.agreement)
    prices = [r[0] for r in m.agreement[1]]
    assert len(prices) == len(set(prices))


@pytest.mark.parametrize(
    "neg",
    [
        ConcederTBNegotiator,
        LinearTBNegotiator,
        AspirationNegotiator,
        NaiveTitForTatNegotiator,
    ],
)
def test_other_builtin_negotiators_reach_agreement(neg, no_enumeration):
    m, os, *_ = run(neg, BoulwareTBNegotiator)
    assert m.agreement is not None and os.is_valid(m.agreement)


def test_war_declares_infinite_spaces_unsupported():
    m, *_ = run(WARNegotiator, BoulwareTBNegotiator)
    assert "infinite outcome spaces" in (m.state.error_details or "")
