from __future__ import annotations
import math
import random
from importlib.resources import files
from math import isnan

import hypothesis.strategies as st
import numpy as np
from negmas.outcomes.contiguous_issue import ContiguousIssue
import pytest
from hypothesis import given, settings
from hypothesis.core import example
from pytest import mark

from negmas.common import PreferencesChangeType
from negmas.inout import Scenario
from negmas.outcomes import enumerate_issues, issues_from_xml_str, make_issue
from negmas.outcomes.outcome_space import CartesianOutcomeSpace, make_os
from negmas.preferences import (
    AffineUtilityFunction,
    HyperRectangleUtilityFunction,
    LinearAdditiveUtilityFunction,
    LinearUtilityFunction,
    MappingUtilityFunction,
    UtilityFunction,
    pareto_frontier,
)
from negmas.preferences.crisp.const import ConstUtilityFunction
from negmas.preferences.inv_ufun import PresortingInverseUtilityFunction
from negmas.preferences.ops import (
    calc_outcome_distances,
    calc_outcome_optimality,
    calc_reserved_value,
    calc_scenario_stats,
    estimate_max_dist,
    estimate_max_dist_using_outcomes,
    is_rational,
    make_rank_ufun,
    normalize,
    pareto_frontier_bf,
    pareto_frontier_numpy,
    scale_max,
)
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator


@mark.parametrize(["n_issues"], [(2,), (3,)])
def test_preferences_range_linear_late_issues(n_issues):
    issues = tuple(make_issue(values=(0.0, 1.0), name=f"i{i}") for i in range(n_issues))
    rs = [(i + 1.0) * random.random() for i in range(n_issues)]
    ufun = LinearUtilityFunction(weights=rs, reserved_value=0.0, issues=issues)
    assert ufun([0.0] * n_issues) == 0.0
    assert ufun([1.0] * n_issues) == sum(rs)
    rng = ufun.minmax(issues=issues)
    assert rng[0] >= 0.0
    assert rng[1] <= sum(rs)


@mark.parametrize(["n_issues"], [(2,), (3,)])
def test_preferences_range_linear(n_issues):
    issues = tuple(make_issue(values=(0.0, 1.0), name=f"i{i}") for i in range(n_issues))
    rs = [(i + 1.0) * random.random() for i in range(n_issues)]
    ufun = LinearUtilityFunction(weights=rs, reserved_value=0.0, issues=issues)
    assert ufun([0.0] * n_issues) == 0.0
    assert ufun([1.0] * n_issues) == sum(rs)
    rng = ufun.minmax()
    assert rng[0] >= 0.0
    assert rng[1] <= sum(rs)


@mark.parametrize(["n_issues"], [(2,), (3,)])
def test_preferences_range_general(n_issues):
    issues = tuple(make_issue(values=(0.0, 1.0), name=f"i{i}") for i in range(n_issues))
    rs = [(i + 1.0) * random.random() for i in range(n_issues)]
    ufun = MappingUtilityFunction(
        mapping=lambda x: sum(r * v for r, v in zip(rs, x)) if x else float("nan")
    )
    assert ufun([0.0] * n_issues) == 0.0
    assert ufun([1.0] * n_issues) == sum(rs)
    rng = ufun.minmax(issues=issues)
    assert rng[0] >= 0.0
    assert rng[1] <= sum(rs)


@given(
    n_outcomes=st.integers(2, 1000),
    n_negotiators=st.integers(2, 5),
    normalized=st.booleans(),
    sort=st.booleans(),
    r0=st.floats(-1.0, 1.0),
    r1=st.floats(-1.0, 1.0),
)
@settings(deadline=60_000)
@example(
    n_outcomes=2, n_negotiators=2, normalized=False, sort=False, r0=0.0, r1=0.5888671875
)
@example(n_outcomes=2, n_negotiators=2, normalized=False, sort=False, r0=0.0, r1=0.0)
def test_calc_outcome_stats(n_outcomes, n_negotiators, normalized, sort, r0, r1):
    def _test_optim_allow_above1(d, outcome, lst):
        if not lst:
            assert isnan(d)
            return
        assert 0 <= d
        if outcome in lst:
            assert abs(1 - d) < 1e-12
            return

    def _test_optim(d, outcome, lst):
        if not lst:
            assert isnan(d)
            return
        assert 0 <= d <= 1
        if outcome in lst:
            assert abs(1 - d) < 1e-12
            return
        assert 1 - d > 1e-12

    def _test_dist(d, outcome, lst):
        if not lst:
            assert isnan(d)
            return
        if outcome in lst:
            assert abs(d) < 1e-12
            return
        assert abs(d) > 1e-12

    np.random.seed(0)
    os = make_os([make_issue(n_outcomes)])
    outcomes = os.enumerate_or_sample()
    utils = np.random.rand(n_negotiators, n_outcomes)
    if not normalized:
        utils *= 100
        r0 *= 100
        r1 *= 100
    ufuns = [
        MappingUtilityFunction(
            dict(zip(outcomes, _)), outcome_space=os, reserved_value=r0 if not i else r1
        )
        for i, _ in enumerate(utils)
    ]
    stats = calc_scenario_stats(tuple(ufuns))
    [tuple(u(_) for u in ufuns) for _ in outcomes]
    mxoverall = estimate_max_dist(ufuns)
    mxpareto = estimate_max_dist_using_outcomes(ufuns, stats.pareto_utils)
    assert mxoverall >= mxpareto
    for outcome in outcomes:
        outils = tuple(u(outcome) for u in ufuns)
        dists = calc_outcome_distances(outils, stats)
        optim_overall = calc_outcome_optimality(dists, stats, max_dist=mxoverall)
        # optim_pareto = calc_outcome_optimality(dists, stats, max_dist=mxpareto)
        # optim_exact = calc_outcome_optimality(dists, stats, outcome_utils=allutils)
        if is_rational(ufuns, outcome):
            _test_optim_allow_above1(
                optim_overall.max_welfare_optimality,
                outcome,
                stats.max_welfare_outcomes,
            )
            for d, o1, lst in zip(
                (
                    dists.pareto_dist,
                    dists.nash_dist,
                    dists.kalai_dist,
                    dists.ks_dist,
                    # dists.max_relative_welfare,
                ),
                (
                    optim_overall.pareto_optimality,
                    optim_overall.nash_optimality,
                    optim_overall.kalai_optimality,
                    optim_overall.ks_optimality,
                    # optim_overall.max_relative_welfare_optimality,
                ),
                (
                    stats.pareto_outcomes,
                    stats.nash_outcomes,
                    stats.kalai_outcomes,
                    stats.ks_outcomes,
                    # stats.max_relative_welfare_outcomes,
                ),
                strict=True,
            ):
                _test_dist(d, outcome, lst)
                _test_optim(o1, outcome, lst)
        else:
            for lst in (
                stats.pareto_outcomes,
                stats.nash_outcomes,
                stats.kalai_outcomes,
                stats.ks_outcomes,
                # stats.max_relative_welfare_outcomes,
            ):
                assert outcome not in lst
                for d in (
                    dists.pareto_dist,
                    dists.nash_dist,
                    dists.kalai_dist,
                    dists.ks_dist,
                    # dists.max_relative_welfare,
                ):
                    _test_dist(d, outcome, lst)
                for o1 in (
                    optim_overall.pareto_optimality,
                    optim_overall.nash_optimality,
                    optim_overall.kalai_optimality,
                    optim_overall.ks_optimality,
                    # optim_overall.max_relative_welfare_optimality,
                ):
                    _test_optim(o1, outcome, lst)
                _test_optim_allow_above1(
                    optim_overall.max_welfare_optimality,
                    outcome,
                    stats.max_welfare_outcomes,
                )


@given(n_outcomes=st.integers(2, 1000), r0=st.floats(-1.0, 1.0), f=st.floats(0.0, 1.0))
@settings(deadline=60_000)
@example(n_outcomes=2, r0=0.0, f=0.5)
def test_calc_reserved(n_outcomes, r0, f):
    np.random.seed(0)
    os = make_os([make_issue(n_outcomes)])
    outcomes = os.enumerate_or_sample()
    utils = np.random.rand(n_outcomes)
    ufun = MappingUtilityFunction(
        dict(zip(outcomes, utils)), outcome_space=os, reserved_value=r0
    )
    r = calc_reserved_value(ufun, f)
    ufun.reserved_value = r
    nrational = sum(is_rational([ufun], _) for _ in outcomes)
    assert nrational == math.ceil(f * n_outcomes), (
        f"Got {nrational} outcomes for reserved value {r} of {n_outcomes} outcomes when using fraction {f}"
    )


def test_calc_reserved_fifty_fifty():
    folder_name = str(files("negmas").joinpath("tests/data/FiftyFifty"))

    d = Scenario.from_genius_folder(folder_name, ignore_discount=True)
    assert d is not None and d.outcome_space is not None and d.ufuns is not None
    d.normalize()
    n_outcomes = d.outcome_space.cardinality
    outcomes = d.outcome_space.enumerate_or_sample()
    f = 1.0
    for ufun in d.ufuns:
        r = calc_reserved_value(ufun, f)
        ufun.reserved_value = r
        nrational = sum(is_rational([ufun], _) for _ in outcomes)
        assert nrational == math.ceil(f * n_outcomes), (
            f"Got {nrational} outcomes for reserved value {r} of {n_outcomes} outcomes when using fraction {f}"
        )
        assert r <= 0.0, f"{r=}"


@given(
    n_outcomes=st.integers(2, 1000),
    n_negotiators=st.integers(2, 5),
    normalize=st.booleans(),
    r0=st.floats(-1.0, 1.0),
    r1=st.floats(-1.0, 1.0),
)
@settings(deadline=60_000)
def test_ranks_match_bf(n_outcomes, n_negotiators, normalize, r0, r1):
    np.random.seed(0)
    os = make_os([make_issue(n_outcomes)])
    outcomes = os.enumerate_or_sample()
    utils = np.random.rand(n_negotiators, n_outcomes)
    ufuns = [
        MappingUtilityFunction(
            dict(zip(outcomes, _)), outcome_space=os, reserved_value=r0 if not i else r1
        )
        for i, _ in enumerate(utils)
    ]
    rank_ufuns = [make_rank_ufun(_, normalize=normalize) for _ in ufuns]
    rank_ufuns_bf = [make_rank_ufun(_, normalize=normalize) for _ in ufuns]
    for u, ubf in zip(rank_ufuns, rank_ufuns_bf):
        assert abs(u.reserved_value - ubf.reserved_value) < 1e-10
        for outcome in list(outcomes) + [None]:
            assert abs(u(outcome) - ubf(outcome)) < 1e-12


@given(
    n_outcomes=st.integers(2, 1000),
    n_negotiators=st.integers(2, 5),
    sort=st.booleans(),
    r0=st.floats(-1.0, 1.0),
    r1=st.floats(-1.0, 1.0),
)
@settings(deadline=60_000)
def test_calc_stats_with_ranks(n_outcomes, n_negotiators, sort, r0, r1):
    def _test(x, bf):
        assert len(x) == len(bf), f"stats:{bf}\nglobal:{x}"
        assert len(bf) == len(set(bf)), f"stats:{bf}\nglobal:{x}"
        assert len(x) == len(set(x)), f"stats:{bf}\nglobal:{x}"
        if sort:
            assert all(list(a == b for a, b in zip(x, bf))), f"b:{bf}\nglobal:{x}"
        else:
            assert set(x) == set(bf), f"bf:{bf}\nglobal:{x}"

    np.random.seed(0)
    os = make_os([make_issue(n_outcomes)])
    outcomes = os.enumerate_or_sample()
    utils = np.random.rand(n_negotiators, n_outcomes)
    ufuns = [
        MappingUtilityFunction(
            dict(zip(outcomes, _)), outcome_space=os, reserved_value=r0 if not i else r1
        )
        for i, _ in enumerate(utils)
    ]
    stats = calc_scenario_stats(tuple(ufuns))
    rank_ufuns = [make_rank_ufun(_) for _ in ufuns]
    rank_stats = calc_scenario_stats(rank_ufuns)
    # applying ranking may change the relative order of pareto outcomes (by changing welfare) but not the set
    _test(sorted(stats.pareto_outcomes), sorted(rank_stats.pareto_outcomes))


@given(
    n_outcomes=st.integers(2, 1000),
    n_negotiators=st.integers(2, 5),
    sort=st.booleans(),
    r0=st.floats(-1.0, 1.0),
    r1=st.floats(-1.0, 1.0),
)
@settings(deadline=60_000)
def test_calc_stats(n_outcomes, n_negotiators, sort, r0, r1):
    def _test(x, bf):
        assert len(x) == len(bf), f"stats:{bf}\nglobal:{x}"
        assert len(bf) == len(set(bf)), f"stats:{bf}\nglobal:{x}"
        assert len(x) == len(set(x)), f"stats:{bf}\nglobal:{x}"
        if sort:
            assert all(list(a == b for a, b in zip(x, bf))), f"b:{bf}\nglobal:{x}"
        else:
            assert set(x) == set(bf), f"bf:{bf}\nglobal:{x}"

    np.random.seed(0)
    os = make_os([make_issue(n_outcomes)])
    outcomes = list(os.enumerate_or_sample())
    utils = np.random.rand(n_negotiators, n_outcomes)
    ufuns = [
        MappingUtilityFunction(
            dict(zip(outcomes, _)), outcome_space=os, reserved_value=r0 if not i else r1
        )
        for i, _ in enumerate(utils)
    ]
    stats = calc_scenario_stats(tuple(ufuns))
    bf, bfoutcomes = stats.pareto_utils, stats.pareto_outcomes
    x, xindices = pareto_frontier(ufuns, outcomes, sort_by_welfare=sort)
    xoutcomes = [outcomes[_] for _ in xindices]
    _test(x, bf)
    _test(xoutcomes, bfoutcomes)


@given(
    n_outcomes=st.integers(2, 1000),
    n_negotiators=st.integers(2, 5),
    sort=st.booleans(),
    r0=st.floats(-1.0, 1.0),
    r1=st.floats(-1.0, 1.0),
)
@settings(deadline=60_000)
@example(n_outcomes=4, n_negotiators=2, sort=False, r0=0.0, r1=0.0)
@example(n_outcomes=2, n_negotiators=2, sort=False, r0=0.0, r1=0.0)
def test_mechanism_pareto_frontier_matches_global(
    n_outcomes, n_negotiators, sort, r0, r1
):
    np.random.seed(0)
    os = make_os([make_issue(n_outcomes)])
    outcomes = os.enumerate_or_sample()
    utils = np.random.rand(n_negotiators, n_outcomes)
    m = SAOMechanism(outcome_space=os)
    ufuns = [
        MappingUtilityFunction(
            dict(zip(outcomes, _)), outcome_space=os, reserved_value=r0 if not i else r1
        )
        for i, _ in enumerate(utils)
    ]
    for u in ufuns:
        m.add(AspirationNegotiator(), preferences=u)
    bf, bfoutcomes = m.pareto_frontier(sort_by_welfare=sort)
    # bf = [_[0] for _ in results]
    # bfoutcomes = [_[1] for _ in results]
    outcomes = list(outcomes)
    x, xindices = pareto_frontier(ufuns, outcomes, sort_by_welfare=sort)
    xoutcomes = [outcomes[_] for _ in xindices]
    assert len(x) == len(bf), f"mech:{bf}\nglobal:{x}"
    assert len(bf) == len(set(bf)), f"mech:{bf}\nglobal:{x}"
    assert len(x) == len(set(x)), f"mech:{bf}\nglobal:{x}"
    if sort:
        assert all(list(a == b for a, b in zip(x, bf))), f"b:{bf}\nglobal:{x}"
    else:
        assert set(x) == set(bf), f"bf:{bf}\nglobal:{x}"

    assert len(xoutcomes) == len(bfoutcomes), f"mech:{bfoutcomes}\nglobal:{xoutcomes}"
    assert len(bfoutcomes) == len(set(bfoutcomes)), (
        f"mech:{bfoutcomes}\nglobal:{xoutcomes}"
    )
    assert len(xoutcomes) == len(set(xoutcomes)), (
        f"mech:{bfoutcomes}\nglobal:{xoutcomes}"
    )
    if sort:
        assert all(list(a == b for a, b in zip(xoutcomes, bfoutcomes))), (
            f"bfoutcomes:{bfoutcomes}\nglobal:{xoutcomes}"
        )
    else:
        assert set(xoutcomes) == set(bfoutcomes), (
            f"bfoutcomes:{bfoutcomes}\nglobal:{xoutcomes}"
        )


@given(
    n_outcomes=st.integers(2, 1000), n_negotiators=st.integers(2, 5), sort=st.booleans()
)
@settings(deadline=60_000)
def test_pareto_frontier_numpy_matches_bf2(n_outcomes, n_negotiators, sort):
    np.random.seed(0)
    utils = np.random.rand(n_outcomes, n_negotiators)
    bf = list(pareto_frontier_bf(utils, sort_by_welfare=sort))
    x = list(pareto_frontier_numpy(utils, sort_by_welfare=sort))
    assert len(x) == len(bf), f"bf:{bf}\nfast:{x}"
    assert len(bf) == len(set(bf)), f"bf:{bf}\nfast:{x}"
    assert len(x) == len(set(x)), f"bf:{bf}\nfast:{x}"
    if sort:
        assert all(list(a == b for a, b in zip(x, bf))), f"bf:{bf}\nfast:{x}"
    else:
        assert set(x) == set(bf), f"bf:{bf}\nfast:{x}"


@given(
    n_outcomes=st.integers(2, 1000), n_negotiators=st.integers(2, 5), sort=st.booleans()
)
@settings(deadline=60_000)
def test_pareto_frontier_numpy_matches_bf(n_outcomes, n_negotiators, sort):
    np.random.seed(0)
    utils = np.random.rand(n_outcomes, n_negotiators)
    bf = list(pareto_frontier_bf(utils, sort_by_welfare=sort))
    x = list(pareto_frontier_numpy(utils, sort_by_welfare=sort))
    assert len(x) == len(bf), f"bf:{bf}\nfast:{x}"
    assert len(bf) == len(set(bf)), f"bf:{bf}\nfast:{x}"
    assert len(x) == len(set(x)), f"bf:{bf}\nfast:{x}"
    if sort:
        assert all(list(a == b for a, b in zip(x, bf))), f"bf:{bf}\nfast:{x}"
    else:
        assert set(x) == set(bf), f"bf:{bf}\nfast:{x}"


def test_pareto_frontier_does_not_depend_on_order():
    u1 = [
        0.5337723805661662,
        0.8532272031479199,
        0.4781281413197942,
        0.7242899747791032,
        0.3461879818432919,
        0.2608677043479706,
        0.9419131964655383,
        0.29368079952747694,
        0.6093201983562316,
        0.7066918086398718,
    ]
    u2 = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    welfare = [_1 + _2 for _1, _2 in zip(u1, u2)]
    assert welfare.index(max(welfare)) == 3

    f1 = MappingUtilityFunction(lambda o: u1[o[0]] if o else float("nan"))
    f2 = MappingUtilityFunction(lambda o: u2[o[0]] if o else float("nan"))
    assert all(f1((i,)) == u1[i] for i in range(10))
    assert all(f2((i,)) == u2[i] for i in range(10))
    p1, l1 = pareto_frontier(
        [f1, f2], sort_by_welfare=True, outcomes=[(_,) for _ in range(10)]
    )
    p2, l2 = pareto_frontier(
        [f2, f1], sort_by_welfare=True, outcomes=[(_,) for _ in range(10)]
    )

    assert len(p1) == len(p2)
    # assert l2 == list(reversed(l1))

    assert set(l1) == {6, 3}
    assert set(l2) == {6, 3}
    assert set(p1) == {(0.9419131964655383, 0.0), (0.7242899747791032, 1.0)}
    assert set(p2) == {(1.0, 0.7242899747791032), (0.0, 0.9419131964655383)}
    # reverse order of p2
    p2 = [(_[1], _[0]) for _ in p2]
    for a in p1:
        assert a in p2


def test_linear_utility():
    buyer_utility = LinearAdditiveUtilityFunction(
        {  # type: ignore
            "cost": lambda x: -x,
            "number of items": lambda x: 0.5 * x,
            "delivery": {"delivered": 10.0, "not delivered": -2.0},
        },
        issues=[
            make_issue((0.0, 1.0), "cost"),
            make_issue(10, "number of items"),
            make_issue(["delivered", "not delivered"], "delivery"),
        ],
    )
    assert buyer_utility((1.0, 3, "not delivered")) == -1.0 + 0.5 * 3 - 2.0


def test_linear_utility_construction():
    buyer_utility = LinearAdditiveUtilityFunction(
        {  # type: ignore
            "cost": lambda x: -x,
            "number of items": lambda x: 0.5 * x,
            "delivery": {"delivered": 10.0, "not delivered": -2.0},
        },
        issues=[
            make_issue((0.0, 1.0), "cost"),
            make_issue(10, "number of items"),
            make_issue(["delivered", "not delivered"], "delivery"),
        ],
    )
    assert isinstance(buyer_utility, LinearAdditiveUtilityFunction)
    with pytest.raises(ValueError):
        LinearAdditiveUtilityFunction(
            {  # type: ignore
                "cost": lambda x: -x,
                "number of items": lambda x: 0.5 * x,
                "delivery": {"delivered": 10.0, "not delivered": -2.0},
            }
        )


def test_hypervolume_utility():
    f = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[
            5.0,
            2.0,
            lambda x: 2 * x[2] + x[0] if x is not None else float("nan"),
        ],
    )
    f_ignore_input = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]],
        ignore_issues_not_in_input=True,
    )
    f_ignore_failing = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]],
        ignore_failing_range_utilities=True,
    )
    f_ignore_both = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]],
        ignore_failing_range_utilities=True,
        ignore_issues_not_in_input=True,
    )

    g = HyperRectangleUtilityFunction(
        outcome_ranges=[{0: (1.0, 2.0), 1: (1.0, 2.0)}, {0: (1.4, 2.0), 2: (2.0, 3.0)}],
        utilities=[2.0, lambda x: 2 * x[2] + x[0]],
    )
    g_ignore_input = HyperRectangleUtilityFunction(
        outcome_ranges=[{0: (1.0, 2.0), 1: (1.0, 2.0)}, {0: (1.4, 2.0), 2: (2.0, 3.0)}],
        utilities=[2.0, lambda x: 2 * x[2] + x[0]],
        ignore_issues_not_in_input=True,
    )
    g_ignore_failing = HyperRectangleUtilityFunction(
        outcome_ranges=[{0: (1.0, 2.0), 1: (1.0, 2.0)}, {0: (1.4, 2.0), 2: (2.0, 3.0)}],
        utilities=[2.0, lambda x: 2 * x[2] + x[0]],
        ignore_failing_range_utilities=True,
    )
    g_ignore_both = HyperRectangleUtilityFunction(
        outcome_ranges=[{0: (1.0, 2.0), 1: (1.0, 2.0)}, {0: (1.4, 2.0), 2: (2.0, 3.0)}],
        utilities=[2.0, lambda x: 2 * x[2] + x[0]],
        ignore_failing_range_utilities=True,
        ignore_issues_not_in_input=True,
    )

    funs = [
        g,
        g_ignore_input,
        g_ignore_failing,
        g_ignore_both,
        f,
        f_ignore_input,
        f_ignore_failing,
        f_ignore_both,
    ]
    outcomes = [
        [1.5, 1.5, 2.5],  # belongs to all volumes
        [1.5, 1.5, 1.0],  # belongs to first
        {0: 1.5, 2: 2.5},
        {0: 11.5, 1: 11.5, 2: 12.5},
        [1.5],
        {2: 2.5},
    ]
    expected = [
        [8.5, 8.5, 8.5, 8.5, 13.5, 13.5, 13.5, 13.5],
        [2.0, 2.0, 2.0, 2.0, 7.0, 7.0, 7.0, 7.0],
        [None, 6.5, None, 6.5, None, 11.5, None, 11.5],
        [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
        [None, 0.0, None, 0.0, None, 5.0, None, 5.0],
        [None, 0.0, None, 0.0, None, 5.0, None, 5.0],
    ]

    for outcome, expectation in zip(outcomes, expected):
        utilities = [f(outcome) for f in funs]
        for i, (u, e) in enumerate(zip(utilities, expectation)):
            # print(i, utilities, outcome)
            assert e is None and np.isnan(u) or u == e


@mark.parametrize("utype", (LinearUtilityFunction, LinearAdditiveUtilityFunction))
def test_dict_conversion(utype):
    # def test_dict_conversion():
    #     utype = LinearUtilityFunction
    issues = [make_issue(10)]
    u = utype.random(issues=issues, normalized=False)
    d = u.to_dict()
    u2 = utype.from_dict(d)
    for o in enumerate_issues(issues):
        assert abs(u(o) - u2(o)) < 1e-3


def test_inverse_genius_domain():
    with open(
        str(files("negmas").joinpath("tests/data/Laptop/Laptop-C-domain.xml"))
    ) as ff:
        issues, _ = issues_from_xml_str(ff.read())
    with open(
        str(files("negmas").joinpath("tests/data/Laptop/Laptop-C-prof1.xml"))
    ) as ff:
        u, _ = UtilityFunction.from_xml_str(ff.read(), issues=issues)
    assert u is not None
    inv = PresortingInverseUtilityFunction(u)
    inv.init()
    for i in range(100):
        v = u(inv.one_in((i / 100.0, i / 100.0), normalized=True))
        assert v - 1e-3 <= v <= v + 0.1


def test_random_linear_utils_are_normalized():
    from negmas.preferences import LinearAdditiveUtilityFunction as U2
    from negmas.preferences import LinearUtilityFunction as U1

    eps = 1e-6
    issues = [make_issue(10), make_issue(5), make_issue(2)]

    for U in (U1, U2):
        u = U.random(issues=issues, normalized=True)
        if U == U2:
            assert 1 - eps <= sum(u.weights) <= 1 + eps
        else:
            assert sum(u.weights) <= 1 + eps
        outcomes = enumerate_issues(issues)
        for w in outcomes:
            assert -1e-6 <= u(w) <= 1 + 1e-6, f"{str(u)}"


def _order(u: list[float]):
    return tuple(_[1] for _ in sorted(zip(u, range(len(u)))))


def _check_order(u: list[float]):
    for a, b in zip(u[1:], u[:-1]):
        assert a <= b


def _relative_fraction(u: list[float]):
    return np.asarray([a / b if b > 1e-3 else 0 for a, b in zip(u[1:], u[:-1])])


rngs = [
    (0.0, 1.0),
    (1.0, 2.0),
    # (float("-inf"), 1.0),
    # (0.0, float("inf")),
    # (float("-inf"), float("inf")),
    # # (1.0, 2.0),
    # (float("-inf"), 2.0),
    # (1.0, float("inf")),
]


@given(
    weights=st.lists(st.integers(-10, 10), min_size=2, max_size=2),
    bias=st.floats(min_value=-5.0, max_value=5.0),
    rng=st.sampled_from(rngs),
)
@example(weights=[2, -1], bias=0.0, rng=(0.0, 1.0))
def test_can_normalize_affine_and_linear_ufun(weights, bias, rng):
    issues = [make_issue(10), make_issue(5)]
    ufun = AffineUtilityFunction(weights=weights, bias=bias, issues=issues)
    outcomes = enumerate_issues(issues)
    u1 = [ufun(w) for w in outcomes]

    if (sum(weights) > 1e-6 and rng == (0.0, float("inf"))) or (
        sum(weights) < 1e-6
        and rng[1] > rng[0] + 1e-6
        and rng[0] != float("-inf")
        and rng[1] != float("inf")
    ):
        with pytest.raises(ValueError):
            nfun = ufun.normalize(rng)
        return
    try:
        nfun = ufun.normalize(rng)
    except ValueError as e:
        # todo if the scale is negative, we should raise this exception. I do not now how to expect that correctl yet
        if "scale" in str(e):
            return
        raise e

    assert (
        isinstance(ufun, AffineUtilityFunction)
        or isinstance(ufun, LinearUtilityFunction)
        or isinstance(ufun, ConstUtilityFunction)
    ), "Normalization of ufun of type "
    f"LinearUtilityFunction should generate an IndependentIssuesUFun but we got {type(nfun).__name__}"
    u2 = [nfun(w) for w in outcomes]

    assert max(u2) < rng[1] + 1e-3 and min(u2) > rng[0] - 1e-3, (
        f"Limits are not correct\n{u1}\n{u2}"
    )

    if rng[0] == float("-inf") and rng[1] == float("inf"):
        assert ufun is nfun, "Normalizing with an infinite range should do nothing"

    # if bias == 0.0 and rng[0] == 0.0 and sum(weights) > 1e-5:
    #     scale = [a / b for a, b in zip(u1, u2) if b != 0 and a != 0]
    #     for a, b in zip(u1, u2):
    #         assert (
    #             abs(a - b) < 1e-5 or abs(b) > 1e-5
    #         ), f"zero values are mapped to zero values"
    #         assert (
    #             abs(a - b) < 1e-5 or abs(a) > 1e-5
    #         ), f"zero values are mapped to zero values"
    #     assert max(scale) - min(scale) < 1e-3, f"ufun did not scale uniformly"
    # order1, order2 = _order(u1), _order(u2)
    # assert order1 == order2, f"normalization changed the order of outcomes\n{u1[37:41]}\n{u2[37:41]}"

    if (
        (rng[1] == float("inf") or rng[0] == float("-inf"))
        and abs(bias) < 1e-3
        and sum(weights) > 1e-5
    ):
        relative1 = _relative_fraction(u1)
        relative2 = _relative_fraction(u2)
        assert np.abs(relative1 - relative2).max() < 1e-3, (
            f"One side normalization should not change the result of dividing outcomes\n{u1}\n{u2}"
        )


def test_normalization():
    with open(
        str(files("negmas").joinpath("tests/data/Laptop/Laptop-C-domain.xml"))
    ) as ff:
        os = CartesianOutcomeSpace.from_xml_str(ff.read())
    issues = os.issues
    outcomes = list(os.enumerate())
    with open(
        str(files("negmas").joinpath("tests/data/Laptop/Laptop-C-prof1.xml"))
    ) as ff:
        u, _ = UtilityFunction.from_xml_str(ff.read(), issues=issues)
    assert u is not None
    assert abs(float(u(("Dell", "60 Gb", "19'' LCD")) - 21.987727736172488)) < 0.000001
    assert abs(float(u(("HP", "80 Gb", "20'' LCD")) - 22.68559475583014)) < 0.000001
    utils = [float(u(_)) for _ in outcomes]
    max_util, min_util = max(utils), min(utils)
    gt_range = dict(
        zip(outcomes, [(_ - min_util) / (max_util - min_util) for _ in utils])
    )
    gt_max = dict(zip(outcomes, [_ / max_util for _ in utils]))

    with open(
        str(files("negmas").joinpath("tests/data/Laptop/Laptop-C-prof1.xml"))
    ) as ff:
        u, _ = UtilityFunction.from_xml_str(ff.read(), issues=issues)
    assert u is not None
    u = normalize(u, to=(0.0, 1.0))
    utils = [float(u(_)) for _ in outcomes]
    max_util, min_util = max(utils), min(utils)
    assert abs(max_util - 1.0) < 0.001
    assert abs(min_util) < 0.001

    for k, v in gt_range.items():
        assert abs(float(u(k)) - v) < 1e-3, f"Failed for {k} got {(u(k))} expected {v}"

    with open(
        str(files("negmas").joinpath("tests/data/Laptop/Laptop-C-prof1.xml"))
    ) as ff:
        u, _ = UtilityFunction.from_xml_str(ff.read(), issues=issues)
    assert u is not None
    u = scale_max(u, 1.0)
    utils = [u(_) for _ in outcomes]
    max_util, min_util = max(utils), min(utils)
    assert abs(max_util - 1.0) < 0.001

    for k, v in gt_max.items():
        assert abs(v - u(k)) < 1e-3, f"Failed for {k} got {(u(k))} expected {v}"


def test_rank_only_ufun_randomize_no_reserve():
    from negmas.preferences import RankOnlyUtilityFunction

    issues = [make_issue((0, 9)), make_issue((1, 5))]
    outcomes = list(make_os(issues).enumerate_or_sample())
    assert len(outcomes) == 10 * 5
    ufun = LinearUtilityFunction(
        weights=[1, 1], issues=issues, reserved_value=float("-inf")
    )
    ro = RankOnlyUtilityFunction(ufun, randomize_equal=True)
    assert isinstance(ro(None), float)
    assert isinstance(ro.reserved_value, float)
    assert ro.reserved_value == ro(None) == float("-inf")
    assert min(ro._mapping.values()) == 0
    assert max(ro._mapping.values()) == 10 * 5 - 1
    assert any(ro((0, _)) == 0 for _ in range(1, 6)), (
        f"{[(_, ro(_)) for _ in [None] + outcomes]}"
    )
    assert ro((9, 5)) == 10 * 5 - 1
    mapping = ro.to_mapping_ufun()
    assert mapping.reserved_value == ro(None)
    assert any(mapping((0, _)) == 0 for _ in range(1, 6)), (
        f"{[(_, mapping(_)) for _ in [None] + outcomes]}"
    )
    assert mapping((9, 5)) == 10 * 5 - 1


def test_rank_only_ufun_randomize():
    from negmas.preferences import RankOnlyUtilityFunction

    issues = [make_issue((0, 9)), make_issue((1, 5))]
    outcomes = list(make_os(issues).enumerate_or_sample())
    assert len(outcomes) == 10 * 5
    ufun = LinearUtilityFunction(weights=[1, 1], issues=issues, reserved_value=1.5)
    ro = RankOnlyUtilityFunction(ufun, randomize_equal=True)
    assert isinstance(ro(None), int)
    assert isinstance(ro.reserved_value, int)
    assert ro.reserved_value == ro(None)
    assert min(ro._mapping.values()) == 0
    assert max(ro._mapping.values()) == 10 * 5
    assert any(ro((0, _)) == 0 for _ in range(1, 6)), (
        f"{[(_, ro(_)) for _ in [None] + outcomes]}"
    )
    assert ro((9, 5)) == 10 * 5
    mapping = ro.to_mapping_ufun()
    assert isinstance(mapping(None), int)
    assert isinstance(mapping.reserved_value, int)
    assert mapping.reserved_value == ro(None)
    assert any(mapping((0, _)) == 0 for _ in range(1, 6)), (
        f"{[(_, mapping(_)) for _ in [None] + outcomes]}"
    )
    assert mapping((9, 5)) == 10 * 5


def test_rank_only_ufun_no_randomize():
    from negmas.preferences import RankOnlyUtilityFunction

    issues = [make_issue((0, 9)), make_issue((1, 5))]
    outcomes = list(make_os(issues).enumerate_or_sample())
    assert len(outcomes) == 10 * 5
    ufun = LinearUtilityFunction(weights=[1, 1], issues=issues, reserved_value=0.5)
    ro = RankOnlyUtilityFunction(ufun, randomize_equal=False)
    assert min(ro._mapping.values()) == 0
    assert max(ro._mapping.values()) < 10 * 5
    assert ro((1, 1)) == ro((0, 2))
    assert ro((2, 1)) == ro((1, 2)) == ro((0, 3))
    assert all(ro((9, 5)) > ro(_) for _ in outcomes if _ != (9, 5)), (
        f"{[(_, ro(_)) for _ in outcomes]}"
    )
    mapping = ro.to_mapping_ufun()
    assert isinstance(mapping(None), int)
    assert isinstance(mapping.reserved_value, int)
    assert mapping.reserved_value == ro(None)
    assert all(mapping((9, 5)) > mapping(_) for _ in outcomes if _ != (9, 5)), (
        f"{[(_, mapping(_)) for _ in outcomes]}"
    )
    assert mapping((1, 1)) == mapping((0, 2))
    assert mapping((2, 1)) == mapping((1, 2)) == mapping((0, 3))


def test_triangular():
    from negmas.preferences.value_fun import TriangularFun

    issue = ContiguousIssue(100)
    bias, scale = -1, 3
    f = TriangularFun(start=10, middle=20, end=30, bias=bias, scale=scale)
    mn, mx = f.minmax(issue)
    assert mn == bias
    assert mx == bias + scale
    for i in range(10):
        assert f(i) == bias
    for i in range(30, 100):
        assert f(i) == bias
    for i in range(10, 20):
        assert f(i) < f(i + 1)
    for i in range(20, 30):
        assert f(i) > f(i + 1)
    assert f(20) == bias + scale
    t = f.to_table(issue)
    mnt, mxt = t.minmax(issue)
    assert mnt == mn, f"{t.mapping}"
    assert mxt == mx, f"{t.mapping}"
    assert bias + scale in list(t.mapping.values())
    k = list(t.mapping.keys())
    for i in range(100):
        assert i in k


# def test_calc_outcome_stats_example():
#     n_outcomes = 2
#     n_negotiators = 2
#     normalized = False
#     sort = False
#     r0 = 0.0
#     r1 = 1.0
#
#     def _test_optim_allow_above1(d, outcome, lst):
#         if not lst:
#             assert isnan(d)
#             return
#         assert 0 <= d
#         if outcome in lst:
#             assert abs(1 - d) < 1e-12
#             return
#
#     def _test_optim(d, outcome, lst):
#         if not lst:
#             assert isnan(d)
#             return
#         assert 0 <= d <= 1
#         if outcome in lst:
#             assert abs(1 - d) < 1e-12
#             return
#         assert 1 - d > 1e-12
#
#     def _test_dist(d, outcome, lst):
#         if not lst:
#             assert isnan(d)
#             return
#         if outcome in lst:
#             assert abs(d) < 1e-12
#             return
#         assert abs(d) > 1e-12
#
#     np.random.seed(0)
#     os = make_os([make_issue(n_outcomes)])
#     outcomes = os.enumerate_or_sample()
#     utils = np.random.rand(n_negotiators, n_outcomes)
#     if not normalized:
#         utils *= 100
#         r0 *= 100
#         r1 *= 100
#     ufuns = [
#         MappingUtilityFunction(
#             dict(zip(outcomes, _)), outcome_space=os, reserved_value=r0 if not i else r1
#         )
#         for i, _ in enumerate(utils)
#     ]
#     stats = calc_scenario_stats(ufuns)
#     allutils = [tuple(u(_) for u in ufuns) for _ in outcomes]
#     mxoverall = estimate_max_dist(ufuns)
#     mxpareto = estimate_max_dist_using_outcomes(ufuns, stats.pareto_utils)
#     assert mxoverall >= mxpareto
#     for outcome in outcomes:
#         outils = tuple(u(outcome) for u in ufuns)
#         dists = calc_outcome_distances(outils, stats)
#         optim_overall = calc_outcome_optimality(dists, stats, max_dist=mxoverall)
#         # optim_pareto = calc_outcome_optimality(dists, stats, max_dist=mxpareto)
#         # optim_exact = calc_outcome_optimality(dists, stats, outcome_utils=allutils)
#         if is_rational(ufuns, outcome):
#             _test_optim_allow_above1(
#                 optim_overall.max_welfare_optimality,
#                 outcome,
#                 stats.max_welfare_outcomes,
#             )
#             for d, o1, lst in zip(
#                 (
#                     dists.pareto_dist,
#                     dists.nash_dist,
#                     dists.kalai_dist,
#                     dists.ks_dist,
#                     # dists.max_relative_welfare,
#                 ),
#                 (
#                     optim_overall.pareto_optimality,
#                     optim_overall.nash_optimality,
#                     optim_overall.kalai_optimality,
#                     optim_overall.ks_optimality,
#                     # optim_overall.max_relative_welfare_optimality,
#                 ),
#                 (
#                     stats.pareto_outcomes,
#                     stats.nash_outcomes,
#                     stats.kalai_outcomes,
#                     stats.ks_outcomes,
#                     # stats.max_relative_welfare_outcomes,
#                 ),
#                 strict=True,
#             ):
#                 _test_dist(d, outcome, lst)
#                 _test_optim(o1, outcome, lst)
#         else:
#             for lst in (
#                 stats.pareto_outcomes,
#                 stats.nash_outcomes,
#                 stats.kalai_outcomes,
#                 stats.ks_outcomes,
#                 # stats.max_relative_welfare_outcomes,
#             ):
#                 assert outcome not in lst
#                 for d in (
#                     dists.pareto_dist,
#                     dists.nash_dist,
#                     dists.kalai_dist,
#                     dists.ks_dist,
#                     # dists.max_relative_welfare,
#                 ):
#                     _test_dist(d, outcome, lst)
#                 for o1 in (
#                     optim_overall.pareto_optimality,
#                     optim_overall.nash_optimality,
#                     optim_overall.kalai_optimality,
#                     optim_overall.ks_optimality,
#                     # optim_overall.max_relative_welfare_optimality,
#                 ):
#                     _test_optim(o1, outcome, lst)
#                 _test_optim_allow_above1(
#                     optim_overall.max_welfare_optimality,
#                     outcome,
#                     stats.max_welfare_outcomes,
#                 )


@mark.parametrize("discounted_class", ["ExpDiscountedUFun", "LinDiscountedUFun"])
def test_discounted_ufun_minmax_uses_outcome_space_fallback(discounted_class):
    """Test that minmax() works without explicit outcome_space when the ufun has one set."""
    from negmas.preferences.discounted import ExpDiscountedUFun, LinDiscountedUFun

    issues = [make_issue(10), make_issue(5)]
    os = make_os(issues)

    # Create a base ufun with outcome_space set
    base_ufun = LinearUtilityFunction.random(issues=issues, normalized=True)
    base_ufun.outcome_space = os

    # Create the discounted ufun with outcome_space set
    if discounted_class == "ExpDiscountedUFun":
        discounted_ufun = ExpDiscountedUFun(
            ufun=base_ufun, discount=0.9, outcome_space=os
        )
    else:
        discounted_ufun = LinDiscountedUFun(ufun=base_ufun, cost=0.1, outcome_space=os)

    # This should work without passing outcome_space explicitly (the fix)
    minmax_result = discounted_ufun.minmax()

    # Verify we get valid results
    assert isinstance(minmax_result, tuple)
    assert len(minmax_result) == 2
    assert minmax_result[0] <= minmax_result[1]

    # Should match calling with explicit outcome_space
    minmax_explicit = discounted_ufun.minmax(outcome_space=os)
    assert minmax_result == minmax_explicit


@mark.parametrize("discounted_class", ["ExpDiscountedUFun", "LinDiscountedUFun"])
def test_discounted_ufun_shares_outcome_space_with_inner_ufun(discounted_class):
    """Test that DiscountedUtilityFunction shares outcome_space with its inner ufun."""
    from negmas.preferences.discounted import ExpDiscountedUFun, LinDiscountedUFun

    issues = [make_issue(10), make_issue(5)]
    os = make_os(issues)

    # Create a base ufun and clear its outcome_space
    base_ufun = LinearUtilityFunction.random(issues=issues, normalized=True)
    base_ufun.outcome_space = None
    assert base_ufun.outcome_space is None

    # Create the discounted ufun WITH outcome_space set
    if discounted_class == "ExpDiscountedUFun":
        discounted_ufun = ExpDiscountedUFun(
            ufun=base_ufun, discount=0.9, outcome_space=os
        )
    else:
        discounted_ufun = LinDiscountedUFun(ufun=base_ufun, cost=0.1, outcome_space=os)

    # The inner ufun should now have the outcome_space set
    assert discounted_ufun.ufun.outcome_space is os
    assert discounted_ufun.outcome_space is os


@mark.parametrize("discounted_class", ["ExpDiscountedUFun", "LinDiscountedUFun"])
def test_discounted_ufun_inherits_outcome_space_from_inner_ufun(discounted_class):
    """Test that DiscountedUtilityFunction inherits outcome_space from inner ufun if not provided."""
    from negmas.preferences.discounted import ExpDiscountedUFun, LinDiscountedUFun

    issues = [make_issue(10), make_issue(5)]
    os = make_os(issues)

    # Create a base ufun WITH outcome_space set
    base_ufun = LinearUtilityFunction.random(issues=issues, normalized=True)
    base_ufun.outcome_space = os

    # Create the discounted ufun WITHOUT outcome_space set
    if discounted_class == "ExpDiscountedUFun":
        discounted_ufun = ExpDiscountedUFun(ufun=base_ufun, discount=0.9)
    else:
        discounted_ufun = LinDiscountedUFun(ufun=base_ufun, cost=0.1)

    # The discounted ufun should inherit outcome_space from inner ufun
    assert discounted_ufun.outcome_space is os
    assert discounted_ufun.ufun.outcome_space is os


# Tests for Stability Criteria
class TestStabilityCriteria:
    """Tests for the new stability criteria system."""

    def test_stability_flags_basic_operations(self):
        """Test basic flag operations on Stability enum."""
        from negmas.preferences import STABLE_MIN, STABLE_MAX, STATIONARY, VOLATILE

        # Test combining flags
        combined = STABLE_MIN | STABLE_MAX
        assert combined.has_stable_min
        assert combined.has_stable_max
        assert not combined.has_stable_ordering
        assert not combined.is_stationary
        assert not combined.is_volatile

        # Test STATIONARY has all flags
        assert STATIONARY.has_stable_min
        assert STATIONARY.has_stable_max
        assert STATIONARY.has_stable_reserved_value
        assert STATIONARY.has_fixed_reserved_value
        assert STATIONARY.has_stable_rational_outcomes
        assert STATIONARY.has_stable_irrational_outcomes
        assert STATIONARY.has_stable_ordering
        assert STATIONARY.has_stable_diff_ratios
        assert STATIONARY.is_stationary

        # Test VOLATILE has no flags
        assert not VOLATILE.has_stable_min
        assert not VOLATILE.has_stable_max
        assert VOLATILE.is_volatile

    def test_stationary_ufun_default_stability(self):
        """Test that stationary ufuns have STATIONARY stability by default."""
        from negmas.preferences import STATIONARY

        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues)

        assert ufun.stability == STATIONARY
        assert ufun.is_stationary()
        assert not ufun.is_volatile()
        assert not ufun.is_session_dependent()
        assert not ufun.is_state_dependent()

    def test_stationary_mixin_sets_stability(self):
        """Test that StationaryMixin properly sets stability."""
        from negmas.preferences import STATIONARY

        issues = [make_issue(10), make_issue(5)]

        # LinearAdditiveUtilityFunction uses StationaryMixin
        ufun = LinearAdditiveUtilityFunction.random(issues=issues)
        assert ufun.stability == STATIONARY
        assert ufun.is_stationary()

        # MappingUtilityFunction uses StationaryMixin
        mapping_ufun = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"), issues=issues
        )
        assert mapping_ufun.stability == STATIONARY
        assert mapping_ufun.is_stationary()

    def test_discounted_ufun_stability(self):
        """Test that discounted ufuns have proper stability (not VOLATILE).

        Discounted ufuns preserve:
        - STABLE_ORDERING: discounting (multiplication/subtraction) preserves relative ordering
        - STABLE_DIFF_RATIOS: linear transformations preserve difference ratios
        - They are state-dependent (not STATE_INDEPENDENT) but not volatile
        """
        from negmas.preferences import VOLATILE
        from negmas.preferences.discounted import ExpDiscountedUFun, LinDiscountedUFun
        from negmas.preferences.stability import (
            STABLE_ORDERING,
            STABLE_DIFF_RATIOS,
            STATE_INDEPENDENT,
            SESSION_INDEPENDENT,
        )

        issues = [make_issue(10), make_issue(5)]
        base_ufun = LinearUtilityFunction.random(issues=issues, normalized=True)

        # Test ExpDiscountedUFun
        exp_ufun = ExpDiscountedUFun(ufun=base_ufun, discount=0.9)
        # Should NOT be volatile - discounting preserves ordering
        assert exp_ufun.stability != VOLATILE
        assert not exp_ufun.is_volatile()
        # Should be state-dependent
        assert exp_ufun.is_state_dependent()
        assert not exp_ufun.is_stationary()
        # Should preserve ordering and diff ratios
        assert exp_ufun.stability & STABLE_ORDERING
        assert exp_ufun.stability & STABLE_DIFF_RATIOS
        # Should NOT have STATE_INDEPENDENT (it depends on state)
        assert not (exp_ufun.stability & STATE_INDEPENDENT)
        # Should inherit SESSION_INDEPENDENT from inner ufun
        assert exp_ufun.stability & SESSION_INDEPENDENT

        # Test LinDiscountedUFun
        lin_ufun = LinDiscountedUFun(ufun=base_ufun, cost=0.1)
        # Should NOT be volatile - discounting preserves ordering
        assert lin_ufun.stability != VOLATILE
        assert not lin_ufun.is_volatile()
        # Should be state-dependent
        assert lin_ufun.is_state_dependent()
        assert not lin_ufun.is_stationary()
        # Should preserve ordering and diff ratios
        assert lin_ufun.stability & STABLE_ORDERING
        assert lin_ufun.stability & STABLE_DIFF_RATIOS
        # Should NOT have STATE_INDEPENDENT (it depends on state)
        assert not (lin_ufun.stability & STATE_INDEPENDENT)
        # Should inherit SESSION_INDEPENDENT from inner ufun
        assert lin_ufun.stability & SESSION_INDEPENDENT

    def test_discounted_ufun_no_discount_inherits_full_stability(self):
        """Test that discounted ufuns with no effective discount inherit full stability."""
        from negmas.preferences.discounted import ExpDiscountedUFun, LinDiscountedUFun

        issues = [make_issue(10), make_issue(5)]
        base_ufun = LinearUtilityFunction.random(issues=issues, normalized=True)

        # ExpDiscountedUFun with discount=1.0 has no effect
        exp_ufun = ExpDiscountedUFun(ufun=base_ufun, discount=1.0)
        assert exp_ufun.stability == base_ufun.stability

        # LinDiscountedUFun with cost=0 has no effect
        lin_ufun = LinDiscountedUFun(ufun=base_ufun, cost=0.0)
        assert lin_ufun.stability == base_ufun.stability

    def test_all_discounted_ufuns_are_state_dependent(self):
        """Test that ALL discounted ufuns with effective discounting are state dependent.

        This is a critical property: discounted ufuns depend on negotiation state
        (time, step, etc.) to compute the discount factor, so they MUST NOT have
        the STATE_INDEPENDENT flag set.

        Note: State independence is orthogonal to session independence:
        - STATE_INDEPENDENT: Does not depend on MechanismState (time, step, offers, etc.)
        - SESSION_INDEPENDENT: Does not depend on NMI (n_negotiators, mechanism params, etc.)

        A ufun can be session-independent but state-dependent (like discounted ufuns),
        or state-independent but session-dependent, or both, or neither.

        Discounted ufuns should:
        - Always clear STATE_INDEPENDENT (they depend on state for discount calculation)
        - Inherit SESSION_INDEPENDENT from inner ufun (they don't add or remove it)
        """
        from negmas.preferences.discounted import ExpDiscountedUFun, LinDiscountedUFun
        from negmas.preferences.stability import (
            STATE_INDEPENDENT,
            SESSION_INDEPENDENT,
            STATIONARY,
        )

        issues = [make_issue(10), make_issue(5)]

        # Test with stationary base ufun (has both STATE_INDEPENDENT and SESSION_INDEPENDENT)
        stationary_base = LinearUtilityFunction.random(issues=issues, normalized=True)
        assert stationary_base.stability == STATIONARY
        assert stationary_base.stability & STATE_INDEPENDENT
        assert stationary_base.stability & SESSION_INDEPENDENT

        # Test various discount values for ExpDiscountedUFun
        for discount in [0.5, 0.9, 0.99, 1.1, 2.0]:
            exp_ufun = ExpDiscountedUFun(ufun=stationary_base, discount=discount)
            # Must be state dependent (discount depends on state.step or state.time)
            assert not (exp_ufun.stability & STATE_INDEPENDENT), (
                f"ExpDiscountedUFun with discount={discount} should be state dependent"
            )
            assert exp_ufun.is_state_dependent()
            # Should still be session independent (inherited from inner ufun)
            assert exp_ufun.stability & SESSION_INDEPENDENT, (
                f"ExpDiscountedUFun with discount={discount} should inherit session independence"
            )

        # Test various cost values for LinDiscountedUFun
        for cost in [0.01, 0.1, 0.5, 1.0, -0.1]:
            lin_ufun = LinDiscountedUFun(ufun=stationary_base, cost=cost)
            # Must be state dependent (cost depends on state.step or state.time)
            assert not (lin_ufun.stability & STATE_INDEPENDENT), (
                f"LinDiscountedUFun with cost={cost} should be state dependent"
            )
            assert lin_ufun.is_state_dependent()
            # Should still be session independent (inherited from inner ufun)
            assert lin_ufun.stability & SESSION_INDEPENDENT, (
                f"LinDiscountedUFun with cost={cost} should inherit session independence"
            )

        # Test with dynamic_reservation=False (should still be state dependent)
        exp_ufun_fixed = ExpDiscountedUFun(
            ufun=stationary_base, discount=0.9, dynamic_reservation=False
        )
        assert not (exp_ufun_fixed.stability & STATE_INDEPENDENT)
        assert exp_ufun_fixed.stability & SESSION_INDEPENDENT

        lin_ufun_fixed = LinDiscountedUFun(
            ufun=stationary_base, cost=0.1, dynamic_reservation=False
        )
        assert not (lin_ufun_fixed.stability & STATE_INDEPENDENT)
        assert lin_ufun_fixed.stability & SESSION_INDEPENDENT

        # Edge case: discount=1.0 or cost=0.0 means no effective discounting
        # These should inherit full stability from inner ufun (including STATE_INDEPENDENT)
        exp_no_discount = ExpDiscountedUFun(ufun=stationary_base, discount=1.0)
        assert exp_no_discount.stability & STATE_INDEPENDENT, (
            "ExpDiscountedUFun with discount=1.0 should inherit STATE_INDEPENDENT"
        )

        lin_no_cost = LinDiscountedUFun(ufun=stationary_base, cost=0.0)
        assert lin_no_cost.stability & STATE_INDEPENDENT, (
            "LinDiscountedUFun with cost=0.0 should inherit STATE_INDEPENDENT"
        )

        exp_none_discount = ExpDiscountedUFun(ufun=stationary_base, discount=None)
        assert exp_none_discount.stability & STATE_INDEPENDENT, (
            "ExpDiscountedUFun with discount=None should inherit STATE_INDEPENDENT"
        )

        lin_none_cost = LinDiscountedUFun(ufun=stationary_base, cost=None)
        assert lin_none_cost.stability & STATE_INDEPENDENT, (
            "LinDiscountedUFun with cost=None should inherit STATE_INDEPENDENT"
        )

    def test_discounted_ufuns_inherit_session_dependence(self):
        """Test that discounted ufuns inherit session dependence from inner ufun.

        Discounted ufuns should NOT set SESSION_INDEPENDENT themselves - they should
        only inherit it from the inner ufun. If the inner ufun is session-dependent,
        the discounted ufun should also be session-dependent.

        This is different from STATE_INDEPENDENT which is always cleared for
        effective discounting (since discounting depends on state).
        """
        from negmas.preferences.discounted import ExpDiscountedUFun, LinDiscountedUFun
        from negmas.preferences.stability import (
            STATE_INDEPENDENT,
            SESSION_INDEPENDENT,
            STATIONARY,
        )

        issues = [make_issue(10), make_issue(5)]

        # Create a session-dependent ufun (has STATE_INDEPENDENT but NOT SESSION_INDEPENDENT)
        session_dep_stability = (
            STATIONARY & ~SESSION_INDEPENDENT
        )  # Remove session independence
        session_dep_base = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else 0.0,
            issues=issues,
            stability=session_dep_stability,
        )
        assert session_dep_base.stability & STATE_INDEPENDENT
        assert not (session_dep_base.stability & SESSION_INDEPENDENT)

        # Discounted ufuns from session-dependent base should also be session-dependent
        exp_ufun = ExpDiscountedUFun(ufun=session_dep_base, discount=0.9)
        assert not (exp_ufun.stability & SESSION_INDEPENDENT), (
            "ExpDiscountedUFun should inherit session dependence from inner ufun"
        )
        assert not (exp_ufun.stability & STATE_INDEPENDENT), (
            "ExpDiscountedUFun should still be state dependent"
        )

        lin_ufun = LinDiscountedUFun(ufun=session_dep_base, cost=0.1)
        assert not (lin_ufun.stability & SESSION_INDEPENDENT), (
            "LinDiscountedUFun should inherit session dependence from inner ufun"
        )
        assert not (lin_ufun.stability & STATE_INDEPENDENT), (
            "LinDiscountedUFun should still be state dependent"
        )

        # Create a session-independent ufun for comparison
        stationary_base = LinearUtilityFunction.random(issues=issues, normalized=True)
        assert stationary_base.stability & SESSION_INDEPENDENT

        # Discounted ufuns from session-independent base should be session-independent
        exp_ufun_si = ExpDiscountedUFun(ufun=stationary_base, discount=0.9)
        assert exp_ufun_si.stability & SESSION_INDEPENDENT, (
            "ExpDiscountedUFun should inherit session independence from inner ufun"
        )

        lin_ufun_si = LinDiscountedUFun(ufun=stationary_base, cost=0.1)
        assert lin_ufun_si.stability & SESSION_INDEPENDENT, (
            "LinDiscountedUFun should inherit session independence from inner ufun"
        )

    def test_custom_stability_via_constructor(self):
        """Test passing custom stability via constructor."""
        from negmas.preferences import STABLE_MIN, STABLE_MAX, STABLE_ORDERING

        issues = [make_issue(10), make_issue(5)]

        # Create with partial stability
        partial_stability = STABLE_MIN | STABLE_MAX | STABLE_ORDERING
        ufun = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=partial_stability,
        )

        assert ufun.stability == partial_stability
        assert ufun.has_stable_min
        assert ufun.has_stable_max
        assert ufun.has_stable_ordering
        assert not ufun.has_stable_reserved_value
        assert not ufun.is_stationary()
        assert not ufun.is_volatile()

    def test_stability_properties_on_preferences(self):
        """Test that stability properties are accessible on Preferences base class."""
        from negmas.preferences import STABLE_SCALE

        issues = [make_issue(10), make_issue(5)]
        stability = STABLE_SCALE  # STABLE_MIN | STABLE_MAX | STABLE_RESERVED_VALUE

        ufun = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=stability,
        )

        assert ufun.has_stable_scale
        assert ufun.has_stable_min
        assert ufun.has_stable_max
        assert ufun.has_stable_reserved_value
        assert not ufun.has_fixed_reserved_value
        assert not ufun.has_stable_ordering

    def test_stability_serialization(self):
        """Test that stability is correctly serialized and deserialized."""
        from negmas.preferences import STABLE_MIN, STABLE_MAX, STABLE_ORDERING

        issues = [make_issue(10), make_issue(5)]
        stability = STABLE_MIN | STABLE_MAX | STABLE_ORDERING

        ufun = LinearUtilityFunction.random(issues=issues)
        ufun._stability = stability  # Override for testing

        # Serialize
        d = ufun.to_dict()
        assert "stability" in d
        assert d["stability"] == int(stability)

        # Deserialize
        ufun2 = LinearUtilityFunction.from_dict(d)
        assert ufun2.stability == stability
        assert ufun2.has_stable_min
        assert ufun2.has_stable_max
        assert ufun2.has_stable_ordering
        assert not ufun2.has_stable_reserved_value

    def test_stability_str_representation(self):
        """Test string representation of stability flags."""
        from negmas.preferences import STABLE_MIN, STABLE_MAX, STATIONARY, VOLATILE

        assert str(STATIONARY) == "STATIONARY"
        assert str(VOLATILE) == "VOLATILE"

        combined = STABLE_MIN | STABLE_MAX
        str_repr = str(combined)
        assert "STABLE_MIN" in str_repr
        assert "STABLE_MAX" in str_repr

    def test_backward_compatibility_is_volatile_is_stationary(self):
        """Test that is_volatile() and is_stationary() methods work correctly."""
        from negmas.preferences import VOLATILE

        issues = [make_issue(10), make_issue(5)]

        # Stationary ufun
        stationary_ufun = LinearUtilityFunction.random(issues=issues)
        assert stationary_ufun.is_stationary()
        assert not stationary_ufun.is_volatile()

        # Volatile ufun (manually set for testing)
        volatile_ufun = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=VOLATILE,
        )
        assert volatile_ufun.is_volatile()
        assert not volatile_ufun.is_stationary()

    def test_changes_returns_empty_for_stationary(self):
        """Test that changes() returns empty list for stationary ufuns."""
        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues)

        assert ufun.is_stationary()
        assert ufun.changes() == []

    def test_stability_setter(self):
        """Test that stability can be set after construction."""
        from negmas.preferences import STABLE_MIN, VOLATILE

        issues = [make_issue(10), make_issue(5)]
        ufun = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"), issues=issues
        )

        # Initially stationary (from StationaryMixin)
        assert ufun.is_stationary()

        # Change to volatile
        ufun.stability = VOLATILE
        assert ufun.is_volatile()
        assert not ufun.is_stationary()

        # Change to partial stability
        ufun.stability = STABLE_MIN
        assert ufun.has_stable_min
        assert not ufun.has_stable_max
        assert not ufun.is_stationary()
        assert not ufun.is_volatile()

    def test_stationary_ufun_caches_minmax(self):
        """Test that stationary ufuns cache minmax results."""
        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues)

        assert ufun.is_stationary()

        # First call computes and caches
        result1 = ufun.minmax()
        assert ufun._cached_minmax is not None

        # Second call should return cached value
        result2 = ufun.minmax()
        assert result1 == result2

        # Clear caches
        ufun.clear_caches()
        assert ufun._cached_minmax is None

    def test_stationary_ufun_caches_extreme_outcomes(self):
        """Test that stationary ufuns cache extreme_outcomes results."""
        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues)

        assert ufun.is_stationary()

        # First call computes and caches
        result1 = ufun.extreme_outcomes()
        assert ufun._cached_extreme_outcomes is not None

        # Second call should return cached value
        result2 = ufun.extreme_outcomes()
        assert result1 == result2

        # Clear caches
        ufun.clear_caches()
        assert ufun._cached_extreme_outcomes is None

    def test_volatile_ufun_does_not_cache_minmax(self):
        """Test that volatile ufuns do not cache minmax results."""
        from negmas.preferences import VOLATILE

        issues = [make_issue(10), make_issue(5)]
        ufun = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=VOLATILE,
        )

        assert ufun.is_volatile()

        # Call minmax
        _ = ufun.minmax()

        # Should not be cached for volatile ufuns
        assert ufun._cached_minmax is None

    def test_volatile_ufun_does_not_cache_extreme_outcomes(self):
        """Test that volatile ufuns do not cache extreme_outcomes results."""
        from negmas.preferences import VOLATILE

        issues = [make_issue(10), make_issue(5)]
        ufun = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=VOLATILE,
        )

        assert ufun.is_volatile()

        # Call extreme_outcomes
        _ = ufun.extreme_outcomes()

        # Should not be cached for volatile ufuns
        assert ufun._cached_extreme_outcomes is None

    def test_partial_stability_caching_minmax(self):
        """Test that minmax caching works based on STABLE_MIN and STABLE_MAX flags."""
        from negmas.preferences import STABLE_MIN, STABLE_MAX, STABLE_ORDERING

        issues = [make_issue(10), make_issue(5)]

        # Only STABLE_ORDERING - should NOT cache minmax
        ufun1 = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=STABLE_ORDERING,
        )
        _ = ufun1.minmax()
        assert ufun1._cached_minmax is None

        # STABLE_MIN only - should NOT cache minmax (needs both)
        ufun2 = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=STABLE_MIN,
        )
        _ = ufun2.minmax()
        assert ufun2._cached_minmax is None

        # STABLE_MIN | STABLE_MAX - should cache minmax
        ufun3 = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=STABLE_MIN | STABLE_MAX,
        )
        _ = ufun3.minmax()
        assert ufun3._cached_minmax is not None

    def test_partial_stability_caching_extreme_outcomes(self):
        """Test that extreme_outcomes caching works based on STABLE_ORDERING or STABLE_DIFF_RATIOS."""
        from negmas.preferences import STABLE_ORDERING, STABLE_DIFF_RATIOS, STABLE_MIN

        issues = [make_issue(10), make_issue(5)]

        # Only STABLE_MIN - should NOT cache extreme_outcomes
        ufun1 = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=STABLE_MIN,
        )
        _ = ufun1.extreme_outcomes()
        assert ufun1._cached_extreme_outcomes is None

        # STABLE_ORDERING - should cache extreme_outcomes
        ufun2 = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=STABLE_ORDERING,
        )
        _ = ufun2.extreme_outcomes()
        assert ufun2._cached_extreme_outcomes is not None

        # STABLE_DIFF_RATIOS - should cache extreme_outcomes
        ufun3 = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else float("nan"),
            issues=issues,
            stability=STABLE_DIFF_RATIOS,
        )
        _ = ufun3.extreme_outcomes()
        assert ufun3._cached_extreme_outcomes is not None

    def test_caching_with_different_outcome_space(self):
        """Test that caching is skipped when a different outcome_space is passed."""
        issues1 = [make_issue(10), make_issue(5)]
        issues2 = [make_issue(8), make_issue(4)]

        ufun = LinearUtilityFunction.random(issues=issues1)
        assert ufun.is_stationary()

        # Call with own outcome space - should cache
        _ = ufun.minmax()
        assert ufun._cached_minmax is not None

        # Clear cache
        ufun.clear_caches()

        # Call with different outcome space - should NOT cache
        os2 = make_os(issues2)
        _ = ufun.minmax(outcome_space=os2)
        assert ufun._cached_minmax is None

    def test_linear_additive_caching(self):
        """Test that LinearAdditiveUtilityFunction caches extreme_outcomes correctly."""
        issues = [make_issue(10), make_issue(5)]
        ufun = LinearAdditiveUtilityFunction.random(issues=issues)

        assert ufun.is_stationary()

        # First call computes and caches
        result1 = ufun.extreme_outcomes()
        assert ufun._cached_extreme_outcomes is not None

        # Second call returns cached value
        result2 = ufun.extreme_outcomes()
        assert result1 == result2

        # Verify cache hit by checking it's the same object
        cached = ufun._cached_extreme_outcomes
        _ = ufun.extreme_outcomes()
        assert ufun._cached_extreme_outcomes is cached

    def test_affine_ufun_caching(self):
        """Test that AffineUtilityFunction caches extreme_outcomes correctly."""
        issues = [make_issue(10), make_issue(5)]
        ufun = AffineUtilityFunction.random(issues=issues)

        assert ufun.is_stationary()

        # First call computes and caches
        result1 = ufun.extreme_outcomes()
        assert ufun._cached_extreme_outcomes is not None

        # Second call returns cached value
        result2 = ufun.extreme_outcomes()
        assert result1 == result2

    def test_stability_setter_records_change(self):
        """Test that stability setter records the change in _changes list."""
        from negmas.preferences import STABLE_MIN, VOLATILE
        from negmas.common import PreferencesChangeType

        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues)

        # Clear any existing changes
        ufun.reset_changes()
        assert len(ufun._changes) == 0

        # Change stability
        old_stability = ufun.stability
        ufun.stability = VOLATILE

        # Check that change was recorded
        assert len(ufun._changes) == 1
        change = ufun._changes[0]
        # STATIONARY -> VOLATILE is a reduction (losing all bits)
        assert change.type == PreferencesChangeType.StabilityReduced
        assert change.data["old"] == old_stability
        assert change.data["new"] == VOLATILE

        # Make another change
        ufun.stability = STABLE_MIN
        assert len(ufun._changes) == 2
        change2 = ufun._changes[1]
        # VOLATILE -> STABLE_MIN is an increase (gaining a bit)
        assert change2.type == PreferencesChangeType.StabilityIncreased
        assert change2.data["old"] == VOLATILE
        assert change2.data["new"] == STABLE_MIN

    def test_stability_setter_notifies_owner(self):
        """Test that stability setter notifies the owner via on_preferences_changed.

        Note: Owner is only set when the negotiator enters a negotiation via
        _on_negotiation_start, not during set_preferences. This test verifies
        that when owner IS set, the notification is sent correctly.
        """
        from negmas.preferences import VOLATILE
        from negmas.common import PreferencesChangeType

        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues)

        # Track notifications
        notifications = []

        class TrackingNegotiator(AspirationNegotiator):
            def on_preferences_changed(self, changes):
                notifications.append(changes)
                super().on_preferences_changed(changes)

        negotiator = TrackingNegotiator()
        negotiator.set_preferences(ufun)

        # Owner is NOT set via set_preferences anymore
        # Owner is only set when entering a negotiation via _on_negotiation_start
        assert ufun.owner is None

        # Manually set owner to test notification (simulating negotiation start)
        ufun.owner = negotiator
        assert ufun.owner is negotiator

        # Clear notifications
        notifications.clear()

        # Change stability
        old_stability = ufun.stability
        ufun.stability = VOLATILE

        # Check that owner was notified
        assert len(notifications) == 1
        notification = notifications[0]
        # notification is [PreferencesChange(...)]
        assert len(notification) == 1
        pref_change = notification[0]
        # STATIONARY -> VOLATILE is a reduction
        assert pref_change.type == PreferencesChangeType.StabilityReduced
        assert pref_change.data["old"] == old_stability
        assert pref_change.data["new"] == VOLATILE

    def test_stability_setter_no_notification_without_owner(self):
        """Test that stability setter doesn't notify when there's no owner."""
        from negmas.preferences import VOLATILE

        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues)

        # No owner
        assert ufun.owner is None

        # Should not raise
        ufun.stability = VOLATILE

        # Change was still recorded internally
        assert len(ufun._changes) > 0
        # STATIONARY -> VOLATILE is a reduction
        assert any(
            c.type == PreferencesChangeType.StabilityReduced for c in ufun._changes
        )

    def test_negotiator_set_preferences_does_not_set_owner(self):
        """Test that Negotiator.set_preferences does NOT set owner.

        Owner is only set when the negotiator enters a negotiation via
        _on_negotiation_start, not during set_preferences.
        """
        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues)

        # Before set_preferences, no owner
        assert ufun.owner is None

        negotiator = AspirationNegotiator()
        negotiator.set_preferences(ufun)

        # After set_preferences, owner should NOT be set
        # Owner is only set when entering a negotiation
        assert ufun.owner is None
        assert negotiator.preferences is ufun

    def test_owner_set_during_negotiation_start(self):
        """Test that owner is set when negotiator enters a negotiation and cleared when it ends."""
        from negmas.sao import SAOMechanism

        issues = [make_issue(10), make_issue(5)]
        ufun1 = LinearUtilityFunction.random(issues=issues)
        ufun2 = LinearUtilityFunction.random(issues=issues)

        # Track owner during on_negotiation_start
        owners_during_start = {}

        class TrackingNegotiator(AspirationNegotiator):
            def on_negotiation_start(self, state):
                owners_during_start[self.id] = (self.ufun.owner, self)
                super().on_negotiation_start(state)

        n1 = TrackingNegotiator(ufun=ufun1)
        n2 = TrackingNegotiator(ufun=ufun2)

        # Before negotiation, no owner
        assert ufun1.owner is None
        assert ufun2.owner is None

        m = SAOMechanism(issues=issues, n_steps=10)
        m.add(n1)
        m.add(n2)

        # After add but before negotiation starts, still no owner
        assert ufun1.owner is None

        # Start the negotiation (first step)
        m.step()

        # Owner was correctly set during on_negotiation_start
        assert owners_during_start[n1.id][0] is n1
        assert owners_during_start[n2.id][0] is n2

        # If negotiation ended, owner should be cleared
        if not m.running:
            assert ufun1.owner is None
            assert ufun2.owner is None
        else:
            # If still running, owner should still be set
            assert ufun1.owner is n1
            assert ufun2.owner is n2

    def test_stability_change_notification_during_negotiation(self):
        """Test stability change notification works during a negotiation."""
        from negmas.preferences import VOLATILE
        from negmas.common import PreferencesChangeType

        issues = [make_issue(5), make_issue(5)]
        os = make_os(issues)

        # Track notifications
        notifications = []

        class TrackingNegotiator(AspirationNegotiator):
            def on_preferences_changed(self, changes):
                notifications.append(changes)
                super().on_preferences_changed(changes)

        ufun1 = LinearUtilityFunction.random(issues=issues, normalized=True)
        ufun2 = LinearUtilityFunction.random(issues=issues, normalized=True)

        n1 = TrackingNegotiator()
        n2 = AspirationNegotiator()

        m = SAOMechanism(outcome_space=os, n_steps=10)
        m.add(n1, preferences=ufun1)
        m.add(n2, preferences=ufun2)

        # Run the negotiation - this sets the owner via _on_negotiation_start
        m.run()

        # After negotiation ends, owner is still set (cleared only on explicit leave)
        # For this test, we manually set owner to test notification
        ufun1.owner = n1

        # Clear notifications from setup and negotiation
        notifications.clear()

        # Change stability after negotiation
        ufun1.stability = VOLATILE

        # Owner should have been notified
        assert len(notifications) == 1
        notification = notifications[0]
        # Notification format is [PreferencesChange(...)]
        assert len(notification) == 1
        # STATIONARY -> VOLATILE is a reduction
        assert notification[0].type == PreferencesChangeType.StabilityReduced


class TestUFunConstraint:
    """Tests for UFunConstraint adapter."""

    def test_constraint_basic_functionality(self):
        """Test that UFunConstraint applies constraint correctly."""
        from negmas.preferences.adapters import UFunConstraint

        issues = [make_issue(6), make_issue(6)]  # values 0-5 for each issue
        base_ufun = LinearUtilityFunction(weights=[1.0, 1.0], issues=issues)

        # Constraint: sum of values must be <= 6
        constrained = UFunConstraint(ufun=base_ufun, constraint=lambda o: sum(o) <= 6)

        # Valid outcomes
        assert constrained((2, 3)) == 5.0  # sum=5, valid
        assert constrained((0, 0)) == 0.0  # sum=0, valid
        assert constrained((3, 3)) == 6.0  # sum=6, valid (boundary)

        # Invalid outcomes
        assert constrained((4, 4)) == float("-inf")  # sum=8, invalid
        assert constrained((5, 5)) == float("-inf")  # sum=10, invalid

    def test_constraint_inherits_stability(self):
        """Test that UFunConstraint properly inherits and ANDs stability."""
        from negmas.preferences.adapters import UFunConstraint
        from negmas.preferences.stability import (
            STABLE_ORDERING,
            STABLE_DIFF_RATIOS,
            STABLE_RATIONAL_OUTCOMES,
            STABLE_IRRATIONAL_OUTCOMES,
            STATIONARY,
        )

        issues = [make_issue(6), make_issue(6)]
        base_ufun = LinearUtilityFunction(weights=[1.0, 1.0], issues=issues)

        # Base ufun is stationary
        assert base_ufun.stability == STATIONARY

        constrained = UFunConstraint(ufun=base_ufun, constraint=lambda o: sum(o) <= 6)

        # Should preserve ordering and diff ratios (among valid outcomes)
        assert constrained.stability & STABLE_ORDERING
        assert constrained.stability & STABLE_DIFF_RATIOS
        # Should preserve rational/irrational status
        assert constrained.stability & STABLE_RATIONAL_OUTCOMES
        assert constrained.stability & STABLE_IRRATIONAL_OUTCOMES
        # Should NOT be volatile
        assert not constrained.is_volatile()

    def test_constraint_with_complex_predicate(self):
        """Test UFunConstraint with a more complex constraint."""
        from negmas.preferences.adapters import UFunConstraint

        issues = [make_issue(6), make_issue(6), make_issue(6)]
        base_ufun = LinearUtilityFunction(weights=[1.0, 2.0, 3.0], issues=issues)

        # Constraint: first value must be >= second value
        constrained = UFunConstraint(ufun=base_ufun, constraint=lambda o: o[0] >= o[1])

        # (3, 2, 1) -> 3*1 + 2*2 + 1*3 = 3 + 4 + 3 = 10, valid
        assert constrained((3, 2, 1)) == 3.0 + 4.0 + 3.0  # 10.0, valid
        # (2, 2, 1) -> 2*1 + 2*2 + 1*3 = 2 + 4 + 3 = 9, valid (boundary)
        assert constrained((2, 2, 1)) == 2.0 + 4.0 + 3.0  # 9.0, valid (boundary)
        assert constrained((1, 2, 3)) == float("-inf")  # invalid

    def test_constraint_reserved_value(self):
        """Test that UFunConstraint handles reserved value correctly."""
        from negmas.preferences.adapters import UFunConstraint

        issues = [make_issue(6), make_issue(6)]
        base_ufun = LinearUtilityFunction(
            weights=[1.0, 1.0], issues=issues, reserved_value=0.5
        )

        constrained = UFunConstraint(ufun=base_ufun, constraint=lambda o: sum(o) <= 6)

        # Reserved value inherited from base
        assert constrained.reserved_value == 0.5
        # Calling with None returns reserved value
        assert constrained(None) == 0.5


class TestCompositeUfunStability:
    """Tests for composite utility function stability."""

    def test_weighted_ufun_ands_stability(self):
        """Test that WeightedUtilityFunction ANDs stability of all components."""
        from negmas.preferences.complex import WeightedUtilityFunction
        from negmas.preferences.stability import (
            STABLE_ORDERING,
            STABLE_DIFF_RATIOS,
            STATIONARY,
        )

        issues = [make_issue(5)]
        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.5], issues=issues)

        # Both are stationary
        assert u1.stability == STATIONARY
        assert u2.stability == STATIONARY

        combined = WeightedUtilityFunction(ufuns=[u1, u2], weights=[0.6, 0.4])

        # Combined should have full stability (AND of both)
        assert combined.stability == STATIONARY
        assert combined.is_stationary()
        assert not combined.is_volatile()
        # Should preserve ordering and diff ratios (linear combination)
        assert combined.stability & STABLE_ORDERING
        assert combined.stability & STABLE_DIFF_RATIOS

    def test_weighted_ufun_with_volatile_component(self):
        """Test WeightedUtilityFunction when one component is volatile."""
        from negmas.preferences.complex import WeightedUtilityFunction
        from negmas.preferences.stability import VOLATILE

        issues = [make_issue(5)]
        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        u2 = MappingUtilityFunction(
            mapping=lambda x: sum(x) if x else 0.0, issues=issues, stability=VOLATILE
        )

        combined = WeightedUtilityFunction(ufuns=[u1, u2], weights=[0.5, 0.5])

        # Combined should be volatile (weakest link)
        assert combined.is_volatile()
        assert combined.stability == VOLATILE

    def test_complex_nonlinear_ufun_clears_ordering_and_diff_ratios(self):
        """Test that ComplexNonlinearUtilityFunction clears STABLE_ORDERING and STABLE_DIFF_RATIOS."""
        from negmas.preferences.complex import ComplexNonlinearUtilityFunction
        from negmas.preferences.stability import (
            STABLE_ORDERING,
            STABLE_DIFF_RATIOS,
            STATIONARY,
        )

        issues = [make_issue(5)]
        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.5], issues=issues)

        # Both are stationary with all flags
        assert u1.stability == STATIONARY
        assert u2.stability == STATIONARY

        # Product combination - arbitrary function
        combined = ComplexNonlinearUtilityFunction(
            ufuns=[u1, u2], combination_function=lambda vals: vals[0] * vals[1]
        )

        # Should NOT have STABLE_ORDERING or STABLE_DIFF_RATIOS
        # because the combination function is arbitrary
        assert not (combined.stability & STABLE_ORDERING)
        assert not (combined.stability & STABLE_DIFF_RATIOS)
        # But should inherit other stability flags
        assert not combined.is_volatile()

    def test_complex_nonlinear_ufun_with_explicit_stability(self):
        """Test that ComplexNonlinearUtilityFunction respects explicit stability."""
        from negmas.preferences.complex import ComplexNonlinearUtilityFunction
        from negmas.preferences.stability import (
            STABLE_ORDERING,
            STABLE_DIFF_RATIOS,
            STATIONARY,
        )

        issues = [make_issue(5)]
        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.5], issues=issues)

        # If user knows their function preserves ordering, they can specify it
        combined = ComplexNonlinearUtilityFunction(
            ufuns=[u1, u2],
            combination_function=lambda vals: vals[0] + vals[1],  # linear
            stability=STATIONARY,  # user asserts full stability
        )

        assert combined.stability == STATIONARY
        assert combined.stability & STABLE_ORDERING
        assert combined.stability & STABLE_DIFF_RATIOS


# Tests for reserved_value with Distribution support
class TestReservedValueDistribution:
    """Tests for the reserved_value property supporting Distribution inputs."""

    def test_reserved_value_float_input_returns_float(self):
        """Test that passing float as reserved_value returns float."""
        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues, reserved_value=0.5)
        assert isinstance(ufun.reserved_value, float)
        assert ufun.reserved_value == 0.5

    def test_reserved_value_distribution_input_returns_mean(self):
        """Test that passing Distribution as reserved_value returns its mean."""
        from negmas.helpers.prob import ScipyDistribution

        issues = [make_issue(10), make_issue(5)]
        dist = ScipyDistribution(
            type="uniform", loc=0.2, scale=0.6
        )  # mean = 0.2 + 0.6/2 = 0.5
        # Use direct constructor, not random() which treats reserved_value as a range
        ufun = LinearUtilityFunction(
            weights=[1.0, 1.0], issues=issues, reserved_value=dist
        )

        assert isinstance(ufun.reserved_value, float)
        assert abs(ufun.reserved_value - 0.5) < 1e-10

    def test_reserved_distribution_from_float_returns_delta(self):
        """Test that reserved_distribution returns delta distribution when reserved_value is float."""
        from negmas.helpers.prob import Real

        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues, reserved_value=0.5)

        dist = ufun.reserved_distribution
        # Delta distribution should have mean equal to the value
        assert abs(dist.mean() - 0.5) < 1e-10
        # And zero variance (it's a delta/Real distribution)
        assert isinstance(dist, Real) or dist.scale == 0.0

    def test_reserved_distribution_from_distribution_returns_same(self):
        """Test that reserved_distribution returns the original distribution."""
        from negmas.helpers.prob import ScipyDistribution

        issues = [make_issue(10), make_issue(5)]
        dist = ScipyDistribution(type="uniform", loc=0.2, scale=0.6)
        # Use direct constructor, not random()
        ufun = LinearUtilityFunction(
            weights=[1.0, 1.0], issues=issues, reserved_value=dist
        )

        returned_dist = ufun.reserved_distribution
        assert abs(returned_dist.mean() - dist.mean()) < 1e-10

    def test_reserved_value_setter_accepts_float(self):
        """Test that reserved_value setter accepts float."""
        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues, reserved_value=0.0)
        ufun.reserved_value = 0.75
        assert ufun.reserved_value == 0.75

    def test_reserved_value_setter_accepts_distribution(self):
        """Test that reserved_value setter accepts Distribution."""
        from negmas.helpers.prob import ScipyDistribution

        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues, reserved_value=0.0)

        dist = ScipyDistribution(type="uniform", loc=0.0, scale=1.0)  # mean = 0.5
        ufun.reserved_value = dist
        assert abs(ufun.reserved_value - 0.5) < 1e-10

    def test_call_with_none_returns_reserved_value(self):
        """Test that calling ufun with None returns reserved_value as float."""
        issues = [make_issue(10), make_issue(5)]
        ufun = LinearUtilityFunction.random(issues=issues, reserved_value=0.3)
        result = ufun(None)
        assert isinstance(result, float)
        assert result == 0.3

    def test_call_with_none_distribution_returns_mean(self):
        """Test that calling ufun with None when reserved is Distribution returns mean."""
        from negmas.helpers.prob import ScipyDistribution

        issues = [make_issue(10), make_issue(5)]
        dist = ScipyDistribution(type="uniform", loc=0.2, scale=0.6)  # mean = 0.5
        # Use direct constructor
        ufun = LinearUtilityFunction(
            weights=[1.0, 1.0], issues=issues, reserved_value=dist
        )

        result = ufun(None)
        assert isinstance(result, float)
        assert abs(result - 0.5) < 1e-10

    def test_prob_ufun_call_with_none_returns_distribution(self):
        """Test that ProbUtilityFunction returns Distribution when called with None."""
        from negmas.helpers.prob import ScipyDistribution
        from negmas.preferences.prob.mapping import ProbMappingUtilityFunction

        mapping = {
            ("a",): ScipyDistribution(type="uniform", loc=0.0, scale=0.5),
            ("b",): ScipyDistribution(type="uniform", loc=0.5, scale=0.5),
        }
        ufun = ProbMappingUtilityFunction(mapping=mapping, reserved_value=0.3)

        result = ufun(None)
        # ProbUtilityFunction should return a Distribution
        assert hasattr(result, "mean")
        assert abs(result.mean() - 0.3) < 1e-10

    def test_prob_ufun_call_with_none_distribution_reserved(self):
        """Test that ProbUtilityFunction returns the reserved Distribution when called with None."""
        from negmas.helpers.prob import ScipyDistribution
        from negmas.preferences.prob.mapping import ProbMappingUtilityFunction

        mapping = {
            ("a",): ScipyDistribution(type="uniform", loc=0.0, scale=0.5),
            ("b",): ScipyDistribution(type="uniform", loc=0.5, scale=0.5),
        }
        dist = ScipyDistribution(type="uniform", loc=0.2, scale=0.6)  # mean = 0.5
        ufun = ProbMappingUtilityFunction(mapping=mapping, reserved_value=dist)

        result = ufun(None)
        assert hasattr(result, "mean")
        assert abs(result.mean() - 0.5) < 1e-10

    def test_discounted_ufun_with_distribution_reserved_value(self):
        """Test that discounted ufuns work with Distribution reserved_value."""
        from negmas.helpers.prob import ScipyDistribution
        from negmas.preferences.discounted import ExpDiscountedUFun

        issues = [make_issue(10), make_issue(5)]
        dist = ScipyDistribution(type="uniform", loc=0.2, scale=0.6)  # mean = 0.5
        base_ufun = LinearUtilityFunction.random(issues=issues, normalized=True)

        discounted = ExpDiscountedUFun(
            ufun=base_ufun, discount=0.9, reserved_value=dist
        )

        assert isinstance(discounted.reserved_value, float)
        assert abs(discounted.reserved_value - 0.5) < 1e-10

    def test_backward_compatibility_float_reserved_value(self):
        """Test that existing code using float reserved_value still works."""
        issues = [make_issue(10), make_issue(5)]

        # Direct constructor with float reserved_value (the main backward compat case)
        # Note: .random() treats reserved_value as a range tuple, so we use direct constructor
        ufun1 = LinearUtilityFunction(
            weights=[1.0, 1.0], issues=issues, reserved_value=0.0
        )
        ufun2 = LinearUtilityFunction(
            weights=[1.0, 1.0], issues=issues, reserved_value=float("-inf")
        )
        ufun3 = LinearUtilityFunction(
            weights=[1.0, 1.0], issues=issues, reserved_value=1.0
        )

        assert ufun1.reserved_value == 0.0
        assert ufun2.reserved_value == float("-inf")
        assert ufun3.reserved_value == 1.0

        # Operations should still work
        assert ufun1(None) == 0.0
        assert ufun2(None) == float("-inf")
        assert ufun3(None) == 1.0
