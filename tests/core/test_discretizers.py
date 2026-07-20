"""Tests for ``negmas.preferences.discretizers`` and ``SubsetCartesianOutcomeSpace``."""

from __future__ import annotations

import pytest

from negmas.outcomes import (
    DiscreteCartesianOutcomeSpace,
    SubsetCartesianOutcomeSpace,
    make_issue,
    make_os,
)
from negmas.preferences import LinearAdditiveUtilityFunction, MappingUtilityFunction
from negmas.outcomes.discretizers import (
    DEFAULT_LEVELS,
    BalancedOutcomeCountsInUFunBinsDiscretizer,
    BalancedOutcomeCountsInUFunsBinsDiscretizer,
    BalancedUFunsVarianceDiscretizer,
    BalancedUFunVarianceDiscretizer,
    BaseDiscretizer,
    DefaultDiscretizer,
    Discretizer,
    GridBasedDiscretizer,
)


def continuous_os(n_issues: int = 2):
    return make_os([make_issue((0.0, 1.0), f"i{i}") for i in range(n_issues)])


def discrete_os(*cards: int) -> DiscreteCartesianOutcomeSpace:
    os = make_os([make_issue(c, f"i{i}") for i, c in enumerate(cards)])
    assert isinstance(os, DiscreteCartesianOutcomeSpace)
    return os


# --------------------------------------------------------------------------- #
# Protocol / base class
# --------------------------------------------------------------------------- #
def test_grid_discretizer_satisfies_protocol():
    d = GridBasedDiscretizer(max_outcomes=100, min_levels=5)
    assert isinstance(d, Discretizer)
    assert isinstance(d, BaseDiscretizer)


def test_default_discretizer_is_grid_based():
    assert DefaultDiscretizer is GridBasedDiscretizer


def test_base_discretizer_none_semantics():
    d = GridBasedDiscretizer()
    assert d.max_outcomes is None and d.min_levels is None
    assert d.max_cardinality == float("inf")
    assert d.levels == DEFAULT_LEVELS


def test_base_discretizer_values():
    d = GridBasedDiscretizer(max_outcomes=50, min_levels=7)
    assert d.max_cardinality == 50
    assert d.levels == 7


# --------------------------------------------------------------------------- #
# GridBasedDiscretizer
# --------------------------------------------------------------------------- #
def test_grid_continuous_basic():
    d = GridBasedDiscretizer(max_outcomes=1000, min_levels=5)
    dos = d(continuous_os(2))
    assert isinstance(dos, DiscreteCartesianOutcomeSpace)
    assert dos.cardinality == 25


def test_grid_matches_to_discrete_when_fits():
    os = continuous_os(2)
    d = GridBasedDiscretizer(max_outcomes=None, min_levels=6)
    assert d(os).cardinality == os.to_discrete(levels=6).cardinality


def test_grid_steps_levels_down_to_fit_cap():
    os = continuous_os(2)
    # 5*5=25 > 20 -> steps to 4*4 = 16
    assert GridBasedDiscretizer(max_outcomes=20, min_levels=5)(os).cardinality == 16


def test_grid_default_levels_when_none():
    os = continuous_os(2)
    # DEFAULT_LEVELS ** 2 outcomes when no cap
    assert GridBasedDiscretizer()(os).cardinality == DEFAULT_LEVELS**2


def test_grid_discrete_passthrough():
    os = discrete_os(3, 4)  # 12
    dos = GridBasedDiscretizer(max_outcomes=100)(os)
    assert dos is os


def test_grid_discrete_over_cap_raises():
    os = discrete_os(5, 5)  # 25
    with pytest.raises(ValueError):
        GridBasedDiscretizer(max_outcomes=10)(os)


def test_grid_continuous_impossible_cap_raises():
    # a huge discrete issue combined with continuous cannot fit even at 1 level
    os = make_os([make_issue(100, "d"), make_issue((0.0, 1.0), "c")])
    with pytest.raises(ValueError):
        GridBasedDiscretizer(max_outcomes=50, min_levels=5)(os)


def test_grid_result_outcomes_are_valid():
    d = GridBasedDiscretizer(max_outcomes=1000, min_levels=4)
    dos = d(continuous_os(2))
    for o in dos.enumerate():
        assert o in dos


# --------------------------------------------------------------------------- #
# SubsetCartesianOutcomeSpace
# --------------------------------------------------------------------------- #
def subset_space():
    issues = (make_issue(["a", "b", "c"], "x"), make_issue([0, 1, 2], "y"))  # 9
    return SubsetCartesianOutcomeSpace(
        issues, outcomes=[("a", 0), ("b", 1), ("c", 2), ("a", 2)]
    )


def test_subset_is_discrete_cartesian():
    os = subset_space()
    assert isinstance(os, DiscreteCartesianOutcomeSpace)
    assert os.is_discrete() and os.is_finite()


def test_subset_cardinality_is_subset_not_grid():
    os = subset_space()
    assert os.cardinality == 4
    assert len(os) == 4


def test_subset_enumerate_only_selected():
    os = subset_space()
    assert sorted(os.enumerate()) == [("a", 0), ("a", 2), ("b", 1), ("c", 2)]


def test_subset_enumerate_or_sample_routes_to_subset():
    os = subset_space()
    assert sorted(os.enumerate_or_sample()) == sorted(os.enumerate())


def test_subset_validity():
    os = subset_space()
    assert ("a", 0) in os  # selected
    assert ("b", 0) not in os  # valid grid combo, not selected
    assert os.is_valid(("c", 2))
    assert not os.is_valid(("c", 0))


def test_subset_issues_preserved():
    os = subset_space()
    assert os.issue_names == ["x", "y"]
    assert len(os.issues) == 2


def test_subset_dedup_preserves_order():
    issues = (make_issue(["a", "b"], "x"), make_issue([0, 1], "y"))
    os = SubsetCartesianOutcomeSpace(
        issues, outcomes=[("a", 0), ("a", 0), ("b", 1), ("a", 0)]
    )
    assert os.cardinality == 2
    assert list(os.outcomes) == [("a", 0), ("b", 1)]


def test_subset_to_discrete_returns_self():
    os = subset_space()
    assert os.to_discrete() is os
    assert os.to_largest_discrete(5) is os


def test_subset_limit_cardinality():
    os = subset_space()
    limited = os.limit_cardinality(2)
    assert limited.cardinality == 2
    assert os.limit_cardinality(100) is os


def test_subset_sample_without_replacement():
    os = subset_space()
    s = list(os.sample(3, with_replacement=False))
    assert len(s) == 3
    assert all(x in os for x in s)
    assert len(set(s)) == 3


def test_subset_sample_with_replacement():
    os = subset_space()
    s = list(os.sample(10, with_replacement=True))
    assert len(s) == 10
    assert all(x in os for x in s)


def test_subset_random_outcome_is_selected():
    os = subset_space()
    for _ in range(20):
        assert os.random_outcome() in os


def test_subset_from_outcome_set_infers_issues():
    os = SubsetCartesianOutcomeSpace.from_outcome_set([("a", 0), ("b", 1)])
    assert len(os.issues) == 2
    assert os.cardinality == 2


def test_subset_constraint_filters_enumeration():
    os = subset_space()
    os.add_constraint(lambda o: o[0] != "a")
    assert ("a", 0) not in os
    assert os.cardinality == 2  # ('b',1) and ('c',2)


def test_subset_iteration_and_bool():
    os = subset_space()
    assert bool(os)
    assert sorted(iter(os)) == sorted(os.enumerate())
    empty = SubsetCartesianOutcomeSpace(os.issues, outcomes=[])
    assert not bool(empty)
    assert empty.cardinality == 0


# --------------------------------------------------------------------------- #
# Balanced (utility-aware) discretizers
# --------------------------------------------------------------------------- #
def _ufun(os):
    return LinearAdditiveUtilityFunction.random(os, reserved_value=0.0)


ALL_BALANCED_SINGLE = [
    BalancedUFunVarianceDiscretizer,
    BalancedOutcomeCountsInUFunBinsDiscretizer,
]
ALL_BALANCED_MULTI = [
    BalancedUFunsVarianceDiscretizer,
    BalancedOutcomeCountsInUFunsBinsDiscretizer,
]


@pytest.mark.parametrize("cls", ALL_BALANCED_SINGLE)
def test_balanced_single_returns_subset_space(cls):
    os = continuous_os(2)
    u = _ufun(os)
    d = cls(u, n_bins=5, max_outcomes=20)
    assert isinstance(d, Discretizer)
    r = d(os)
    assert isinstance(r, SubsetCartesianOutcomeSpace)
    assert r.cardinality <= 20
    assert r.cardinality > 0
    assert all(o in r for o in r.enumerate())


@pytest.mark.parametrize("cls", ALL_BALANCED_MULTI)
def test_balanced_multi_returns_subset_space(cls):
    os = continuous_os(2)
    u1, u2 = _ufun(os), _ufun(os)
    d = cls([u1, u2], n_bins=4, max_outcomes=30)
    r = d(os)
    assert isinstance(r, SubsetCartesianOutcomeSpace)
    assert r.cardinality <= 30
    assert r.cardinality > 0


@pytest.mark.parametrize("cls", ALL_BALANCED_SINGLE)
def test_balanced_deterministic_on_continuous(cls):
    os = continuous_os(2)
    u = _ufun(os)
    d = cls(u, n_bins=5, max_outcomes=25)
    assert sorted(d(os).enumerate()) == sorted(d(os).enumerate())


@pytest.mark.parametrize("cls", ALL_BALANCED_SINGLE)
def test_balanced_on_discrete_space(cls):
    os = discrete_os(6, 6)  # 36 candidates, enumerated exactly
    u = _ufun(os)
    r = cls(u, n_bins=4, max_outcomes=12)(os)
    assert r.cardinality <= 12
    # every selected outcome is one of the real grid outcomes
    grid = set(os.enumerate())
    assert all(o in grid for o in r.enumerate())


@pytest.mark.parametrize("cls", ALL_BALANCED_SINGLE)
def test_balanced_no_budget_returns_balanced_set(cls):
    os = discrete_os(8, 8)
    u = _ufun(os)
    r = cls(u, n_bins=4)(os)  # no max_outcomes
    assert r.cardinality > 0
    assert r.cardinality <= os.cardinality


@pytest.mark.parametrize("cls", ALL_BALANCED_SINGLE)
def test_balanced_nbins_one(cls):
    os = discrete_os(5, 5)
    u = _ufun(os)
    r = cls(u, n_bins=1, max_outcomes=7)(os)
    assert 0 < r.cardinality <= 7


def test_balanced_validation():
    os = discrete_os(4, 4)
    u = _ufun(os)
    with pytest.raises(ValueError):
        BalancedUFunsVarianceDiscretizer([], n_bins=3)
    with pytest.raises(ValueError):
        BalancedUFunVarianceDiscretizer(u, n_bins=0)


def test_balanced_budget_distributes_across_bins():
    # With equal-width bins over a discrete grid, selection should touch
    # multiple utility bins rather than collapsing to the top outcomes.
    os = discrete_os(10, 10)
    u = _ufun(os)
    r = BalancedOutcomeCountsInUFunBinsDiscretizer(u, n_bins=5, max_outcomes=25)(os)
    utils = sorted(float(u(o)) for o in r.enumerate())
    # the selected utilities should span a wide range (not just the top bin)
    assert utils[-1] - utils[0] > 0


# --- semantics: the four discretizers must actually differ ------------------ #
def _skewed_ufun(os, power=2.0):
    """A monotone, skewed single-issue ufun (u = v**power)."""
    mapping = {(v,): float(v) ** power for v in range(os.issues[0].cardinality)}
    return MappingUtilityFunction(mapping, outcome_space=os)  # type: ignore[arg-type]


def test_quantile_vs_equal_width_differ_on_skewed_data():
    os = discrete_os(30)
    u = _skewed_ufun(os)
    variance = set(
        BalancedUFunVarianceDiscretizer(u, n_bins=5, max_outcomes=10)(os).enumerate()
    )
    counts = set(
        BalancedOutcomeCountsInUFunBinsDiscretizer(u, n_bins=5, max_outcomes=10)(
            os
        ).enumerate()
    )
    # Quantile (equal-frequency) and equal-width bins select different subsets
    # when the utility distribution is skewed.
    assert variance != counts


@pytest.mark.parametrize("cls", ALL_BALANCED_MULTI)
def test_multi_ufun_depends_on_all_ufuns(cls):
    # 2-issue discrete space; three genuinely different ufuns.
    os = discrete_os(8, 8)
    outcomes = list(os.enumerate())
    u1 = MappingUtilityFunction(
        {o: float(o[0] + o[1]) for o in outcomes}, outcome_space=os
    )
    u2 = MappingUtilityFunction(
        {o: float(o[0] - o[1]) for o in outcomes}, outcome_space=os
    )
    u3 = MappingUtilityFunction(
        {o: float(o[0] * o[1]) for o in outcomes}, outcome_space=os
    )
    with_u2 = set(cls([u1, u2], n_bins=3, max_outcomes=15)(os).enumerate())
    with_u3 = set(cls([u1, u3], n_bins=3, max_outcomes=15)(os).enumerate())
    # Swapping the second ufun changes the joint binning, hence the selection.
    assert with_u2 != with_u3


def test_equal_width_meets_budget_on_skewed_data():
    # Skewed distribution: sparse high-utility tail. Water-filling should still
    # deliver the full budget when enough candidates exist.
    os = discrete_os(40)
    u = _skewed_ufun(os, power=3.0)
    r = BalancedOutcomeCountsInUFunBinsDiscretizer(u, n_bins=8, max_outcomes=20)(os)
    assert r.cardinality == 20


def test_grid_matches_to_discrete_outcomes_when_fits():
    os = continuous_os(2)
    got = set(GridBasedDiscretizer(min_levels=6)(os).enumerate())
    expected = set(os.to_discrete(levels=6).enumerate())
    assert got == expected


# --------------------------------------------------------------------------- #
# DiscreteCartesianOutcomeSpace.limit_cardinality (bug-fix regression)
# --------------------------------------------------------------------------- #
def test_limit_cardinality_max_only_reduces():
    os = discrete_os(5, 5)  # 25
    r = os.limit_cardinality(10)  # previously a no-op returning 25
    assert r.cardinality <= 10
    assert r.cardinality > 0


def test_limit_cardinality_returns_self_when_within_both():
    os = discrete_os(5, 5)  # 25
    assert os.limit_cardinality(100) is os
    assert os.limit_cardinality(100, levels=5) is os


def test_limit_cardinality_with_levels_no_crash():
    os = discrete_os(5, 5)
    for lv in (3, 4, 5):
        r = os.limit_cardinality(10, levels=lv)
        assert r.cardinality <= 10
        assert all(i.cardinality <= lv for i in r.issues)


def test_limit_cardinality_shrinks_categorical():
    os = make_os(
        [
            make_issue([f"a{i}" for i in range(10)], "x"),
            make_issue([f"b{i}" for i in range(10)], "y"),
        ]
    )  # 100 categorical
    r = os.limit_cardinality(9)
    assert r.cardinality <= 9
    assert all(i.cardinality <= 3 for i in r.issues)


def test_limit_cardinality_balanced_and_bounded():
    os = make_os([make_issue(1000, "a"), make_issue(980, "b")])  # 980_000
    r = os.limit_cardinality(10_000)
    assert r.cardinality <= 10_000
    per = sorted(i.cardinality for i in r.issues)
    # balanced: neither issue gutted to 1 while the other keeps everything
    assert per[0] > 1


def test_limit_cardinality_levels_only():
    os = discrete_os(5, 5)
    r = os.limit_cardinality(levels=3)
    assert all(i.cardinality <= 3 for i in r.issues)


def test_limit_cardinality_never_increases_discrete_issue():
    os = discrete_os(3, 4)  # small
    r = os.limit_cardinality(1000, levels=10)
    assert r is os  # already within both; must not pad issues up to 10


# --------------------------------------------------------------------------- #
# to_largest_discrete (bug-fix regression)
# --------------------------------------------------------------------------- #
def test_to_largest_discrete_continuous_fits():
    os = continuous_os(2)
    assert os.to_largest_discrete(levels=10, max_cardinality=1000).cardinality == 100


def test_to_largest_discrete_steps_down():
    os = continuous_os(2)
    # previously RAISED (checked fixed `levels`); now steps down to 6*6=36 <= 40
    assert os.to_largest_discrete(levels=10, max_cardinality=40).cardinality == 36


def test_to_largest_discrete_raises_when_impossible():
    os = make_os([make_issue((0.0, 1.0), "c"), make_issue(100, "d")])
    with pytest.raises(ValueError):
        os.to_largest_discrete(levels=10, max_cardinality=50)


def test_to_largest_discrete_discrete_returns_self():
    os = discrete_os(20, 3)
    # discrete spaces are returned unchanged (use limit_cardinality to shrink)
    assert os.to_largest_discrete(levels=10, max_cardinality=1000) is os


# --------------------------------------------------------------------------- #
# Balanced discretizers: hole-free full-grid mode
# --------------------------------------------------------------------------- #
def _skewed_2d(nx=20, ny=20):
    os = make_os([make_issue(nx, "x"), make_issue(ny, "y")])
    outs = list(os.enumerate())
    u = MappingUtilityFunction(
        {o: float(o[0] + o[1]) ** 2 for o in outs}, outcome_space=os
    )  # type: ignore[arg-type]
    return os, u


def _chi2_over_pool_bins(space, ufun, pool_utils, n_bins=5):
    import numpy as np

    edges = np.linspace(pool_utils.min(), pool_utils.max(), n_bins + 1)
    gu = np.array([float(ufun(o)) for o in space.enumerate()])
    c = np.clip(np.digitize(gu, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(c, minlength=n_bins).astype(float)
    return float(((counts - len(gu) / n_bins) ** 2).sum())


@pytest.mark.parametrize("backend", ["coordinate", "scipy"])
def test_full_grid_is_holeless_and_bounded(backend):
    import numpy as np

    os, u = _skewed_2d()
    r = BalancedOutcomeCountsInUFunBinsDiscretizer(
        u, n_bins=5, max_outcomes=25, full_grid=True, grid_optimizer=backend
    )(os)
    assert isinstance(r, DiscreteCartesianOutcomeSpace)
    assert not isinstance(r, SubsetCartesianOutcomeSpace)
    per = [i.cardinality for i in r.issues]
    # no holes: full Cartesian product, nothing dropped
    assert r.cardinality == int(np.prod(per)) == len(list(r.enumerate()))
    assert r.cardinality <= 25


@pytest.mark.parametrize("backend", ["coordinate", "scipy"])
def test_full_grid_beats_naive_even_grid(backend):
    import numpy as np

    os, u = _skewed_2d()
    r = BalancedOutcomeCountsInUFunBinsDiscretizer(
        u, n_bins=5, max_outcomes=25, full_grid=True, grid_optimizer=backend
    )(os)
    per = [i.cardinality for i in r.issues]
    pool = np.array([float(u(o)) for o in os.enumerate()])
    naive = make_os(
        [
            make_issue(list(np.linspace(0, 19, per[0]).round().astype(int)), "x"),
            make_issue(list(np.linspace(0, 19, per[1]).round().astype(int)), "y"),
        ]
    )
    assert _chi2_over_pool_bins(r, u, pool) < _chi2_over_pool_bins(naive, u, pool)


def test_full_grid_continuous_input():
    os = continuous_os(2)
    u = _ufun(os)
    r = BalancedUFunVarianceDiscretizer(u, n_bins=4, max_outcomes=16, full_grid=True)(
        os
    )
    assert isinstance(r, DiscreteCartesianOutcomeSpace)
    assert not isinstance(r, SubsetCartesianOutcomeSpace)
    assert r.cardinality == len(list(r.enumerate())) <= 16


def test_full_grid_multi_ufun():
    os = make_os([make_issue(12, "x"), make_issue(12, "y")])
    outs = list(os.enumerate())
    u1 = MappingUtilityFunction({o: float(o[0] + o[1]) for o in outs}, outcome_space=os)  # type: ignore[arg-type]
    u2 = MappingUtilityFunction({o: float(o[0] - o[1]) for o in outs}, outcome_space=os)  # type: ignore[arg-type]
    r = BalancedOutcomeCountsInUFunsBinsDiscretizer(
        [u1, u2], n_bins=3, max_outcomes=16, full_grid=True
    )(os)
    assert isinstance(r, DiscreteCartesianOutcomeSpace)
    assert r.cardinality == len(list(r.enumerate())) <= 16


def test_full_grid_invalid_optimizer():
    os, u = _skewed_2d(6, 6)
    with pytest.raises(ValueError):
        BalancedUFunVarianceDiscretizer(u, n_bins=3, grid_optimizer="nope")


# --------------------------------------------------------------------------- #
# to_discrete(method=...) dispatch
# --------------------------------------------------------------------------- #
def test_to_discrete_default_grid_unchanged():
    os = continuous_os(2)
    assert os.to_discrete(levels=5).cardinality == 25
    with pytest.raises(ValueError):
        os.to_discrete(levels=10, max_cardinality=40)  # default grid still raises


def test_to_discrete_method_grid_based_steps_down():
    os = continuous_os(2)
    r = os.to_discrete(levels=10, max_cardinality=40, method="grid_based")
    assert r.cardinality == 36  # GridBasedDiscretizer steps levels down


def test_to_discrete_method_balanced_by_name():
    os = discrete_os(10, 10)
    u = _ufun(os)
    r = os.to_discrete(
        max_cardinality=20, method="balanced_ufun_variance", ufun=u, n_bins=4
    )
    assert isinstance(r, SubsetCartesianOutcomeSpace)
    assert r.cardinality <= 20


def test_to_discrete_method_instance_and_class():
    os = discrete_os(10, 10)
    u = _ufun(os)
    inst = BalancedUFunVarianceDiscretizer(u, n_bins=4, max_outcomes=16, full_grid=True)
    r = os.to_discrete(method=inst)
    assert isinstance(r, DiscreteCartesianOutcomeSpace)
    assert r.cardinality <= 16
    r2 = os.to_discrete(
        max_cardinality=15, method=BalancedUFunVarianceDiscretizer, ufun=u, n_bins=3
    )
    assert r2.cardinality <= 15


def test_to_discrete_unknown_method_raises():
    os = continuous_os(2)
    with pytest.raises(ValueError):
        os.to_discrete(method="does_not_exist")


def test_to_discrete_registry_covers_all_discretizers():
    from negmas.outcomes.discretizers import DISCRETIZERS, get_discretizer

    for name in (
        "grid_based",
        "balanced_ufun_variance",
        "balanced_ufuns_variance",
        "balanced_outcome_counts_in_ufun_bins",
        "balanced_outcome_counts_in_ufuns_bins",
    ):
        assert name in DISCRETIZERS
        assert get_discretizer(name) is DISCRETIZERS[name]


# --------------------------------------------------------------------------- #
# Stability: repeated calls return the same output (no random sampling)
# --------------------------------------------------------------------------- #
def test_discretization_is_stable_large_discrete():
    # A discrete space larger than max_candidates previously triggered random
    # sampling in candidate generation; it must now be deterministic.
    big = make_os([make_issue(200, "x"), make_issue(200, "y")])  # 40_000 > 10_000
    u = MappingUtilityFunction(
        {o: float(o[0] * o[1]) for o in big.enumerate()}, outcome_space=big
    )  # type: ignore[arg-type]
    for cls in ALL_BALANCED_SINGLE:
        d = cls(u, n_bins=5, max_outcomes=25)
        assert sorted(d(big).enumerate()) == sorted(d(big).enumerate())
        dg = cls(u, n_bins=5, max_outcomes=25, full_grid=True)
        assert sorted(dg(big).enumerate()) == sorted(dg(big).enumerate())


def test_discretization_is_stable_grid_and_to_discrete():
    os = continuous_os(3)  # dense grid (100**3) exceeds max_candidates internally
    g = GridBasedDiscretizer(max_outcomes=64, min_levels=10)
    assert sorted(g(os).enumerate()) == sorted(g(os).enumerate())
    assert sorted(os.to_discrete(levels=5).enumerate()) == sorted(
        os.to_discrete(levels=5).enumerate()
    )
