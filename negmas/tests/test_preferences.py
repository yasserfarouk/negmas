from __future__ import annotations

import random

import hypothesis.strategies as st
import numpy as np
import pkg_resources
import pytest
from hypothesis import given, settings
from hypothesis.core import example
from pytest import mark

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
    normalize,
    pareto_frontier_bf,
    pareto_frontier_of,
    scale_max,
)


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
        mapping=lambda x: sum(r * v for r, v in zip(rs, x)),
    )
    assert ufun([0.0] * n_issues) == 0.0
    assert ufun([1.0] * n_issues) == sum(rs)
    rng = ufun.minmax(issues=issues)
    assert rng[0] >= 0.0
    assert rng[1] <= sum(rs)


@given(
    n_outcomes=st.integers(2, 1000),
    n_negotiators=st.integers(2, 5),
    unique=st.booleans(),
    sort=st.booleans(),
)
@settings(deadline=60_000)
@example(n_outcomes=4, n_negotiators=2, unique=False, sort=False)
@example(n_outcomes=2, n_negotiators=2, unique=False, sort=False)
def test_pareto_frontier_matches_bf(n_outcomes, n_negotiators, unique, sort):

    np.random.seed(0)
    utils = np.random.rand(n_outcomes, n_negotiators)
    bf = list(
        pareto_frontier_bf(utils, sort_by_welfare=sort, unique_utility_values=unique)
    )
    x = list(
        pareto_frontier_of(utils, sort_by_welfare=sort, unique_utility_values=unique)
    )
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

    f1 = MappingUtilityFunction(lambda o: u1[o[0]])
    f2 = MappingUtilityFunction(lambda o: u2[o[0]])
    assert all(f1((i,)) == u1[i] for i in range(10))
    assert all(f2((i,)) == u2[i] for i in range(10))
    p1, l1 = pareto_frontier(
        [f1, f2],
        sort_by_welfare=True,
        unique_utility_values=False,
        outcomes=[(_,) for _ in range(10)],
    )
    p2, l2 = pareto_frontier(
        [f2, f1],
        sort_by_welfare=True,
        unique_utility_values=False,
        outcomes=[(_,) for _ in range(10)],
    )

    assert set(p1) == {(0.9419131964655383, 0.0), (0.7242899747791032, 1.0)}
    assert set(p2) == {(1.0, 0.7242899747791032), (0.0, 0.9419131964655383)}
    assert set(l1) == {6, 3}
    assert len(p1) == len(p2)
    # assert l2 == list(reversed(l1))

    # reverse order of p2
    p2 = [(_[1], _[0]) for _ in p2]
    for a in p1:
        assert a in p2


def test_linear_utility():
    buyer_utility = LinearAdditiveUtilityFunction(
        {
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
        {
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
            {
                "cost": lambda x: -x,
                "number of items": lambda x: 0.5 * x,
                "delivery": {"delivered": 10.0, "not delivered": -2.0},
            },
        )


def test_hypervolume_utility():
    f = HyperRectangleUtilityFunction(
        outcome_ranges=[
            None,
            {0: (1.0, 2.0), 1: (1.0, 2.0)},
            {0: (1.4, 2.0), 2: (2.0, 3.0)},
        ],
        utilities=[5.0, 2.0, lambda x: 2 * x[2] + x[0]],
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
            assert u == e


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
    issues, _ = issues_from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-domain.xml"
            ),
        ).read(),
    )
    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
        ).read(),
        issues=issues,
    )
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
    ), f"Normalization of ufun of type "
    f"LinearUtilityFunction should generate an IndependentIssuesUFun but we got {type(nfun).__name__}"
    u2 = [nfun(w) for w in outcomes]

    assert (
        max(u2) < rng[1] + 1e-3 and min(u2) > rng[0] - 1e-3
    ), f"Limits are not correct\n{u1}\n{u2}"

    if rng[0] == float("-inf") and rng[1] == float("inf"):
        assert ufun is nfun, "Normalizing with an infinite range should do nothing"

    if bias == 0.0 and rng[0] == 0.0 and sum(weights) > 1e-5:
        scale = [a / b for a, b in zip(u1, u2) if b != 0 and a != 0]
        for a, b in zip(u1, u2):
            assert (
                abs(a - b) < 1e-5 or abs(b) > 1e-5
            ), f"zero values are mapped to zero values"
            assert (
                abs(a - b) < 1e-5 or abs(a) > 1e-5
            ), f"zero values are mapped to zero values"
        assert max(scale) - min(scale) < 1e-3, f"ufun did not scale uniformly"
    # order1, order2 = _order(u1), _order(u2)
    # assert order1 == order2, f"normalization changed the order of outcomes\n{u1[37:41]}\n{u2[37:41]}"

    if (
        (rng[1] == float("inf") or rng[0] == float("-inf"))
        and abs(bias) < 1e-3
        and sum(weights) > 1e-5
    ):
        relative1 = _relative_fraction(u1)
        relative2 = _relative_fraction(u2)
        assert (
            np.abs(relative1 - relative2).max() < 1e-3
        ), f"One side normalization should not change the result of dividing outcomes\n{u1}\n{u2}"


def test_normalization():
    os = CartesianOutcomeSpace.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-domain.xml"
            ),
        ).read(),
    )
    issues = os.issues
    outcomes = list(os.enumerate())
    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
        ).read(),
        issues=issues,
    )
    assert abs(u(("Dell", "60 Gb", "19'' LCD")) - 21.987727736172488) < 0.000001
    assert abs(u(("HP", "80 Gb", "20'' LCD")) - 22.68559475583014) < 0.000001
    utils = [u(_) for _ in outcomes]
    max_util, min_util = max(utils), min(utils)
    gt_range = dict(
        zip(outcomes, [(_ - min_util) / (max_util - min_util) for _ in utils])
    )
    gt_max = dict(zip(outcomes, [_ / max_util for _ in utils]))

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
        ).read(),
        issues=issues,
    )
    u = normalize(u, to=(0.0, 1.0))
    utils = [u(_) for _ in outcomes]
    max_util, min_util = max(utils), min(utils)
    assert abs(max_util - 1.0) < 0.001
    assert abs(min_util) < 0.001

    for k, v in gt_range.items():
        assert abs(v - u(k)) < 1e-3, f"Failed for {k} got {(u(k))} expected {v}"

    u, _ = UtilityFunction.from_xml_str(
        open(
            pkg_resources.resource_filename(
                "negmas", resource_name="tests/data/Laptop/Laptop-C-prof1.xml"
            ),
        ).read(),
        issues=issues,
    )
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
    assert any(
        ro((0, _)) == 0 for _ in range(1, 6)
    ), f"{[(_, ro(_)) for _ in [None] + outcomes]}"
    assert ro((9, 5)) == 10 * 5 - 1
    mapping = ro.to_mapping_ufun()
    assert mapping.reserved_value == ro(None)
    assert any(
        mapping((0, _)) == 0 for _ in range(1, 6)
    ), f"{[(_, mapping(_)) for _ in [None] + outcomes]}"
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
    assert any(
        ro((0, _)) == 0 for _ in range(1, 6)
    ), f"{[(_, ro(_)) for _ in [None] + outcomes]}"
    assert ro((9, 5)) == 10 * 5
    mapping = ro.to_mapping_ufun()
    assert isinstance(mapping(None), int)
    assert isinstance(mapping.reserved_value, int)
    assert mapping.reserved_value == ro(None)
    assert any(
        mapping((0, _)) == 0 for _ in range(1, 6)
    ), f"{[(_, mapping(_)) for _ in [None] + outcomes]}"
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
    assert all(
        ro((9, 5)) > ro(_) for _ in outcomes if _ != (9, 5)
    ), f"{[(_, ro(_)) for _ in outcomes]}"
    mapping = ro.to_mapping_ufun()
    assert isinstance(mapping(None), int)
    assert isinstance(mapping.reserved_value, int)
    assert mapping.reserved_value == ro(None)
    assert all(
        mapping((9, 5)) > mapping(_) for _ in outcomes if _ != (9, 5)
    ), f"{[(_, mapping(_)) for _ in outcomes]}"
    assert mapping((1, 1)) == mapping((0, 2))
    assert mapping((2, 1)) == mapping((1, 2)) == mapping((0, 3))


if __name__ == "__main__":
    pytest.main(args=[__file__])
