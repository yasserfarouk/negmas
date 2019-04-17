import pytest

from negmas import *


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
    p1, l1 = pareto_frontier([f1, f2], outcomes=[(_,) for _ in range(10)])
    p2, l2 = pareto_frontier([f2, f1], outcomes=[(_,) for _ in range(10)])

    assert p1 == [(0.9419131964655383, 0.0), (0.7242899747791032, 1.0)]
    assert p2 == [(1.0, 0.7242899747791032), (0.0, 0.9419131964655383)]
    assert l1 == [6, 3]
    assert l2 == list(reversed(l1))
    assert len(p1) == len(p2)

    # reverse order of p2
    p2 = [(_[1], _[0]) for _ in p2]
    for a in p1:
        assert a in p2


def test_linear_utility():
    buyer_utility = LinearUtilityAggregationFunction(
        {
            "cost": lambda x: -x,
            "number of items": lambda x: 0.5 * x,
            "delivery": {"delivered": 10.0, "not delivered": -2.0},
        }
    )
    assert (
        buyer_utility({"cost": 1.0, "number of items": 3, "delivery": "not delivered"})
        == -1.0 + 1.5 - 2.0
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


if __name__ == "__main__":
    pytest.main(args=[__file__])
