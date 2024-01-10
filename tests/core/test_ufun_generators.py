import random

from hypothesis import example, given
from hypothesis import strategies as st
from pytest_check import pytest

from negmas.helpers.misc import distribute_integer_randomly
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.preferences.generators import (
    GENERATOR_MAP,
    generate_multi_issue_ufuns,
    generate_single_issue_ufuns,
    generate_utility_values,
    make_curve_pareto,
    make_endpoints,
    make_non_pareto,
    make_pareto,
    make_piecewise_linear_pareto,
    make_zero_sum_pareto,
    sample_between,
)


def dominates(x, y):
    return any(a > b for a, b in zip(x, y))


@given(
    n_pareto=st.integers(2, 40),
    n_segments_min=st.integers(1, 10),
    n_segments_range=st.integers(0, 10),
)
@example(
    n_pareto=2,
    n_segments_min=2,
    n_segments_range=1,
)
def test_make_piecewise_pareto2(n_pareto, n_segments_min, n_segments_range):
    make_piecewise_linear_pareto(
        n_pareto,
        n_segments=(n_segments_min, n_segments_min + n_segments_range)
        if n_segments_range
        else n_segments_min,
    )


@given(n=st.integers(0, 100), m=st.integers(1, 200), min_per_bin=st.integers(0, 200))
def test_distribute_integer_randomly(n, m, min_per_bin):
    lst = distribute_integer_randomly(n, m)
    assert len(lst) == m
    assert sum(lst) == n


@given(n=st.integers(0, 100), m=st.integers(1, 200))
def test_distribute_integer_randomly_on_none(n, m):
    lst = distribute_integer_randomly(n, m, min_per_bin=None)
    assert len(lst) == m
    assert sum(lst) == n
    if lst:
        assert max(lst) - min(lst) <= 1


@given(
    start=st.floats(0.0, 1.0),
    rng=st.floats(0.0, 1.0),
    n=st.integers(1, 100),
    endpoint=st.booleans(),
    main_range_min=st.floats(0.0, 0.5),
    main_range_range=st.floats(0.0, 0.5),
)
def test_sample_between(start, rng, n, endpoint, main_range_min, main_range_range):
    main_range_min = round(main_range_min, 6)
    main_range_range = round(main_range_range, 6)
    end = start + rng
    lst = sample_between(
        start, end, n, endpoint, (main_range_min, main_range_min + main_range_range)
    )
    assert len(lst) == n
    assert all(start <= _ <= end for _ in lst)


@given(n_segments=st.integers(0, 100))
def test_make_endpoints(n_segments):
    points = make_endpoints(n_segments)
    prev = (float("-inf"), float("inf"))
    for point in points:
        assert len(point) == 2
        assert point[0] > prev[0]
        assert point[1] < prev[1]
        prev = point


@given(
    n_segments=st.integers(0, 100),
    n_outcomes=st.integers(0, 100),
)
@example(n_segments=3, n_outcomes=2)
def test_make_pareto(n_segments, n_outcomes):
    points = make_pareto(make_endpoints(n_segments), n_outcomes)
    assert len(points) == n_outcomes
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                continue
            assert (
                any(a < b for a, b in zip(p1, p2))
                or (all(a == b for a, b in zip(p1, p2)))
                or n_segments == 0
            )


@given(
    n_segments=st.integers(1, 100),
    n_pareto=st.integers(1, 100),
    n_non=st.integers(0, 400),
)
@example(n_segments=1, n_pareto=2, n_non=1)
def test_make_non_pareto(n_segments, n_pareto, n_non):
    pareto_points = make_pareto(make_endpoints(n_segments), n_pareto)
    assert len(pareto_points) == n_pareto
    points = make_non_pareto(pareto_points, n_non)
    assert len(points) == n_non

    for non_pareto in points:
        assert any(dominates(x, non_pareto) for x in pareto_points)


@given(
    n_pareto=st.integers(1, 100),
)
def test_make_zero_sum_pareto(n_pareto):
    points = make_zero_sum_pareto(n_pareto)
    assert len(points) == n_pareto
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                continue
            assert any(a < b for a, b in zip(p1, p2)) or (
                all(a == b for a, b in zip(p1, p2))
            )


@given(
    n_segments=st.integers(0, 100),
    n_pareto=st.integers(1, 100),
)
def test_make_piecewise_pareto(n_pareto, n_segments):
    points = make_piecewise_linear_pareto(n_pareto, n_segments=n_segments)
    assert len(points) == n_pareto
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i == j:
                continue
            assert any(a < b for a, b in zip(p1, p2)) or (
                all(a == b for a, b in zip(p1, p2))
            )


@given(
    shape=st.floats(1e-3, 5.0),
    n_pareto=st.integers(1, 100),
)
def test_make_curve_pareto(n_pareto, shape):
    pareto = make_curve_pareto(n_pareto, shape=shape)
    assert len(pareto) == n_pareto
    for i, p1 in enumerate(pareto):
        for j, p2 in enumerate(pareto):
            if i == j:
                continue
            assert any(a < b for a, b in zip(p1, p2)) or (
                all(a == b for a, b in zip(p1, p2))
            )


@given(
    n_pareto=st.integers(1, 100),
    n_non=st.integers(0, 400),
    generator=st.sampled_from(list(GENERATOR_MAP.keys())),
)
def test_generate_utility_values(n_pareto, n_non, generator):
    n_outcomes = n_pareto + n_non
    points = generate_utility_values(
        n_pareto=n_pareto,
        n_outcomes=n_outcomes,
        pareto_first=True,
        pareto_generator=generator,
    )
    pareto = points[:n_pareto]
    non_paretos = points[n_pareto:]

    for i, p1 in enumerate(pareto):
        for j, p2 in enumerate(pareto):
            if i == j:
                continue
            assert any(a < b for a, b in zip(p1, p2)) or (
                all(a == b for a, b in zip(p1, p2))
            ), f"{p1} and {p2} should be Pareto optimal but neither dominates the other"
    for y in non_paretos:
        assert any(
            dominates(x, y) for x in pareto
        ), f"{y} is non-pareto but not dominated by any pareto outcome"


@pytest.mark.parametrize(
    "ufun_type, val_type, numeric, linear",
    [
        (LinearAdditiveUtilityFunction, int, True, True),
        (LinearAdditiveUtilityFunction, str, False, True),
        (MappingUtilityFunction, int, True, False),
        (MappingUtilityFunction, str, False, False),
    ],
)
def test_generate_singleissue_utility_values_example(
    ufun_type, val_type, numeric, linear
):
    ufuns = generate_single_issue_ufuns(
        n_pareto=100,
        n_outcomes=30,
        n_ufuns=2,
        pareto_generator=random.choice(list(GENERATOR_MAP.keys())),
        linear=linear,
        numeric=numeric,
    )
    assert len(ufuns) > 0
    assert ufuns[0].outcome_space is not None
    assert isinstance(ufuns[0].outcome_space, CartesianOutcomeSpace)
    assert all([isinstance(_, ufun_type) for _ in ufuns])
    outcome = ufuns[0].outcome_space.random_outcome()
    assert all([isinstance(_, val_type) for _ in outcome])


@pytest.mark.parametrize(
    "ufun_type, val_type, numeric, linear",
    [
        (LinearAdditiveUtilityFunction, int, True, True),
        (LinearAdditiveUtilityFunction, str, False, True),
        (MappingUtilityFunction, int, True, False),
        (MappingUtilityFunction, str, False, False),
    ],
)
def test_generate_multiissue_utility_values_example(
    ufun_type, val_type, numeric, linear
):
    ufuns = generate_multi_issue_ufuns(
        5,
        n_values=10,
        sizes=None,
        n_ufuns=2,
        pareto_generators=tuple(GENERATOR_MAP.keys()),
        linear=linear,
        numeric=numeric,
    )
    assert len(ufuns) > 0
    assert ufuns[0].outcome_space is not None
    assert isinstance(ufuns[0].outcome_space, CartesianOutcomeSpace)
    assert all([isinstance(_, ufun_type) for _ in ufuns])
    outcome = ufuns[0].outcome_space.random_outcome()
    assert all([isinstance(_, val_type) for _ in outcome])


@given(
    fractions=st.lists(st.floats(0, 1), min_size=2, max_size=2),
    generator=st.sampled_from(list(GENERATOR_MAP.keys())),
)
def test_generate_singleissue_utility_rational_fraction(fractions, generator):
    ufuns = generate_single_issue_ufuns(
        n_pareto=100,
        n_outcomes=30,
        n_ufuns=2,
        pareto_generator=generator,
        rational_fractions=fractions,
        reservation_selector=lambda a, _: a,
    )
    for ufun, f in zip(ufuns, fractions):
        assert ufun.outcome_space
        outcomes = list(ufun.outcome_space.enumerate_or_sample())
        n_outcomes = ufun.outcome_space.cardinality
        assert (
            abs(
                int(f * n_outcomes + 0.5)
                - len(
                    [
                        v
                        for outcome in outcomes
                        if (v := float(ufun(outcome))) >= ufun.reserved_value
                    ]
                )
            )
            < 2
        )


@pytest.mark.parametrize(
    "ufun_type, val_type, numeric, linear",
    [
        (LinearAdditiveUtilityFunction, int, True, True),
        (LinearAdditiveUtilityFunction, str, False, True),
        (MappingUtilityFunction, int, True, False),
        (MappingUtilityFunction, str, False, False),
    ],
)
def test_generate_multiissue_utility_values_example(
    ufun_type, val_type, numeric, linear
):
    ufuns = generate_multi_issue_ufuns(
        5,
        n_values=10,
        sizes=None,
        n_ufuns=2,
        pareto_generators=tuple(GENERATOR_MAP.keys()),
        linear=linear,
        numeric=numeric,
    )
    assert len(ufuns) > 0
    assert ufuns[0].outcome_space is not None
    assert isinstance(ufuns[0].outcome_space, CartesianOutcomeSpace)
    assert all([isinstance(_, ufun_type) for _ in ufuns])
    outcome = ufuns[0].outcome_space.random_outcome()
    assert all([isinstance(_, val_type) for _ in outcome])
