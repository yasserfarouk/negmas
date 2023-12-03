import hypothesis.strategies as st
import pytest
from hypothesis import example, given
from rich import print

from negmas.outcomes import make_issue, make_os
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.preferences.inv_ufun import (
    PresortingInverseUtilityFunction,
    PresortingInverseUtilityFunctionBruteForce,
)


def make_ufun(nissues=1, nvalues=10, r=4):
    os = make_os([make_issue(nvalues) for _ in range(nissues)])
    outcomes = list(os.enumerate_or_sample())
    ufun = MappingUtilityFunction(
        dict(zip(outcomes, range(len(outcomes)), strict=True)),
        reserved_value=r,
        outcome_space=os,
    )
    return outcomes, ufun


@pytest.mark.parametrize(
    "cls",
    [PresortingInverseUtilityFunction, PresortingInverseUtilityFunctionBruteForce],
)
def test_inv_simple_case_sort_all(cls):
    outcomes, ufun = make_ufun()
    inverter = cls(ufun, rational_only=False)
    inverter.init()
    inverted_outcomes = [outcomes[-i] for i in range(1, len(outcomes) + 1)]
    for i in range(len(outcomes)):
        assert (
            inverter.outcome_at(i) == inverted_outcomes[i]
        ), f"{i}: found {inverter.outcome_at(i)} expected {inverted_outcomes[i]}"


@pytest.mark.parametrize(
    "cls",
    [PresortingInverseUtilityFunction, PresortingInverseUtilityFunctionBruteForce],
)
def test_inv_simple_case_sort_rational(cls):
    outcomes, ufun = make_ufun()
    inverter = cls(ufun, rational_only=True)
    inverter.init()
    inverted_outcomes = [outcomes[-i] for i in range(1, len(outcomes) + 1)]
    for i in range(5):
        assert (
            inverter.outcome_at(i) == inverted_outcomes[i]
        ), f"{i}: found {inverter.outcome_at(i)} expected {inverted_outcomes[i]}"
    for i in range(6, 10):
        assert (
            inverter.outcome_at(i) == outcomes[i - 6]
        ), f"{i}: found {inverter.outcome_at(i)} expected {outcomes[i]}"


@given(
    rational_only=st.booleans(),
    normalized=st.booleans(),
    nissues=st.integers(1, 4),
    nvalues=st.integers(1, 4),
    mn=st.floats(0.0, 1.0),
    mx=st.floats(0.0, 1.0),
    n=st.integers(1, 10),
    r=st.floats(1, 10),
)
def test_inv_some(rational_only, normalized, nissues, nvalues, mn, mx, n, r):
    _, ufun = make_ufun(nissues, nvalues, r)
    fast = PresortingInverseUtilityFunction(ufun, rational_only=rational_only)
    fast.init()
    d = float(ufun.max()) - float(ufun.min())
    true_range = (min(mn, mx) * d + ufun.min(), max(mn, mx) * d + ufun.min())
    if normalized:
        rng = (min(mn, mx), max(mn, mx))
    else:
        rng = true_range
    o = fast.some(rng, normalized, n=n)
    assert all([true_range[0] <= ufun(_) <= true_range[1] for _ in o])


@given(
    rational_only=st.booleans(),
    nissues=st.integers(1, 4),
    nvalues=st.integers(1, 4),
    r=st.floats(1, 4),
)
@example(rational_only=True, nissues=1, nvalues=2, r=1)
@example(rational_only=True, nissues=1, nvalues=2, r=2.0)
def test_inv_matches_bruteforce_outcome_at(rational_only, nissues, nvalues, r):
    outcomes, ufun = make_ufun(nissues, nvalues, r)
    fast = PresortingInverseUtilityFunction(ufun, rational_only=rational_only)
    brute = PresortingInverseUtilityFunctionBruteForce(
        ufun, rational_only=rational_only
    )
    fast.init()
    brute.init()
    assert fast.extreme_outcomes() == brute.extreme_outcomes()
    assert fast.minmax() == brute.minmax()
    assert fast.best() == brute.best()
    assert fast.worst() == brute.worst()
    assert fast.min() == brute.min()
    assert fast.max() == brute.max()
    for indx in range(len(outcomes) + 1):
        fo = fast.outcome_at(indx)
        fb = brute.outcome_at(indx)
        assert fo == fb or (
            ufun(fo) < ufun.reserved_value and ufun(fb) < ufun.reserved_value
        )
        fu, bu = fast.utility_at(indx), brute.utility_at(indx) or (
            ufun(fo) < ufun.reserved_value and ufun(fb) < ufun.reserved_value
        )
        assert (
            abs(fu - bu) < 1e-6
            or (bu == float("-inf") and fu == float("-inf"))
            or (ufun(fo) < ufun.reserved_value and ufun(fb) < ufun.reserved_value)
        )


@given(
    rational_only=st.booleans(),
    nissues=st.integers(1, 4),
    nvalues=st.integers(1, 4),
    mn=st.floats(0.0, 1.0),
    mx=st.floats(0.0, 1.0),
    r=st.floats(0.0, 1.0),
)
@example(rational_only=False, nissues=1, nvalues=1, mn=0.0, mx=0.0, r=0.0)
def test_inv_matches_bruteforce_within_indices(
    rational_only, nissues, nvalues, mn, mx, r
):
    outcomes, ufun = make_ufun(nissues, nvalues, r)
    fast = PresortingInverseUtilityFunction(ufun, rational_only=rational_only)
    brute = PresortingInverseUtilityFunctionBruteForce(
        ufun, rational_only=rational_only
    )
    fast.init()
    brute.init()
    rng = (int(min(mn, mx) * len(outcomes)), int(max(mn, mx) * len(outcomes)))
    x = sorted(fast.within_fractions(rng))
    y = sorted(brute.within_fractions(rng))
    assert abs(len(x) - len(y)) <= 1
    assert all([abs(a[0] - b[0]) < 1.1 for a, b in zip(x, y)]) or all(
        [abs(a[0] - b[0]) < 1.1 for a, b in zip(reversed(x), reversed(y))]
    )


@given(
    rational_only=st.booleans(),
    nissues=st.integers(1, 4),
    nvalues=st.integers(1, 4),
    mn=st.floats(0.0, 1.0),
    mx=st.floats(0.0, 1.0),
    r=st.floats(0.0, 1.0),
)
@example(rational_only=False, nissues=1, nvalues=2, mn=0.5, mx=0.5, r=0.0)
@example(rational_only=False, nissues=1, nvalues=2, mn=1.0, mx=0.5, r=0.0)
@example(rational_only=True, nissues=1, nvalues=2, mn=0.0, mx=1.0, r=1.0)
@example(rational_only=False, nissues=1, nvalues=3, mn=0.0, mx=0.5, r=0.0)
@example(rational_only=False, nissues=1, nvalues=2, mn=0.0, mx=1.0, r=0.0)
@example(rational_only=False, nissues=1, nvalues=2, mn=0.0, mx=0.5, r=0.0)
def test_inv_matches_bruteforce_within_fractions(
    rational_only, nissues, nvalues, mn, mx, r
):
    _, ufun = make_ufun(nissues, nvalues, r)
    fast = PresortingInverseUtilityFunction(ufun, rational_only=rational_only)
    brute = PresortingInverseUtilityFunctionBruteForce(
        ufun, rational_only=rational_only
    )
    fast.init()
    brute.init()

    rng = (min(mn, mx), max(mn, mx))
    x = sorted(fast.within_fractions(rng))
    y = sorted(brute.within_fractions(rng))
    assert abs(len(x) - len(y)) <= 1
    assert all([abs(a[0] - b[0]) < 1.1 for a, b in zip(x, y)]) or all(
        [abs(a[0] - b[0]) < 1.1 for a, b in zip(reversed(x), reversed(y))]
    )


@given(
    rational_only=st.booleans(),
    nissues=st.integers(1, 4),
    nvalues=st.integers(1, 4),
    mn=st.floats(0.0, 1.0),
    mx=st.floats(0.0, 1.0),
    r=st.floats(0.0, 1.0),
)
@example(rational_only=False, nissues=1, nvalues=2, mn=0.0, mx=1.0, r=0.0)
@example(rational_only=True, nissues=1, nvalues=1, mn=0.0, mx=0.0, r=1.0)
def test_inv_matches_bruteforce_best_worst(rational_only, nissues, nvalues, mn, mx, r):
    _, ufun = make_ufun(nissues, nvalues, r)
    fast = PresortingInverseUtilityFunction(ufun, rational_only=rational_only)
    brute = PresortingInverseUtilityFunctionBruteForce(
        ufun, rational_only=rational_only
    )
    fast.init()
    brute.init()
    rng = (min(mn, mx), max(mn, mx))
    x = fast.best_in(rng, normalized=True, cycle=False)
    y = brute.best_in(rng, normalized=True)
    assert x == y
    x = fast.worst_in(rng, normalized=True, cycle=False)
    y = brute.worst_in(rng, normalized=True)
    assert x == y


@given(
    rational_only=st.booleans(),
    nissues=st.integers(1, 4),
    nvalues=st.integers(2, 4),
    mn=st.floats(0.0, 1.0),
    mx=st.floats(0.0, 1.0),
    r=st.floats(0.0, 1.0),
)
@example(rational_only=False, nissues=1, nvalues=3, mn=0.5, mx=0.5, r=0.0)
@example(rational_only=True, nissues=1, nvalues=2, mn=0.0, mx=1.0, r=1.0)
@example(rational_only=False, nissues=1, nvalues=2, mn=0.0, mx=1.0, r=0.0)
@example(rational_only=True, nissues=1, nvalues=1, mn=0.0, mx=0.0, r=1.0)
def test_inv_matches_bruteforce_all(rational_only, nissues, nvalues, mn, mx, r):
    _, ufun = make_ufun(nissues, nvalues, r)
    fast = PresortingInverseUtilityFunction(ufun, rational_only=rational_only)
    brute = PresortingInverseUtilityFunctionBruteForce(
        ufun, rational_only=rational_only
    )
    fast.init()
    brute.init()
    rng = (min(mn, mx), max(mn, mx))

    x = sorted(fast.all(rng, normalized=False))
    y = sorted(brute.all(rng, normalized=False))
    assert abs(len(x) - len(y)) <= 1
    assert all([abs(a[0] - b[0]) < 1.1 for a, b in zip(x, y)]) or all(
        [abs(a[0] - b[0]) < 1.1 for a, b in zip(reversed(x), reversed(y))]
    )


@given(
    rational_only=st.booleans(),
    normalized=st.booleans(),
    nissues=st.integers(1, 4),
    nvalues=st.integers(2, 4),
    mn=st.floats(0.0, 1.0),
    mx=st.floats(0.0, 1.0),
    r=st.floats(0.0, 1.0),
)
@example(
    rational_only=True,
    normalized=False,  # or any other generated value
    nissues=1,  # or any other generated value
    nvalues=2,  # or any other generated value
    mn=0.0,
    mx=0.0,
    r=1.0,
)
def test_inv_one_in(rational_only, normalized, nissues, nvalues, mn, mx, r):
    _, ufun = make_ufun(nissues, nvalues, r)
    fast = PresortingInverseUtilityFunction(ufun, rational_only=rational_only)
    fast.init()
    umn, umx = ufun.minmax()
    assert ufun.outcome_space is not None
    all_values = sorted(ufun(_) for _ in ufun.outcome_space.enumerate_or_sample())
    umn, umx = float(umn), float(umx)
    d = umx - umn
    true_range = (min(mn, mx) * d + umn, max(mn, mx) * d + umn)
    if normalized:
        rng = (min(mn, mx), max(mn, mx))
    else:
        rng = true_range
    outcome_found = False
    for u in all_values:
        if true_range[0] <= u <= true_range[1]:
            outcome_found = True
            break
    o = fast.one_in(rng, normalized)
    assert (
        o is not None
        or true_range[0] > umx
        or true_range[1] < umn
        or (not outcome_found)
        or r > umn
    ), f"We should always find an outcome if the range {true_range} is within {umn, umx}\n{all_values=}\n{outcome_found=}, ufun range: {(umn, umx)}"
    assert o is None or true_range[0] - 1e-4 <= ufun(o) <= true_range[1] + 1e-4
