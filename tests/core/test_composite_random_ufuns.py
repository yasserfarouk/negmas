"""Regression tests for the random generators of composite ufuns."""

from __future__ import annotations

import pytest

from negmas.outcomes import make_issue, make_os
from negmas.preferences import (
    AffineUtilityFunction,
    ComplexNonlinearUtilityFunction,
    ExpDiscountedUFun,
    LinDiscountedUFun,
    WeightedUtilityFunction,
)


@pytest.mark.parametrize("cls", [ExpDiscountedUFun, LinDiscountedUFun])
@pytest.mark.parametrize("as_space", [False, True], ids=["issues", "outcome_space"])
def test_discounted_random_accepts_issues_and_outcome_spaces(cls, as_space):
    issues = [make_issue((0, 9), "x"), make_issue((0, 4), "y")]
    os = make_os(issues)
    u = cls.random(os if as_space else issues)
    assert all(isinstance(float(u(o)), float) for o in os.enumerate())


@pytest.mark.parametrize("cls", [ExpDiscountedUFun, LinDiscountedUFun])
def test_discounted_random_with_an_issues_based_base_type(cls):
    issues = [make_issue((0, 9), "x"), make_issue((0, 4), "y")]
    u = cls.random(issues, base_preferences_type=AffineUtilityFunction)
    assert isinstance(u.ufun, AffineUtilityFunction)


@pytest.mark.parametrize(
    "cls", [WeightedUtilityFunction, ComplexNonlinearUtilityFunction]
)
def test_composite_random_samples_the_reserved_value(cls):
    os = make_os([make_issue(4, "a"), make_issue(3, "b")])
    for _ in range(5):
        u = cls.random(os, (0.2, 0.4))
        assert isinstance(u.reserved_value, float)
        assert 0.2 <= u.reserved_value <= 0.4
    assert cls.random(os, 0.3).reserved_value == pytest.approx(0.3)


@pytest.mark.parametrize(
    "cls", [WeightedUtilityFunction, ComplexNonlinearUtilityFunction]
)
def test_composite_random_components_have_no_reserved_value(cls):
    os = make_os([make_issue(4, "a"), make_issue(3, "b")])
    u = cls.random(os, 0.0, n_ufuns=3)
    parts = u.ufuns if hasattr(u, "ufuns") else u.values
    assert all(v.reserved_value == 0.0 for v in parts)
