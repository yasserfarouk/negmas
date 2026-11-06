"""Regression tests for the random generators of composite ufuns."""

from __future__ import annotations

import pytest

from negmas.outcomes import make_issue, make_os
from negmas.preferences import (
    AffineUtilityFunction,
    ExpDiscountedUFun,
    LinDiscountedUFun,
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
