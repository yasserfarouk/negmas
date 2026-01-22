"""
Tests for Scenario improvements from negobench integration.

This test module covers:
1. is_normalized() with both independent and common-scale modes
2. is_linear property
3. scale_min() and scale_max() with outcome_space parameter
"""

from __future__ import annotations

import pytest
from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import (
    ConstUtilityFunction,
    ExpDiscountedUFun,
    LinearAdditiveUtilityFunction,
    LinearUtilityFunction,
)
from negmas.preferences.crisp.nonlinear import HyperRectangleUtilityFunction


class TestIsNormalizedIndependent:
    """Tests for is_normalized() with independent=True mode."""

    def test_independent_normalized_returns_true(self):
        """Test that independently normalized scenario returns True."""
        issues = (make_issue(10), make_issue(5))
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        scenario.normalize(independent=True, to=(0.0, 1.0))

        # Each ufun should span [0, 1]
        assert scenario.is_normalized((0.0, 1.0), independent=True)

    def test_independent_not_normalized_returns_false(self):
        """Test that non-normalized scenario returns False in independent mode."""
        issues = (make_issue(10), make_issue(5))
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))

        # Not normalized yet
        assert not scenario.is_normalized((0.0, 1.0), independent=True)

    def test_independent_with_different_ranges(self):
        """Test that independent mode requires all ufuns to reach both bounds."""
        issues = (make_issue(10), make_issue(5))
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        # Common-scale normalization - not all ufuns reach max
        scenario.normalize(independent=False, to=(0.0, 1.0))

        # Should fail in independent mode because not all ufuns reach 1.0
        assert not scenario.is_normalized((0.0, 1.0), independent=True)


class TestIsNormalizedCommonScale:
    """Tests for is_normalized() with independent=False mode (common-scale)."""

    def test_common_scale_normalized_returns_true(self):
        """Test that common-scale normalized scenario returns True."""
        issues = (make_issue(10), make_issue(5))
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        scenario.normalize(independent=False, to=(0.0, 1.0), guarantee_min=True)

        # Should pass common-scale check
        assert scenario.is_normalized((0.0, 1.0), independent=False)

    def test_common_scale_with_proportional_ranges(self):
        """Test common-scale with ufuns having proportional ranges."""
        issues = (make_issue([0, 5, 10], "x"), make_issue([0, 2, 4], "y"))
        os = make_os(issues)

        # u1: range [0, 10], u2: range [0, 4]
        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        scenario.normalize(independent=False, to=(0.0, 1.0), guarantee_max=False)

        # After normalization with guarantee_max=False:
        # - All ufuns are scaled by the same factor based on global range
        # - Then shifted to align minimum with to[0]
        # - u1 should be [0, 1.0] (has global max)
        # - u2 should be [0, 0.4] (proportional to its range)
        # Both within [0, 1], at least one reaches each bound
        assert scenario.is_normalized((0.0, 1.0), independent=False)

        # Verify ranges
        mn1, mx1 = scenario.ufuns[0].minmax()
        mn2, mx2 = scenario.ufuns[1].minmax()

        assert abs(mn1 - 0.0) < 1e-6
        assert abs(mx1 - 1.0) < 1e-6
        assert abs(mn2 - 0.0) < 1e-6
        assert 0.3 < mx2 < 0.5  # Approximately 0.4

    def test_common_scale_none_bounds(self):
        """Test is_normalized with None bounds."""
        issues = (make_issue(10),)
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.5], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        scenario.normalize(independent=False, to=(0.0, 1.0))

        # Check only max constraint
        assert scenario.is_normalized((None, 1.0), independent=False)

        # Check only min constraint
        assert scenario.is_normalized((0.0, None), independent=False)

    def test_common_scale_not_normalized_returns_false(self):
        """Test that non-normalized scenario returns False."""
        issues = (make_issue(10), make_issue(5))
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))

        # Not normalized yet
        assert not scenario.is_normalized((0.0, 1.0), independent=False)


class TestIsLinear:
    """Tests for is_linear property."""

    def test_linear_scenario_returns_true(self):
        """Test that scenario with all linear ufuns returns True."""
        issues = (make_issue(10), make_issue(5))
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert scenario.is_linear

    def test_linear_additive_returns_true(self):
        """Test that LinearAdditiveUtilityFunction is considered linear."""
        issues = (make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y"))
        os = make_os(issues)

        u1 = LinearAdditiveUtilityFunction(
            weights={"x": 0.5, "y": 0.5},
            values={"x": {0: 0.0, 1: 0.5, 2: 1.0}, "y": {0: 0.0, 1: 0.5, 2: 1.0}},
            outcome_space=os,
        )
        u2 = LinearAdditiveUtilityFunction(
            weights={"x": 0.7, "y": 0.3},
            values={"x": {0: 0.0, 1: 0.5, 2: 1.0}, "y": {0: 0.0, 1: 0.5, 2: 1.0}},
            outcome_space=os,
        )

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert scenario.is_linear

    def test_const_ufun_returns_true(self):
        """Test that ConstUtilityFunction is considered linear."""
        issues = (make_issue(10),)
        os = make_os(issues)

        u1 = ConstUtilityFunction(value=5.0, outcome_space=os)
        u2 = ConstUtilityFunction(value=10.0, outcome_space=os)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert scenario.is_linear

    def test_discounted_linear_returns_true(self):
        """Test that discounted linear ufun is considered linear."""
        issues = (make_issue(10),)
        os = make_os(issues)

        base_ufun = LinearUtilityFunction(weights=[1.0], issues=issues)
        u1 = ExpDiscountedUFun(ufun=base_ufun, discount=0.9)
        u2 = LinearUtilityFunction(weights=[0.5], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert scenario.is_linear

    def test_nonlinear_returns_false(self):
        """Test that scenario with nonlinear ufuns returns False."""
        issues = (make_issue(values=[0, 1, 2, 3, 4], name="x"),)
        os = make_os(issues)

        # Create a hyperrectangle ufun (nonlinear)
        u1 = HyperRectangleUtilityFunction(
            outcome_ranges=[{0: (0.0, 2.0), 1: (1.0, 3.0)}],
            utilities=[10.0],
            issues=issues,
        )
        u2 = LinearUtilityFunction(weights=[1.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert not scenario.is_linear

    def test_mixed_linear_nonlinear_returns_false(self):
        """Test that mixing linear and nonlinear ufuns returns False."""
        issues = (make_issue(values=[0, 1, 2, 3, 4], name="x"),)
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        u2 = HyperRectangleUtilityFunction(
            outcome_ranges=[{0: (0.0, 2.0)}], utilities=[5.0], issues=issues
        )

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        assert not scenario.is_linear


class TestScaleWithOutcomeSpace:
    """Tests for scale_min() and scale_max() with outcome_space parameter."""

    def test_scale_min_with_outcome_space(self):
        """Test scale_min with explicit outcome_space parameter."""
        issues = (make_issue([1, 5, 10], "x"),)  # Range [1, 10] to avoid zero
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[2.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        scenario.scale_min(to=5.0, outcome_space=os)

        # Both should have min=5.0
        mn1, _ = scenario.ufuns[0].minmax()
        mn2, _ = scenario.ufuns[1].minmax()

        assert abs(mn1 - 5.0) < 1e-6
        assert abs(mn2 - 5.0) < 1e-6

    def test_scale_max_with_outcome_space(self):
        """Test scale_max with explicit outcome_space parameter."""
        issues = (make_issue(10),)
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[2.0], issues=issues)

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        scenario.scale_max(to=100.0, outcome_space=os)

        # Both should have max=100.0
        _, mx1 = scenario.ufuns[0].minmax()
        _, mx2 = scenario.ufuns[1].minmax()

        assert abs(mx1 - 100.0) < 1e-6
        assert abs(mx2 - 100.0) < 1e-6

    def test_scale_min_without_outcome_space_uses_default(self):
        """Test that scale_min without outcome_space uses ufun's default."""
        issues = (make_issue([1, 5, 10], "x"),)  # Range [1, 10] to avoid zero
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        scenario = Scenario(outcome_space=os, ufuns=(u1,))

        # Should work without explicit outcome_space
        scenario.scale_min(to=2.0)

        mn, _ = scenario.ufuns[0].minmax()
        assert abs(mn - 2.0) < 1e-6


class TestIsNormalizedPositive:
    """Tests for the positive parameter in is_normalized()."""

    def test_positive_constraint_with_negative_min(self):
        """Test that positive=True fails when min is negative."""
        issues = (make_issue(10),)
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        scenario = Scenario(outcome_space=os, ufuns=(u1,))

        # Shift to negative range
        scenario.ufuns = tuple(u.shift_by(-5.0) for u in scenario.ufuns)

        # Should fail positive constraint
        assert not scenario.is_normalized(
            (None, None), positive=True, independent=False
        )

    def test_positive_constraint_with_non_negative_min(self):
        """Test that positive=True passes when min is non-negative."""
        issues = (make_issue(10),)
        os = make_os(issues)

        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)
        scenario = Scenario(outcome_space=os, ufuns=(u1,))
        scenario.normalize(to=(0.0, 1.0))

        # Should pass positive constraint
        assert scenario.is_normalized((None, None), positive=True, independent=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
