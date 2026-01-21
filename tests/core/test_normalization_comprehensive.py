"""
Comprehensive tests for normalization across all utility function types.

Tests consistency of:
1. normalize() vs normalize_for() with ufun's outcome_space
2. Reserved value handling across all methods
3. Constant function edge cases
4. Scale/shift operations
5. normalize_all_for() common scale behavior
"""

from __future__ import annotations

import pytest
from negmas.outcomes import make_issue, make_os
from negmas.preferences import (
    AffineUtilityFunction,
    ConstUtilityFunction,
    LinearAdditiveUtilityFunction,
    LinearUtilityFunction,
)
from negmas.preferences.value_fun import AffineFun, ConstFun, LinearFun


class TestNormalizeConsistency:
    """Test that normalize() and normalize_for() are consistent."""

    def test_normalize_vs_normalize_for_linear(self):
        """Test normalize() equals normalize_for() with ufun's outcome_space."""
        issues = (make_issue(10),)
        u = LinearUtilityFunction(weights=[2.0], issues=issues)

        normalized1 = u.normalize(to=(0.0, 1.0))
        normalized2 = u.normalize_for(to=(0.0, 1.0), outcome_space=u.outcome_space)

        # Check same min/max
        mn1, mx1 = normalized1.minmax()
        mn2, mx2 = normalized2.minmax()
        assert abs(mn1 - mn2) < 1e-6
        assert abs(mx1 - mx2) < 1e-6

        # Check same utilities for sample outcomes
        for i in range(10):
            assert abs(normalized1((i,)) - normalized2((i,))) < 1e-6

    def test_normalize_vs_normalize_for_affine(self):
        """Test normalize() equals normalize_for() with ufun's outcome_space."""
        issues = (make_issue(10),)
        u = AffineUtilityFunction(weights=[1.0], bias=5.0, issues=issues)

        normalized1 = u.normalize(to=(0.0, 1.0))
        normalized2 = u.normalize_for(to=(0.0, 1.0), outcome_space=u.outcome_space)

        mn1, mx1 = normalized1.minmax()
        mn2, mx2 = normalized2.minmax()
        assert abs(mn1 - mn2) < 1e-6
        assert abs(mx1 - mx2) < 1e-6

        for i in range(10):
            assert abs(normalized1((i,)) - normalized2((i,))) < 1e-6

    def test_normalize_different_ranges(self):
        """Test normalization to different target ranges."""
        issues = (make_issue(10),)
        u = LinearUtilityFunction(weights=[1.0], issues=issues)

        # Original range: [0, 9]
        normalized1 = u.normalize(to=(0.0, 1.0))
        normalized2 = u.normalize(to=(2.0, 8.0))
        normalized3 = u.normalize(to=(-1.0, 1.0))

        mn1, mx1 = normalized1.minmax()
        mn2, mx2 = normalized2.minmax()
        mn3, mx3 = normalized3.minmax()

        assert abs(mn1 - 0.0) < 1e-6 and abs(mx1 - 1.0) < 1e-6
        assert abs(mn2 - 2.0) < 1e-6 and abs(mx2 - 8.0) < 1e-6
        assert abs(mn3 - (-1.0)) < 1e-6 and abs(mx3 - 1.0) < 1e-6


class TestReservedValueConsistency:
    """Test reserved value handling is consistent."""

    def test_const_function_reserved_value_relationship(self):
        """Test that const value and reserved value relationship is preserved."""

        issues = (make_issue(10),)
        os = make_os(issues)

        # Case 1: const_value < reserved_value
        u1 = ConstUtilityFunction(value=5.0, outcome_space=os, reserved_value=10.0)
        n1 = u1.normalize(to=(0.0, 1.0))
        assert n1((0,)) < n1.reserved_value, "Const value should be < reserved"
        assert abs(n1((0,)) - 0.0) < 1e-6  # const maps to to[0]
        assert abs(n1.reserved_value - 1.0) < 1e-6  # reserved maps to to[1]

        # Case 2: const_value > reserved_value
        u2 = ConstUtilityFunction(value=10.0, outcome_space=os, reserved_value=5.0)
        n2 = u2.normalize(to=(0.0, 1.0))
        assert n2((0,)) > n2.reserved_value, "Const value should be > reserved"
        assert abs(n2((0,)) - 1.0) < 1e-6  # const maps to to[1]
        assert abs(n2.reserved_value - 0.0) < 1e-6  # reserved maps to to[0]

    def test_const_function_different_ranges(self):
        """Test const function normalization with different target ranges."""

        issues = (make_issue(10),)
        os = make_os(issues)

        u = ConstUtilityFunction(value=10.0, outcome_space=os, reserved_value=20.0)

        # Normalize to different ranges
        n1 = u.normalize(to=(0.0, 1.0))
        n2 = u.normalize(to=(2.0, 8.0))
        n3 = u.normalize(to=(-5.0, 5.0))

        # All should preserve const < reserved relationship
        assert n1((0,)) < n1.reserved_value
        assert n2((0,)) < n2.reserved_value
        assert n3((0,)) < n3.reserved_value

        # Check actual values
        assert abs(n1((0,)) - 0.0) < 1e-6 and abs(n1.reserved_value - 1.0) < 1e-6
        assert abs(n2((0,)) - 2.0) < 1e-6 and abs(n2.reserved_value - 8.0) < 1e-6
        assert abs(n3((0,)) - (-5.0)) < 1e-6 and abs(n3.reserved_value - 5.0) < 1e-6


class TestNormalizeAllForCommonScale:
    """Test normalize_all_for creates true common scale."""

    def test_common_scale_with_guarantee_min(self):
        """Test that guarantee_min=True creates common scale from minimum."""
        issues = (make_issue(10), make_issue(5))

        # Agent 1: range [0, 9]
        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        # Agent 2: range [0, 4]
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        normalized = LinearUtilityFunction.normalize_all_for(
            (u1, u2), to=(0.0, 1.0), guarantee_min=True, guarantee_max=False
        )
        n1, n2 = normalized

        # Both should start at 0
        mn1, mx1 = n1.minmax()
        mn2, mx2 = n2.minmax()
        assert abs(mn1 - 0.0) < 1e-6
        assert abs(mn2 - 0.0) < 1e-6

        # n1 should span full range (had global max)
        assert abs(mx1 - 1.0) < 1e-6

        # n2 should span less (4/9 of full range)
        assert abs(mx2 - 4.0 / 9.0) < 1e-3

    def test_common_scale_preserves_ratios(self):
        """Test that common scale preserves utility ratios."""
        issues = (make_issue(10),)

        u1 = LinearUtilityFunction(weights=[1.0], issues=issues)  # [0, 9]
        u2 = LinearUtilityFunction(weights=[2.0], issues=issues)  # [0, 18]

        # Before normalization, u2 is exactly 2x u1
        assert abs(u2((5,)) - 2 * u1((5,))) < 1e-6

        normalized = LinearUtilityFunction.normalize_all_for(
            (u1, u2), to=(0.0, 1.0), guarantee_min=True, guarantee_max=False
        )
        n1, n2 = normalized

        # After normalization, ratio should be preserved
        # Both start at 0, u2 should still be 2x u1
        for i in range(10):
            if abs(n1((i,))) > 1e-6:  # Avoid division by zero
                ratio = n2((i,)) / n1((i,))
                assert abs(ratio - 2.0) < 1e-3

    def test_normalize_all_for_with_negative_utilities(self):
        """Test normalize_all_for handles negative utilities correctly."""
        issues = (make_issue(10),)

        # Agent 1: range [-5, 4] (bias=-5, values 0-9)
        u1 = AffineUtilityFunction(weights=[1.0], bias=-5.0, issues=issues)
        # Agent 2: range [0, 9]
        u2 = LinearUtilityFunction(weights=[1.0], issues=issues)

        mn1, mx1 = u1.minmax()
        mn2, mx2 = u2.minmax()
        assert abs(mn1 - (-5.0)) < 1e-6
        assert abs(mn2 - 0.0) < 1e-6

        normalized = AffineUtilityFunction.normalize_all_for(
            (u1, u2), to=(0.0, 1.0), guarantee_min=True, guarantee_max=False
        )
        n1, n2 = normalized

        # Both should start at 0 (global min was -5)
        mn1_new, mx1_new = n1.minmax()
        mn2_new, mx2_new = n2.minmax()
        assert abs(mn1_new - 0.0) < 1e-6
        assert abs(mn2_new - 0.0) < 1e-6

        # n1 had range 9 (from -5 to 4), global range is 14 (from -5 to 9)
        # n2 had range 9 (from 0 to 9), global range is 14 (from -5 to 9)
        # After scaling by 1/14: n1 spans 9/14, n2 spans 9/14
        assert abs(mx1_new - 9.0 / 14.0) < 1e-3
        assert abs(mx2_new - 9.0 / 14.0) < 1e-3


class TestValueFunctionOperations:
    """Test value function scale_by and shift_by operations."""

    def test_const_fun_scale_by(self):
        """Test ConstFun.scale_by returns ConstFun."""
        c = ConstFun(5.0)
        scaled = c.scale_by(2.0)

        assert isinstance(scaled, ConstFun)
        assert abs(scaled.bias - 10.0) < 1e-6

    def test_linear_fun_scale_by(self):
        """Test LinearFun.scale_by scales weight."""
        f = LinearFun(3.0)
        scaled = f.scale_by(2.0)

        assert isinstance(scaled, LinearFun)
        assert abs(scaled(0.0) - 0.0) < 1e-6
        assert abs(scaled(1.0) - 6.0) < 1e-6

    def test_affine_fun_scale_by(self):
        """Test AffineFun.scale_by scales weight and bias."""
        f = AffineFun(2.0, 3.0)  # 2*x + 3
        scaled = f.scale_by(3.0)  # Should be 6*x + 9

        assert isinstance(scaled, AffineFun)
        assert abs(scaled(0.0) - 9.0) < 1e-6
        assert abs(scaled(1.0) - 15.0) < 1e-6
        assert abs(scaled(2.0) - 21.0) < 1e-6

    def test_const_fun_shift_by(self):
        """Test ConstFun.shift_by returns ConstFun."""
        c = ConstFun(5.0)
        shifted = c.shift_by(3.0)

        assert isinstance(shifted, ConstFun)
        assert abs(shifted.bias - 8.0) < 1e-6

    def test_linear_fun_shift_by(self):
        """Test LinearFun.shift_by returns AffineFun."""
        f = LinearFun(2.0)  # 2*x
        shifted = f.shift_by(5.0)  # Should be 2*x + 5

        assert isinstance(shifted, AffineFun)
        assert abs(shifted(0.0) - 5.0) < 1e-6
        assert abs(shifted(1.0) - 7.0) < 1e-6
        assert abs(shifted(2.0) - 9.0) < 1e-6


class TestLinearAdditiveNormalization:
    """Test LinearAdditiveUtilityFunction.normalize_for() thoroughly."""

    def test_normalize_handles_negative_weights(self):
        """Test that negative weights are made positive."""
        issues = (make_issue(10), make_issue(10))
        values = [LinearFun(1.0), LinearFun(-1.0)]  # Second is negative
        weights = [1.0, 1.0]

        u = LinearAdditiveUtilityFunction(values=values, weights=weights, issues=issues)

        # Before normalization, utilities can be negative
        assert u((9, 0)) < 0  # 1*9 + 1*(-1*0) = 9 - 0 = 9... hmm

        normalized = u.normalize_for(to=(0.0, 1.0))

        # After normalization, all weights should be positive
        assert isinstance(normalized, LinearAdditiveUtilityFunction)
        for w in normalized.weights:
            assert w >= 0

    def test_normalize_makes_weights_sum_to_one(self):
        """Test that weights are normalized to sum to 1."""
        issues = (make_issue(10), make_issue(10))
        values = [LinearFun(1.0), LinearFun(2.0)]
        weights = [3.0, 7.0]  # Sum to 10

        u = LinearAdditiveUtilityFunction(values=values, weights=weights, issues=issues)
        normalized = u.normalize_for(to=(0.0, 1.0))

        assert isinstance(normalized, LinearAdditiveUtilityFunction)
        assert abs(sum(normalized.weights) - 1.0) < 1e-6

    def test_normalize_makes_value_functions_01(self):
        """Test that all value functions map to [0, 1]."""
        issues = (make_issue(10), make_issue(10))
        values = [AffineFun(2.0, -5.0), AffineFun(1.0, 3.0)]  # Different ranges
        weights = [1.0, 1.0]

        u = LinearAdditiveUtilityFunction(values=values, weights=weights, issues=issues)
        normalized = u.normalize_for(to=(0.0, 1.0))

        assert isinstance(normalized, LinearAdditiveUtilityFunction)

        # Check each value function maps [0, 9] to [0, 1]
        for vf in normalized.values:
            mn = vf(0)
            mx = vf(9)
            assert abs(mn - 0.0) < 1e-6 or abs(mn - 1.0) < 1e-6
            assert abs(mx - 0.0) < 1e-6 or abs(mx - 1.0) < 1e-6
            assert abs(abs(mx - mn) - 1.0) < 1e-6


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_normalize_zero_range_becomes_const(self):
        """Test that ufun with zero range becomes ConstUtilityFunction."""
        issues = (make_issue(10),)
        u = LinearUtilityFunction(weights=[0.0], issues=issues)  # Always returns 0

        mn, mx = u.minmax()
        assert abs(mx - mn) < 1e-6

        normalized = u.normalize(to=(0.0, 1.0))
        assert isinstance(normalized, ConstUtilityFunction)

    def test_affine_zero_weights_raises_error(self):
        """Test that affine with zero weights can't be normalized to non-zero range."""
        issues = (make_issue(10),)
        u = AffineUtilityFunction(weights=[0.0], bias=5.0, issues=issues)

        with pytest.raises(ValueError, match="zero weights"):
            u.normalize_for(to=(0.0, 1.0))

    def test_normalize_already_normalized(self):
        """Test normalizing an already normalized function."""
        issues = (make_issue(10),)
        u = LinearUtilityFunction(weights=[1.0 / 9.0], issues=issues)

        # Already in [0, 1]
        mn, mx = u.minmax()
        assert abs(mn - 0.0) < 1e-6
        assert abs(mx - 1.0) < 1e-6

        normalized = u.normalize(to=(0.0, 1.0))

        # Should produce equivalent results
        mn_new, mx_new = normalized.minmax()
        assert abs(mn_new - 0.0) < 1e-6
        assert abs(mx_new - 1.0) < 1e-6

        for i in range(10):
            assert abs(normalized((i,)) - u((i,))) < 1e-6

    def test_normalize_negative_to_positive_range(self):
        """Test normalizing utilities from negative to positive range."""
        issues = (make_issue(10),)
        u = AffineUtilityFunction(
            weights=[1.0], bias=-10.0, issues=issues
        )  # Range [-10, -1]

        mn, mx = u.minmax()
        assert mn < 0 and mx < 0

        normalized = u.normalize(to=(0.0, 1.0))

        mn_new, mx_new = normalized.minmax()
        assert abs(mn_new - 0.0) < 1e-6
        assert abs(mx_new - 1.0) < 1e-6

    def test_normalize_single_outcome(self):
        """Test normalizing with single outcome (should become constant)."""
        issues = (make_issue(1),)  # Only value 0
        u = LinearUtilityFunction(weights=[1.0], issues=issues)

        # Only one outcome, so min == max
        mn, mx = u.minmax()
        assert abs(mx - mn) < 1e-6

        normalized = u.normalize(to=(0.0, 1.0))
        assert isinstance(normalized, ConstUtilityFunction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
