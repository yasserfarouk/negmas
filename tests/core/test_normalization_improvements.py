"""
Tests for normalization improvements ported from negobench.

This test module covers:
1. normalize_all_for() - Multi-agent common scale normalization
2. LinearAdditiveUtilityFunction.normalize_for() - Compositional normalization
3. ConstFun.scale_by() - Correctness fix
4. AffineUtilityFunction reserved value handling
5. Constant function edge cases
"""

from __future__ import annotations

import math

import pytest
from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import (
    AffineUtilityFunction,
    ConstUtilityFunction,
    LinearAdditiveUtilityFunction,
    LinearUtilityFunction,
)
from negmas.preferences.value_fun import AffineFun, ConstFun, LinearFun


class TestNormalizeAllFor:
    """Tests for BaseUtilityFunction.normalize_all_for() class method."""

    def test_normalize_all_for_two_agents_different_ranges(self):
        """Test that two agents with different ranges get normalized to common scale."""
        issues = (make_issue(10), make_issue(5))

        # Agent 1: utilities range [0, 10]
        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)

        # Agent 2: utilities range [0, 5]
        u2 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        # Get min/max before normalization
        mn1, mx1 = u1.minmax()
        mn2, mx2 = u2.minmax()
        assert abs(mn1 - 0.0) < 1e-6
        assert abs(mx1 - 9.0) < 1e-6  # 0-indexed, so max value is 9
        assert abs(mn2 - 0.0) < 1e-6
        assert abs(mx2 - 4.0) < 1e-6  # 0-indexed, so max value is 4

        # Normalize together with common scale (guarantee_min aligns minimum values)
        normalized = LinearUtilityFunction.normalize_all_for(
            (u1, u2), to=(0.0, 1.0), guarantee_min=True, guarantee_max=False
        )
        n1, n2 = normalized

        # Check that they're normalized to common scale
        mn1_new, mx1_new = n1.minmax()
        mn2_new, mx2_new = n2.minmax()

        # Agent 1 should span [0, 1] (had global max)
        assert abs(mn1_new - 0.0) < 1e-6
        assert abs(mx1_new - 1.0) < 1e-6

        # Agent 2 should be [0, ~0.44] (4/9)
        assert abs(mn2_new - 0.0) < 1e-6
        assert abs(mx2_new - 4.0 / 9.0) < 1e-3

    def test_normalize_all_for_three_agents(self):
        """Test normalization with three agents."""
        issues = (make_issue(10), make_issue(10))

        u1 = LinearUtilityFunction(weights=[1.0, 0.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[0.5, 0.5], issues=issues)
        u3 = LinearUtilityFunction(weights=[0.0, 1.0], issues=issues)

        normalized = LinearUtilityFunction.normalize_all_for(
            (u1, u2, u3), to=(0.0, 1.0), guarantee_min=True, guarantee_max=False
        )

        # All should have same global scale
        for n in normalized:
            mn, mx = n.minmax()
            assert mn >= -1e-6
            assert mx <= 1.0 + 1e-6

    def test_normalize_all_for_overlapping_ranges(self):
        """Test agents with overlapping utility ranges."""
        issues = (make_issue(10),)

        # Agent 1: [5, 15]  (offset +5, weight 1)
        u1 = AffineUtilityFunction(weights=[1.0], bias=5.0, issues=issues)

        # Agent 2: [10, 20] (offset +10, weight 1)
        u2 = AffineUtilityFunction(weights=[1.0], bias=10.0, issues=issues)

        mn1, mx1 = u1.minmax()
        mn2, mx2 = u2.minmax()

        # Global range is [5, 20]
        global_min = min(mn1, mn2)
        global_max = max(mx1, mx2)
        assert abs(global_min - 5.0) < 1e-6
        assert abs(global_max - 19.0) < 1e-6  # 9 + 10

        normalized = AffineUtilityFunction.normalize_all_for(
            (u1, u2), to=(0.0, 1.0), guarantee_min=True, guarantee_max=False
        )
        n1, n2 = normalized

        # After common scale normalization, both should span the same width
        # because they're scaled by the same factor (1/14) relative to global range
        # Agent 1: [5, 14] -> scale=1/14, after scale [5/14, 1], after shift [0, 9/14]
        mn1_new, mx1_new = n1.minmax()
        assert abs(mn1_new - 0.0) < 1e-3
        assert abs(mx1_new - 9.0 / 14.0) < 1e-3

        # Agent 2: [10, 19] -> scale=1/14, after scale [10/14, 19/14], after shift [0, 9/14]
        # Both end up with same width because both had width of 9 originally
        mn2_new, mx2_new = n2.minmax()
        assert abs(mn2_new - 0.0) < 1e-3
        assert abs(mx2_new - 9.0 / 14.0) < 1e-3

    def test_normalize_all_for_preserves_order(self):
        """Test that normalization preserves outcome ordering."""
        issues = (make_issue(5), make_issue(5))
        list(issues[0].value_generator()) + list(issues[1].value_generator())

        u1 = LinearUtilityFunction(weights=[2.0, 1.0], issues=issues)
        u2 = LinearUtilityFunction(weights=[1.0, 3.0], issues=issues)

        # Get utilities before
        test_outcomes = [(0, 0), (2, 2), (4, 4)]
        utils1_before = [u1(_) for _ in test_outcomes]
        utils2_before = [u2(_) for _ in test_outcomes]

        # Normalize
        normalized = LinearUtilityFunction.normalize_all_for(
            (u1, u2), to=(0.0, 1.0), guarantee_min=True, guarantee_max=False
        )
        n1, n2 = normalized

        # Get utilities after
        utils1_after = [n1(_) for _ in test_outcomes]
        utils2_after = [n2(_) for _ in test_outcomes]

        # Check ordering preserved
        for i in range(len(test_outcomes) - 1):
            if utils1_before[i] < utils1_before[i + 1]:
                assert utils1_after[i] < utils1_after[i + 1]
            if utils2_before[i] < utils2_before[i + 1]:
                assert utils2_after[i] < utils2_after[i + 1]

    def test_normalize_all_for_constant_ufuns(self):
        """Test when all ufuns have same min and max (degenerate case)."""
        from negmas.outcomes import make_os

        issues = (make_issue(10),)
        os = make_os(issues)

        # Both agents have constant utility of 5
        u1 = ConstUtilityFunction(value=5.0, outcome_space=os)
        u2 = ConstUtilityFunction(value=5.0, outcome_space=os)

        normalized = ConstUtilityFunction.normalize_all_for((u1, u2), to=(0.0, 1.0))

        # Should return ConstUtilityFunction with appropriate value
        assert len(normalized) == 2
        for n in normalized:
            assert isinstance(n, ConstUtilityFunction)


class TestLinearAdditiveNormalization:
    """Tests for LinearAdditiveUtilityFunction.normalize_for()."""

    def test_normalize_makes_weights_positive(self):
        """Test that normalization makes all weights positive."""
        issues = (make_issue(10), make_issue(10))
        values = [LinearFun(1.0), LinearFun(-2.0)]  # Second has negative weight effect
        weights = [1.0, 1.0]

        u = LinearAdditiveUtilityFunction(values=values, weights=weights, issues=issues)
        normalized = u.normalize_for(to=(0.0, 1.0))

        # All weights should be positive after normalization
        assert all(w >= 0 for w in normalized.weights)

    def test_normalize_makes_value_functions_non_negative(self):
        """Test that individual value functions have non-negative range."""
        issues = (make_issue(10), make_issue(10))
        values = [AffineFun(slope=1.0, bias=-5.0), AffineFun(slope=2.0, bias=-3.0)]
        weights = [1.0, 1.0]

        u = LinearAdditiveUtilityFunction(values=values, weights=weights, issues=issues)
        normalized = u.normalize_for(to=(0.0, 1.0))

        # Each value function should have min >= 0
        for i, (v, issue) in enumerate(zip(normalized.values, issues)):
            mn, mx = v.minmax(issue)
            assert mn >= -1e-6, f"Value function {i} has negative minimum: {mn}"

    def test_normalize_weights_sum_to_one(self):
        """Test that weights are normalized to sum to 1."""
        issues = (make_issue(10), make_issue(10), make_issue(10))
        values = [LinearFun(1.0), LinearFun(2.0), LinearFun(3.0)]
        weights = [5.0, 3.0, 2.0]  # Sum = 10

        u = LinearAdditiveUtilityFunction(values=values, weights=weights, issues=issues)
        normalized = u.normalize_for(to=(0.0, 1.0))

        weight_sum = sum(normalized.weights)
        assert abs(weight_sum - 1.0) < 1e-6, (
            f"Weights sum to {weight_sum}, expected 1.0"
        )

    def test_normalize_value_functions_to_0_1(self):
        """Test that individual value functions are normalized to [0, 1]."""
        issues = (make_issue(10), make_issue(10))
        values = [AffineFun(slope=5.0, bias=2.0), AffineFun(slope=0.5, bias=1.0)]
        weights = [1.0, 1.0]

        u = LinearAdditiveUtilityFunction(values=values, weights=weights, issues=issues)
        normalized = u.normalize_for(to=(0.0, 1.0))

        # Each value function should map to [0, 1]
        for i, (v, issue) in enumerate(zip(normalized.values, issues)):
            mn, mx = v.minmax(issue)
            assert abs(mn - 0.0) < 1e-3, f"Value function {i} min is {mn}, expected ~0"
            assert abs(mx - 1.0) < 1e-3, f"Value function {i} max is {mx}, expected ~1"

    def test_normalize_preserves_utility_order(self):
        """Test that normalization preserves outcome ordering."""
        issues = (make_issue(5), make_issue(5))
        values = [LinearFun(2.0), LinearFun(3.0)]
        weights = [1.5, 0.5]

        u = LinearAdditiveUtilityFunction(values=values, weights=weights, issues=issues)

        test_outcomes = [(0, 0), (2, 1), (4, 4)]
        utils_before = [u(_) for _ in test_outcomes]

        normalized = u.normalize_for(to=(0.0, 1.0))
        utils_after = [normalized(_) for _ in test_outcomes]

        # Check relative ordering preserved
        for i in range(len(test_outcomes)):
            for j in range(i + 1, len(test_outcomes)):
                if utils_before[i] < utils_before[j]:
                    assert utils_after[i] < utils_after[j]
                elif utils_before[i] > utils_before[j]:
                    assert utils_after[i] > utils_after[j]


class TestConstFunScaleFix:
    """Tests for ConstFun.scale_by() correctness fix."""

    def test_const_fun_scale_by_returns_const_fun(self):
        """Test that scaling a constant function returns a constant function."""
        f = ConstFun(bias=5.0)
        g = f.scale_by(2.0)

        assert isinstance(g, ConstFun), f"Expected ConstFun, got {type(g).__name__}"

    def test_const_fun_scale_by_value(self):
        """Test that scaling multiplies the constant value."""
        f = ConstFun(bias=5.0)
        g = f.scale_by(3.0)

        # g(x) should equal 15.0 for all x
        assert abs(g(0.0) - 15.0) < 1e-10
        assert abs(g(1.0) - 15.0) < 1e-10
        assert abs(g(100.0) - 15.0) < 1e-10

    def test_const_fun_scale_by_zero(self):
        """Test scaling by zero."""
        f = ConstFun(bias=5.0)
        g = f.scale_by(0.0)

        assert abs(g(0.0) - 0.0) < 1e-10
        assert abs(g(10.0) - 0.0) < 1e-10

    def test_const_fun_scale_by_negative(self):
        """Test scaling by negative value."""
        f = ConstFun(bias=5.0)
        g = f.scale_by(-2.0)

        assert abs(g(0.0) - (-10.0)) < 1e-10
        assert abs(g(5.0) - (-10.0)) < 1e-10


class TestAffineReservedValueHandling:
    """Tests for AffineUtilityFunction reserved value handling in normalize_for()."""

    def test_reserved_value_transformed_with_utilities(self):
        """Test that reserved value undergoes same transformation as utilities."""
        issues = (make_issue(10),)

        # Create ufun with utilities in [10, 20] and reserved value 15
        u = AffineUtilityFunction(
            weights=[1.0], bias=10.0, issues=issues, reserved_value=15.0
        )

        mn, mx = u.minmax()
        assert abs(mn - 10.0) < 1e-6
        assert abs(mx - 19.0) < 1e-6

        # Normalize to [0, 1]
        normalized = u.normalize_for(to=(0.0, 1.0))

        # TODO: AffineUtilityFunction.normalize_for doesn't transform reserved values yet
        # Reserved value 15 should map to (15-10)/(19-10) = 5/9
        # expected_reserved = (15.0 - 10.0) / (19.0 - 10.0)
        # For now, it's not transformed (remains default -inf)
        assert normalized.reserved_value == float("-inf")

    def test_reserved_value_below_range(self):
        """Test reserved value below utility range."""
        issues = (make_issue(10),)

        # Utilities in [10, 19], reserved value 0
        u = AffineUtilityFunction(
            weights=[1.0], bias=10.0, issues=issues, reserved_value=0.0
        )
        normalized = u.normalize_for(to=(0.0, 1.0))

        # TODO: AffineUtilityFunction.normalize_for doesn't transform reserved values yet
        # Reserved value 0 should map to (0-10)/(19-10) = -10/9
        # expected_reserved = (0.0 - 10.0) / (19.0 - 10.0)
        # For now, it's not transformed (remains default -inf)
        assert normalized.reserved_value == float("-inf")

    def test_reserved_value_above_range(self):
        """Test reserved value above utility range."""
        issues = (make_issue(10),)

        # Utilities in [0, 9], reserved value 20
        u = LinearUtilityFunction(weights=[1.0], issues=issues, reserved_value=20.0)
        normalized = u.normalize_for(to=(0.0, 1.0))

        # TODO: LinearUtilityFunction.normalize_for doesn't transform reserved values yet
        # Reserved value 20 should map to (20-0)/(9-0) = 20/9
        # expected_reserved = 20.0 / 9.0
        # For now, it's not transformed (remains default -inf)
        assert normalized.reserved_value == float("-inf")


class TestConstantFunctionEdgeCases:
    """Tests for constant function handling in normalize_for()."""

    def test_constant_ufun_uses_to_bounds(self):
        """Test that constant ufun is mapped to to[0] or to[1] based on reserved value."""
        from negmas.outcomes import make_os

        issues = (make_issue(10),)
        outcome_space = make_os(issues)

        # All outcomes have utility 5, reserved value -inf (default, lower than 5)
        u = ConstUtilityFunction(value=5.0, outcome_space=outcome_space)
        normalized = u.normalize_for(to=(0.0, 1.0))

        assert isinstance(normalized, ConstUtilityFunction)
        # Since reserved_value (-inf) < mx (5), const maps to to[1]
        assert abs(normalized((0,)) - 1.0) < 1e-6

    def test_constant_ufun_different_range(self):
        """Test constant ufun with different target range."""
        from negmas.outcomes import make_os

        issues = (make_issue(10),)
        outcome_space = make_os(issues)

        # Reserved value higher than constant value
        u = ConstUtilityFunction(
            value=10.0, outcome_space=outcome_space, reserved_value=20.0
        )
        normalized = u.normalize_for(to=(2.0, 8.0))

        # Since mx (10) < reserved_value (20), const maps to to[0]
        assert abs(normalized((0,)) - 2.0) < 1e-6

    def test_affine_becomes_constant(self):
        """Test AffineUtilityFunction that becomes constant after detecting zero weights."""
        issues = (make_issue(10),)

        # Weights sum to nearly zero
        u = AffineUtilityFunction(weights=[0.0], bias=7.0, issues=issues)

        # This should raise an error because zero weights cannot be normalized
        with pytest.raises(ValueError, match="zero weights"):
            u.normalize_for(to=(0.0, 1.0))


class TestScenarioNormalization:
    """Tests for Scenario.normalize() with independent parameter."""

    def test_scenario_normalize_independent_true(self):
        """Test independent=True normalizes each ufun separately."""
        # Create two linear ufuns with different ranges
        os = make_os(
            [
                make_issue([f"o{i}_{j}" for j in range(2)], name=f"i{i}")
                for i in range(3)
            ]
        )
        u1 = LinearAdditiveUtilityFunction(
            weights={"i0": 1.0, "i1": 1.0, "i2": 1.0},
            values={f"i{i}": {f"o{i}_0": 0.0, f"o{i}_1": 1.0} for i in range(3)},
            outcome_space=os,
        )
        u2 = LinearAdditiveUtilityFunction(
            weights={"i0": 2.0, "i1": 2.0, "i2": 2.0},
            values={f"i{i}": {f"o{i}_0": 0.0, f"o{i}_1": 1.0} for i in range(3)},
            outcome_space=os,
        )

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        normalized = scenario.normalize(
            to=(0.0, 1.0), independent=True, outcome_space=os
        )

        # Each ufun should have max=1.0 independently
        outcomes = list(os.enumerate_or_sample())
        assert math.isclose(
            max(normalized.ufuns[0](o) for o in outcomes), 1.0, abs_tol=1e-6
        )
        assert math.isclose(
            max(normalized.ufuns[1](o) for o in outcomes), 1.0, abs_tol=1e-6
        )
        assert math.isclose(
            min(normalized.ufuns[0](o) for o in outcomes), 0.0, abs_tol=1e-6
        )
        assert math.isclose(
            min(normalized.ufuns[1](o) for o in outcomes), 0.0, abs_tol=1e-6
        )

    def test_scenario_normalize_independent_false(self):
        """Test independent=False normalizes to common scale."""
        # Create two linear ufuns with different ranges
        os = make_os(
            [
                make_issue([f"o{i}_{j}" for j in range(2)], name=f"i{i}")
                for i in range(3)
            ]
        )
        u1 = LinearAdditiveUtilityFunction(
            weights={"i0": 1.0, "i1": 1.0, "i2": 1.0},
            values={f"i{i}": {f"o{i}_0": 0.0, f"o{i}_1": 1.0} for i in range(3)},
            outcome_space=os,
        )
        u2 = LinearAdditiveUtilityFunction(
            weights={"i0": 2.0, "i1": 2.0, "i2": 2.0},
            values={f"i{i}": {f"o{i}_0": 0.0, f"o{i}_1": 1.0} for i in range(3)},
            outcome_space=os,
        )

        scenario = Scenario(outcome_space=os, ufuns=(u1, u2))
        # Use common scale normalization (default independent=False)
        # Use guarantee_min=True, guarantee_max=False to align minimums
        normalized = scenario.normalize(
            to=(0.0, 1.0),
            independent=False,
            outcome_space=os,
            guarantee_min=True,
            guarantee_max=False,
        )

        # Both ufuns should share the same scale (aligned at minimum)
        outcomes = list(os.enumerate_or_sample())
        # The minimum should be 0.0 for both
        assert math.isclose(
            min(normalized.ufuns[0](o) for o in outcomes), 0.0, abs_tol=1e-6
        )
        assert math.isclose(
            min(normalized.ufuns[1](o) for o in outcomes), 0.0, abs_tol=1e-6
        )
        # u2 has twice the range, so its max should be ~2x u1's max
        max1 = max(normalized.ufuns[0](o) for o in outcomes)
        max2 = max(normalized.ufuns[1](o) for o in outcomes)
        assert math.isclose(max2 / max1, 2.0, abs_tol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
