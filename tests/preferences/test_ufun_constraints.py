"""Tests for utility function constraint functionality."""

from __future__ import annotations

import math

import pytest

from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction, MappingUtilityFunction


class TestUtilityFunctionConstraints:
    """Tests for utility function constraint functionality."""

    def test_construction_with_constraints(self):
        """Test that constraints can be passed during construction."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        assert len(ufun.constraints) == 1
        assert ufun.constraints[0] is constraint

    def test_construction_without_constraints(self):
        """Test that utility functions work without constraints."""
        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        )

        assert len(ufun.constraints) == 0
        # Should work normally
        assert isinstance(ufun((1, 0)), float)

    def test_constraint_returns_negative_infinity(self):
        """Test that constrained outcomes return -inf."""

        def constraint(outcome):
            return outcome[0] != 1

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        # Valid outcomes should return normal utility
        u_valid = ufun((0, 0))
        assert isinstance(u_valid, float)
        assert not math.isinf(u_valid)

        # Invalid outcomes should return -inf
        u_invalid = ufun((1, 0))
        assert u_invalid == float("-inf")
        assert math.isinf(u_invalid) and u_invalid < 0

    def test_add_constraint(self):
        """Test adding constraints after construction."""
        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        )

        def constraint(outcome):
            return outcome[0] > 0

        ufun.add_constraint(constraint)

        assert len(ufun.constraints) == 1
        assert constraint in ufun.constraints

        # Should now return -inf for x=0
        assert ufun((0, 0)) == float("-inf")

    def test_remove_constraint(self):
        """Test removing constraints."""

        def constraint1(outcome):
            return outcome[0] > 0

        def constraint2(outcome):
            return outcome[1] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint1, constraint2],
        )

        assert len(ufun.constraints) == 2

        ufun.remove_constraint(constraint1)

        assert len(ufun.constraints) == 1
        assert constraint1 not in ufun.constraints
        assert constraint2 in ufun.constraints

        # Now only constraint2 should apply
        assert ufun((0, 1)) != float("-inf")  # constraint1 removed
        assert ufun((1, 0)) == float("-inf")  # constraint2 still applies

    def test_remove_nonexistent_constraint(self):
        """Test that removing a non-existent constraint doesn't raise an error."""

        def constraint(outcome):
            return True

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        )

        # Should not raise an error
        ufun.remove_constraint(constraint)

    def test_clear_constraints(self):
        """Test clearing all constraints."""

        def constraint1(outcome):
            return outcome[0] > 0

        def constraint2(outcome):
            return outcome[1] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint1, constraint2],
        )

        assert len(ufun.constraints) == 2

        # Both should return -inf
        assert ufun((0, 0)) == float("-inf")

        ufun.clear_constraints()

        assert len(ufun.constraints) == 0

        # Now should return normal utility
        u = ufun((0, 0))
        assert not math.isinf(u)

    def test_satisfies_constraints(self):
        """Test the satisfies_constraints method."""

        def constraint(outcome):
            return outcome[0] + outcome[1] <= 2

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[constraint],
        )

        assert ufun.satisfies_constraints((0, 0))
        assert ufun.satisfies_constraints((1, 1))
        assert ufun.satisfies_constraints((2, 0))
        assert not ufun.satisfies_constraints((2, 2))
        assert not ufun.satisfies_constraints((1, 2))

    def test_multiple_constraints(self):
        """Test multiple constraints work together (AND logic)."""

        def constraint1(outcome):
            return outcome[0] > 0

        def constraint2(outcome):
            return outcome[1] < 2

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[constraint1, constraint2],
        )

        # Only outcomes with x>0 AND y<2 should be valid
        assert ufun((1, 0)) != float("-inf")
        assert ufun((2, 1)) != float("-inf")
        assert ufun((0, 1)) == float("-inf")  # fails constraint1
        assert ufun((1, 2)) == float("-inf")  # fails constraint2
        assert ufun((0, 2)) == float("-inf")  # fails both

    def test_constraint_with_none_outcome(self):
        """Test that None outcome returns reserved value."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
            reserved_value=0.3,
        )

        # None should return reserved value, not check constraints
        assert ufun(None) == 0.3

    def test_mapping_utility_function_with_constraints(self):
        """Test constraints work with MappingUtilityFunction."""

        def constraint(outcome):
            return outcome[0] != 1

        mapping = {
            (0, 0): 0.1,
            (0, 1): 0.2,
            (1, 0): 0.5,
            (1, 1): 0.7,
            (2, 0): 0.3,
            (2, 1): 0.6,
        }

        ufun = MappingUtilityFunction(mapping=mapping, constraints=[constraint])

        # Valid outcomes should return mapped utility
        assert ufun((0, 0)) == 0.1
        assert ufun((2, 1)) == 0.6

        # Invalid outcomes should return -inf
        assert ufun((1, 0)) == float("-inf")
        assert ufun((1, 1)) == float("-inf")

    def test_cache_cleared_on_constraint_change(self):
        """Test that caches are cleared when constraints change."""
        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        )

        # Trigger some caching by calling extreme_outcomes
        _ = ufun.extreme_outcomes()
        assert ufun._cached_extreme_outcomes is not None

        def constraint(outcome):
            return outcome[0] > 0

        # Adding constraint should clear caches
        ufun.add_constraint(constraint)
        assert ufun._cached_extreme_outcomes is None

        # Trigger caching again
        _ = ufun.extreme_outcomes()
        assert ufun._cached_extreme_outcomes is not None

        # Removing constraint should clear caches
        ufun.remove_constraint(constraint)
        assert ufun._cached_extreme_outcomes is None

        # Trigger caching again
        _ = ufun.extreme_outcomes()
        assert ufun._cached_extreme_outcomes is not None

        # Clearing constraints should clear caches
        ufun.clear_constraints()
        assert ufun._cached_extreme_outcomes is None


class TestConstraintEdgeCases:
    """Tests for edge cases and error handling."""

    def test_constraint_returns_non_boolean(self):
        """Test that constraints can return truthy/falsy values."""

        def constraint(outcome):
            return outcome[0]  # Returns 0 (falsy) or positive (truthy)

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        # x=0 should return -inf (falsy)
        assert ufun((0, 0)) == float("-inf")

        # x>0 should return normal utility (truthy)
        assert ufun((1, 0)) != float("-inf")

    def test_constraint_raises_exception(self):
        """Test behavior when constraint raises an exception."""

        def bad_constraint(outcome):
            raise ValueError("Test error")

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[bad_constraint],
        )

        # Should propagate the exception
        with pytest.raises(ValueError, match="Test error"):
            ufun((1, 0))

    def test_constraint_with_invalid_outcome_space(self):
        """Test constraint interaction with invalid outcomes."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
            invalid_value=float("-inf"),
        )

        # Outcome not in outcome space should return invalid_value
        # before checking constraints
        invalid_outcome = (5, 0)  # x=5 not in [0,1,2]
        assert ufun(invalid_outcome) == float("-inf")

    def test_all_outcomes_constrained(self):
        """Test behavior when all outcomes are constrained."""

        def impossible_constraint(outcome):
            return False  # Nothing satisfies this

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1], "x"), make_issue([0, 1], "y")],
            constraints=[impossible_constraint],
        )

        # All outcomes should return -inf
        for x in [0, 1]:
            for y in [0, 1]:
                assert ufun((x, y)) == float("-inf")

    def test_constraint_with_different_outcome_types(self):
        """Test constraints work with different outcome representations."""

        def constraint(outcome):
            # Should work whether outcome is tuple or dict
            if isinstance(outcome, dict):
                return outcome["x"] > 0
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        # Tuple outcome
        assert ufun((0, 0)) == float("-inf")
        assert ufun((1, 0)) != float("-inf")


class TestConstraintIntegration:
    """Integration tests with other features."""

    def test_constraints_with_extreme_outcomes(self):
        """Test that extreme_outcomes respects constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        worst, best = ufun.extreme_outcomes()

        # Both worst and best should satisfy constraint
        assert worst[0] > 0
        assert best[0] > 0

    def test_constraints_with_minmax(self):
        """Test that minmax values respect constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        min_val, max_val = ufun.minmax()

        # Min should not be -inf (which would be for constrained outcomes)
        assert not math.isinf(min_val)
        assert min_val >= 0.0
        assert max_val <= 1.0

    def test_constraints_with_reserved_value(self):
        """Test interaction between constraints and reserved value."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
            reserved_value=0.5,
        )

        # Constrained outcome should return -inf, not reserved value
        assert ufun((0, 0)) == float("-inf")

        # None should return reserved value
        assert ufun(None) == 0.5

    def test_constraints_preserved_in_operations(self):
        """Test that constraints are preserved through utility function operations."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        # Normalize should preserve constraints
        normalized = ufun.normalize()
        assert len(normalized.constraints) == len(ufun.constraints)
        assert normalized((0, 0)) == float("-inf")

    def test_constraints_with_outcome_space_filtering(self):
        """Test that utility function and outcome space constraints work together."""
        from negmas.outcomes import DiscreteCartesianOutcomeSpace

        def os_constraint(outcome):
            return outcome[0] != 1

        def ufun_constraint(outcome):
            return outcome[1] != 1

        os = DiscreteCartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[os_constraint],
        )

        ufun = LinearAdditiveUtilityFunction.random(
            outcome_space=os, constraints=[ufun_constraint]
        )

        # Outcomes should be filtered by outcome space
        outcomes = list(os.enumerate())
        assert all(o[0] != 1 for o in outcomes)

        # And ufun should return -inf for y=1
        for outcome in outcomes:
            if outcome[1] == 1:
                assert ufun(outcome) == float("-inf")
            else:
                assert ufun(outcome) != float("-inf")
