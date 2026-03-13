"""Tests for outcome space constraint functionality."""

from __future__ import annotations

import pytest

from negmas.outcomes import (
    CartesianOutcomeSpace,
    DiscreteCartesianOutcomeSpace,
    EnumeratingOutcomeSpace,
    SingletonOutcomeSpace,
    make_issue,
)


class TestCartesianOutcomeSpaceConstraints:
    """Tests for CartesianOutcomeSpace constraints."""

    def test_construction_with_constraints(self):
        """Test that constraints can be passed during construction."""

        def constraint(outcome):
            return outcome[0] != 1

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        assert len(os.constraints) == 1
        assert os.constraints[0] is constraint

    def test_construction_without_constraints(self):
        """Test that outcome spaces work without constraints."""
        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        )

        assert len(os.constraints) == 0
        assert os.cardinality == 6

    def test_add_constraint(self):
        """Test adding constraints after construction."""
        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        )

        def constraint(outcome):
            return outcome[0] > 0

        os.add_constraint(constraint)

        assert len(os.constraints) == 1
        assert constraint in os.constraints

    def test_remove_constraint(self):
        """Test removing constraints."""

        def constraint1(outcome):
            return outcome[0] > 0

        def constraint2(outcome):
            return outcome[1] > 0

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint1, constraint2],
        )

        assert len(os.constraints) == 2

        os.remove_constraint(constraint1)

        assert len(os.constraints) == 1
        assert constraint1 not in os.constraints
        assert constraint2 in os.constraints

    def test_remove_nonexistent_constraint(self):
        """Test that removing a non-existent constraint doesn't raise an error."""

        def constraint(outcome):
            return True

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        )

        # Should not raise an error
        os.remove_constraint(constraint)

    def test_clear_constraints(self):
        """Test clearing all constraints."""

        def constraint1(outcome):
            return outcome[0] > 0

        def constraint2(outcome):
            return outcome[1] > 0

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint1, constraint2],
        )

        assert len(os.constraints) == 2

        os.clear_constraints()

        assert len(os.constraints) == 0

    def test_satisfies_constraints(self):
        """Test the satisfies_constraints method."""

        def constraint(outcome):
            return outcome[0] + outcome[1] <= 2

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[constraint],
        )

        assert os.satisfies_constraints((0, 0))
        assert os.satisfies_constraints((1, 1))
        assert os.satisfies_constraints((2, 0))
        assert not os.satisfies_constraints((2, 2))
        assert not os.satisfies_constraints((1, 2))

    def test_sample_with_constraints(self):
        """Test that sample respects constraints."""

        def constraint(outcome):
            return outcome[0] != 1

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        samples = list(os.sample(10, with_replacement=True))

        # All samples should satisfy the constraint
        assert all(s[0] != 1 for s in samples)

    def test_random_outcome_with_constraints(self):
        """Test that random_outcome respects constraints."""

        def constraint(outcome):
            return outcome[0] != 1

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        # Generate multiple random outcomes
        for _ in range(20):
            outcome = os.random_outcome()
            assert outcome[0] != 1

    def test_random_outcome_with_impossible_constraints(self):
        """Test that random_outcome raises error when no valid outcome exists."""

        def impossible_constraint(outcome):
            return False  # Nothing satisfies this

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1], "x"), make_issue([0, 1], "y")],
            constraints=[impossible_constraint],
        )

        with pytest.raises(ValueError, match="Could not generate a random outcome"):
            os.random_outcome()

    def test_multiple_constraints(self):
        """Test multiple constraints work together (AND logic)."""

        def constraint1(outcome):
            return outcome[0] > 0

        def constraint2(outcome):
            return outcome[1] < 2

        os = CartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[constraint1, constraint2],
        )

        # Only outcomes with x>0 AND y<2 should satisfy
        assert os.satisfies_constraints((1, 0))
        assert os.satisfies_constraints((2, 1))
        assert not os.satisfies_constraints((0, 1))  # fails constraint1
        assert not os.satisfies_constraints((1, 2))  # fails constraint2


class TestDiscreteCartesianOutcomeSpaceConstraints:
    """Tests for DiscreteCartesianOutcomeSpace constraints."""

    def test_enumerate_with_constraints(self):
        """Test that enumerate filters by constraints."""

        def constraint(outcome):
            return outcome[0] + outcome[1] <= 2

        os = DiscreteCartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[constraint],
        )

        outcomes = list(os.enumerate())

        # Should have 6 outcomes: (0,0), (0,1), (0,2), (1,0), (1,1), (2,0)
        assert len(outcomes) == 6
        assert all(o[0] + o[1] <= 2 for o in outcomes)

    def test_enumerate_without_constraints(self):
        """Test that enumerate works normally without constraints."""
        os = DiscreteCartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")]
        )

        outcomes = list(os.enumerate())

        # Should have all 9 outcomes
        assert len(outcomes) == 9

    def test_cardinality_unaffected_by_constraints(self):
        """Test that cardinality property doesn't reflect constraints for CartesianOutcomeSpace.

        Note: Cardinality in CartesianOutcomeSpace returns the theoretical maximum,
        not the actual count after filtering. Use len(list(enumerate())) for filtered count.
        """

        def constraint(outcome):
            return outcome[0] > 0

        os = DiscreteCartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        # Cardinality returns theoretical maximum (3 * 2 = 6)
        assert os.cardinality == 6

        # Actual filtered count is different
        actual_count = len(list(os.enumerate()))
        assert actual_count == 4  # Only outcomes with x>0: (1,0), (1,1), (2,0), (2,1)


class TestEnumeratingOutcomeSpaceConstraints:
    """Tests for EnumeratingOutcomeSpace constraints."""

    def test_construction_with_constraints(self):
        """Test that constraints can be passed during construction."""

        def constraint(outcome):
            return sum(outcome) < 5

        eos = EnumeratingOutcomeSpace(
            baseset={(1, 2), (3, 4), (5, 6)}, constraints=[constraint]
        )

        assert len(eos.constraints) == 1

    def test_enumerate_with_constraints(self):
        """Test that enumerate filters by constraints."""

        def constraint(outcome):
            return sum(outcome) < 10

        eos = EnumeratingOutcomeSpace(
            baseset={(1, 2), (3, 4), (5, 6), (10, 5)}, constraints=[constraint]
        )

        outcomes = list(eos.enumerate())

        # Should filter out (10, 5) and (5, 6)
        assert len(outcomes) == 2
        assert (1, 2) in outcomes
        assert (3, 4) in outcomes
        assert all(sum(o) < 10 for o in outcomes)

    def test_sample_with_constraints(self):
        """Test that sample respects constraints."""

        def constraint(outcome):
            return outcome[0] < 3

        eos = EnumeratingOutcomeSpace(
            baseset={(1, 0), (2, 0), (3, 0), (4, 0)}, constraints=[constraint]
        )

        samples = list(eos.sample(2, with_replacement=False))

        assert len(samples) == 2
        assert all(s[0] < 3 for s in samples)

    def test_cardinality_with_constraints(self):
        """Test that cardinality reflects constraints for EnumeratingOutcomeSpace."""

        def constraint(outcome):
            return outcome[0] > 1

        eos = EnumeratingOutcomeSpace(
            baseset={(0, 0), (1, 0), (2, 0), (3, 0)}, constraints=[constraint]
        )

        # Should only count outcomes satisfying constraint
        assert eos.cardinality == 2  # (2, 0) and (3, 0)

    def test_cardinality_without_constraints(self):
        """Test that cardinality is fast without constraints."""
        eos = EnumeratingOutcomeSpace(baseset={(0, 0), (1, 0), (2, 0), (3, 0)})

        # Should return length of baseset quickly
        assert eos.cardinality == 4

    def test_is_valid_checks_constraints(self):
        """Test that is_valid checks constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        eos = EnumeratingOutcomeSpace(
            baseset={(0, 0), (1, 0), (2, 0)}, constraints=[constraint]
        )

        assert not eos.is_valid((0, 0))  # In baseset but fails constraint
        assert eos.is_valid((1, 0))  # In baseset and passes constraint
        assert not eos.is_valid((5, 0))  # Not in baseset


class TestSingletonOutcomeSpaceConstraints:
    """Tests for SingletonOutcomeSpace constraints (inherits from DiscreteCartesianOutcomeSpace)."""

    def test_singleton_inherits_constraints(self):
        """Test that SingletonOutcomeSpace inherits constraint functionality."""

        def constraint(outcome):
            return outcome[0] > 0

        sos = SingletonOutcomeSpace(outcome=(1, 2), constraints=[constraint])

        assert len(sos.constraints) == 1
        assert sos.satisfies_constraints((1, 2))


class TestConstraintEdgeCases:
    """Tests for edge cases and error handling."""

    def test_constraint_with_empty_outcome_space(self):
        """Test constraints with empty outcome spaces."""

        def constraint(outcome):
            return outcome[0] > 5

        os = DiscreteCartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        outcomes = list(os.enumerate())
        assert len(outcomes) == 0  # No outcomes satisfy constraint

    def test_constraint_returns_non_boolean(self):
        """Test that constraints can return truthy/falsy values."""

        def constraint(outcome):
            return outcome[0]  # Returns 0 (falsy) or positive (truthy)

        os = DiscreteCartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        outcomes = list(os.enumerate())

        # Should filter out outcomes where x=0
        assert all(o[0] != 0 for o in outcomes)
        assert len(outcomes) == 4  # (1,0), (1,1), (2,0), (2,1)

    def test_constraint_raises_exception(self):
        """Test behavior when constraint raises an exception."""

        def bad_constraint(outcome):
            raise ValueError("Test error")

        os = DiscreteCartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[bad_constraint],
        )

        # Should propagate the exception
        with pytest.raises(ValueError, match="Test error"):
            list(os.enumerate())

    def test_constraints_with_tuple_conversion(self):
        """Test that constraints work with different outcome representations."""

        def constraint(outcome):
            return outcome[0] + outcome[1] > 2

        os = DiscreteCartesianOutcomeSpace(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[constraint],
        )

        outcomes = list(os.enumerate())

        # All outcomes should satisfy the constraint
        assert all(o[0] + o[1] > 2 for o in outcomes)
