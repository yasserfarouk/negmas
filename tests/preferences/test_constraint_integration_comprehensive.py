"""Comprehensive integration tests for constraint support across all utility function operations."""

import math
from negmas.outcomes import make_issue
from negmas.preferences import (
    LinearAdditiveUtilityFunction,
    AffineUtilityFunction,
    PAUtilityFunction,
    GPAUtilityFunction,
    GLAUtilityFunction,
)
from negmas.preferences.value_fun import AffineFun


class TestConstraintIntegrationComprehensive:
    """Test that all utility function operations properly handle constraints."""

    def test_minmax_respects_constraints(self):
        """Test that minmax() respects constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        min_val, max_val = ufun.minmax()

        # Min should not be -inf (which would be for constrained outcomes)
        assert not math.isinf(min_val), (
            "minmax returned -inf, constraints not respected"
        )
        assert not math.isinf(max_val), "minmax returned inf unexpectedly"

        # Verify that outcomes producing min/max values satisfy constraints
        worst, best = ufun.extreme_outcomes()
        assert constraint(worst), f"Worst outcome {worst} violates constraint"
        assert constraint(best), f"Best outcome {best} violates constraint"

    def test_min_respects_constraints(self):
        """Test that min() respects constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        min_val = ufun.min()

        # Min should not be -inf
        assert not math.isinf(min_val), "min() returned -inf, constraints not respected"

    def test_max_respects_constraints(self):
        """Test that max() respects constraints."""

        def constraint(outcome):
            return outcome[0] < 2  # Exclude best outcome

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        max_val = ufun.max()

        # Max should be finite and correspond to a valid outcome
        assert not math.isinf(max_val), "max() returned inf unexpectedly"

        # Best outcome should satisfy constraint
        best = ufun.best()
        assert constraint(best), f"Best outcome {best} violates constraint"

    def test_extreme_outcomes_respects_constraints(self):
        """Test that extreme_outcomes() respects constraints."""

        def constraint(outcome):
            return outcome[0] > 0 and outcome[1] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[constraint],
        )

        worst, best = ufun.extreme_outcomes()

        assert constraint(worst), f"Worst outcome {worst} violates constraint"
        assert constraint(best), f"Best outcome {best} violates constraint"

    def test_normalize_preserves_constraints(self):
        """Test that normalize() preserves constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        normalized = ufun.normalize()

        # Constraints should be preserved
        assert len(normalized.constraints) == 1
        assert constraint in normalized.constraints

        # Normalized ufun should respect constraints
        worst, best = normalized.extreme_outcomes()
        assert constraint(worst), (
            f"Normalized worst outcome {worst} violates constraint"
        )
        assert constraint(best), f"Normalized best outcome {best} violates constraint"

    def test_normalize_for_preserves_constraints(self):
        """Test that normalize_for() preserves constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        normalized = ufun.normalize_for(to=(0.0, 10.0))

        # Constraints should be preserved
        assert len(normalized.constraints) == 1

        # Normalized ufun should respect constraints
        worst, best = normalized.extreme_outcomes()
        assert constraint(worst), (
            f"Normalized worst outcome {worst} violates constraint"
        )
        assert constraint(best), f"Normalized best outcome {best} violates constraint"

    def test_scale_by_preserves_constraints(self):
        """Test that scale_by() preserves constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        scaled = ufun.scale_by(2.0)

        # Constraints should be preserved
        assert len(scaled.constraints) == 1

        # Scaled ufun should respect constraints
        worst, best = scaled.extreme_outcomes()
        assert constraint(worst), f"Scaled worst outcome {worst} violates constraint"
        assert constraint(best), f"Scaled best outcome {best} violates constraint"

    def test_shift_by_preserves_constraints(self):
        """Test that shift_by() preserves constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")],
            constraints=[constraint],
        )

        shifted = ufun.shift_by(5.0)

        # Constraints should be preserved
        assert len(shifted.constraints) == 1

        # Shifted ufun should respect constraints
        worst, best = shifted.extreme_outcomes()
        assert constraint(worst), f"Shifted worst outcome {worst} violates constraint"
        assert constraint(best), f"Shifted best outcome {best} violates constraint"

    def test_affine_utility_function_operations_preserve_constraints(self):
        """Test that AffineUtilityFunction operations preserve constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        issues = [make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        ufun = AffineUtilityFunction(
            weights=[1.0, 0.5], bias=0.0, issues=issues, constraints=[constraint]
        )

        # Test normalize
        normalized = ufun.normalize()
        assert len(normalized.constraints) == 1
        worst, best = normalized.extreme_outcomes()
        assert constraint(worst)
        assert constraint(best)

        # Test scale_by
        scaled = ufun.scale_by(2.0)
        assert len(scaled.constraints) == 1
        worst, best = scaled.extreme_outcomes()
        assert constraint(worst)
        assert constraint(best)

        # Test shift_by
        shifted = ufun.shift_by(5.0)
        assert len(shifted.constraints) == 1
        worst, best = shifted.extreme_outcomes()
        assert constraint(worst)
        assert constraint(best)

    def test_pa_utility_function_operations_preserve_constraints(self):
        """Test that PAUtilityFunction operations preserve constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        issues = [make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        ufun = PAUtilityFunction(
            values=[AffineFun(1.0, 0.0), AffineFun(0.5, 0.0)],
            terms=[(1.0, (1, 0)), (0.5, (0, 1))],
            bias=0.0,
            issues=issues,
            constraints=[constraint],
        )

        # Test scale_by
        scaled = ufun.scale_by(2.0)
        assert len(scaled.constraints) == 1

        # Test shift_by
        shifted = ufun.shift_by(5.0)
        assert len(shifted.constraints) == 1

    def test_gpa_utility_function_operations_preserve_constraints(self):
        """Test that GPAUtilityFunction operations preserve constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        issues = [make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        ufun = GPAUtilityFunction(
            factors=[((0, 1), AffineFun(1.0, 0.0))],
            terms=[(1.0, (1,))],
            bias=0.0,
            issues=issues,
            constraints=[constraint],
        )

        # Test scale_by
        scaled = ufun.scale_by(2.0)
        assert len(scaled.constraints) == 1

        # Test shift_by
        shifted = ufun.shift_by(5.0)
        assert len(shifted.constraints) == 1

    def test_gla_utility_function_operations_preserve_constraints(self):
        """Test that GLAUtilityFunction operations preserve constraints."""

        def constraint(outcome):
            return outcome[0] > 0

        issues = [make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        ufun = GLAUtilityFunction(
            factors=[
                ((0,), AffineFun(1.0, 0.0), 1.0),
                ((1,), AffineFun(0.5, 0.0), 0.5),
            ],
            bias=0.0,
            issues=issues,
            constraints=[constraint],
        )

        # Test scale_by
        scaled = ufun.scale_by(2.0)
        assert len(scaled.constraints) == 1

        # Test shift_by
        shifted = ufun.shift_by(5.0)
        assert len(shifted.constraints) == 1

    def test_constraints_affect_normalization_range(self):
        """Test that constraints properly affect the normalization range."""

        def constraint(outcome):
            # Exclude the extremes
            return 0 < outcome[0] < 2

        issues = [make_issue([0, 1, 2], "x"), make_issue([0, 1], "y")]
        ufun = LinearAdditiveUtilityFunction(
            values=[AffineFun(1.0, 0.0), AffineFun(0.5, 0.0)],
            weights=[1.0, 1.0],
            issues=issues,
            constraints=[constraint],
        )

        # Get min/max with constraints
        min_constrained, max_constrained = ufun.minmax()

        # Normalize
        normalized = ufun.normalize(to=(0.0, 1.0))

        # Check that normalized values are in expected range
        norm_min, norm_max = normalized.minmax()

        # Should be close to 0 and 1
        assert abs(norm_min - 0.0) < 0.01, (
            f"Normalized min {norm_min} should be close to 0"
        )
        assert abs(norm_max - 1.0) < 0.01, (
            f"Normalized max {norm_max} should be close to 1"
        )

    def test_multiple_constraints_all_operations(self):
        """Test that multiple constraints work across all operations."""

        def constraint1(outcome):
            return outcome[0] > 0

        def constraint2(outcome):
            return outcome[1] > 0

        ufun = LinearAdditiveUtilityFunction.random(
            issues=[make_issue([0, 1, 2], "x"), make_issue([0, 1, 2], "y")],
            constraints=[constraint1, constraint2],
        )

        # Test extreme_outcomes
        worst, best = ufun.extreme_outcomes()
        assert constraint1(worst) and constraint2(worst)
        assert constraint1(best) and constraint2(best)

        # Test minmax
        min_val, max_val = ufun.minmax()
        assert not math.isinf(min_val)
        assert not math.isinf(max_val)

        # Test normalize
        normalized = ufun.normalize()
        assert len(normalized.constraints) == 2
        worst, best = normalized.extreme_outcomes()
        assert constraint1(worst) and constraint2(worst)
        assert constraint1(best) and constraint2(best)

        # Test scale_by
        scaled = ufun.scale_by(2.0)
        assert len(scaled.constraints) == 2
        worst, best = scaled.extreme_outcomes()
        assert constraint1(worst) and constraint2(worst)
        assert constraint1(best) and constraint2(best)

        # Test shift_by
        shifted = ufun.shift_by(5.0)
        assert len(shifted.constraints) == 2
        worst, best = shifted.extreme_outcomes()
        assert constraint1(worst) and constraint2(worst)
        assert constraint1(best) and constraint2(best)
