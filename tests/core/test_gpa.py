"""Tests for GPAUtilityFunction (Generalized Polynomial Aggregation)."""

from __future__ import annotations

import pytest
from pytest import approx

from negmas.outcomes import make_issue
from negmas.preferences.value_fun import AffineFun, AffineMultiFun
from negmas.preferences.crisp.gpa import GPAUtilityFunction


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestGPABasic:
    """Basic tests for GPAUtilityFunction."""

    def test_single_factor_single_term(self):
        """GPA with a single factor raised to power 1."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), AffineFun(slope=1.0, bias=0.0))]
        terms = [(1.0, (1,))]  # Just f0
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((0,)) == 0.0
        assert f((5,)) == 5.0
        assert f((9,)) == 9.0

    def test_single_factor_power(self):
        """GPA with a single factor raised to a power."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), AffineFun(slope=1.0, bias=0.0))]
        terms = [(1.0, (2,))]  # f0^2
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((2,)) == 4.0
        assert f((3,)) == 9.0
        assert f((5,)) == 25.0

    def test_multiple_factors_linear(self):
        """GPA with multiple factors, each in separate terms (like linear aggregation)."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        factors = [
            (("A",), AffineFun(slope=1.0, bias=0.0)),
            (("B",), AffineFun(slope=1.0, bias=0.0)),
        ]
        # u = 2*a + 3*b (like weighted linear)
        terms = [(2.0, (1, 0)), (3.0, (0, 1))]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((1, 1)) == 5.0  # 2*1 + 3*1
        assert f((2, 3)) == 13.0  # 2*2 + 3*3 = 4 + 9

    def test_polynomial_interaction(self):
        """GPA with polynomial interaction between factors."""
        issues = [make_issue(10, "A"), make_issue(10, "B"), make_issue(10, "C")]
        factors = [
            (("A",), AffineFun(slope=1.0, bias=0.0)),
            (("B",), AffineFun(slope=1.0, bias=0.0)),
            (("C",), AffineFun(slope=1.0, bias=0.0)),
        ]
        # u = a^3 + b^2*c + a*b*c^2
        terms = [
            (1.0, (3, 0, 0)),  # a^3
            (1.0, (0, 2, 1)),  # b^2 * c
            (1.0, (1, 1, 2)),  # a * b * c^2
        ]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        # For (2, 3, 4): a=2, b=3, c=4
        # u = 8 + 36 + 96 = 140
        assert f((2, 3, 4)) == 140.0

    def test_bias_term(self):
        """GPA with bias term."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), AffineFun(slope=1.0, bias=0.0))]
        terms = [(1.0, (1,))]
        f = GPAUtilityFunction(factors=factors, terms=terms, bias=10.0, issues=issues)

        assert f((5,)) == 15.0  # 5 + 10

    def test_none_outcome(self):
        """GPA returns reserved_value for None outcome."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), AffineFun(slope=1.0, bias=0.0))]
        terms = [(1.0, (1,))]
        f = GPAUtilityFunction(
            factors=factors, terms=terms, issues=issues, reserved_value=0.5
        )

        assert f(None) == 0.5


# =============================================================================
# Multi-Issue Factor Tests
# =============================================================================


class TestGPAMultiIssueFunctions:
    """Tests for GPA with multi-issue factor functions."""

    def test_affine_multi_factor(self):
        """GPA with AffineMultiFun factor."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        # Factor that combines both issues: f = A + 2*B
        factors = [((0, 1), AffineMultiFun(slope=(1.0, 2.0), bias=0.0))]
        terms = [(1.0, (2,))]  # f^2
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        # f(2, 3) -> factor = 2 + 6 = 8 -> 8^2 = 64
        assert f((2, 3)) == 64.0

    def test_mixed_single_and_multi_factors(self):
        """GPA with both single and multi-issue factors."""
        issues = [make_issue(10, "A"), make_issue(10, "B"), make_issue(10, "C")]
        factors = [
            (("A",), AffineFun(slope=1.0, bias=0.0)),  # f0 = A
            ((1, 2), AffineMultiFun(slope=(1.0, 1.0), bias=0.0)),  # f1 = B + C
        ]
        # u = f0 * f1 = A * (B + C)
        terms = [(1.0, (1, 1))]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        # f(2, 3, 4) -> f0=2, f1=7 -> 2*7 = 14
        assert f((2, 3, 4)) == 14.0


# =============================================================================
# Issue Indexing Tests
# =============================================================================


class TestGPAIssueIndexing:
    """Tests for issue indexing (by name and integer)."""

    def test_index_by_name(self):
        """GPA with issue indices as names."""
        issues = [make_issue(10, "price"), make_issue(10, "quantity")]
        factors = [
            (("price",), AffineFun(slope=1.0, bias=0.0)),
            (("quantity",), AffineFun(slope=1.0, bias=0.0)),
        ]
        terms = [(1.0, (1, 1))]  # price * quantity
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((5, 4)) == 20.0

    def test_index_by_integer(self):
        """GPA with issue indices as integers."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        factors = [
            ((0,), AffineFun(slope=1.0, bias=0.0)),
            ((1,), AffineFun(slope=1.0, bias=0.0)),
        ]
        terms = [(1.0, (1, 1))]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((5, 4)) == 20.0

    def test_invalid_issue_name_raises(self):
        """GPA raises ValueError for invalid issue name."""
        issues = [make_issue(10, "A")]
        factors = [(("B",), AffineFun(slope=1.0, bias=0.0))]  # B doesn't exist
        terms = [(1.0, (1,))]

        with pytest.raises(ValueError, match="Issue name 'B' not found"):
            GPAUtilityFunction(factors=factors, terms=terms, issues=issues)


# =============================================================================
# Auto-Conversion Tests
# =============================================================================


class TestGPAAutoConversion:
    """Tests for automatic conversion of factor function types."""

    def test_dict_to_table(self):
        """GPA converts dict to TableFun."""
        issues = [make_issue(["a", "b", "c"], "choice")]
        factors = [(("choice",), {"a": 1.0, "b": 2.0, "c": 3.0})]
        terms = [(1.0, (2,))]  # factor^2
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f(("a",)) == 1.0
        assert f(("b",)) == 4.0
        assert f(("c",)) == 9.0

    def test_lambda_to_lambdafun(self):
        """GPA converts callable to LambdaFun."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), lambda x: x**2)]
        terms = [(1.0, (1,))]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((3,)) == 9.0  # 3^2

    def test_float_to_const(self):
        """GPA converts float to constant function."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), 5.0)]  # Constant factor
        terms = [(1.0, (1,))]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((0,)) == 5.0
        assert f((9,)) == 5.0  # Always 5


# =============================================================================
# Validation Tests
# =============================================================================


class TestGPAValidation:
    """Tests for validation of inputs."""

    def test_mismatched_powers_raises(self):
        """GPA raises ValueError if powers length doesn't match factors."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        factors = [
            (("A",), AffineFun(slope=1.0, bias=0.0)),
            (("B",), AffineFun(slope=1.0, bias=0.0)),
        ]
        terms = [(1.0, (1,))]  # Only 1 power, but 2 factors

        with pytest.raises(ValueError, match="powers length"):
            GPAUtilityFunction(factors=factors, terms=terms, issues=issues)


# =============================================================================
# Serialization Tests
# =============================================================================


class TestGPASerialization:
    """Tests for serialization/deserialization."""

    def test_to_dict_from_dict(self):
        """GPA can be serialized and deserialized."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        factors = [
            (("A",), AffineFun(slope=2.0, bias=1.0)),
            (("B",), AffineFun(slope=3.0, bias=0.0)),
        ]
        terms = [(1.0, (2, 0)), (1.0, (1, 1))]
        f = GPAUtilityFunction(factors=factors, terms=terms, bias=5.0, issues=issues)

        d = f.to_dict()
        f2 = GPAUtilityFunction.from_dict(d)

        # Test evaluation is preserved
        for outcome in [(0, 0), (1, 2), (5, 5), (9, 9)]:
            assert f(outcome) == approx(f2(outcome))


# =============================================================================
# Method Tests
# =============================================================================


class TestGPAMethods:
    """Tests for GPA methods."""

    def test_shift_by(self):
        """GPA shift_by adds to bias."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), AffineFun(slope=1.0, bias=0.0))]
        terms = [(1.0, (1,))]
        f = GPAUtilityFunction(factors=factors, terms=terms, bias=0.0, issues=issues)

        f2 = f.shift_by(10.0)
        assert f2((5,)) == f((5,)) + 10.0

    def test_scale_by(self):
        """GPA scale_by multiplies terms and bias."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), AffineFun(slope=1.0, bias=0.0))]
        terms = [(1.0, (1,))]
        f = GPAUtilityFunction(factors=factors, terms=terms, bias=5.0, issues=issues)

        f2 = f.scale_by(2.0)
        assert f2((5,)) == f((5,)) * 2.0

    def test_properties(self):
        """GPA properties work correctly."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        factors = [
            (("A",), AffineFun(slope=1.0, bias=0.0)),
            (("B",), AffineFun(slope=1.0, bias=0.0)),
        ]
        terms = [(2.0, (1, 0)), (3.0, (0, 1))]
        f = GPAUtilityFunction(factors=factors, terms=terms, bias=1.0, issues=issues)

        assert len(f.factors) == 2
        assert len(f.terms) == 2
        assert f.bias == 1.0


# =============================================================================
# Random Generation Tests
# =============================================================================


class TestGPARandom:
    """Tests for random GPA generation."""

    def test_random_generation(self):
        """GPA can be randomly generated."""
        issues = [make_issue(10, "A"), make_issue(10, "B"), make_issue(10, "C")]
        f = GPAUtilityFunction.random(issues=issues)

        # Should be callable
        result = f((1, 2, 3))
        assert isinstance(result, float)

    def test_random_with_params(self):
        """GPA random respects parameters."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        f = GPAUtilityFunction.random(
            issues=issues, n_factors=3, n_terms=4, max_power=2
        )

        assert len(f.factors) == 3
        assert len(f.terms) == 4


# =============================================================================
# Edge Cases
# =============================================================================


class TestGPAEdgeCases:
    """Edge case tests for GPA."""

    def test_zero_power(self):
        """Terms with power 0 treat factor as 1."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        factors = [
            (("A",), AffineFun(slope=1.0, bias=0.0)),
            (("B",), AffineFun(slope=1.0, bias=0.0)),
        ]
        # u = 5 * a^0 * b^2 = 5 * 1 * b^2 = 5 * b^2
        terms = [(5.0, (0, 2))]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((999, 3)) == 45.0  # A value doesn't matter, only B: 5*9

    def test_multiple_terms_same_factor(self):
        """Multiple terms can use the same factor with different powers."""
        issues = [make_issue(10, "A")]
        factors = [(("A",), AffineFun(slope=1.0, bias=0.0))]
        # u = x + x^2 + x^3
        terms = [(1.0, (1,)), (1.0, (2,)), (1.0, (3,))]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        # f(2) = 2 + 4 + 8 = 14
        assert f((2,)) == 14.0

    def test_negative_coefficients(self):
        """Terms can have negative coefficients."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        factors = [
            (("A",), AffineFun(slope=1.0, bias=0.0)),
            (("B",), AffineFun(slope=1.0, bias=0.0)),
        ]
        # u = a - b (difference)
        terms = [(1.0, (1, 0)), (-1.0, (0, 1))]
        f = GPAUtilityFunction(factors=factors, terms=terms, issues=issues)

        assert f((5, 3)) == 2.0  # 5 - 3
        assert f((3, 5)) == -2.0  # 3 - 5
