"""Tests for PAUtilityFunction (Polynomial Aggregation)."""

from __future__ import annotations

import pytest
from pytest import approx

from negmas.outcomes import make_issue
from negmas.preferences.value_fun import AffineFun, IdentityFun, LinearFun, TableFun
from negmas.preferences.crisp.pa import PAUtilityFunction


# =============================================================================
# Basic Functionality Tests
# =============================================================================


class TestPABasic:
    """Basic tests for PAUtilityFunction."""

    def test_single_issue_single_term(self):
        """PA with a single value function and single term."""
        issues = [make_issue(10, "A")]
        values = [IdentityFun()]
        terms = [(1.0, (1,))]  # Just v0
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f((0,)) == 0.0
        assert f((5,)) == 5.0
        assert f((9,)) == 9.0

    def test_single_issue_power(self):
        """PA with a single value function raised to a power."""
        issues = [make_issue(10, "A")]
        values = [IdentityFun()]
        terms = [(1.0, (2,))]  # v0^2
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f((2,)) == 4.0
        assert f((3,)) == 9.0
        assert f((5,)) == 25.0

    def test_multiple_issues_linear(self):
        """PA with multiple issues, linear aggregation (like weighted sum)."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        values = [IdentityFun(), IdentityFun()]
        # u = 2*v_A + 3*v_B (like weighted linear)
        terms = [(2.0, (1, 0)), (3.0, (0, 1))]
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f((1, 1)) == 5.0  # 2*1 + 3*1
        assert f((2, 3)) == 13.0  # 2*2 + 3*3 = 4 + 9

    def test_polynomial_interaction(self):
        """PA with polynomial interaction between values."""
        issues = [make_issue(10, "A"), make_issue(10, "B"), make_issue(10, "C")]
        values = [IdentityFun(), IdentityFun(), IdentityFun()]
        # u = v_A^3 + v_B^2*v_C + v_A*v_B*v_C^2
        terms = [
            (1.0, (3, 0, 0)),  # v_A^3
            (1.0, (0, 2, 1)),  # v_B^2 * v_C
            (1.0, (1, 1, 2)),  # v_A * v_B * v_C^2
        ]
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        # For (2, 3, 4): v_A=2, v_B=3, v_C=4
        # u = 8 + 36 + 96 = 140
        assert f((2, 3, 4)) == 140.0

    def test_bias_term(self):
        """PA with bias term."""
        issues = [make_issue(10, "A")]
        values = [IdentityFun()]
        terms = [(1.0, (1,))]
        f = PAUtilityFunction(values=values, terms=terms, bias=10.0, issues=issues)

        assert f((5,)) == 15.0  # 5 + 10

    def test_none_outcome(self):
        """PA returns reserved_value for None outcome."""
        issues = [make_issue(10, "A")]
        values = [IdentityFun()]
        terms = [(1.0, (1,))]
        f = PAUtilityFunction(
            values=values, terms=terms, issues=issues, reserved_value=0.5
        )

        assert f(None) == 0.5


# =============================================================================
# Value Function Types Tests
# =============================================================================


class TestPAValueFunctions:
    """Tests for PA with different value function types."""

    def test_with_affine_fun(self):
        """PA with AffineFun value functions."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        # v_A = 2*A + 1, v_B = B
        values = [AffineFun(slope=2.0, bias=1.0), IdentityFun()]
        terms = [(1.0, (1, 1))]  # v_A * v_B
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        # f(3, 4) = (2*3 + 1) * 4 = 7 * 4 = 28
        assert f((3, 4)) == 28.0

    def test_with_linear_fun(self):
        """PA with LinearFun value functions."""
        issues = [make_issue(10, "A")]
        values = [LinearFun(slope=3.0)]  # v = 3*A
        terms = [(1.0, (2,))]  # v^2 = 9*A^2
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f((2,)) == 36.0  # (3*2)^2 = 36

    def test_with_table_fun(self):
        """PA with TableFun value functions."""
        issues = [make_issue(["a", "b", "c"], "choice")]
        values = [TableFun(mapping={"a": 1.0, "b": 2.0, "c": 3.0})]
        terms = [(1.0, (2,))]  # v^2
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f(("a",)) == 1.0  # 1^2
        assert f(("b",)) == 4.0  # 2^2
        assert f(("c",)) == 9.0  # 3^2


# =============================================================================
# Dict-based Value Functions Tests
# =============================================================================


class TestPADictValues:
    """Tests for PA with dict-based value functions."""

    def test_values_as_dict(self):
        """PA accepts dict mapping issue names to value functions."""
        issues = [make_issue(10, "price"), make_issue(10, "quantity")]
        values = {
            "price": AffineFun(slope=1.0, bias=0.0),
            "quantity": AffineFun(slope=2.0, bias=0.0),
        }
        terms = [(1.0, (1, 1))]  # v_price * v_quantity
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        # f(3, 4) = 3 * 8 = 24
        assert f((3, 4)) == 24.0

    def test_values_as_list_of_dicts(self):
        """PA accepts list of dicts as value functions (converted to TableFun)."""
        issues = [make_issue(["a", "b"], "X")]
        values = [{"a": 10.0, "b": 20.0}]
        terms = [(1.0, (1,))]
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f(("a",)) == 10.0
        assert f(("b",)) == 20.0


# =============================================================================
# Lambda Value Functions Tests
# =============================================================================


class TestPALambdaValues:
    """Tests for PA with lambda/callable value functions."""

    def test_lambda_value_function(self):
        """PA converts callable to LambdaFun."""
        issues = [make_issue(10, "A")]
        values = [lambda x: x**2]  # v = A^2
        terms = [(1.0, (1,))]  # Just v
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f((3,)) == 9.0  # 3^2

    def test_multiple_lambda_functions(self):
        """PA with multiple lambda value functions."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        values = [lambda x: float(x), lambda x: float(x)]  # Identity
        # u = v_A^3 + v_B^2
        terms = [(1.0, (3, 0)), (1.0, (0, 2))]
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f((2, 3)) == 17.0  # 8 + 9


# =============================================================================
# Validation Tests
# =============================================================================


class TestPAValidation:
    """Tests for validation of inputs."""

    def test_mismatched_powers_raises(self):
        """PA raises ValueError if powers length doesn't match values."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        values = [IdentityFun(), IdentityFun()]
        terms = [(1.0, (1,))]  # Only 1 power, but 2 values

        with pytest.raises(ValueError, match="powers length"):
            PAUtilityFunction(values=values, terms=terms, issues=issues)


# =============================================================================
# Serialization Tests
# =============================================================================


class TestPASerialization:
    """Tests for serialization/deserialization."""

    def test_to_dict_from_dict(self):
        """PA can be serialized and deserialized."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        values = [AffineFun(slope=2.0, bias=1.0), AffineFun(slope=3.0, bias=0.0)]
        terms = [(1.0, (2, 0)), (1.0, (1, 1))]
        f = PAUtilityFunction(values=values, terms=terms, bias=5.0, issues=issues)

        d = f.to_dict()
        f2 = PAUtilityFunction.from_dict(d)

        # Test evaluation is preserved
        for outcome in [(0, 0), (1, 2), (5, 5), (9, 9)]:
            assert f(outcome) == approx(f2(outcome))


# =============================================================================
# Method Tests
# =============================================================================


class TestPAMethods:
    """Tests for PA methods."""

    def test_shift_by(self):
        """PA shift_by adds to bias."""
        issues = [make_issue(10, "A")]
        values = [IdentityFun()]
        terms = [(1.0, (1,))]
        f = PAUtilityFunction(values=values, terms=terms, bias=0.0, issues=issues)

        f2 = f.shift_by(10.0)
        assert f2((5,)) == f((5,)) + 10.0

    def test_scale_by(self):
        """PA scale_by multiplies terms and bias."""
        issues = [make_issue(10, "A")]
        values = [IdentityFun()]
        terms = [(1.0, (1,))]
        f = PAUtilityFunction(values=values, terms=terms, bias=5.0, issues=issues)

        f2 = f.scale_by(2.0)
        assert f2((5,)) == f((5,)) * 2.0

    def test_properties(self):
        """PA properties work correctly."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        values = [IdentityFun(), IdentityFun()]
        terms = [(2.0, (1, 0)), (3.0, (0, 1))]
        f = PAUtilityFunction(values=values, terms=terms, bias=1.0, issues=issues)

        assert len(f.values) == 2
        assert len(f.terms) == 2
        assert f.bias == 1.0


# =============================================================================
# Random Generation Tests
# =============================================================================


class TestPARandom:
    """Tests for random PA generation."""

    def test_random_generation(self):
        """PA can be randomly generated."""
        issues = [make_issue(10, "A"), make_issue(10, "B"), make_issue(10, "C")]
        f = PAUtilityFunction.random(issues=issues)

        # Should be callable
        result = f((1, 2, 3))
        assert isinstance(result, float)

    def test_random_with_params(self):
        """PA random respects parameters."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        f = PAUtilityFunction.random(issues=issues, n_terms=4, max_power=2)

        assert len(f.values) == 2  # One per issue
        assert len(f.terms) == 4


# =============================================================================
# Edge Cases
# =============================================================================


class TestPAEdgeCases:
    """Edge case tests for PA."""

    def test_zero_power(self):
        """Terms with power 0 treat value as 1."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        values = [IdentityFun(), IdentityFun()]
        # u = 5 * v_A^0 * v_B^2 = 5 * 1 * v_B^2 = 5 * v_B^2
        terms = [(5.0, (0, 2))]
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f((999, 3)) == 45.0  # A value doesn't matter, only B: 5*9

    def test_multiple_terms_same_value(self):
        """Multiple terms can use the same value with different powers."""
        issues = [make_issue(10, "A")]
        values = [IdentityFun()]
        # u = x + x^2 + x^3
        terms = [(1.0, (1,)), (1.0, (2,)), (1.0, (3,))]
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        # f(2) = 2 + 4 + 8 = 14
        assert f((2,)) == 14.0

    def test_negative_coefficients(self):
        """Terms can have negative coefficients."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        values = [IdentityFun(), IdentityFun()]
        # u = v_A - v_B (difference)
        terms = [(1.0, (1, 0)), (-1.0, (0, 1))]
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        assert f((5, 3)) == 2.0  # 5 - 3
        assert f((3, 5)) == -2.0  # 3 - 5


# =============================================================================
# Equivalence to LinearAdditiveUtilityFunction Tests
# =============================================================================


class TestPAEquivalenceToLinear:
    """Tests showing PA can express linear additive utilities."""

    def test_equivalent_to_weighted_sum(self):
        """PA with powers of (1,0,...) etc. is like weighted sum."""
        issues = [make_issue(10, "A"), make_issue(10, "B"), make_issue(10, "C")]
        values = [IdentityFun(), IdentityFun(), IdentityFun()]
        # Weighted sum: u = 0.5*A + 0.3*B + 0.2*C
        terms = [(0.5, (1, 0, 0)), (0.3, (0, 1, 0)), (0.2, (0, 0, 1))]
        f = PAUtilityFunction(values=values, terms=terms, issues=issues)

        # f(4, 6, 5) = 0.5*4 + 0.3*6 + 0.2*5 = 2 + 1.8 + 1 = 4.8
        assert f((4, 6, 5)) == approx(4.8)
