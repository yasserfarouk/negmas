"""Tests for GLAUtilityFunction (Generalized Linear Aggregation)."""

from __future__ import annotations

import math
import random

import pytest
from pytest import approx

from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.crisp import GLAUtilityFunction
from negmas.preferences.value_fun import (
    AffineFun,
    AffineMultiFun,
    LambdaMultiFun,
    LinearMultiFun,
    TableFun,
    TableMultiFun,
)


# =============================================================================
# Basic GLAUtilityFunction Tests
# =============================================================================


class TestGLABasic:
    """Basic tests for GLAUtilityFunction."""

    def test_single_factor_affine(self):
        """GLA with a single affine factor."""
        issues = [make_issue(10, "A")]
        f = GLAUtilityFunction(
            factors=[(("A",), AffineFun(slope=2.0, bias=1.0))],
            weights=[1.0],
            issues=issues,
        )
        # f(x) = 2*x + 1
        assert f((0,)) == 1.0
        assert f((5,)) == 11.0
        assert f((9,)) == 19.0

    def test_single_factor_table(self):
        """GLA with a single table factor."""
        issues = [make_issue(["a", "b", "c"], "Color")]
        f = GLAUtilityFunction(
            factors=[(("Color",), TableFun(mapping={"a": 1.0, "b": 2.0, "c": 3.0}))],
            weights=[1.0],
            issues=issues,
        )
        assert f(("a",)) == 1.0
        assert f(("b",)) == 2.0
        assert f(("c",)) == 3.0

    def test_multiple_single_issue_factors(self):
        """GLA with multiple single-issue factors."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0)),
                (("B",), AffineFun(slope=2.0, bias=0.0)),
            ],
            weights=[1.0, 1.0],
            issues=issues,
        )
        # f(a, b) = 1*a + 2*b
        assert f((3, 2)) == 3 + 4  # 7
        assert f((0, 0)) == 0
        assert f((5, 4)) == 5 + 8  # 13

    def test_weighted_factors(self):
        """GLA with weighted factors."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0)),
                (("B",), AffineFun(slope=1.0, bias=0.0)),
            ],
            weights=[0.5, 2.0],
            issues=issues,
        )
        # f(a, b) = 0.5*a + 2.0*b
        assert f((4, 2)) == 0.5 * 4 + 2.0 * 2  # 6

    def test_bias_term(self):
        """GLA with a bias term."""
        issues = [make_issue(10, "A")]
        f = GLAUtilityFunction(
            factors=[(("A",), AffineFun(slope=1.0, bias=0.0))],
            weights=[1.0],
            bias=10.0,
            issues=issues,
        )
        # f(a) = a + 10
        assert f((0,)) == 10.0
        assert f((5,)) == 15.0

    def test_none_outcome(self):
        """GLA returns reserved_value for None outcome."""
        issues = [make_issue(10, "A")]
        f = GLAUtilityFunction(
            factors=[(("A",), AffineFun(slope=1.0, bias=0.0))],
            weights=[1.0],
            issues=issues,
            reserved_value=-1.0,
        )
        assert f(None) == -1.0


class TestGLAMultiIssueFunctions:
    """Tests for GLA with multi-issue value functions."""

    def test_linear_multi_fun(self):
        """GLA with LinearMultiFun (cross-issue factor)."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[(("A", "B"), LinearMultiFun(slope=(1.0, 2.0)))],
            weights=[1.0],
            issues=issues,
        )
        # f(a, b) = 1*a + 2*b
        assert f((3, 2)) == 3 + 4  # 7
        assert f((0, 0)) == 0

    def test_affine_multi_fun(self):
        """GLA with AffineMultiFun."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[(("A", "B"), AffineMultiFun(slope=(1.0, 2.0), bias=5.0))],
            weights=[1.0],
            issues=issues,
        )
        # f(a, b) = 1*a + 2*b + 5
        assert f((3, 2)) == 3 + 4 + 5  # 12

    def test_table_multi_fun(self):
        """GLA with TableMultiFun."""
        issues = [make_issue(2, "A"), make_issue(2, "B")]
        f = GLAUtilityFunction(
            factors=[
                (
                    ("A", "B"),
                    TableMultiFun(
                        mapping={(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0}
                    ),
                )
            ],
            weights=[1.0],
            issues=issues,
        )
        assert f((0, 0)) == 1.0
        assert f((0, 1)) == 2.0
        assert f((1, 0)) == 3.0
        assert f((1, 1)) == 4.0

    def test_lambda_multi_fun(self):
        """GLA with LambdaMultiFun."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        f = GLAUtilityFunction(
            factors=[(("A", "B"), LambdaMultiFun(f=lambda x: x[0] * x[1]))],
            weights=[1.0],
            issues=issues,
        )
        # f(a, b) = a * b
        assert f((3, 4)) == 12
        assert f((0, 5)) == 0

    def test_mixed_single_and_multi_factors(self):
        """GLA with both single-issue and multi-issue factors."""
        issues = [make_issue(10, "A"), make_issue(5, "B"), make_issue(3, "C")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0)),  # Single issue
                (("B",), AffineFun(slope=2.0, bias=0.0)),  # Single issue
                (("A", "B"), LinearMultiFun(slope=(0.1, 0.1))),  # Cross-issue
                (("C",), AffineFun(slope=3.0, bias=0.0)),  # Single issue
            ],
            weights=[1.0, 1.0, 1.0, 1.0],
            issues=issues,
        )
        # f = a + 2*b + 0.1*a + 0.1*b + 3*c = 1.1*a + 2.1*b + 3*c
        result = f((5, 2, 1))
        expected = 5 + 4 + 0.5 + 0.2 + 3
        assert result == approx(expected)


class TestGLAIssueIndexing:
    """Tests for GLA issue indexing (by name and by index)."""

    def test_index_by_name(self):
        """GLA can reference issues by name."""
        issues = [make_issue(10, "price"), make_issue(5, "quantity")]
        f = GLAUtilityFunction(
            factors=[
                (("price",), AffineFun(slope=-1.0, bias=100.0)),
                (("quantity",), AffineFun(slope=2.0, bias=0.0)),
            ],
            weights=[1.0, 1.0],
            issues=issues,
        )
        # price=50, quantity=3: (100 - 50) + 6 = 56
        assert f((50, 3)) == 56

    def test_index_by_integer(self):
        """GLA can reference issues by integer index."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                ((0,), AffineFun(slope=1.0, bias=0.0)),
                ((1,), AffineFun(slope=2.0, bias=0.0)),
            ],
            weights=[1.0, 1.0],
            issues=issues,
        )
        assert f((3, 2)) == 7

    def test_mixed_indexing(self):
        """GLA can mix name and integer indexing."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0)),
                ((1,), AffineFun(slope=2.0, bias=0.0)),
            ],
            weights=[1.0, 1.0],
            issues=issues,
        )
        assert f((3, 2)) == 7

    def test_invalid_issue_name_raises(self):
        """GLA raises ValueError for invalid issue names."""
        issues = [make_issue(10, "A")]
        with pytest.raises(ValueError, match="Issue name"):
            GLAUtilityFunction(
                factors=[(("NonExistent",), AffineFun(slope=1.0, bias=0.0))],
                issues=issues,
            )


class TestGLAAutoConversion:
    """Tests for automatic conversion of factor functions."""

    def test_dict_to_table_single(self):
        """Dict is converted to TableFun for single-issue factors."""
        issues = [make_issue(["a", "b"], "X")]
        f = GLAUtilityFunction(
            factors=[(("X",), {"a": 1.0, "b": 2.0})], weights=[1.0], issues=issues
        )
        assert f(("a",)) == 1.0
        assert f(("b",)) == 2.0

    def test_dict_to_table_multi(self):
        """Dict is converted to TableMultiFun for multi-issue factors."""
        issues = [make_issue(2, "A"), make_issue(2, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A", "B"), {(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0})
            ],
            weights=[1.0],
            issues=issues,
        )
        assert f((0, 0)) == 1.0
        assert f((1, 1)) == 4.0

    def test_lambda_to_lambdafun_single(self):
        """Lambda is converted to LambdaFun for single-issue factors."""
        issues = [make_issue(10, "A")]
        f = GLAUtilityFunction(
            factors=[(("A",), lambda x: x**2)], weights=[1.0], issues=issues
        )
        assert f((3,)) == 9
        assert f((4,)) == 16

    def test_lambda_to_lambdafun_multi(self):
        """Lambda is converted to LambdaMultiFun for multi-issue factors."""
        issues = [make_issue(10, "A"), make_issue(10, "B")]
        f = GLAUtilityFunction(
            factors=[(("A", "B"), lambda x: x[0] + x[1])], weights=[1.0], issues=issues
        )
        assert f((3, 4)) == 7

    def test_float_to_const_single(self):
        """Float is converted to constant function for single-issue factors."""
        issues = [make_issue(10, "A")]
        f = GLAUtilityFunction(factors=[(("A",), 5.0)], weights=[1.0], issues=issues)
        # Constant factor always returns 5
        assert f((0,)) == 5.0
        assert f((9,)) == 5.0

    def test_float_to_const_multi(self):
        """Float is converted to constant function for multi-issue factors."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[(("A", "B"), 10.0)], weights=[1.0], issues=issues
        )
        assert f((0, 0)) == 10.0
        assert f((5, 3)) == 10.0


# =============================================================================
# GLA equivalence to LinearAdditiveUtilityFunction
# =============================================================================


class TestGLAEquivalenceToLinearAdditive:
    """Tests that GLA with single-issue factors equals LinearAdditiveUtilityFunction."""

    def test_equivalence_with_affine_values(self):
        """GLA with single-issue AffineFuns equals LinearAdditiveUtilityFunction."""
        issues = [make_issue(10, "A"), make_issue(5, "B"), make_issue(8, "C")]
        weights = [0.4, 0.35, 0.25]

        # Create value functions
        value_funs = [
            AffineFun(slope=0.1, bias=0.0),
            AffineFun(slope=0.2, bias=0.0),
            AffineFun(slope=0.125, bias=0.0),
        ]

        # GLA version
        gla = GLAUtilityFunction(
            factors=[((issue.name,), vf) for issue, vf in zip(issues, value_funs)],
            weights=weights,
            issues=issues,
        )

        # LinearAdditiveUtilityFunction version
        lauf = LinearAdditiveUtilityFunction(
            values=value_funs, weights=weights, issues=issues
        )

        # Test on random outcomes
        random.seed(42)
        for _ in range(20):
            outcome = tuple(
                random.randint(0, issue.cardinality - 1) for issue in issues
            )
            assert gla(outcome) == approx(lauf(outcome)), f"Mismatch at {outcome}"

    def test_equivalence_with_table_values(self):
        """GLA with single-issue TableFuns equals LinearAdditiveUtilityFunction."""
        issues = [
            make_issue(["low", "medium", "high"], "Quality"),
            make_issue(["red", "blue", "green"], "Color"),
        ]
        weights = [0.6, 0.4]

        value_funs = [
            TableFun(mapping={"low": 0.0, "medium": 0.5, "high": 1.0}),
            TableFun(mapping={"red": 0.3, "blue": 0.6, "green": 1.0}),
        ]

        # GLA version
        gla = GLAUtilityFunction(
            factors=[((issue.name,), vf) for issue, vf in zip(issues, value_funs)],
            weights=weights,
            issues=issues,
        )

        # LinearAdditiveUtilityFunction version
        lauf = LinearAdditiveUtilityFunction(
            values=value_funs, weights=weights, issues=issues
        )

        # Test all outcomes
        for q in ["low", "medium", "high"]:
            for c in ["red", "blue", "green"]:
                outcome = (q, c)
                assert gla(outcome) == approx(lauf(outcome)), f"Mismatch at {outcome}"

    def test_equivalence_normalized(self):
        """GLA matches normalized LinearAdditiveUtilityFunction."""
        issues = [
            make_issue((0.0, 100.0), "Price"),
            make_issue((0.0, 10.0), "Quantity"),
        ]

        # Normalized weights
        weights = [0.7, 0.3]

        # Value functions that map to [0, 1]
        value_funs = [
            AffineFun(slope=-0.01, bias=1.0),  # Higher price = lower value
            AffineFun(slope=0.1, bias=0.0),  # Higher quantity = higher value
        ]

        gla = GLAUtilityFunction(
            factors=[((issue.name,), vf) for issue, vf in zip(issues, value_funs)],
            weights=weights,
            issues=issues,
        )

        lauf = LinearAdditiveUtilityFunction(
            values=value_funs, weights=weights, issues=issues
        )

        # Test various outcomes
        test_outcomes = [(0.0, 0.0), (50.0, 5.0), (100.0, 10.0), (25.0, 7.5)]
        for outcome in test_outcomes:
            assert gla(outcome) == approx(lauf(outcome)), f"Mismatch at {outcome}"


# =============================================================================
# GLA Serialization Tests
# =============================================================================


class TestGLASerialization:
    """Tests for GLA serialization and deserialization."""

    def test_to_dict_from_dict_affine(self):
        """GLA with AffineFun can be serialized and deserialized."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=2.0, bias=1.0)),
                (("B",), AffineFun(slope=3.0, bias=0.0)),
            ],
            weights=[0.5, 0.5],
            bias=1.0,
            issues=issues,
        )

        d = f.to_dict()
        f2 = GLAUtilityFunction.from_dict(d)

        # Test on various outcomes
        for outcome in [(0, 0), (5, 2), (9, 4)]:
            assert f(outcome) == approx(f2(outcome))

    def test_to_dict_from_dict_table(self):
        """GLA with TableFun can be serialized and deserialized."""
        issues = [make_issue(["a", "b"], "X")]
        f = GLAUtilityFunction(
            factors=[(("X",), TableFun(mapping={"a": 1.0, "b": 2.0}))],
            weights=[1.0],
            issues=issues,
        )

        d = f.to_dict()
        f2 = GLAUtilityFunction.from_dict(d)

        assert f(("a",)) == f2(("a",))
        assert f(("b",)) == f2(("b",))

    def test_to_dict_from_dict_multi_issue(self):
        """GLA with multi-issue functions can be serialized and deserialized."""
        issues = [make_issue(5, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0)),
                (("A", "B"), AffineMultiFun(slope=(0.5, 0.5), bias=0.0)),
            ],
            weights=[1.0, 1.0],
            issues=issues,
        )

        d = f.to_dict()
        f2 = GLAUtilityFunction.from_dict(d)

        for a in range(5):
            for b in range(5):
                assert f((a, b)) == approx(f2((a, b)))


# =============================================================================
# GLA Utility Methods Tests
# =============================================================================


class TestGLAMethods:
    """Tests for GLA utility methods (shift_by, scale_by, etc.)."""

    def test_shift_by(self):
        """shift_by creates a shifted utility function."""
        issues = [make_issue(10, "A")]
        f = GLAUtilityFunction(
            factors=[(("A",), AffineFun(slope=1.0, bias=0.0))],
            weights=[1.0],
            issues=issues,
        )

        f2 = f.shift_by(5.0)

        assert f2((0,)) == f((0,)) + 5.0
        assert f2((5,)) == f((5,)) + 5.0

    def test_scale_by(self):
        """scale_by creates a scaled utility function."""
        issues = [make_issue(10, "A")]
        f = GLAUtilityFunction(
            factors=[(("A",), AffineFun(slope=1.0, bias=0.0))],
            weights=[1.0],
            issues=issues,
        )

        f2 = f.scale_by(2.0)

        assert f2((5,)) == f((5,)) * 2.0
        assert f2((3,)) == f((3,)) * 2.0

    def test_properties(self):
        """Test factors, weights, and bias properties."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0)),
                (("B",), AffineFun(slope=2.0, bias=0.0)),
            ],
            weights=[0.3, 0.7],
            bias=1.5,
            issues=issues,
        )

        assert len(f.factors) == 2
        assert f.weights == [0.3, 0.7]
        assert f.bias == 1.5


# =============================================================================
# GLA Random Generation Tests
# =============================================================================


class TestGLARandom:
    """Tests for GLA random generation."""

    def test_random_generation(self):
        """GLA.random() generates valid utility functions."""
        issues = [make_issue(10, "A"), make_issue(5, "B"), make_issue(3, "C")]
        f = GLAUtilityFunction.random(issues=issues)

        # Should be callable
        outcome = (5, 2, 1)
        result = f(outcome)
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_random_with_n_factors(self):
        """GLA.random() respects n_factors parameter."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction.random(issues=issues, n_factors=3)

        assert len(f.factors) == 3

    def test_random_with_max_factor_issues(self):
        """GLA.random() respects max_factor_issues parameter."""
        issues = [make_issue(10, "A"), make_issue(5, "B"), make_issue(3, "C")]
        f = GLAUtilityFunction.random(issues=issues, n_factors=10, max_factor_issues=2)

        for indices, _ in f.factors:
            assert len(indices) <= 2


# =============================================================================
# GLA Edge Cases
# =============================================================================


class TestGLAEdgeCases:
    """Edge case tests for GLA."""

    def test_empty_factors(self):
        """GLA with no factors returns only bias."""
        issues = [make_issue(10, "A")]
        f = GLAUtilityFunction(factors=[], weights=[], bias=5.0, issues=issues)
        assert f((0,)) == 5.0
        assert f((9,)) == 5.0

    def test_zero_weights(self):
        """GLA with zero weights ignores those factors."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0)),
                (("B",), AffineFun(slope=100.0, bias=0.0)),
            ],
            weights=[1.0, 0.0],  # Second factor has zero weight
            issues=issues,
        )
        # Only A matters
        assert f((5, 0)) == 5.0
        assert f((5, 4)) == 5.0  # B value doesn't matter

    def test_negative_weights(self):
        """GLA handles negative weights."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0)),
                (("B",), AffineFun(slope=1.0, bias=0.0)),
            ],
            weights=[1.0, -1.0],
            issues=issues,
        )
        # f(a, b) = a - b
        assert f((5, 2)) == 3
        assert f((2, 5)) == -3

    def test_weights_in_factor_tuple(self):
        """GLA accepts weights as third element of factor tuple."""
        issues = [make_issue(10, "A"), make_issue(5, "B")]
        f = GLAUtilityFunction(
            factors=[
                (("A",), AffineFun(slope=1.0, bias=0.0), 0.5),
                (("B",), AffineFun(slope=1.0, bias=0.0), 0.5),
            ],
            issues=issues,
        )
        assert f((4, 4)) == 0.5 * 4 + 0.5 * 4  # 4
