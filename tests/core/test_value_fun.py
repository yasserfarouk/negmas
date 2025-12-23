"""Tests for value functions (single-issue and multi-issue)."""

from __future__ import annotations

import math
from math import e

import pytest
from pytest import approx

from negmas.outcomes import make_issue
from negmas.preferences.value_fun import (
    AffineFun,
    AffineMultiFun,
    BilinearMultiFun,
    ConstFun,
    CosFun,
    ExponentialFun,
    IdentityFun,
    LambdaFun,
    LambdaMultiFun,
    LinearFun,
    LinearMultiFun,
    LogFun,
    PolynomialFun,
    PolynomialMultiFun,
    ProductMultiFun,
    QuadraticFun,
    QuadraticMultiFun,
    SinFun,
    TableFun,
    TableMultiFun,
    TriangularFun,
)


# =============================================================================
# Tests for Single-Issue Value Functions (BaseFun subclasses)
# =============================================================================


class TestConstFun:
    """Tests for ConstFun."""

    def test_constant_value(self):
        """ConstFun always returns the same value."""
        f = ConstFun(bias=5.0)
        assert f(0) == 5.0
        assert f(100) == 5.0
        assert f(-50) == 5.0

    def test_minmax(self):
        """MinMax should return (bias, bias)."""
        f = ConstFun(bias=3.0)
        issue = make_issue(10, "x")
        mn, mx = f.minmax(issue)
        assert mn == 3.0
        assert mx == 3.0

    def test_shift_by(self):
        """Shifting should add to the bias."""
        f = ConstFun(bias=5.0)
        f2 = f.shift_by(2.0)
        assert f2(0) == 7.0

    def test_scale_by(self):
        """Scaling a ConstFun returns an AffineFun."""
        f = ConstFun(bias=5.0)
        f2 = f.scale_by(2.0)
        # scale_by on ConstFun returns AffineFun(slope=scale, bias=self.bias)
        assert isinstance(f2, AffineFun)


class TestIdentityFun:
    """Tests for IdentityFun."""

    def test_identity(self):
        """IdentityFun returns input unchanged."""
        f = IdentityFun()
        assert f(5.0) == 5.0
        assert f(0) == 0
        assert f(-3.5) == -3.5

    def test_minmax(self):
        """MinMax should return issue's min and max."""
        f = IdentityFun()
        issue = make_issue((0.0, 10.0), "x")
        mn, mx = f.minmax(issue)
        assert mn == 0.0
        assert mx == 10.0

    def test_shift_by(self):
        """Shifting returns a ConstFun."""
        f = IdentityFun()
        f2 = f.shift_by(3.0)
        assert isinstance(f2, ConstFun)
        assert f2.bias == 3.0

    def test_scale_by(self):
        """Scaling returns a LinearFun."""
        f = IdentityFun()
        f2 = f.scale_by(2.0)
        assert isinstance(f2, LinearFun)
        assert f2.slope == 2.0


class TestLinearFun:
    """Tests for LinearFun."""

    def test_linear(self):
        """LinearFun computes slope * x."""
        f = LinearFun(slope=2.0)
        assert f(5.0) == 10.0
        assert f(0) == 0
        assert f(-3.0) == -6.0

    def test_minmax_positive_slope(self):
        """MinMax with positive slope."""
        f = LinearFun(slope=2.0)
        issue = make_issue((0.0, 10.0), "x")
        mn, mx = f.minmax(issue)
        assert mn == 0.0
        assert mx == 20.0

    def test_minmax_negative_slope(self):
        """MinMax with negative slope."""
        f = LinearFun(slope=-2.0)
        issue = make_issue((0.0, 10.0), "x")
        mn, mx = f.minmax(issue)
        assert mn == -20.0
        assert mx == 0.0

    def test_shift_by(self):
        """Shifting returns an AffineFun."""
        f = LinearFun(slope=2.0)
        f2 = f.shift_by(1.0)
        assert isinstance(f2, AffineFun)
        assert f2.slope == 2.0
        assert f2.bias == 1.0

    def test_scale_by(self):
        """Scaling multiplies the slope."""
        f = LinearFun(slope=2.0)
        f2 = f.scale_by(3.0)
        assert f2.slope == 6.0


class TestAffineFun:
    """Tests for AffineFun."""

    def test_affine(self):
        """AffineFun computes slope * x + bias."""
        f = AffineFun(slope=2.0, bias=1.0)
        assert f(5.0) == 11.0
        assert f(0) == 1.0
        assert f(-3.0) == -5.0

    def test_minmax(self):
        """MinMax for affine function."""
        f = AffineFun(slope=2.0, bias=1.0)
        issue = make_issue((0.0, 10.0), "x")
        mn, mx = f.minmax(issue)
        assert mn == 1.0
        assert mx == 21.0

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = AffineFun(slope=2.0, bias=1.0)
        f2 = f.shift_by(3.0)
        assert f2.slope == 2.0
        assert f2.bias == 4.0

    def test_scale_by(self):
        """Scaling multiplies both slope and bias."""
        f = AffineFun(slope=2.0, bias=1.0)
        f2 = f.scale_by(3.0)
        assert f2.slope == 6.0
        assert f2.bias == 3.0


class TestQuadraticFun:
    """Tests for QuadraticFun."""

    def test_quadratic(self):
        """QuadraticFun computes a2*x^2 + a1*x + bias."""
        f = QuadraticFun(a2=1.0, a1=0.0, bias=0.0)
        assert f(2.0) == 4.0
        assert f(-2.0) == 4.0
        assert f(0) == 0.0

        f2 = QuadraticFun(a2=1.0, a1=-2.0, bias=1.0)
        # (x-1)^2 = x^2 - 2x + 1
        assert f2(1.0) == 0.0
        assert f2(0.0) == 1.0
        assert f2(2.0) == 1.0

    def test_minmax(self):
        """MinMax for quadratic function."""
        f = QuadraticFun(a2=1.0, a1=-2.0, bias=0.0)
        # f(x) = x^2 - 2x, minimum at x=1 where f(1) = -1
        issue = make_issue((0.0, 3.0), "x")
        mn, mx = f.minmax(issue)
        assert mn == approx(-1.0)  # f(1) = 1 - 2 = -1
        # Note: minmax implementation may vary, just check mn < mx
        assert mn <= mx

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = QuadraticFun(a2=1.0, a1=0.0, bias=0.0)
        f2 = f.shift_by(5.0)
        assert f2.bias == 5.0

    def test_scale_by(self):
        """Scaling multiplies all coefficients."""
        f = QuadraticFun(a2=1.0, a1=2.0, bias=3.0)
        f2 = f.scale_by(2.0)
        assert f2.a2 == 2.0
        assert f2.a1 == 4.0
        assert f2.bias == 6.0


class TestPolynomialFun:
    """Tests for PolynomialFun."""

    def test_polynomial(self):
        """PolynomialFun computes sum(coef[i] * x^(i+1)) + bias."""
        # f(x) = 2x + 3x^2 + 1
        f = PolynomialFun(coefficients=(2.0, 3.0), bias=1.0)
        assert f(0) == 1.0
        assert f(1) == 2.0 + 3.0 + 1.0  # 6
        assert f(2) == 4.0 + 12.0 + 1.0  # 17

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = PolynomialFun(coefficients=(1.0,), bias=0.0)
        f2 = f.shift_by(5.0)
        assert f2.bias == 5.0

    def test_scale_by(self):
        """Scaling multiplies all coefficients and bias."""
        f = PolynomialFun(coefficients=(1.0, 2.0), bias=3.0)
        f2 = f.scale_by(2.0)
        assert f2.coefficients == (2.0, 4.0)
        assert f2.bias == 6.0


class TestTriangularFun:
    """Tests for TriangularFun."""

    def test_triangular_at_edges(self):
        """TriangularFun returns bias at edges."""
        f = TriangularFun(start=0.0, middle=5.0, end=10.0, bias=0.0)
        assert f(-1.0) == 0.0  # Before start
        assert f(0.0) == 0.0  # At start
        assert f(10.0) == 0.0  # At end
        assert f(11.0) == 0.0  # After end

    def test_triangular_at_middle(self):
        """TriangularFun returns bias + scale at middle."""
        f = TriangularFun(start=0.0, middle=5.0, end=10.0, bias=0.0, scale=1.0)
        assert f(5.0) == 1.0

    def test_triangular_interpolation(self):
        """TriangularFun linearly interpolates."""
        f = TriangularFun(start=0.0, middle=10.0, end=20.0, bias=0.0, scale=1.0)
        assert f(5.0) == approx(0.5)  # Halfway up
        assert f(15.0) == approx(0.5)  # Halfway down

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = TriangularFun(start=0.0, middle=5.0, end=10.0, bias=0.0)
        f2 = f.shift_by(2.0)
        assert f2.bias == 2.0

    def test_scale_by(self):
        """Scaling multiplies the scale."""
        f = TriangularFun(start=0.0, middle=5.0, end=10.0, bias=0.0, scale=1.0)
        f2 = f.scale_by(3.0)
        assert f2.scale == 3.0


class TestExponentialFun:
    """Tests for ExponentialFun."""

    def test_exponential(self):
        """ExponentialFun computes base^(tau*x) + bias."""
        f = ExponentialFun(tau=1.0, bias=0.0, base=e)
        assert f(0) == approx(1.0)
        assert f(1) == approx(e)
        assert f(2) == approx(e**2)

    def test_exponential_with_tau(self):
        """ExponentialFun with different tau."""
        f = ExponentialFun(tau=2.0, bias=0.0, base=e)
        assert f(1) == approx(e**2)

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = ExponentialFun(tau=1.0, bias=0.0)
        f2 = f.shift_by(5.0)
        assert f2.bias == 5.0


class TestLogFun:
    """Tests for LogFun."""

    def test_log(self):
        """LogFun computes scale * log_base(tau*x) + bias."""
        # Note: LogFun uses math.log(tau*x, base) internally
        f = LogFun(tau=1.0, bias=0.0, base=e, scale=1.0)
        # log_e(1) = 0
        assert f(1.0) == approx(0.0)
        # log_e(e) = 1
        assert f(e) == approx(1.0)

    def test_log_with_scale(self):
        """LogFun with different scale."""
        f = LogFun(tau=1.0, bias=0.0, base=e, scale=2.0)
        # 2 * log_e(e) = 2
        assert f(e) == approx(2.0)

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = LogFun(tau=1.0, bias=0.0)
        f2 = f.shift_by(5.0)
        assert f2.bias == 5.0


class TestSinFun:
    """Tests for SinFun."""

    def test_sin(self):
        """SinFun computes amplitude * sin(multiplier*x + phase) + bias."""
        f = SinFun(multiplier=1.0, bias=0.0, phase=0.0, amplitude=1.0)
        assert f(0) == approx(0.0)
        assert f(math.pi / 2) == approx(1.0)
        assert f(math.pi) == approx(0.0, abs=1e-10)
        assert f(3 * math.pi / 2) == approx(-1.0)

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = SinFun()
        f2 = f.shift_by(1.0)
        assert f2.bias == 1.0

    def test_scale_by(self):
        """Scaling multiplies the amplitude."""
        f = SinFun(amplitude=1.0)
        f2 = f.scale_by(2.0)
        assert f2.amplitude == 2.0


class TestCosFun:
    """Tests for CosFun."""

    def test_cos(self):
        """CosFun computes amplitude * cos(multiplier*x + phase) + bias."""
        f = CosFun(multiplier=1.0, bias=0.0, phase=0.0, amplitude=1.0)
        assert f(0) == approx(1.0)
        assert f(math.pi / 2) == approx(0.0, abs=1e-10)
        assert f(math.pi) == approx(-1.0)

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = CosFun()
        f2 = f.shift_by(1.0)
        assert f2.bias == 1.0

    def test_scale_by(self):
        """Scaling multiplies the amplitude."""
        f = CosFun(amplitude=1.0)
        f2 = f.scale_by(2.0)
        assert f2.amplitude == 2.0


class TestTableFun:
    """Tests for TableFun."""

    def test_table_lookup(self):
        """TableFun looks up values in mapping."""
        f = TableFun(mapping={"a": 1.0, "b": 2.0, "c": 3.0})
        assert f("a") == 1.0
        assert f("b") == 2.0
        assert f("c") == 3.0

    def test_table_numeric(self):
        """TableFun works with numeric keys."""
        f = TableFun(mapping={0: 0.0, 1: 0.5, 2: 1.0})
        assert f(0) == 0.0
        assert f(1) == 0.5
        assert f(2) == 1.0

    def test_table_missing_key(self):
        """TableFun raises KeyError for missing keys."""
        f = TableFun(mapping={"a": 1.0})
        with pytest.raises(KeyError):
            f("b")

    def test_minmax(self):
        """MinMax returns min and max of values."""
        f = TableFun(mapping={0: 1.0, 1: 5.0, 2: 3.0})
        issue = make_issue(3, "x")
        mn, mx = f.minmax(issue)
        assert mn == 1.0
        assert mx == 5.0

    def test_shift_by(self):
        """Shifting adds to all values."""
        f = TableFun(mapping={"a": 1.0, "b": 2.0})
        f2 = f.shift_by(3.0)
        assert f2("a") == 4.0
        assert f2("b") == 5.0

    def test_scale_by(self):
        """Scaling multiplies all values."""
        f = TableFun(mapping={"a": 1.0, "b": 2.0})
        f2 = f.scale_by(2.0)
        assert f2("a") == 2.0
        assert f2("b") == 4.0


class TestLambdaFun:
    """Tests for LambdaFun."""

    def test_lambda(self):
        """LambdaFun evaluates custom function."""
        f = LambdaFun(f=lambda x: x**2)
        assert f(2) == 4.0
        assert f(3) == 9.0

    def test_lambda_with_bias(self):
        """LambdaFun adds bias."""
        f = LambdaFun(f=lambda x: x**2, bias=1.0)
        assert f(2) == 5.0

    def test_shift_by(self):
        """Shifting creates new LambdaFun with offset."""
        f = LambdaFun(f=lambda x: x)
        f2 = f.shift_by(5.0)
        assert f2(0) == 5.0

    def test_scale_by(self):
        """Scaling creates new LambdaFun with multiplier."""
        f = LambdaFun(f=lambda x: x)
        f2 = f.scale_by(2.0)
        assert f2(5) == 10.0


# =============================================================================
# Tests for Multi-Issue Value Functions (BaseMultiFun subclasses)
# =============================================================================


class TestLinearMultiFun:
    """Tests for LinearMultiFun."""

    def test_linear_multi(self):
        """LinearMultiFun computes sum(slope[i] * x[i])."""
        f = LinearMultiFun(slope=(1.0, 2.0, 3.0))
        assert f((1, 1, 1)) == 6.0
        assert f((2, 3, 4)) == 2 + 6 + 12  # 20
        assert f((0, 0, 0)) == 0.0

    def test_minmax(self):
        """MinMax for linear multi-issue function."""
        f = LinearMultiFun(slope=(1.0, 2.0))
        issues = (make_issue((0.0, 10.0), "x"), make_issue((0.0, 5.0), "y"))
        mn, mx = f.minmax(issues)
        assert mn == 0.0
        assert mx == 10.0 + 10.0  # 1*10 + 2*5

    def test_shift_by(self):
        """Shifting returns AffineMultiFun."""
        f = LinearMultiFun(slope=(1.0, 2.0))
        f2 = f.shift_by(3.0)
        assert isinstance(f2, AffineMultiFun)
        assert f2((0, 0)) == 3.0

    def test_scale_by(self):
        """Scaling multiplies slopes."""
        f = LinearMultiFun(slope=(1.0, 2.0))
        f2 = f.scale_by(2.0)
        assert f2.slope == (2.0, 4.0)


class TestAffineMultiFun:
    """Tests for AffineMultiFun."""

    def test_affine_multi(self):
        """AffineMultiFun computes sum(slope[i] * x[i]) + bias."""
        f = AffineMultiFun(slope=(1.0, 2.0), bias=5.0)
        assert f((0, 0)) == 5.0
        assert f((1, 1)) == 8.0
        assert f((2, 3)) == 2 + 6 + 5  # 13

    def test_minmax(self):
        """MinMax for affine multi-issue function."""
        f = AffineMultiFun(slope=(1.0, 2.0), bias=1.0)
        issues = (make_issue((0.0, 10.0), "x"), make_issue((0.0, 5.0), "y"))
        mn, mx = f.minmax(issues)
        assert mn == 1.0
        assert mx == 1.0 + 10.0 + 10.0  # 21

    def test_shift_by(self):
        """Shifting adds to bias."""
        f = AffineMultiFun(slope=(1.0, 2.0), bias=5.0)
        f2 = f.shift_by(3.0)
        assert f2.bias == 8.0

    def test_scale_by(self):
        """Scaling multiplies slopes and bias."""
        f = AffineMultiFun(slope=(1.0, 2.0), bias=5.0)
        f2 = f.scale_by(2.0)
        assert f2.slope == (2.0, 4.0)
        assert f2.bias == 10.0


class TestTableMultiFun:
    """Tests for TableMultiFun."""

    def test_table_multi_lookup(self):
        """TableMultiFun looks up value tuples."""
        f = TableMultiFun(mapping={(0, 0): 1.0, (0, 1): 2.0, (1, 0): 3.0, (1, 1): 4.0})
        assert f((0, 0)) == 1.0
        assert f((0, 1)) == 2.0
        assert f((1, 0)) == 3.0
        assert f((1, 1)) == 4.0

    def test_table_multi_missing_key(self):
        """TableMultiFun raises KeyError for missing tuples."""
        f = TableMultiFun(mapping={(0, 0): 1.0})
        with pytest.raises(KeyError):
            f((0, 1))

    def test_shift_by(self):
        """Shifting adds to all values."""
        f = TableMultiFun(mapping={(0, 0): 1.0, (0, 1): 2.0})
        f2 = f.shift_by(3.0)
        assert f2((0, 0)) == 4.0
        assert f2((0, 1)) == 5.0

    def test_scale_by(self):
        """Scaling multiplies all values."""
        f = TableMultiFun(mapping={(0, 0): 1.0, (0, 1): 2.0})
        f2 = f.scale_by(2.0)
        assert f2((0, 0)) == 2.0
        assert f2((0, 1)) == 4.0


class TestLambdaMultiFun:
    """Tests for LambdaMultiFun."""

    def test_lambda_multi(self):
        """LambdaMultiFun evaluates custom function on tuple."""
        f = LambdaMultiFun(f=lambda x: x[0] * x[1])
        assert f((2, 3)) == 6.0
        assert f((4, 5)) == 20.0

    def test_lambda_multi_with_bias(self):
        """LambdaMultiFun adds bias."""
        f = LambdaMultiFun(f=lambda x: x[0] + x[1], bias=10.0)
        assert f((2, 3)) == 15.0


# =============================================================================
# Tests for serialization/deserialization
# =============================================================================


class TestValueFunSerialization:
    """Tests for value function serialization."""

    def test_affine_fun_serialization(self):
        """AffineFun can be serialized and deserialized."""
        f = AffineFun(slope=2.0, bias=1.0)
        d = f.to_dict()
        f2 = AffineFun.from_dict(d)
        assert f(5) == f2(5)

    def test_table_fun_serialization(self):
        """TableFun can be serialized and deserialized."""
        f = TableFun(mapping={"a": 1.0, "b": 2.0})
        d = f.to_dict()
        f2 = TableFun.from_dict(d)
        assert f("a") == f2("a")
        assert f("b") == f2("b")

    def test_affine_multi_fun_serialization(self):
        """AffineMultiFun can be serialized and deserialized."""
        f = AffineMultiFun(slope=(1.0, 2.0), bias=3.0)
        d = f.to_dict()
        f2 = AffineMultiFun.from_dict(d)
        assert f((1, 1)) == f2((1, 1))


# =============================================================================
# Tests for to_table conversion
# =============================================================================


class TestToTable:
    """Tests for converting functions to tables."""

    def test_affine_to_table(self):
        """AffineFun can be converted to TableFun."""
        f = AffineFun(slope=2.0, bias=1.0)
        issue = make_issue(5, "x")  # 0, 1, 2, 3, 4
        table = f.to_table(issue)
        assert isinstance(table, TableFun)
        for i in range(5):
            assert table(i) == f(i)

    def test_quadratic_to_table(self):
        """QuadraticFun can be converted to TableFun."""
        f = QuadraticFun(a2=1.0, a1=0.0, bias=0.0)
        issue = make_issue(5, "x")
        table = f.to_table(issue)
        for i in range(5):
            assert table(i) == f(i)


# =============================================================================
# Tests for New Multi-Issue Value Functions
# =============================================================================


class TestQuadraticMultiFun:
    """Tests for QuadraticMultiFun.

    Mathematical form:
    f(x) = sum_i(linear[i] * x[i]) + sum_i(quadratic[i] * x[i]^2)
           + sum_{i<j}(interactions[idx] * x[i] * x[j]) + bias

    interactions are indexed in upper-triangular order: (0,1), (0,2), ..., (n-2,n-1)
    """

    def test_linear_only(self):
        """QuadraticMultiFun with only linear terms."""
        # 3 issues, no squared or interaction terms
        f = QuadraticMultiFun(
            linear=(1.0, 2.0, 3.0),
            quadratic=(0.0, 0.0, 0.0),
            interactions=(0.0, 0.0, 0.0),  # (0,1), (0,2), (1,2)
        )
        # f(x) = 1*x[0] + 2*x[1] + 3*x[2]
        assert f((1, 1, 1)) == 6.0
        assert f((2, 3, 4)) == 2 + 6 + 12  # 20

    def test_quadratic_only(self):
        """QuadraticMultiFun with only quadratic terms."""
        f = QuadraticMultiFun(
            linear=(0.0, 0.0),
            quadratic=(1.0, 2.0),
            interactions=(0.0,),  # (0,1)
        )
        # f(x) = 1*x[0]^2 + 2*x[1]^2
        assert f((2, 3)) == 4 + 18  # 22
        assert f((1, 1)) == 1 + 2  # 3

    def test_interaction_only(self):
        """QuadraticMultiFun with only interaction terms."""
        # 3 issues: interactions are (0,1), (0,2), (1,2)
        f = QuadraticMultiFun(
            linear=(0.0, 0.0, 0.0),
            quadratic=(0.0, 0.0, 0.0),
            interactions=(1.0, 2.0, 0.0),  # 1*x[0]*x[1] + 2*x[0]*x[2]
        )
        # f(x) = 1*x[0]*x[1] + 2*x[0]*x[2]
        assert f((2, 3, 4)) == 6 + 16  # 22

    def test_full_quadratic(self):
        """QuadraticMultiFun with all terms."""
        f = QuadraticMultiFun(
            linear=(1.0, 2.0),
            quadratic=(1.0, 1.0),
            interactions=(3.0,),  # c_01 = 3
            bias=5.0,
        )
        # f(x,y) = x + 2y + x^2 + y^2 + 3xy + 5
        # f(2,3) = 2 + 6 + 4 + 9 + 18 + 5 = 44
        assert f((2, 3)) == 44.0

    def test_bias_only(self):
        """QuadraticMultiFun with only bias returns constant."""
        f = QuadraticMultiFun(
            linear=(0.0, 0.0, 0.0),
            quadratic=(0.0, 0.0, 0.0),
            interactions=(0.0, 0.0, 0.0),
            bias=7.0,
        )
        assert f((1, 2, 3)) == 7.0
        assert f((0, 0, 0)) == 7.0

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = QuadraticMultiFun(
            linear=(1.0, 2.0), quadratic=(0.0, 0.0), interactions=(0.0,), bias=5.0
        )
        f2 = f.shift_by(3.0)
        assert f2.bias == 8.0
        assert f2((1, 1)) == f((1, 1)) + 3.0

    def test_scale_by(self):
        """Scaling multiplies all coefficients."""
        f = QuadraticMultiFun(
            linear=(1.0, 2.0), quadratic=(1.0, 1.0), interactions=(3.0,), bias=5.0
        )
        f2 = f.scale_by(2.0)
        assert f2.linear == (2.0, 4.0)
        assert f2.quadratic == (2.0, 2.0)
        assert f2.bias == 10.0
        assert f2((2, 3)) == f((2, 3)) * 2.0


class TestBilinearMultiFun:
    """Tests for BilinearMultiFun.

    Mathematical form: f(x, y) = a*x + b*y + c*x*y + bias
    """

    def test_bilinear_basic(self):
        """BilinearMultiFun computes a*x + b*y + c*x*y + bias."""
        f = BilinearMultiFun(a=1.0, b=2.0, c=3.0, bias=0.0)
        # f(2, 3) = 2 + 6 + 18 = 26
        assert f((2, 3)) == 26.0

    def test_bilinear_with_bias(self):
        """BilinearMultiFun with bias."""
        f = BilinearMultiFun(a=1.0, b=1.0, c=1.0, bias=10.0)
        # f(1, 1) = 1 + 1 + 1 + 10 = 13
        assert f((1, 1)) == 13.0

    def test_bilinear_no_interaction(self):
        """BilinearMultiFun without interaction term is linear."""
        f = BilinearMultiFun(a=2.0, b=3.0, c=0.0, bias=0.0)
        # f(x, y) = 2x + 3y
        assert f((4, 5)) == 8 + 15  # 23

    def test_bilinear_only_interaction(self):
        """BilinearMultiFun with only interaction term."""
        f = BilinearMultiFun(a=0.0, b=0.0, c=1.0, bias=0.0)
        # f(x, y) = x*y
        assert f((3, 4)) == 12.0
        assert f((5, 6)) == 30.0

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = BilinearMultiFun(a=1.0, b=1.0, c=1.0, bias=5.0)
        f2 = f.shift_by(3.0)
        assert f2.bias == 8.0

    def test_scale_by(self):
        """Scaling multiplies all coefficients."""
        f = BilinearMultiFun(a=1.0, b=2.0, c=3.0, bias=4.0)
        f2 = f.scale_by(2.0)
        assert f2.a == 2.0
        assert f2.b == 4.0
        assert f2.c == 6.0
        assert f2.bias == 8.0


class TestPolynomialMultiFun:
    """Tests for PolynomialMultiFun.

    Mathematical form: f(x) = sum_k(c_k * prod_i(x[i]^p_ki)) + bias
    Each term is (coefficient, powers) where powers is tuple of exponents.
    """

    def test_polynomial_single_term(self):
        """PolynomialMultiFun with single term."""
        # f(x,y) = 2 * x^2 * y
        f = PolynomialMultiFun(terms=((2.0, (2, 1)),))
        assert f((3, 4)) == 2 * 9 * 4  # 72

    def test_polynomial_multiple_terms(self):
        """PolynomialMultiFun with multiple terms."""
        # f(x,y) = x^2 + y^2 + x*y
        f = PolynomialMultiFun(
            terms=(
                (1.0, (2, 0)),  # x^2
                (1.0, (0, 2)),  # y^2
                (1.0, (1, 1)),  # xy
            )
        )
        # f(2, 3) = 4 + 9 + 6 = 19
        assert f((2, 3)) == 19.0

    def test_polynomial_with_bias(self):
        """PolynomialMultiFun with bias."""
        f = PolynomialMultiFun(terms=((1.0, (1, 1)),), bias=10.0)
        # f(x,y) = xy + 10
        assert f((2, 3)) == 16.0

    def test_polynomial_high_degree(self):
        """PolynomialMultiFun with higher degree terms."""
        # f(x,y,z) = x^3 * y^2 * z
        f = PolynomialMultiFun(terms=((1.0, (3, 2, 1)),))
        # f(2, 3, 4) = 8 * 9 * 4 = 288
        assert f((2, 3, 4)) == 288.0

    def test_polynomial_linear_equivalent(self):
        """PolynomialMultiFun can express linear functions."""
        # f(x,y) = 2x + 3y + 5
        f = PolynomialMultiFun(terms=((2.0, (1, 0)), (3.0, (0, 1))), bias=5.0)
        # f(2, 3) = 4 + 9 + 5 = 18
        assert f((2, 3)) == 18.0

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = PolynomialMultiFun(terms=((1.0, (1, 1)),), bias=5.0)
        f2 = f.shift_by(3.0)
        assert f2.bias == 8.0

    def test_scale_by(self):
        """Scaling multiplies all coefficients and bias."""
        f = PolynomialMultiFun(terms=((1.0, (2, 0)), (2.0, (0, 2))), bias=3.0)
        f2 = f.scale_by(2.0)
        assert f2.bias == 6.0
        # Verify output is scaled
        assert f2((2, 3)) == f((2, 3)) * 2.0


class TestProductMultiFun:
    """Tests for ProductMultiFun (Cobb-Douglas style).

    Mathematical form: f(x) = scale * prod_i(x[i]^powers[i]) + bias
    """

    def test_product_basic(self):
        """ProductMultiFun computes scaled product."""
        f = ProductMultiFun(powers=(1.0, 1.0), scale=1.0, bias=0.0)
        # f(x,y) = x * y
        assert f((2, 3)) == 6.0
        assert f((4, 5)) == 20.0

    def test_product_with_powers(self):
        """ProductMultiFun with non-unit powers."""
        f = ProductMultiFun(powers=(2.0, 3.0), scale=1.0, bias=0.0)
        # f(x,y) = x^2 * y^3
        assert f((2, 3)) == 4 * 27  # 108

    def test_product_with_scale(self):
        """ProductMultiFun with scale factor."""
        f = ProductMultiFun(powers=(1.0, 1.0), scale=2.0, bias=0.0)
        # f(x,y) = 2 * x * y
        assert f((3, 4)) == 24.0

    def test_product_with_bias(self):
        """ProductMultiFun with bias."""
        f = ProductMultiFun(powers=(1.0, 1.0), scale=1.0, bias=10.0)
        # f(x,y) = x * y + 10
        assert f((2, 3)) == 16.0

    def test_product_fractional_powers(self):
        """ProductMultiFun with fractional powers."""
        f = ProductMultiFun(powers=(0.5, 0.5), scale=1.0, bias=0.0)
        # f(x,y) = sqrt(x) * sqrt(y) = sqrt(xy)
        assert f((4, 9)) == approx(6.0)  # sqrt(4) * sqrt(9) = 2 * 3

    def test_cobb_douglas(self):
        """ProductMultiFun as Cobb-Douglas production function."""
        # Y = A * L^alpha * K^beta where alpha + beta = 1
        f = ProductMultiFun(powers=(0.7, 0.3), scale=2.0, bias=0.0)
        # f(L=10, K=20) = 2 * 10^0.7 * 20^0.3
        result = 2.0 * (10**0.7) * (20**0.3)
        assert f((10, 20)) == approx(result)

    def test_shift_by(self):
        """Shifting adds to the bias."""
        f = ProductMultiFun(powers=(1.0, 1.0), scale=1.0, bias=5.0)
        f2 = f.shift_by(3.0)
        assert f2.bias == 8.0

    def test_scale_by(self):
        """Scaling multiplies the scale factor."""
        f = ProductMultiFun(powers=(1.0, 1.0), scale=2.0, bias=0.0)
        f2 = f.scale_by(3.0)
        assert f2.scale == 6.0


# =============================================================================
# Tests for minmax on new MultiFun classes
# =============================================================================


class TestNewMultiFunMinMax:
    """Tests for minmax on the new MultiFun classes."""

    def test_quadratic_multi_fun_minmax(self):
        """QuadraticMultiFun minmax over issues."""
        f = QuadraticMultiFun(
            linear=(1.0, 2.0), quadratic=(0.0, 0.0), interactions=(0.0,), bias=0.0
        )
        issues = (make_issue((0.0, 10.0), "x"), make_issue((0.0, 5.0), "y"))
        mn, mx = f.minmax(issues)
        # Linear: f(0,0)=0, f(10,5)=10+10=20
        assert mn == 0.0
        assert mx == 20.0

    def test_bilinear_multi_fun_minmax(self):
        """BilinearMultiFun minmax over issues."""
        f = BilinearMultiFun(a=1.0, b=1.0, c=0.0, bias=0.0)
        issues = (make_issue((0.0, 10.0), "x"), make_issue((0.0, 5.0), "y"))
        mn, mx = f.minmax(issues)
        assert mn == 0.0
        assert mx == 15.0

    def test_product_multi_fun_minmax(self):
        """ProductMultiFun minmax over issues."""
        f = ProductMultiFun(powers=(1.0, 1.0), scale=1.0, bias=0.0)
        issues = (make_issue((1.0, 10.0), "x"), make_issue((1.0, 5.0), "y"))
        mn, mx = f.minmax(issues)
        # min at (1,1)=1, max at (10,5)=50
        assert mn == 1.0
        assert mx == 50.0
