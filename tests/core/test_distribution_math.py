"""Exhaustive tests for the `Distribution` arithmetic/algebra design.

These tests pin down the exact math implemented by `ScipyDistribution` and
`Real` for the two supported families:

- Uniform ``U(loc=a, scale=w)`` with support ``[a, a + w]`` (scipy convention:
  ``loc`` is the lower bound, ``scale`` is the width). Mean ``a + w/2``.
- Normal ``N(loc=m, scale=s)``: ``loc`` is the mean, ``scale`` the std-dev.

Every operator is expected to return a NEW distribution (operators never mutate
their operands) and to only combine distributions of the *same* family.
"""

from __future__ import annotations

import math

import pytest

from negmas.helpers.prob import (
    NORMAL,
    UNIFORM,
    Real,
    ScipyDistribution,
    UniformDistribution,
    NormalDistribution,
    canonical_distribution_type,
    make_distribution,
)

EPS = 1e-9


def U(a, w):
    return ScipyDistribution(UNIFORM, loc=a, scale=w)


def N(m, s):
    return ScipyDistribution(NORMAL, loc=m, scale=s)


# --------------------------------------------------------------------------- #
# type-name canonicalization (single source of truth)                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "alias,canonical",
    [
        ("uniform", UNIFORM),
        ("Uniform", UNIFORM),
        ("unif", UNIFORM),
        ("U", UNIFORM),
        ("norm", NORMAL),
        ("normal", NORMAL),
        ("Normal", NORMAL),
        ("gaussian", NORMAL),
        ("GAUSS", NORMAL),
    ],
)
def test_canonical_type(alias, canonical):
    assert canonical_distribution_type(alias) == canonical


def test_type_names_agree_across_constructors():
    assert ScipyDistribution("norm", loc=0, scale=1).type() == NORMAL
    assert ScipyDistribution("normal", loc=0, scale=1).type() == NORMAL
    assert NormalDistribution(loc=0, scale=1).type() == NORMAL
    assert UniformDistribution(loc=0, scale=1).type() == UNIFORM
    assert NormalDistribution(loc=0, scale=1).is_gaussian()
    assert UniformDistribution(loc=0, scale=1).is_uniform()


# --------------------------------------------------------------------------- #
# accessors: mean / min / max / loc / scale                                   #
# --------------------------------------------------------------------------- #
def test_uniform_accessors():
    u = U(0.2, 0.4)  # support [0.2, 0.6]
    assert abs(u.loc - 0.2) < EPS
    assert abs(u.scale - 0.4) < EPS
    assert abs(u.mean() - 0.4) < EPS  # a + w/2
    assert abs(float(u) - 0.4) < EPS
    assert abs(u.min - 0.2) < EPS  # true lower bound
    assert abs(u.max - 0.6) < EPS  # true upper bound


def test_normal_accessors():
    n = N(1.0, 2.0)
    assert abs(n.mean() - 1.0) < EPS  # loc is the mean
    assert abs(float(n) - 1.0) < EPS
    assert abs(n.min - (-1.0)) < EPS  # m - s heuristic
    assert abs(n.max - 3.0) < EPS  # m + s heuristic


def test_normal_mean_no_longer_raises():
    # regression: mean() used to raise NotImplementedError for non-uniform.
    assert N(5.0, 1.0).mean() == 5.0


def test_crisp_mean():
    assert U(0.3, 0.0).mean() == 0.3
    assert N(0.3, 0.0).mean() == 0.3


# --------------------------------------------------------------------------- #
# scalar shift (add / sub)                                                    #
# --------------------------------------------------------------------------- #
def test_uniform_scalar_shift():
    u = U(0.2, 0.4)
    a = u + 1.0
    assert (abs(a.loc - 1.2) < EPS) and (abs(a.scale - 0.4) < EPS)
    s = u - 0.1
    assert (abs(s.loc - 0.1) < EPS) and (abs(s.scale - 0.4) < EPS)
    # reflected
    assert abs((1.0 + u).loc - 1.2) < EPS
    r = 1.0 - u  # c - X : support [c-max, c-min] -> U(0.4, 0.4)
    assert (abs(r.loc - 0.4) < EPS) and (abs(r.scale - 0.4) < EPS)
    assert abs(r.mean() - (1.0 - u.mean())) < EPS


def test_normal_scalar_shift():
    n = N(1.0, 2.0)
    assert abs((n + 3.0).loc - 4.0) < EPS and abs((n + 3.0).scale - 2.0) < EPS
    assert abs((5.0 - n).loc - 4.0) < EPS and abs((5.0 - n).scale - 2.0) < EPS


# --------------------------------------------------------------------------- #
# scalar multiplication                                                       #
# --------------------------------------------------------------------------- #
def test_uniform_scalar_mul():
    u = U(0.2, 0.4)  # support [0.2, 0.6], mean 0.4
    m = 0.5 * u
    assert abs(m.loc - 0.1) < EPS and abs(m.scale - 0.2) < EPS
    assert abs(m.mean() - 0.2) < EPS
    # negative weight flips the support but keeps a non-negative scale
    neg = -1.0 * u
    assert neg.scale >= 0
    assert abs(neg.min - (-0.6)) < EPS and abs(neg.max - (-0.2)) < EPS
    assert abs(neg.mean() - (-0.4)) < EPS


def test_normal_scalar_mul():
    n = N(1.0, 2.0)
    m = 3.0 * n
    assert abs(m.loc - 3.0) < EPS and abs(m.scale - 6.0) < EPS
    neg = -2.0 * n
    assert abs(neg.loc - (-2.0)) < EPS and abs(neg.scale - 4.0) < EPS  # |k|*s


# --------------------------------------------------------------------------- #
# same-family sum / difference                                                #
# --------------------------------------------------------------------------- #
def test_uniform_sum_support_add():
    a, b = U(0.2, 0.4), U(0.1, 0.2)
    s = a + b  # loc adds, scale adds (exact support of the sum)
    assert abs(s.loc - 0.3) < EPS and abs(s.scale - 0.6) < EPS
    assert abs(s.mean() - (a.mean() + b.mean())) < EPS


def test_uniform_diff():
    a, b = U(0.2, 0.4), U(0.1, 0.2)  # supports [0.2,0.6], [0.1,0.3]
    d = a - b  # support [a1-max2, max1-a2] = [-0.1, 0.5]
    assert abs(d.loc - (-0.1)) < EPS and abs(d.scale - 0.6) < EPS
    assert abs(d.mean() - (a.mean() - b.mean())) < EPS


def test_convex_combination_mean_exact():
    # The elicitation hot path: p*u + (1-p)*v. Mean must be exact.
    u, v, p = U(0.2, 0.4), U(0.1, 0.2), 0.6
    c = p * u + (1 - p) * v
    assert abs(float(c) - (p * u.mean() + (1 - p) * v.mean())) < EPS


def test_normal_sum_variance_add():
    n = N(0.0, 3.0) + N(0.0, 4.0)
    assert abs(n.loc - 0.0) < EPS
    assert abs(n.scale - 5.0) < EPS  # sqrt(9+16)
    d = N(2.0, 3.0) - N(1.0, 4.0)
    assert abs(d.loc - 1.0) < EPS and abs(d.scale - 5.0) < EPS


def test_mixed_family_add_raises():
    with pytest.raises(TypeError):
        _ = U(0.0, 1.0) + N(0.0, 1.0)
    with pytest.raises(TypeError):
        _ = N(0.0, 1.0) - U(0.0, 1.0)


# --------------------------------------------------------------------------- #
# intersection (&)                                                            #
# --------------------------------------------------------------------------- #
def test_uniform_intersection_overlap():
    i = U(0.2, 0.4) & U(0.5, 0.3)  # [0.2,0.6] & [0.5,0.8] = [0.5,0.6]
    assert abs(i.loc - 0.5) < EPS and abs(i.scale - 0.1) < EPS


def test_uniform_intersection_symmetric():
    x, y = U(0.2, 0.4), U(0.5, 0.3)
    a, b = x & y, y & x
    assert abs(a.loc - b.loc) < EPS and abs(a.scale - b.scale) < EPS


def test_uniform_intersection_empty_is_guarded():
    # disjoint supports must NOT yield a negative scale (np.seterr(all='raise'))
    e = U(0.0, 0.1) & U(0.5, 0.1)
    assert e.scale == 0.0
    assert e.scale >= 0


def test_normal_intersection_precision_weighted():
    # product of two identical Gaussians N(0,1): var 1/(1+1)=0.5 -> scale sqrt(0.5)
    i = N(0.0, 1.0) & N(0.0, 1.0)
    assert abs(i.loc - 0.0) < EPS
    assert abs(i.scale - math.sqrt(0.5)) < EPS
    # different means: precision-weighted average, here symmetric -> midpoint
    j = N(0.0, 1.0) & N(2.0, 1.0)
    assert abs(j.loc - 1.0) < EPS


def test_intersection_mixed_family_raises():
    with pytest.raises(TypeError):
        _ = U(0.0, 1.0) & N(0.0, 1.0)


# --------------------------------------------------------------------------- #
# immutability: operators never mutate operands                              #
# --------------------------------------------------------------------------- #
def test_operators_do_not_mutate():
    u = U(0.2, 0.4)
    before = (u.loc, u.scale)
    _ = u + 1.0
    _ = u * 2.0
    _ = u - 0.1
    _ = u & U(0.3, 0.4)
    _ = u + U(0.1, 0.2)
    assert (u.loc, u.scale) == before


# --------------------------------------------------------------------------- #
# Real (delta / point mass)                                                    #
# --------------------------------------------------------------------------- #
def test_real_is_crisp_scalar():
    r = Real(0.5)
    assert r.is_crisp()
    assert float(r) == 0.5
    assert r.mean() == 0.5
    assert r.min == r.max == 0.5


def test_real_scalar_arithmetic():
    r = Real(0.5)
    assert float(r + 0.25) == 0.75
    assert float(r - 0.25) == 0.25
    assert float(r * 2.0) == 1.0
    assert float(2.0 * r) == 1.0
    assert float(1.0 - r) == 0.5
    assert float(0.25 + r) == 0.75


def test_real_with_distribution_shifts_and_scales():
    r, u = Real(0.5), U(0.2, 0.4)
    # delta(0.5) + U shifts U by 0.5
    a = r + u
    assert abs(a.loc - 0.7) < EPS and abs(a.scale - 0.4) < EPS
    # delta(0.5) * U scales U by 0.5
    m = r * u
    assert abs(m.loc - 0.1) < EPS and abs(m.scale - 0.2) < EPS


def test_real_intersection_returns_point():
    i = Real(0.5) & U(0.0, 1.0)
    assert i.is_crisp() and float(i) == 0.5


def test_crisp_distribution_acts_as_scalar_in_add():
    # A near-zero-scale ScipyDistribution should shift, not require same family.
    crisp = U(0.5, 0.0)
    n = N(1.0, 2.0)
    shifted = n + crisp
    assert abs(shifted.loc - 1.5) < EPS and abs(shifted.scale - 2.0) < EPS


# --------------------------------------------------------------------------- #
# comparisons and helpers                                                     #
# --------------------------------------------------------------------------- #
def test_always_less_greater():
    assert U(0.0, 0.2) < U(0.5, 0.2)  # [0,0.2] entirely below [0.5,0.7]
    assert U(0.5, 0.2) > U(0.0, 0.2)
    assert not (U(0.0, 0.5) < U(0.3, 0.5))  # overlapping


def test_make_distribution():
    assert isinstance(make_distribution(0.5), Real)
    u = U(0.2, 0.4)
    assert make_distribution(u) is u


def test_scipy_prob_uses_pdf():
    # regression: prob() used to call the non-existent scipy `.prob` method.
    u = U(0.0, 1.0)  # density 1/w = 1.0 inside support
    assert abs(u.prob(0.5) - 1.0) < EPS
    assert u.prob(2.0) == 0.0
    assert abs(u.cum_prob(0.0, 0.5) - 0.5) < EPS


def test_real_prob_indicator():
    # regression: Real.prob used a broken chained comparison.
    r = Real(0.5)
    assert r.prob(0.5) == 1.0
    assert r.prob(0.9) == 0.0
    assert r.cum_prob(0.0, 1.0) == 1
    assert r.cum_prob(0.6, 1.0) == 0


def test_equal_same_family_distributions():
    # regression: __eq__ compared bound methods, so equal dists were unequal.
    assert U(0.2, 0.4) == U(0.2, 0.4)
    assert N(1.0, 2.0) == N(1.0, 2.0)
    assert U(0.2, 0.4) != U(0.2, 0.5)
