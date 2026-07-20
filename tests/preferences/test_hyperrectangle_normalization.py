"""Range-normalization of HyperRectangleUtilityFunction (non-linear ufuns).

Ported from negobench: a HyperRectangle ufun is affinely mapped to [0, 1] with
EXACT extrema found via a max-weight clique over the rectangle-overlap graph
(Helly's theorem for axis-aligned boxes), so no enumeration of the (possibly
astronomically large) outcome space is needed.
"""

from __future__ import annotations

import itertools

import pytest

from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences.crisp.nonlinear import HyperRectangleUtilityFunction

EPS = 1e-9


def _brute_extremes(ufun, issues):
    us = [float(ufun(o)) for o in itertools.product(*[list(i.all) for i in issues])]
    return min(us), max(us)


def _small_ufun():
    # Integer issues 0..9; rectangle bounds are fractional so no discrete value
    # ever lands on a boundary (keeps clique membership == outcome_in_range).
    issues = [make_issue(10, "0"), make_issue(10, "1"), make_issue(10, "2")]
    os_ = make_os(issues, name="small")
    f = HyperRectangleUtilityFunction(
        outcome_ranges=[
            {0: (0.5, 4.5), 1: (0.5, 4.5)},
            {1: (2.5, 7.5), 2: (0.5, 3.5)},
            {0: (6.5, 9.5)},
            {0: (0.5, 4.5), 2: (5.5, 9.5)},
        ],
        utilities=[3.0, 5.0, -2.0, 4.0],
        weights=[1.0, 1.0, 1.0, 1.0],
        bias=1.0,
        outcome_space=os_,
        reserved_value=0.5,
    )
    return f, issues


def test_clique_extremes_match_bruteforce():
    f, issues = _small_ufun()
    mn, mx = f._range_extremes()
    bmn, bmx = _brute_extremes(f, issues)
    assert abs(mn - bmn) < EPS, (mn, bmn)
    assert abs(mx - bmx) < EPS, (mx, bmx)
    # minmax() must agree with the clique extremes (no outcomes passed)
    assert f.minmax() == pytest.approx((bmn, bmx), abs=EPS)


def test_normalize_maps_to_unit_and_is_affine():
    f, issues = _small_ufun()
    n = f.normalize_for((0.0, 1.0))
    outs = list(itertools.product(*[list(i.all) for i in issues]))
    u_raw = [float(f(o)) for o in outs]
    u_norm = [float(n(o)) for o in outs]
    # (3,4,10) range and reaches both ends
    assert min(u_norm) == pytest.approx(0.0, abs=1e-9)
    assert max(u_norm) == pytest.approx(1.0, abs=1e-9)
    # (1) single affine map reproduces every outcome
    a = (u_norm[u_raw.index(max(u_raw))] - u_norm[u_raw.index(min(u_raw))]) / (
        max(u_raw) - min(u_raw)
    )
    c = u_norm[u_raw.index(min(u_raw))] - a * min(u_raw)
    for ur, un in zip(u_raw, u_norm):
        assert abs(un - (a * ur + c)) < 1e-9, (ur, un, a, c)
    # (2) reserved value follows the same map
    assert abs(n.reserved_value - (a * f.reserved_value + c)) < 1e-9


def test_large_space_no_enumeration():
    # 30 issues (0..9) -> 10^30 outcomes; must normalize via the clique, not by
    # enumeration. 60 overlapping rectangles.
    n_issues, n_rects = 30, 60
    rng = [(i * 7) % 100 for i in range(n_rects * 4)]  # deterministic pseudo-values
    rects = []
    for r in range(n_rects):
        rect = {}
        for k in range(3):  # each rectangle constrains 3 issues
            iss = (r * 3 + k) % n_issues
            lo = rng[(r * 4 + k) % len(rng)] % 6
            rect[iss] = (float(lo), float(lo + 3))
        rects.append(rect)
    issues = [make_issue(10, str(i)) for i in range(n_issues)]
    os_ = make_os(issues, name="big")
    f = HyperRectangleUtilityFunction(
        outcome_ranges=rects,
        utilities=[float((r % 9) + 1) for r in range(n_rects)],
        weights=[1.0] * n_rects,
        bias=0.0,
        outcome_space=os_,
        reserved_value=0.0,
    )
    mn, mx = f._range_extremes()  # returns quickly despite 10^30 outcomes
    assert mx > mn
    n = f.normalize_for((0.0, 1.0))
    assert n.minmax() == pytest.approx((0.0, 1.0), abs=1e-9)


def test_serialization_round_trip_preserves_ranges():
    # to_dict/from_dict turns range tuples into lists; _coerce_ranges must restore
    # them so eval keeps treating them as ranges (not discrete value sets).
    f, issues = _small_ufun()
    d = f.to_dict()
    f2 = HyperRectangleUtilityFunction.from_dict(d)
    outs = list(itertools.product(*[list(i.all) for i in issues]))
    for o in outs:
        assert float(f(o)) == pytest.approx(float(f2(o)), abs=1e-9)
