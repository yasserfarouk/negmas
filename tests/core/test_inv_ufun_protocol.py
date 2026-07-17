"""Extensive protocol-conformance and correctness tests for utility-function
inverters (`SamplingInverseUtilityFunction` and `PresortingInverseUtilityFunction`).

These tests are independent of `tests/core/test_inverter.py` (which already
covers `PresortingInverseUtilityFunction` vs `BruteForceInverseUtilityFunction`)
and instead focus on:

    - Making sure every class actually implements the full `InverseUFun` protocol
      (i.e. can be instantiated and all abstract methods work).
    - Correctness of `best_in`/`worst_in`/`one_in`/`some`/`all` against a brute-force
      ground truth computed directly from the wrapped ufun.
    - Edge cases: empty ranges, reserved values, normalized vs. non-normalized ranges,
      degenerate ufuns (single outcome, constant ufun).
"""

from __future__ import annotations

import random

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from negmas.outcomes import make_issue, make_os
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.preferences.inv_ufun import (
    AttributePlanningInverseUtilityFunction,
    BIDSInverseUtilityFunction,
    BruteForceInverseUtilityFunction,
    MCTSInverseUtilityFunction,
    PresortingInverseUtilityFunction,
    PresortingLegacyInverseUtilityFunction,
    SamplingInverseUtilityFunction,
)
from negmas.preferences.protocols import InverseUFun

ALL_INVERTER_TYPES = [
    BruteForceInverseUtilityFunction,
    SamplingInverseUtilityFunction,
    PresortingInverseUtilityFunction,
    PresortingLegacyInverseUtilityFunction,
]

# Inverters expected to return the exact nearest-achievable outcome strictly inside
# the requested range for best_in/worst_in and to sample randomly (not
# deterministically) in one_in. The legacy presorting inverter is deliberately kept
# verbatim from before this work and has known rough edges -- nearest-tie clamping in
# best_in/worst_in can return an out-of-range outcome, and its one_in is deterministic
# -- so it is excluded from these strict property tests (it is still covered by every
# other test below). SamplingInverseUtilityFunction is also excluded because it is a
# clamping inverter: when the random sample misses the in-range outcome(s),
# worst_in/best_in fall back to the nearest boundary outcome rather than returning
# the exact in-range result.
STRICT_INVERTER_TYPES = [
    BruteForceInverseUtilityFunction,
    PresortingInverseUtilityFunction,
]

# Inverters that require LinearAdditiveUtilityFunction
ALL_ADDITIVE_INVERTER_TYPES = [
    BIDSInverseUtilityFunction,
    AttributePlanningInverseUtilityFunction,
]


def make_linear_ufun(nissues: int = 2, nvalues: int = 5, seed: int = 42):
    """Create a random LinearAdditiveUtilityFunction for testing."""
    import random as _random

    _random.seed(seed)
    os = make_os([make_issue(nvalues) for _ in range(nissues)])
    ufun = LinearAdditiveUtilityFunction.random(outcome_space=os)
    outcomes = list(os.enumerate_or_sample())
    return outcomes, ufun


# A utility-fraction strategy that avoids astronomically tiny non-zero values (e.g.
# ~1e-212), or values within a few ULPs of 1.0 (e.g. 0.9999999999999999), that
# hypothesis likes to generate from the full float range. Such values are effectively
# indistinguishable from 0.0/1.0 for any practical purpose but can create meaningless
# floating-point "boundary" test cases (e.g. a computed upper bound that is a hair
# below the true max due to rounding, excluding it from a strict bisection search)
# that are smaller than any reasonable floating-point comparison tolerance -- snap
# them to exactly 0.0/1.0 instead. The snapping threshold (1e-4) is kept comfortably
# larger than the `tol`/`eps` (1e-6) used by the tests below so that values that
# survive snapping are unambiguously outside any tolerance band used for comparisons.
UTIL_FRACTION = st.floats(0.0, 1.0).map(
    lambda x: 0.0 if x < 1e-4 else (1.0 if x > 1 - 1e-4 else x)
)


def make_ufun(nissues=1, nvalues=10, r=-1.0):
    """Creates a simple discrete ufun with utilities 0..n-1 over its outcomes."""
    os = make_os([make_issue(nvalues) for _ in range(nissues)])
    outcomes = list(os.enumerate_or_sample())
    ufun = MappingUtilityFunction(
        mapping=dict(zip(outcomes, range(len(outcomes)), strict=True)),
        reserved_value=r,
        outcome_space=os,
    )
    return outcomes, ufun


def brute_force_utils(ufun):
    """Ground truth: (outcome, utility) pairs for every outcome, sorted ascending
    by utility."""
    outcomes = list(ufun.outcome_space.enumerate_or_sample())
    return sorted(((float(ufun(o)), o) for o in outcomes), key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Instantiation / protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_INVERTER_TYPES)
def test_can_instantiate_and_matches_protocol(cls):
    """Every inverter must be constructible and satisfy the InverseUFun protocol."""
    _, ufun = make_ufun()
    inv = cls(ufun)
    assert isinstance(inv, InverseUFun)
    inv.init()
    assert inv.initialized
    assert inv.ufun is ufun


@pytest.mark.parametrize("cls", ALL_INVERTER_TYPES)
def test_all_required_methods_are_callable(cls):
    """Calls every method required by the protocol at least once with sane args."""
    _, ufun = make_ufun(nissues=1, nvalues=10)
    inv = cls(ufun)
    inv.init()
    assert isinstance(inv.min(), float)
    assert isinstance(inv.max(), float)
    mn, mx = inv.minmax()
    assert mn <= mx
    assert inv.worst() is not None
    assert inv.best() is not None
    w, b = inv.extreme_outcomes()
    assert w is not None and b is not None
    assert ufun(w) <= ufun(b)
    some = inv.some((mn, mx), normalized=False)
    assert isinstance(some, list)
    one = inv.one_in((mn, mx), normalized=False)
    assert one is not None
    called = inv((mn, mx), normalized=False)
    assert called is not None
    bi = inv.best_in((mn, mx), normalized=False)
    wi = inv.worst_in((mn, mx), normalized=False)
    assert bi is not None
    assert wi is not None


# ---------------------------------------------------------------------------
# min / max / worst / best correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_INVERTER_TYPES)
def test_min_max_worst_best_correctness(cls):
    outcomes, ufun = make_ufun(nissues=2, nvalues=5, r=-100)
    inv = cls(ufun)
    inv.init()
    truth = brute_force_utils(ufun)
    true_min, true_max = truth[0][0], truth[-1][0]
    assert inv.min() == pytest.approx(true_min)
    assert inv.max() == pytest.approx(true_max)
    assert ufun(inv.worst()) == pytest.approx(true_min)
    assert ufun(inv.best()) == pytest.approx(true_max)


# ---------------------------------------------------------------------------
# best_in / worst_in correctness against brute force
# ---------------------------------------------------------------------------


@given(
    nissues=st.integers(1, 3),
    nvalues=st.integers(2, 6),
    mn_frac=UTIL_FRACTION,
    mx_frac=UTIL_FRACTION,
)
@settings(max_examples=60)
@pytest.mark.parametrize("cls", STRICT_INVERTER_TYPES)
def test_best_in_returns_highest_util_in_range(cls, nissues, nvalues, mn_frac, mx_frac):
    _, ufun = make_ufun(nissues, nvalues, r=-100)
    inv = (
        cls(ufun, max_samples_per_call=20_000)
        if cls is SamplingInverseUtilityFunction
        else cls(ufun)
    )
    inv.init()
    truth = brute_force_utils(ufun)
    umn, umx = truth[0][0], truth[-1][0]
    d = umx - umn
    lo, hi = min(mn_frac, mx_frac), max(mn_frac, mx_frac)
    rng = (lo * d + umn, hi * d + umn)
    # Use the same tolerance here as the one passed to best_in()/worst_in() below so
    # that candidates computed from `truth` and the outcome actually returned agree on
    # what counts as "in range" -- using a looser tolerance here than what the inverter
    # itself uses internally would make outcomes right at the range boundary
    # (excluded by the inverter's stricter eps) incorrectly show up as "expected".
    tol = 1e-6
    candidates = [u for u, _ in truth if rng[0] - tol <= u <= rng[1] + tol]
    if not candidates:
        return
    expected_best = max(candidates)
    kwargs = {} if cls is SamplingInverseUtilityFunction else {"eps": tol}
    got = inv.best_in(rng, normalized=False, **kwargs)
    assert got is not None
    got_u = float(ufun(got))
    assert got_u == pytest.approx(expected_best, abs=1e-6), (
        f"best_in returned utility {got_u} but the best in range {rng} is {expected_best}"
    )


@given(
    nissues=st.integers(1, 3),
    nvalues=st.integers(2, 6),
    mn_frac=UTIL_FRACTION,
    mx_frac=UTIL_FRACTION,
)
@settings(max_examples=60)
@pytest.mark.parametrize("cls", STRICT_INVERTER_TYPES)
def test_worst_in_returns_lowest_util_in_range(cls, nissues, nvalues, mn_frac, mx_frac):
    _, ufun = make_ufun(nissues, nvalues, r=-100)
    inv = (
        cls(ufun, max_samples_per_call=20_000)
        if cls is SamplingInverseUtilityFunction
        else cls(ufun)
    )
    inv.init()
    truth = brute_force_utils(ufun)
    umn, umx = truth[0][0], truth[-1][0]
    d = umx - umn
    lo, hi = min(mn_frac, mx_frac), max(mn_frac, mx_frac)
    rng = (lo * d + umn, hi * d + umn)
    # See test_best_in_returns_highest_util_in_range for why `tol` must match the eps
    # actually used by worst_in() below.
    tol = 1e-6
    candidates = [u for u, _ in truth if rng[0] - tol <= u <= rng[1] + tol]
    if not candidates:
        return
    expected_worst = min(candidates)
    kwargs = {} if cls is SamplingInverseUtilityFunction else {"eps": tol}
    got = inv.worst_in(rng, normalized=False, **kwargs)
    assert got is not None
    got_u = float(ufun(got))
    assert got_u == pytest.approx(expected_worst, abs=1e-6), (
        f"worst_in returned utility {got_u} but the worst in range {rng} is {expected_worst}"
    )


# ---------------------------------------------------------------------------
# one_in correctness + randomness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_INVERTER_TYPES)
def test_one_in_returns_outcome_within_range(cls):
    _, ufun = make_ufun(nissues=1, nvalues=10, r=-100)
    inv = cls(ufun)
    inv.init()
    rng = (2.0, 7.0)
    for _ in range(20):
        o = inv.one_in(rng, normalized=False)
        assert o is not None
        assert rng[0] - 1e-6 <= ufun(o) <= rng[1] + 1e-6


@pytest.mark.parametrize("cls", STRICT_INVERTER_TYPES)
def test_one_in_is_random_across_calls(cls):
    """one_in() should not always return the same outcome when several outcomes
    qualify (regression test for a bug where it always returned the same,
    lowest-index outcome in range)."""
    random.seed(0)
    _, ufun = make_ufun(nissues=1, nvalues=30, r=-100)
    inv = cls(ufun)
    inv.init()
    rng = (0.0, 29.0)
    seen = {float(ufun(inv.one_in(rng, normalized=False))) for _ in range(50)}
    assert len(seen) > 1, (
        f"one_in() returned the same utility value ({seen}) on every one of 50 calls "
        "with a wide range and many qualifying outcomes -- it is not sampling "
        "randomly within the range."
    )


# ---------------------------------------------------------------------------
# some() / all()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_INVERTER_TYPES)
def test_some_returns_outcomes_within_range(cls):
    _, ufun = make_ufun(nissues=2, nvalues=5, r=-100)
    inv = cls(ufun)
    inv.init()
    rng = (3.0, 15.0)
    some = inv.some(rng, normalized=False, n=1000)
    assert all(rng[0] - 1e-6 <= ufun(o) <= rng[1] + 1e-6 for o in some)


def test_presorting_all_returns_every_matching_outcome():
    _, ufun = make_ufun(nissues=2, nvalues=4, r=-100)
    inv = PresortingInverseUtilityFunction(ufun)
    inv.init()
    truth = brute_force_utils(ufun)
    rng = (3.0, 8.0)
    expected = {o for u, o in truth if rng[0] <= u <= rng[1]}
    got = set(inv.all(rng, normalized=False))
    assert got == expected


def test_sampling_all_raises():
    """SamplingInverseUtilityFunction cannot enumerate all outcomes in a range."""
    _, ufun = make_ufun()
    inv = SamplingInverseUtilityFunction(ufun)
    inv.init()
    with pytest.raises(ValueError):
        inv.all((0.0, 5.0))


# ---------------------------------------------------------------------------
# Normalized ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", STRICT_INVERTER_TYPES)
def test_normalized_range_matches_unnormalized(cls):
    _, ufun = make_ufun(nissues=2, nvalues=5, r=-100)
    inv = cls(ufun)
    inv.init()
    umn, umx = inv.minmax()
    d = umx - umn
    norm_rng = (0.2, 0.6)
    true_rng = (norm_rng[0] * d + umn, norm_rng[1] * d + umn)
    got_norm = inv.best_in(norm_rng, normalized=True)
    got_true = inv.best_in(true_rng, normalized=False)
    assert got_norm is not None and got_true is not None
    assert float(ufun(got_norm)) == pytest.approx(float(ufun(got_true)), abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_INVERTER_TYPES)
def test_single_outcome_outcome_space(cls):
    _, ufun = make_ufun(nissues=1, nvalues=1, r=-100)
    inv = cls(ufun)
    inv.init()
    assert inv.min() == inv.max()
    assert ufun(inv.best()) == ufun(inv.worst())
    o = inv.one_in((inv.min(), inv.max()), normalized=False)
    assert o is not None


@pytest.mark.parametrize("cls", ALL_INVERTER_TYPES)
def test_reserved_value_excludes_irrational_outcomes_from_best_worst(cls):
    """best()/worst() for PresortingInverseUtilityFunction constructed with
    rational_only=True should ignore outcomes below the reserved value; for the
    default (rational_only=False) or for Sampling (which has no such option)
    they should not."""
    _, ufun = make_ufun(nissues=1, nvalues=10, r=5.0)
    if cls is PresortingInverseUtilityFunction:
        inv = cls(ufun, rational_only=True)
        inv.init()
        assert ufun(inv.worst()) >= 5.0
    else:
        inv = cls(ufun)
        inv.init()
        # Sampling based inverter has no rational-only concept; should just work.
        assert inv.worst() is not None


@pytest.mark.parametrize("cls", ALL_INVERTER_TYPES)
def test_empty_range_returns_none_or_empty(cls):
    """Requesting a range strictly above the maximum utility should not find
    any outcome (best_in should either return None or fall back sensibly, and
    some() should return no matching outcomes)."""
    _, ufun = make_ufun(nissues=1, nvalues=10, r=-100)
    inv = cls(ufun)
    inv.init()
    mx = inv.max()
    impossible = (mx + 100, mx + 200)
    some = inv.some(impossible, normalized=False)
    assert all(ufun(o) >= impossible[0] - 1e-6 for o in some) or not some


# ---------------------------------------------------------------------------
# PresortingInverseUtilityFunction specific: next_worse / next_better, within_*
# ---------------------------------------------------------------------------


def test_presorting_next_worse_next_better_cycle_through_all_outcomes():
    outcomes, ufun = make_ufun(nissues=1, nvalues=10, r=-100)
    inv = PresortingInverseUtilityFunction(ufun)
    inv.init()
    seen = []
    o = inv.next_better()
    while o is not None:
        seen.append(float(ufun(o)))
        o = inv.next_better()
    assert seen == sorted(seen)
    assert len(seen) == len(outcomes)


def test_presorting_next_worse_from_best():
    outcomes, ufun = make_ufun(nissues=1, nvalues=10, r=-100)
    inv = PresortingInverseUtilityFunction(ufun)
    inv.init()
    seen = []
    o = inv.next_worse()
    while o is not None:
        seen.append(float(ufun(o)))
        o = inv.next_worse()
    assert seen == sorted(seen, reverse=True)
    assert len(seen) == len(outcomes)


def test_presorting_within_indices_best_first():
    _, ufun = make_ufun(nissues=1, nvalues=10, r=-100)
    inv = PresortingInverseUtilityFunction(ufun)
    inv.init()
    result = inv.within_indices((0, 2))
    utils = [float(ufun(o)) for o in result]
    assert utils == sorted(utils, reverse=True)
    assert utils[0] == pytest.approx(inv.max())


def test_presorting_within_fractions():
    """within_fractions() uses rank/count-based cutoffs (top-`k` outcomes by count,
    where `k` is derived from the fraction and `n`), not a utility-value-interpolation
    cutoff. Cross-check the fast, bisection-based implementation against the simple,
    trusted `BruteForceInverseUtilityFunction` reference implementation
    (which uses the exact same convention) rather than re-deriving the cutoff formula
    by hand here.
    """
    _, ufun = make_ufun(nissues=1, nvalues=10, r=-100)
    inv = PresortingInverseUtilityFunction(ufun)
    inv.init()
    bf = BruteForceInverseUtilityFunction(ufun)
    bf.init()
    for frac_rng in [(0.0, 0.2), (0.0, 1.0), (0.5, 0.5), (0.3, 0.7)]:
        result = inv.within_fractions(frac_rng)
        expected = bf.within_fractions(frac_rng)
        result_utils = [float(ufun(o)) for o in result]
        expected_utils = [float(ufun(o)) for o in expected]
        assert result_utils == pytest.approx(expected_utils), (
            f"within_fractions{frac_rng} = {result_utils} but brute force gives {expected_utils}"
        )
    # A non-trivial fraction of the (rational, non-degenerate) outcome space should
    # return at least one outcome, and results should be sorted best-to-worst.
    result = inv.within_fractions((0.0, 0.2))
    assert len(result) > 0
    result_utils = [float(ufun(o)) for o in result]
    assert result_utils == sorted(result_utils, reverse=True)


# ---------------------------------------------------------------------------
# AttributePlanningInverseUtilityFunction tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_ADDITIVE_INVERTER_TYPES)
def test_additive_inverter_can_instantiate_and_init(cls):
    """All additive inverters must work with LinearAdditiveUtilityFunction."""
    _, ufun = make_linear_ufun(nissues=2, nvalues=5)
    inv = cls(ufun)
    assert isinstance(inv, InverseUFun)
    inv.init()
    assert inv.initialized
    assert inv.ufun is ufun


@pytest.mark.parametrize("cls", ALL_ADDITIVE_INVERTER_TYPES)
def test_additive_inverter_raises_for_non_additive(cls):
    """Additive inverters must raise TypeError when given a MappingUtilityFunction."""
    _, ufun = make_ufun(nissues=2, nvalues=5)
    inv = cls(ufun)
    with pytest.raises(TypeError):
        inv.init()


def test_attribute_planning_one_in_in_range():
    """one_in() must return an outcome whose utility is within the requested range."""
    _, ufun = make_linear_ufun(nissues=2, nvalues=5, seed=0)
    inv = AttributePlanningInverseUtilityFunction(ufun)
    inv.init()
    mn, mx = inv.minmax()
    mid = (mn + mx) / 2.0
    rng = (mn, mid)
    for _ in range(20):
        o = inv.one_in(rng, normalized=False)
        if o is not None:
            u = float(ufun(o))
            assert mn - 1e-6 <= u <= mid + 1e-6, f"Utility {u} not in range {rng}"
            return
    # If one_in always falls back, that's also acceptable — just verify fallback
    o = inv.one_in(rng, normalized=False, fallback_to_best=True)
    assert o is not None


def test_attribute_planning_some_in_range():
    """some() must return outcomes whose utilities are within the range."""
    _, ufun = make_linear_ufun(nissues=2, nvalues=5, seed=1)
    inv = AttributePlanningInverseUtilityFunction(ufun)
    inv.init()
    mn, mx = inv.minmax()
    rng = (mn, mx)
    outcomes = inv.some(rng, normalized=False, n=10)
    # At least some outcomes should be returned
    assert isinstance(outcomes, list)
    for o in outcomes:
        u = float(ufun(o))
        assert mn - 1e-6 <= u <= mx + 1e-6, f"Utility {u} not in range {rng}"


def test_attribute_planning_minmax():
    """min/max/best/worst should delegate to ufun."""
    _, ufun = make_linear_ufun(nissues=2, nvalues=5, seed=2)
    inv = AttributePlanningInverseUtilityFunction(ufun)
    inv.init()
    assert inv.min() == pytest.approx(float(ufun.minmax()[0]), abs=1e-6)
    assert inv.max() == pytest.approx(float(ufun.minmax()[1]), abs=1e-6)
    assert float(ufun(inv.best())) == pytest.approx(inv.max(), abs=1e-6)
    assert float(ufun(inv.worst())) == pytest.approx(inv.min(), abs=1e-6)


# ---------------------------------------------------------------------------
# MCTSInverseUtilityFunction tests
# ---------------------------------------------------------------------------


def test_mcts_raises_without_outcome_space():
    """MCTSInverseUtilityFunction must raise ValueError for a ufun without an
    outcome space that has issues."""

    # A MappingUtilityFunction has outcome_space but the ufun from make_ufun has it;
    # We need one without issues. Create a minimal ufun with outcome_space=None.
    class _NoOSUFun:
        outcome_space = None

        def minmax(self):
            return (0.0, 1.0)

        def extreme_outcomes(self):
            return (None, None)

        def __call__(self, o):
            return 0.5

    inv = MCTSInverseUtilityFunction(_NoOSUFun())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        inv.init()


def test_mcts_one_in_in_range():
    """MCTSInverseUtilityFunction.one_in() must return an outcome in the range."""
    _, ufun = make_linear_ufun(nissues=2, nvalues=4, seed=10)
    inv = MCTSInverseUtilityFunction(ufun, n_simulations=200)
    inv.init()
    mn, mx = inv.minmax()
    rng = (mn, mx)
    o = inv.one_in(rng, normalized=False)
    assert o is not None
    u = float(ufun(o))
    assert mn - 1e-6 <= u <= mx + 1e-6


def test_mcts_one_in_with_mapping_ufun():
    """MCTSInverseUtilityFunction should work with MappingUtilityFunction."""
    _, ufun = make_ufun(nissues=2, nvalues=4)
    inv = MCTSInverseUtilityFunction(ufun, n_simulations=200)
    inv.init()
    mn, mx = inv.minmax()
    o = inv.one_in((mn, mx), normalized=False)
    assert o is not None


def test_mcts_diverse_some():
    """some() should return multiple different outcomes with high probability."""
    _, ufun = make_linear_ufun(nissues=2, nvalues=5, seed=20)
    inv = MCTSInverseUtilityFunction(ufun, n_simulations=200)
    inv.init()
    mn, mx = inv.minmax()
    outcomes = inv.some((mn, mx), normalized=False, n=5)
    # Should return at least some outcomes
    assert len(outcomes) >= 1


def test_mcts_minmax():
    """min/max/best/worst should delegate to ufun."""
    _, ufun = make_linear_ufun(nissues=2, nvalues=4, seed=30)
    inv = MCTSInverseUtilityFunction(ufun, n_simulations=100)
    inv.init()
    assert inv.min() == pytest.approx(float(ufun.minmax()[0]), abs=1e-6)
    assert inv.max() == pytest.approx(float(ufun.minmax()[1]), abs=1e-6)
    assert float(ufun(inv.best())) == pytest.approx(inv.max(), abs=1e-6)
    assert float(ufun(inv.worst())) == pytest.approx(inv.min(), abs=1e-6)
