"""Comprehensive tests for ALL inverse-utility-function (inverter) types.

This file distinguishes two kinds of inverters:

* **Strict** inverters return ``None`` when no outcome's utility falls inside
  the requested range. They never clamp, fall back, or expand the range.
  `BruteForceInverseUtilityFunction` and
  `PresortingInverseUtilityFunctionBruteForce` are strict.

* **Clamping** inverters apply fallback strategies when no in-range outcome is
  found: they may expand the range upward and/or fall back to the best/worst
  outcome overall. All other inverters (`SamplingInverseUtilityFunction`,
  `PresortingInverseUtilityFunction`, `PresortingLegacyInverseUtilityFunction`,
  `BIDSInverseUtilityFunction`, `MCTSInverseUtilityFunction`,
  `AttributePlanningInverseUtilityFunction`, `AdaptiveInverseUtilityFunction`)
  are clamping.

These tests pin, for every inverter:

1. The fallback contract: strict inverters return ``None`` for out-of-range
   queries; clamping inverters return a fallback outcome (never ``None`` when
   a valid outcome exists in the full outcome space).
2. Integration: `AspirationNegotiator` using any inverter does not break the
   SAO mechanism (no ``None`` proposals propagate).
3. Concession: `AspirationNegotiator` with a linear curve concedes over time
   (later offers have lower utility than earlier ones) on a real ANAC
   scenario.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

import negmas
from negmas.inout import Scenario
from negmas.outcomes import make_issue, make_os
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.inv_ufun import (
    AdaptiveInverseUtilityFunction,
    AttributePlanningInverseUtilityFunction,
    BIDSInverseUtilityFunction,
    BruteForceInverseUtilityFunction,
    MCTSInverseUtilityFunction,
    PresortingInverseUtilityFunction,
    PresortingInverseUtilityFunctionBruteForce,
    PresortingLegacyInverseUtilityFunction,
    SamplingInverseUtilityFunction,
)
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator


# ─── Inverter classification ────────────────────────────────────────────────────
# Strict inverters return None when no in-range outcome exists.
STRICT_INVERTERS = [
    BruteForceInverseUtilityFunction,
    PresortingInverseUtilityFunctionBruteForce,
]
# Clamping inverters apply fallbacks (expand range upward / fall back to best
# or worst) when no in-range outcome is found.
CLAMPING_INVERTERS = [
    SamplingInverseUtilityFunction,
    PresortingInverseUtilityFunction,
    PresortingLegacyInverseUtilityFunction,
    BIDSInverseUtilityFunction,
    MCTSInverseUtilityFunction,
    AttributePlanningInverseUtilityFunction,
    AdaptiveInverseUtilityFunction,
]
ALL_INVERTERS = STRICT_INVERTERS + CLAMPING_INVERTERS


# ─── Fixtures ──────────────────────────────────────────────────────────────────


def _synthetic_scenario(n_issues: int = 4, n_vals: int = 10, seed: int = 42):
    """Build a normalized synthetic scenario with a unique best outcome."""
    issues = [make_issue(n_vals, name=f"i{k}") for k in range(n_issues)]
    os_ = make_os(issues)
    random.seed(seed)
    u1 = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    u2 = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    return issues, os_, u1, u2


def _camera_scenario():
    """Load the ANAC Camera-A scenario (ignoring discount) used by
    coding_agents/plot_inverter_comparison.py. Returns None if unavailable."""
    camera_dir = (
        Path(negmas.__file__).parent.parent.parent
        / "tests/data/scenarios/anac/y2013/cameradomain"
    )
    if not camera_dir.exists():
        return None
    scenario = Scenario.from_genius_folder(
        camera_dir, ignore_discount=True, ignore_reserved=True
    )
    if scenario is None or scenario.outcome_space is None or not scenario.ufuns:
        return None
    scenario = scenario.normalize()
    issues = list(scenario.outcome_space.issues)
    return issues, scenario.outcome_space, scenario.ufuns[0], scenario.ufuns[1]


# ─── 1. Strict inverters return None for out-of-range queries ─────────────────


@pytest.mark.parametrize("inv_cls", STRICT_INVERTERS, ids=lambda c: c.__name__)
def test_strict_inverters_return_none_out_of_range(inv_cls):
    """Strict inverters must return None when the requested range lies entirely
    outside the ufun's utility range (above the max). They never clamp or
    fall back."""
    issues, os_, u1, _u2 = _synthetic_scenario(4, 10, seed=42)
    inv = inv_cls(u1)
    inv.init()
    mn, mx = u1.minmax()
    # Range entirely above the maximum utility.
    above_range = (float(mx) + 0.1, float(mx) + 0.2)
    assert inv.worst_in(above_range, normalized=False) is None
    assert inv.best_in(above_range, normalized=False) is None
    # Range entirely below the minimum utility.
    below_range = (float(mn) - 0.2, float(mn) - 0.1)
    assert inv.worst_in(below_range, normalized=False) is None
    assert inv.best_in(below_range, normalized=False) is None


@pytest.mark.parametrize("inv_cls", STRICT_INVERTERS, ids=lambda c: c.__name__)
def test_strict_inverters_find_in_range_outcome(inv_cls):
    """Strict inverters must find the correct outcome when one exists in the
    requested range (no false None)."""
    issues, os_, u1, _u2 = _synthetic_scenario(4, 10, seed=42)
    inv = inv_cls(u1)
    inv.init()
    mn, mx = u1.minmax()
    best = u1.best()
    # The full utility range contains the best outcome.
    r = inv.best_in((float(mn), float(mx)), normalized=False)
    assert r is not None
    assert r == best


# ─── 2. Clamping inverters never return None when an outcome exists ───────────


@pytest.mark.parametrize("inv_cls", CLAMPING_INVERTERS, ids=lambda c: c.__name__)
def test_clamping_inverters_never_none_when_outcome_exists(inv_cls):
    """Clamping inverters must never return None when the outcome space is
    non-empty, even for a narrow range that contains only the best outcome
    (the worst case for sampling-based inverters). They must fall back to
    best/worst instead."""
    issues, os_, u1, _u2 = _synthetic_scenario(4, 10, seed=42)
    inv = inv_cls(u1)
    inv.init()
    # Narrow normalized range that contains only the best outcome.
    none_count_w = 0
    none_count_b = 0
    for seed in range(50):
        random.seed(seed)
        if inv.worst_in((0.99999997, 1.0), normalized=True) is None:
            none_count_w += 1
        random.seed(seed)
        if inv.best_in((0.99999997, 1.0), normalized=True) is None:
            none_count_b += 1
    assert none_count_w == 0, (
        f"{inv_cls.__name__}.worst_in returned None {none_count_w}/50 times; "
        "clamping inverters must fall back rather than returning None."
    )
    assert none_count_b == 0, (
        f"{inv_cls.__name__}.best_in returned None {none_count_b}/50 times; "
        "clamping inverters must fall back rather than returning None."
    )


@pytest.mark.parametrize("inv_cls", CLAMPING_INVERTERS, ids=lambda c: c.__name__)
def test_clamping_inverters_return_fallback_above_max(inv_cls):
    """Clamping inverters, when queried with a range above the maximum
    utility, must return a fallback (best/worst) rather than None."""
    issues, os_, u1, _u2 = _synthetic_scenario(4, 10, seed=42)
    inv = inv_cls(u1)
    inv.init()
    mn, mx = u1.minmax()
    above_range = (float(mx) + 0.1, float(mx) + 0.2)
    w = inv.worst_in(above_range, normalized=False)
    b = inv.best_in(above_range, normalized=False)
    assert w is not None, (
        f"{inv_cls.__name__}.worst_in returned None for an above-max range; "
        "clamping inverters must fall back."
    )
    assert b is not None, (
        f"{inv_cls.__name__}.best_in returned None for an above-max range; "
        "clamping inverters must fall back."
    )


# ─── 3. one_in is robust for ALL inverters (has fallbacks by design) ─────────


@pytest.mark.parametrize("inv_cls", ALL_INVERTERS, ids=lambda c: c.__name__)
def test_one_in_never_none_when_outcome_exists(inv_cls):
    """one_in has built-in fallbacks (fallback_to_higher, fallback_to_best)
    for every inverter. It must never return None when the outcome space is
    non-empty, even for a narrow range containing only the best outcome."""
    issues, os_, u1, _u2 = _synthetic_scenario(4, 10, seed=42)
    inv = inv_cls(u1)
    inv.init()
    none_count = 0
    for seed in range(50):
        random.seed(seed)
        if inv.one_in((0.99999997, 1.0), normalized=True) is None:
            none_count += 1
    assert none_count == 0, (
        f"{inv_cls.__name__}.one_in returned None {none_count}/50 times; "
        "one_in must always find a fallback outcome."
    )


# ─── 4. AspirationNegotiator does not break with ANY inverter ─────────────────


@pytest.mark.parametrize("inv_cls", ALL_INVERTERS, ids=lambda c: c.__name__)
def test_aspiration_negotiator_does_not_break(inv_cls):
    """AspirationNegotiator with any inverter must not break the SAO
    mechanism. Before the fix, strict inverters (BruteForce) and clamping
    inverters without fallbacks (Sampling) could cause None proposals that
    broke the negotiation."""
    issues, os_, u1, u2 = _synthetic_scenario(4, 10, seed=42)
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=200)
    n1 = AspirationNegotiator(
        ufun_inverter=inv_cls, aspiration_type="boulware", name="A"
    )
    n2 = AspirationNegotiator(
        ufun_inverter=inv_cls, aspiration_type="boulware", name="B"
    )
    session.add(n1, ufun=u1)
    session.add(n2, ufun=u2)
    state = session.run()
    assert not state.broken, (
        f"Negotiation broke after {state.step} steps with {inv_cls.__name__}. "
        "AspirationNegotiator must recover from None proposals."
    )
    assert state.agreement is not None or state.step >= 200


@pytest.mark.parametrize("inv_cls", ALL_INVERTERS, ids=lambda c: c.__name__)
def test_aspiration_negotiator_propose_never_none(inv_cls):
    """AspirationNegotiator.propose must never return None mid-negotiation
    with any inverter (the negotiator falls back to the best outcome if the
    inverter returns None)."""
    issues, os_, u1, u2 = _synthetic_scenario(4, 10, seed=42)
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=100)
    n1 = AspirationNegotiator(
        ufun_inverter=inv_cls, aspiration_type="boulware", name="A"
    )
    n2 = AspirationNegotiator(
        ufun_inverter=inv_cls, aspiration_type="boulware", name="B"
    )
    session.add(n1, ufun=u1)
    session.add(n2, ufun=u2)
    none_steps = []
    original_propose = n1.propose

    def tracking_propose(state, dest=None):
        r = original_propose(state, dest)
        if r is None and state.step < 99:
            none_steps.append(state.step)
        return r

    n1.propose = tracking_propose
    session.run()
    assert not none_steps, (
        f"{inv_cls.__name__}: AspirationNegotiator.propose returned None at "
        f"steps {none_steps}; it must fall back to the best outcome."
    )


# ─── 5. Concession behavior on Camera-A ──────────────────────────────────────


@pytest.mark.parametrize("inv_cls", ALL_INVERTERS, ids=lambda c: c.__name__)
def test_aspiration_negotiator_concedes_on_camera_a(inv_cls):
    """AspirationNegotiator with a linear curve must concede over time on the
    Camera-A ANAC scenario (later offers have lower utility than earlier
    ones). This is the behavioral symptom that the plot script surfaced."""
    sc = _camera_scenario()
    if sc is None:
        pytest.skip("Camera-A scenario data not available")
    issues, os_, u1, u2 = sc
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=1000)
    n1 = AspirationNegotiator(ufun_inverter=inv_cls, aspiration_type="linear", name="A")
    # Use a boulware opponent so the negotiation does not end immediately.
    n2 = AspirationNegotiator(
        ufun_inverter=PresortingInverseUtilityFunction,
        aspiration_type="boulware",
        name="B",
    )
    session.add(n1, ufun=u1)
    session.add(n2, ufun=u2)
    utilities = []
    original_propose = n1.propose

    def tracking_propose(state, dest=None):
        r = original_propose(state, dest)
        if r is not None:
            utilities.append(float(u1(r)))
        return r

    n1.propose = tracking_propose
    session.run()
    assert not session.state.broken, (
        f"{inv_cls.__name__}: negotiation broke on Camera-A."
    )
    assert len(utilities) >= 10, (
        f"{inv_cls.__name__}: expected >=10 proposals, got {len(utilities)}"
    )
    assert utilities[0] >= 0.9 * max(utilities), (
        f"{inv_cls.__name__}: first offer {utilities[0]} not near max {max(utilities)}"
    )
    assert utilities[-1] < utilities[0] - 1e-6, (
        f"{inv_cls.__name__}: no concession (first={utilities[0]}, "
        f"last={utilities[-1]})"
    )
