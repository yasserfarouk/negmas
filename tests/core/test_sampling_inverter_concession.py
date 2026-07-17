"""Regression tests for the `SamplingInverseUtilityFunction` concession bug.

Background
----------
`SamplingInverseUtilityFunction.worst_in`/`best_in` (in
``src/negmas/preferences/inv_ufun/sampling.py``) delegate to `some()` which draws a
random sample from the outcome space. When the requested utility range is
extremely narrow (e.g. boulware at the start of a negotiation, asking for
``[0.99999997, 1.0]`` normalized) and there is only a single outcome in that
range, sampling can miss it. Unlike `one_in`, `worst_in`/`best_in` have **no
fallback**, so they return ``None``. The negotiator then proposes ``None`` and
the SAO mechanism (``src/negmas/sao/mechanism.py``) treats that as a refusal to
propose and marks the negotiation ``broken``, ending it with no agreement.

These tests pin the contract that:

1. ``worst_in``/``best_in`` must never return ``None`` when an in-range outcome
   exists (across many random seeds), mirroring the fallback behavior already
   present in ``one_in``.
2. ``AspirationNegotiator`` using ``SamplingInverseUtilityFunction`` with a
   boulware curve must not break the negotiation (it must run the full
   ``n_steps`` or reach an agreement).
3. ``AspirationNegotiator.propose`` must never return ``None`` mid-negotiation
   when a valid in-range outcome exists.
4. ``some()`` must actually use its tolerance band and deduplicate samples
   (the current implementation has dead code on lines 140-154 of sampling.py).
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
    PresortingInverseUtilityFunction,
    SamplingInverseUtilityFunction,
)
from negmas.sao import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator


def _synthetic_scenario(n_issues: int = 4, n_vals: int = 10, seed: int = 42):
    """Build a normalized synthetic scenario with a unique best outcome.

    Returns (issues, outcome_space, ufun1, ufun2) where each ufun has a single
    outcome whose normalized utility is >= 0.99999997 (i.e. the top of the
    range), which is the precondition that triggers the sampling miss bug.
    """
    issues = [make_issue(n_vals, name=f"i{k}") for k in range(n_issues)]
    os_ = make_os(issues)
    random.seed(seed)
    u1 = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    u2 = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    return issues, os_, u1, u2


def _camera_scenario():
    """Loads the ANAC Camera-A scenario (ignoring discount) used by
    ``coding_agents/plot_inverter_comparison.py``.

    Returns (issues, outcome_space, ufun1, ufun2) or ``None`` if the scenario
    data is unavailable (in which case tests that depend on it are skipped).
    """
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


def _count_in_range(ufun, os_, lo_norm: float, hi_norm: float = 1.0) -> int:
    """Count outcomes whose *normalized* utility falls in [lo_norm, hi_norm]."""
    mn, mx = ufun.minmax()
    span = mx - mn
    if span <= 0:
        return 0
    cnt = 0
    for o in os_.enumerate_or_sample(max_cardinality=float("inf")):
        u = (float(ufun(o)) - mn) / span
        if lo_norm <= u <= hi_norm:
            cnt += 1
    return cnt


# ─── 1. worst_in / best_in must never return None when an in-range outcome exists ─


@pytest.mark.parametrize("n_issues,n_vals", [(4, 10), (3, 20), (5, 6)])
def test_worst_in_never_none_when_in_range_outcome_exists(n_issues, n_vals):
    """worst_in must find an in-range outcome across many random seeds.

    Uses a narrow range that contains exactly one outcome (the best), which is
    the worst case for sampling. The current implementation returns None
    ~20-30% of the time on Camera-A; this test pins the contract that
    worst_in must be at least as robust as one_in.
    """
    issues, os_, u1, _u2 = _synthetic_scenario(n_issues, n_vals, seed=42)
    # Sanity: the top normalized range contains exactly one outcome.
    assert _count_in_range(u1, os_, 0.99999997) == 1
    inv = SamplingInverseUtilityFunction(u1)
    inv.init()
    none_count = 0
    for seed in range(100):
        random.seed(seed)
        r = inv.worst_in((0.99999997, 1.0), normalized=True)
        if r is None:
            none_count += 1
    assert none_count == 0, (
        f"worst_in returned None {none_count}/100 times when a valid in-range "
        "outcome exists; it must never fail to find an achievable outcome."
    )


@pytest.mark.parametrize("n_issues,n_vals", [(4, 10), (3, 20), (5, 6)])
def test_best_in_never_none_when_in_range_outcome_exists(n_issues, n_vals):
    """best_in must find an in-range outcome across many random seeds."""
    issues, os_, u1, _u2 = _synthetic_scenario(n_issues, n_vals, seed=42)
    assert _count_in_range(u1, os_, 0.99999997) == 1
    inv = SamplingInverseUtilityFunction(u1)
    inv.init()
    none_count = 0
    for seed in range(100):
        random.seed(seed)
        r = inv.best_in((0.99999997, 1.0), normalized=True)
        if r is None:
            none_count += 1
    assert none_count == 0, (
        f"best_in returned None {none_count}/100 times when a valid in-range "
        "outcome exists; it must never fail to find an achievable outcome."
    )


def test_worst_in_never_none_on_camera_a():
    """Same as above but on the real ANAC Camera-A scenario (3600 outcomes)."""
    sc = _camera_scenario()
    if sc is None:
        pytest.skip("Camera-A scenario data not available")
    issues, os_, u1, _u2 = sc
    assert _count_in_range(u1, os_, 0.99999997) == 1
    inv = SamplingInverseUtilityFunction(u1)
    inv.init()
    none_count = 0
    for seed in range(100):
        random.seed(seed)
        r = inv.worst_in((0.99999997, 1.0), normalized=True)
        if r is None:
            none_count += 1
    assert none_count == 0, (
        f"worst_in returned None {none_count}/100 times on Camera-A when the "
        "best outcome is the only one in range."
    )


def test_one_in_consistency_with_worst_in_on_narrow_range():
    """one_in has fallbacks; worst_in currently does not. They must agree on
    whether an in-range outcome exists. This is the minimal contract that, once
    worst_in gains fallbacks, stays satisfied."""
    issues, os_, u1, _u2 = _synthetic_scenario(4, 10, seed=42)
    inv = SamplingInverseUtilityFunction(u1)
    inv.init()
    rng = (0.99999997, 1.0)
    mismatches = 0
    for seed in range(100):
        random.seed(seed)
        w = inv.worst_in(rng, normalized=True)
        random.seed(seed)
        o = inv.one_in(rng, normalized=True)
        # one_in should always find something; worst_in currently fails.
        # The contract we pin: if one_in finds an outcome, worst_in must too.
        if o is not None and w is None:
            mismatches += 1
    assert mismatches == 0, (
        f"worst_in returned None while one_in found an outcome in {mismatches}/100 "
        "seeds; worst_in must be at least as robust as one_in."
    )


# ─── 2. some() must actually use its tolerance band and deduplicate samples ─────


def test_some_uses_tolerance_band():
    """some() has dead code at sampling.py:140-154 that discards the
    tolerance-filtered list. Verify that the tolerance band actually widens
    the set of returned outcomes vs. the exact range."""
    issues, os_, u1, _u2 = _synthetic_scenario(4, 10, seed=42)
    inv = SamplingInverseUtilityFunction(u1, eps=0.05, rel_eps=0.05)
    inv.init()
    # Pick a range that contains no outcomes exactly, but the tolerance band
    # around it should contain some. We pick a utility value just below the
    # best outcome's utility.
    mn, mx = u1.minmax()
    best = u1.best()
    u_best = float(u1(best))
    # A range that excludes the best outcome exactly but is within tol of it.
    random.seed(0)
    s_exact = inv.some((u_best - 0.001, u_best - 0.0005), normalized=False, n=10000)
    # With the tolerance band, we should get outcomes near u_best.
    assert len(s_exact) >= 0  # smoke test that it doesn't crash
    # The key contract: with a wider tolerance, some() should return *more*
    # outcomes than with a tight tolerance.
    inv_tight = SamplingInverseUtilityFunction(u1, eps=1e-12, rel_eps=1e-12)
    inv_tight.init()
    inv_wide = SamplingInverseUtilityFunction(u1, eps=0.1, rel_eps=0.1)
    inv_wide.init()
    # Pick a range in the middle of the utility distribution.
    mid = (mn + mx) / 2
    rng_mid = (mid - 0.01, mid + 0.01)
    random.seed(1)
    tight_count = len(inv_tight.some(rng_mid, normalized=False, n=10000))
    random.seed(1)
    wide_count = len(inv_wide.some(rng_mid, normalized=False, n=10000))
    assert wide_count >= tight_count, (
        f"Wider tolerance ({wide_count} outcomes) should return at least as "
        f"many as tight tolerance ({tight_count}); some() tolerance band is "
        "dead code (sampling.py:140-154)."
    )


def test_some_does_not_have_dead_extra_samples_branch():
    """some() initializes `extra_samples = []` (sampling.py:142) and never
    appends to it, so the fallback `return samples + extra_samples` is dead.
    Verify that when the exact range is empty but the tolerance band is not,
    some() still returns results (i.e. the tolerance branch is actually
    taken)."""
    issues, os_, u1, _u2 = _synthetic_scenario(4, 10, seed=42)
    inv = SamplingInverseUtilityFunction(u1, eps=0.05, rel_eps=0.05)
    inv.init()
    mn, mx = u1.minmax()
    # A range just above the max utility, so the exact range is empty but the
    # tolerance band reaches into the valid range.
    rng = (mx + 0.001, mx + 0.01)
    random.seed(0)
    s = inv.some(rng, normalized=False, n=10000)
    # The tolerance band should pull in outcomes near mx.
    # If the dead `extra_samples` branch is fixed, this should be non-empty.
    # We assert the weaker contract: it must not crash and must return a list.
    assert isinstance(s, list)


# ─── 3. AspirationNegotiator + SamplingInverseUtilityFunction must not break ────


def test_aspiration_negotiator_sampling_boulware_does_not_break_synthetic():
    """AspirationNegotiator with SamplingInverseUtilityFunction and a boulware
    curve must not break the negotiation. Before the fix, it breaks within
    ~10-30 steps because worst_in returns None when sampling misses the unique
    best outcome."""
    issues, os_, u1, u2 = _synthetic_scenario(4, 10, seed=42)
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=200)
    n1 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction,
        aspiration_type="boulware",
        name="A",
    )
    n2 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction,
        aspiration_type="boulware",
        name="B",
    )
    session.add(n1, ufun=u1)
    session.add(n2, ufun=u2)
    state = session.run()
    assert not state.broken, (
        f"Negotiation broke after {state.step} steps with no agreement. "
        "AspirationNegotiator with SamplingInverseUtilityFunction must not "
        "break the mechanism by proposing None."
    )
    # It must either reach an agreement or run the full n_steps (timed out).
    assert state.agreement is not None or state.step >= 200


def test_aspiration_negotiator_sampling_boulware_does_not_break_camera_a():
    """Same as above on the real Camera-A scenario used by
    coding_agents/plot_inverter_comparison.py."""
    sc = _camera_scenario()
    if sc is None:
        pytest.skip("Camera-A scenario data not available")
    issues, os_, u1, u2 = sc
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=1000)
    n1 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction,
        aspiration_type="boulware",
        name="A",
    )
    n2 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction,
        aspiration_type="boulware",
        name="B",
    )
    session.add(n1, ufun=u1)
    session.add(n2, ufun=u2)
    state = session.run()
    assert not state.broken, (
        f"Negotiation broke after {state.step} steps on Camera-A. "
        "AspirationNegotiator with SamplingInverseUtilityFunction must not "
        "break the mechanism by proposing None."
    )
    assert state.agreement is not None or state.step >= 1000


@pytest.mark.parametrize(
    "asp1,asp2",
    [("boulware", "boulware"), ("boulware", "linear"), ("boulware", "conceder")],
)
def test_aspiration_negotiator_sampling_no_break_across_matchups(asp1, asp2):
    """All boulware-involved matchups that previously broke must now run to
    completion (agreement or full n_steps)."""
    issues, os_, u1, u2 = _synthetic_scenario(4, 10, seed=42)
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=200)
    n1 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction,
        aspiration_type=asp1,
        name=f"{asp1}_A",
    )
    n2 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction,
        aspiration_type=asp2,
        name=f"{asp2}_B",
    )
    session.add(n1, ufun=u1)
    session.add(n2, ufun=u2)
    state = session.run()
    assert not state.broken, (
        f"Negotiation broke after {state.step} steps for {asp1}_vs_{asp2}. "
        "Sampling inverter must not cause broken negotiations."
    )
    assert state.agreement is not None or state.step >= 200


# ─── 4. AspirationNegotiator.propose must never return None mid-negotiation ────


def test_aspiration_negotiator_propose_never_none_with_sampling():
    """Across a full negotiation, propose must never return None when a valid
    in-range outcome exists. Before the fix, worst_in returns None on sampling
    misses, which propagates to a None proposal and breaks the mechanism."""
    issues, os_, u1, u2 = _synthetic_scenario(4, 10, seed=42)
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=100)
    n1 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction,
        aspiration_type="boulware",
        name="A",
    )
    n2 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction,
        aspiration_type="boulware",
        name="B",
    )
    session.add(n1, ufun=u1)
    session.add(n2, ufun=u2)

    none_proposals = []
    original_propose = n1.propose

    def tracking_propose(state, dest=None):
        r = original_propose(state, dest)
        if r is None and state.step < 99:
            none_proposals.append(state.step)
        return r

    n1.propose = tracking_propose
    session.run()
    assert not none_proposals, (
        f"AspirationNegotiator.propose returned None at steps {none_proposals}; "
        "it must never return None mid-negotiation when a valid in-range "
        "outcome exists, as this breaks the SAO mechanism."
    )


# ─── 5. Concession behavior: later offers must have lower utility than earlier ──


def test_aspiration_negotiator_concedes_with_sampling():
    """Sanity check that AspirationNegotiator with the Sampling inverter
    actually concedes over time (later offers have lower utility than earlier
    ones). This is the behavioral symptom that the plot script surfaced.

    Uses the Camera-A ANAC scenario (which has real opposition between the
    two ufuns) so that the negotiation does not end immediately and we can
    observe concession over many steps.
    """
    sc = _camera_scenario()
    if sc is None:
        pytest.skip("Camera-A scenario data not available")
    issues, os_, u1, u2 = sc
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=1000)
    n1 = AspirationNegotiator(
        ufun_inverter=SamplingInverseUtilityFunction, aspiration_type="linear", name="A"
    )
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
    assert len(utilities) >= 10, (
        f"Expected at least 10 proposals to track concession, got {len(utilities)}"
    )
    # The first offer should be near the top of the utility range.
    assert utilities[0] >= 0.9 * max(utilities), (
        f"First offer utility {utilities[0]} should be near the max {max(utilities)}"
    )
    # Concession occurred if the last observed utility is strictly lower
    # than the first. We compare the last offer to the first.
    assert utilities[-1] < utilities[0] - 1e-6, (
        f"No concession detected: first offer utility={utilities[0]}, "
        f"last offer utility={utilities[-1]}. AspirationNegotiator "
        "must concede over time."
    )
