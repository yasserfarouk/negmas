"""Tests for ParetoSampler implementations: IPSParetoSampler, NB3ParetoSampler, MOBANOSParetoSampler."""

from __future__ import annotations

import random

import pytest

from negmas.outcomes import make_issue, make_os
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.preferences.pareto_sampler import (
    BruteForceParetoSampler,
    IPSParetoSampler,
    MOBANOSParetoSampler,
    NB3ParetoSampler,
    ParetoSampler,
)

ALL_PARETO_SAMPLER_TYPES = [
    BruteForceParetoSampler,
    IPSParetoSampler,
    NB3ParetoSampler,
    MOBANOSParetoSampler,
]


def make_two_additive_ufuns(
    nissues: int = 2, nvalues: int = 5, seed: int = 42
) -> tuple[LinearAdditiveUtilityFunction, LinearAdditiveUtilityFunction]:
    """Create two random LinearAdditiveUtilityFunctions sharing the same outcome space."""
    random.seed(seed)
    os = make_os([make_issue(nvalues) for _ in range(nissues)])
    own = LinearAdditiveUtilityFunction.random(outcome_space=os)
    opp = LinearAdditiveUtilityFunction.random(outcome_space=os)
    return own, opp


def brute_force_pareto(
    own: LinearAdditiveUtilityFunction, opp: LinearAdditiveUtilityFunction
) -> list[tuple[float, float]]:
    """Compute the exact Pareto front by brute force.

    Returns list of (own_util, opp_util) for all non-dominated outcomes.
    """
    os = own.outcome_space
    assert os is not None
    outcomes = list(os.enumerate_or_sample())
    utils = [(float(own(o)), float(opp(o))) for o in outcomes]

    non_dominated = []
    for i, (ua, va) in enumerate(utils):
        dominated = False
        for j, (ub, vb) in enumerate(utils):
            if i == j:
                continue
            if ub >= ua and vb >= va and (ub > ua or vb > va):
                dominated = True
                break
        if not dominated:
            non_dominated.append((ua, va))
    return non_dominated


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_PARETO_SAMPLER_TYPES)
def test_pareto_sampler_instantiation(cls):
    """All ParetoSampler classes must satisfy the ParetoSampler protocol."""
    own, opp = make_two_additive_ufuns()
    sampler = cls()
    assert isinstance(sampler, ParetoSampler)
    sampler.init(own, opp)
    assert sampler.initialized


# ---------------------------------------------------------------------------
# Pareto outcomes non-empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_PARETO_SAMPLER_TYPES)
def test_pareto_outcomes_non_empty(cls):
    """pareto_outcomes() must return at least one outcome for a reasonable ufun pair."""
    own, opp = make_two_additive_ufuns(nissues=2, nvalues=5, seed=1)
    sampler = cls()
    sampler.init(own, opp)
    outcomes = sampler.pareto_outcomes()
    assert len(outcomes) >= 1


# ---------------------------------------------------------------------------
# Non-dominated outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_PARETO_SAMPLER_TYPES)
def test_pareto_outcomes_are_non_dominated(cls):
    """No outcome returned by pareto_outcomes() should dominate another."""
    own, opp = make_two_additive_ufuns(nissues=2, nvalues=3, seed=2)
    sampler = cls()
    sampler.init(own, opp)
    outcomes = sampler.pareto_outcomes()
    if len(outcomes) < 2:
        return
    utils = [(float(own(o)), float(opp(o))) for o in outcomes]
    for i, (ua, va) in enumerate(utils):
        for j, (ub, vb) in enumerate(utils):
            if i == j:
                continue
            # a should not strictly dominate b
            assert not (ua >= ub and va >= vb and (ua > ub or va > vb)), (
                f"Outcome {i} with utils ({ua:.4f}, {va:.4f}) dominates "
                f"outcome {j} with utils ({ub:.4f}, {vb:.4f})"
            )


# ---------------------------------------------------------------------------
# min_util filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_PARETO_SAMPLER_TYPES)
def test_pareto_min_util_filter(cls):
    """pareto_outcomes(min_util=0.5, normalized=True) returns only outcomes
    with own utility >= 0.5 (normalized)."""
    own, opp = make_two_additive_ufuns(nissues=2, nvalues=5, seed=3)
    sampler = cls()
    sampler.init(own, opp)
    min_util_norm = 0.5
    outcomes = sampler.pareto_outcomes(min_util=min_util_norm, normalized=True)
    mn, mx = own.minmax()
    raw_min = mn + min_util_norm * (mx - mn)
    for o in outcomes:
        assert float(own(o)) >= raw_min - 1e-6, (
            f"Outcome utility {float(own(o)):.4f} < raw_min {raw_min:.4f}"
        )


# ---------------------------------------------------------------------------
# best_for_opponent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_PARETO_SAMPLER_TYPES)
def test_best_for_opponent_satisfies_own_constraint(cls):
    """best_for_opponent(min_util=0.3, normalized=True) must return an outcome
    with own utility >= 0.3 (normalized) and should maximise opponent utility
    among all Pareto outcomes meeting the constraint."""
    own, opp = make_two_additive_ufuns(nissues=2, nvalues=5, seed=4)
    sampler = cls()
    sampler.init(own, opp)
    min_util_norm = 0.3
    result = sampler.best_for_opponent(min_util=min_util_norm, normalized=True)
    if result is None:
        # If nothing satisfies the constraint, that's acceptable
        return
    mn, mx = own.minmax()
    raw_min = mn + min_util_norm * (mx - mn)
    own_u = float(own(result))
    assert own_u >= raw_min - 1e-6, (
        f"best_for_opponent returned outcome with own utility {own_u:.4f} "
        f"< required {raw_min:.4f}"
    )
    # Verify it is indeed the best for opponent among feasible Pareto outcomes
    pareto = sampler.pareto_outcomes(min_util=min_util_norm, normalized=True)
    if pareto:
        best_opp_u = float(opp(result))
        for o in pareto:
            assert float(opp(o)) <= best_opp_u + 1e-6, (
                f"Found Pareto outcome with higher opponent utility "
                f"{float(opp(o)):.4f} > best_for_opponent {best_opp_u:.4f}"
            )


# ---------------------------------------------------------------------------
# TypeError for non-additive ufun
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [NB3ParetoSampler, MOBANOSParetoSampler])
def test_raises_for_non_additive_ufun(cls):
    """NB3 and MOBANOS must raise TypeError on init() for non-additive ufuns."""
    os = make_os([make_issue(5), make_issue(5)])
    outcomes = list(os.enumerate_or_sample())
    mapping_ufun = MappingUtilityFunction(
        mapping=dict(zip(outcomes, range(len(outcomes)), strict=True)), outcome_space=os
    )
    own, opp = make_two_additive_ufuns()
    # Non-additive own ufun
    sampler = cls()
    with pytest.raises(TypeError):
        sampler.init(mapping_ufun, opp)
    # Non-additive opp ufun
    sampler2 = cls()
    with pytest.raises(TypeError):
        sampler2.init(own, mapping_ufun)


# ---------------------------------------------------------------------------
# Agreement across algorithms
# ---------------------------------------------------------------------------


def test_nb3_and_mobanos_agree_with_ips():
    """For a small outcome space (2 issues × 3 values), all three algorithms
    should find the same Pareto-optimal utility pairs."""
    own, opp = make_two_additive_ufuns(nissues=2, nvalues=3, seed=10)

    ips = IPSParetoSampler()
    nb3 = NB3ParetoSampler()
    mob = MOBANOSParetoSampler()

    ips.init(own, opp)
    nb3.init(own, opp)
    mob.init(own, opp)

    # Compute true brute-force Pareto front
    bf = brute_force_pareto(own, opp)
    bf_set = {(round(u, 5), round(v, 5)) for u, v in bf}

    def outcome_utils_set(outcomes):
        return {(round(float(own(o)), 5), round(float(opp(o)), 5)) for o in outcomes}

    ips_set = outcome_utils_set(ips.pareto_outcomes())
    nb3_set = outcome_utils_set(nb3.pareto_outcomes())
    mob_set = outcome_utils_set(mob.pareto_outcomes())

    # All algorithms should contain all true Pareto-optimal outcomes
    for u, v in bf_set:
        assert any(abs(uu - u) < 1e-4 and abs(vv - v) < 1e-4 for uu, vv in ips_set), (
            f"IPS missing Pareto point ({u}, {v})"
        )
        assert any(abs(uu - u) < 1e-4 and abs(vv - v) < 1e-4 for uu, vv in nb3_set), (
            f"NB3 missing Pareto point ({u}, {v})"
        )
        assert any(abs(uu - u) < 1e-4 and abs(vv - v) < 1e-4 for uu, vv in mob_set), (
            f"MOBANOS missing Pareto point ({u}, {v})"
        )


# ---------------------------------------------------------------------------
# None opponent ufun
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_PARETO_SAMPLER_TYPES)
def test_pareto_sampler_none_opponent_ufun(cls):
    """When opponent_ufun=None, init() should not raise, pareto_outcomes() should
    return empty list, and best_for_opponent() should return None."""
    own, _ = make_two_additive_ufuns()
    sampler = cls()
    sampler.init(own, None)  # Should not raise
    assert sampler.pareto_outcomes() == []
    assert sampler.best_for_opponent(min_util=0.5) is None


# ---------------------------------------------------------------------------
# Brute-force verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_PARETO_SAMPLER_TYPES)
def test_brute_force_pareto_contained_in_results(cls):
    """For a small outcome space (2 issues × 3 values), all truly Pareto-optimal
    outcomes should appear in the algorithm's output."""
    own, opp = make_two_additive_ufuns(nissues=2, nvalues=3, seed=99)
    sampler = cls()
    sampler.init(own, opp)
    bf = brute_force_pareto(own, opp)
    outcomes = sampler.pareto_outcomes()
    algo_utils = [(round(float(own(o)), 5), round(float(opp(o)), 5)) for o in outcomes]

    for u, v in bf:
        u_r, v_r = round(u, 5), round(v, 5)
        assert any(
            abs(uu - u_r) < 1e-4 and abs(vv - v_r) < 1e-4 for uu, vv in algo_utils
        ), (
            f"{cls.__name__} missing brute-force Pareto point ({u_r}, {v_r}). "
            f"Got: {algo_utils}"
        )
