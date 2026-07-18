"""Extra coverage for the ``negmas.preferences.pareto_sampler`` module.

Complements ``test_pareto_samplers.py`` (which covers the core queries on small
finite additive spaces) by exercising the parts that file does not:

* the ``ufun`` property and the ``initialized`` state transition,
* the ``n`` limit and both raw/normalized ``min_util`` filtering,
* passing ``opponent_ufun`` to the *query* methods (not just ``init``),
* continuous and large outcome spaces,
* the degenerate (constant-utility) ufun,
* ``BruteForceParetoSampler`` on non-additive ufuns (it enumerates, so it works
  where the additive-only samplers correctly raise),
* the ``make_pareto_sampler`` integration point + its caching, and
* the ``_protocol`` backward-compatibility shim.
"""

from __future__ import annotations

import random

import pytest

from negmas.outcomes import make_issue, make_os
from negmas.preferences.crisp.linear import LinearAdditiveUtilityFunction
from negmas.preferences.crisp.mapping import MappingUtilityFunction
from negmas.preferences.value_fun import AffineFun, ConstFun, IdentityFun
from negmas.preferences.pareto_sampler import (
    BruteForceParetoSampler,
    IPSParetoSampler,
    MOBANOSParetoSampler,
    NB3ParetoSampler,
    ParetoSampler,
)

ALL_TYPES = [
    BruteForceParetoSampler,
    IPSParetoSampler,
    NB3ParetoSampler,
    MOBANOSParetoSampler,
]
ADDITIVE_ONLY = [IPSParetoSampler, NB3ParetoSampler, MOBANOSParetoSampler]


def two_additive(nissues=2, nvalues=5, seed=42):
    """Two random additive ufuns sharing one finite outcome space."""
    random.seed(seed)
    os_ = make_os([make_issue(nvalues) for _ in range(nissues)])
    own = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    opp = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    return own, opp


def two_zero_sum(nissues=2, nvalues=6):
    """Two anti-correlated additive ufuns (own increasing, opp decreasing) whose
    Pareto frontier is guaranteed to have many points."""
    issues = [make_issue(nvalues) for _ in range(nissues)]
    os_ = make_os(issues)
    own = LinearAdditiveUtilityFunction(
        values=[IdentityFun() for _ in issues], issues=issues, reserved_value=0.0
    )
    opp = LinearAdditiveUtilityFunction(
        values=[AffineFun(-1, bias=nvalues - 1) for _ in issues],
        issues=issues,
        reserved_value=0.0,
    )
    return own, opp, os_


def is_non_dominated(outcomes, own, opp):
    """True if no returned outcome strictly dominates another."""
    utils = [(float(own(o)), float(opp(o))) for o in outcomes]
    for i, (ua, va) in enumerate(utils):
        for j, (ub, vb) in enumerate(utils):
            if i == j:
                continue
            if ua >= ub and va >= vb and (ua > ub or va > vb):
                return False
    return True


# ---------------------------------------------------------------------------
# ufun property and initialized transition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_ufun_property_returns_own(cls):
    own, opp = two_additive()
    sampler = cls()
    sampler.init(own, opp)
    assert sampler.ufun is own


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_initialized_transition(cls):
    own, opp = two_additive(seed=5)
    sampler = cls()
    assert sampler.initialized is False
    sampler.init(own, opp)
    assert sampler.initialized is True


# ---------------------------------------------------------------------------
# n limit and raw/normalized min_util
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_pareto_outcomes_n_limit(cls):
    own, opp, _ = two_zero_sum(nissues=2, nvalues=6)
    sampler = cls()
    sampler.init(own, opp)
    full = sampler.pareto_outcomes()
    assert len(full) >= 2, "zero-sum frontier should have many points"
    limited = sampler.pareto_outcomes(n=1)
    assert len(limited) == 1
    assert len(sampler.pareto_outcomes(n=len(full) + 10)) == len(full)


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_min_util_raw_matches_normalized(cls):
    own, opp = two_additive(nissues=2, nvalues=5, seed=7)
    sampler = cls()
    sampler.init(own, opp)
    mn, mx = own.minmax()
    norm = 0.5
    raw = mn + norm * (mx - mn)
    got_norm = {
        tuple(o) for o in sampler.pareto_outcomes(min_util=norm, normalized=True)
    }
    got_raw = {
        tuple(o) for o in sampler.pareto_outcomes(min_util=raw, normalized=False)
    }
    assert got_norm == got_raw
    for o in got_raw:
        assert float(own(o)) >= raw - 1e-6


# ---------------------------------------------------------------------------
# opponent_ufun passed to the query methods (not just init)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_opponent_ufun_at_query_time(cls):
    """Construct with no opponent, then supply it at query time."""
    own, opp = two_additive(seed=8)
    sampler = cls()
    sampler.init(own, None)
    assert sampler.initialized is False
    assert sampler.pareto_outcomes() == []
    assert sampler.best_for_opponent(min_util=0.2) is None
    # passing opponent_ufun at query time re-inits and answers
    outcomes = sampler.pareto_outcomes(opponent_ufun=opp)
    assert len(outcomes) >= 1
    assert sampler.initialized is True
    best = sampler.best_for_opponent(min_util=0.0, normalized=True, opponent_ufun=opp)
    assert best is not None


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_best_for_opponent_is_max_among_feasible(cls):
    own, opp = two_additive(nissues=2, nvalues=5, seed=9)
    sampler = cls()
    sampler.init(own, opp)
    best = sampler.best_for_opponent(min_util=0.0, normalized=True)
    if best is None:
        pytest.skip("no feasible outcome")
    feasible = sampler.pareto_outcomes(min_util=0.0, normalized=True)
    best_u = float(opp(best))
    for o in feasible:
        assert float(opp(o)) <= best_u + 1e-6


# ---------------------------------------------------------------------------
# continuous outcome spaces
# ---------------------------------------------------------------------------


def test_bruteforce_on_continuous_space():
    random.seed(3)
    os_ = make_os([make_issue((0.0, 1.0), "x"), make_issue((0.0, 1.0), "y")])
    own = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    opp = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    sampler = BruteForceParetoSampler(max_cardinality=200)
    sampler.init(own, opp)
    assert sampler.initialized
    outcomes = sampler.pareto_outcomes()
    assert len(outcomes) >= 1
    assert is_non_dominated(outcomes, own, opp)
    assert sampler.best_for_opponent(min_util=0.3, normalized=True) is not None


@pytest.mark.parametrize("cls", ADDITIVE_ONLY)
def test_additive_samplers_reject_continuous(cls):
    """Additive samplers enumerate ``issue.all`` and so cannot handle continuous
    issues; they must raise (ValueError) rather than silently mis-answer."""
    random.seed(3)
    os_ = make_os([make_issue((0.0, 1.0), "x"), make_issue((0.0, 1.0), "y")])
    own = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    opp = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    sampler = cls()
    with pytest.raises(ValueError):
        sampler.init(own, opp)


# ---------------------------------------------------------------------------
# non-additive ufuns: BruteForce works, additive-only samplers raise
# ---------------------------------------------------------------------------


def test_bruteforce_on_non_additive_ufun():
    random.seed(11)
    os_ = make_os([make_issue(4, "a"), make_issue(4, "b")])
    outs = list(os_.enumerate_or_sample())
    own = MappingUtilityFunction(
        mapping={o: random.random() for o in outs}, outcome_space=os_
    )
    opp = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    sampler = BruteForceParetoSampler()
    sampler.init(own, opp)
    assert sampler.initialized
    outcomes = sampler.pareto_outcomes()
    assert len(outcomes) >= 1
    assert is_non_dominated(outcomes, own, opp)


# ---------------------------------------------------------------------------
# large discrete space: all four agree and complete quickly
# ---------------------------------------------------------------------------


def test_large_space_agreement():
    own, opp = two_additive(nissues=2, nvalues=40, seed=21)  # 1600 outcomes

    def util_set(sampler):
        sampler.init(own, opp)
        return {
            (round(float(own(o)), 5), round(float(opp(o)), 5))
            for o in sampler.pareto_outcomes()
        }

    bf = util_set(BruteForceParetoSampler())
    assert len(bf) >= 1
    for cls in ADDITIVE_ONLY:
        got = util_set(cls())
        # every exact (brute-force) frontier point must be found by the sampler
        for u, v in bf:
            assert any(abs(uu - u) < 1e-4 and abs(vv - v) < 1e-4 for uu, vv in got), (
                f"{cls.__name__} missing frontier point ({u}, {v})"
            )


# ---------------------------------------------------------------------------
# degenerate (constant-utility) ufun
# ---------------------------------------------------------------------------


def test_degenerate_constant_own_ufun():
    random.seed(31)
    os_ = make_os([make_issue(3, "a"), make_issue(3, "b")])
    own = LinearAdditiveUtilityFunction(
        values=[ConstFun(0.0), ConstFun(0.0)], issues=os_.issues
    )
    opp = LinearAdditiveUtilityFunction.random(outcome_space=os_)
    sampler = BruteForceParetoSampler()
    sampler.init(own, opp)
    assert sampler.initialized
    outcomes = sampler.pareto_outcomes()
    # all outcomes tie on own utility, so the frontier is the opp-maximising set
    assert len(outcomes) >= 1
    # min_util filtering must not crash on the degenerate scale
    assert sampler.best_for_opponent(min_util=0.5, normalized=True) is not None


# ---------------------------------------------------------------------------
# make_pareto_sampler integration point (used by NiceTitForTat) + caching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_make_pareto_sampler_and_caching(cls):
    own, opp = two_additive(seed=41)
    s1 = own.make_pareto_sampler(opponent_ufun=opp, pareto_sampler=cls)
    assert isinstance(s1, cls)
    assert s1.initialized
    # same (type, opponent) -> cached instance is reused
    s2 = own.make_pareto_sampler(opponent_ufun=opp, pareto_sampler=cls)
    assert s2 is s1
    best = s1.best_for_opponent(min_util=0.0, normalized=True, opponent_ufun=opp)
    assert best is not None
    own.forget_pareto_sampler()


def test_make_pareto_sampler_default_is_ips():
    own, opp = two_additive(seed=42)
    s = own.make_pareto_sampler(opponent_ufun=opp)
    assert isinstance(s, IPSParetoSampler)
    own.forget_pareto_sampler()


# ---------------------------------------------------------------------------
# None opponent (no estimate available)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_none_opponent_graceful(cls):
    own, _ = two_additive(seed=43)
    sampler = cls()
    sampler.init(own, None)  # must not raise
    assert sampler.initialized is False
    assert sampler.pareto_outcomes() == []
    assert sampler.pareto_outcomes(min_util=0.5, normalized=True) == []
    assert sampler.best_for_opponent(min_util=0.5) is None


# ---------------------------------------------------------------------------
# _protocol backward-compatibility shim
# ---------------------------------------------------------------------------


def test_protocol_shim_reexports_same_object():
    from negmas.preferences.pareto_sampler._protocol import (
        ParetoSampler as ShimProtocol,
    )
    from negmas.preferences.protocols import ParetoSampler as CanonicalProtocol

    assert ShimProtocol is CanonicalProtocol
    assert ParetoSampler is CanonicalProtocol


@pytest.mark.parametrize("cls", ALL_TYPES)
def test_runtime_protocol_conformance(cls):
    own, opp = two_additive(seed=44)
    sampler = cls()
    sampler.init(own, opp)
    assert isinstance(sampler, ParetoSampler)
