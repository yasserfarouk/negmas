from __future__ import annotations

import random
from statistics import median


from negmas.common import PreferencesChangeType
from negmas.negotiators.helpers import PolyAspiration
from negmas.outcomes import make_issue, make_os
from negmas.preferences.generators import generate_ufuns_for
from negmas.preferences.inv_ufun import (
    AdaptiveInverseUtilityFunction,
    AttributePlanningInverseUtilityFunction,
    BIDSInverseUtilityFunction,
    BruteForceInverseUtilityFunction,
    MCTSInverseUtilityFunction,
    PresortingInverseUtilityFunction,
    PresortingLegacyInverseUtilityFunction,
    SamplingInverseUtilityFunction,
)
from negmas.sao import SAOMechanism, SAONegotiator, ResponseType


class _AspirationInverterNegotiator(SAONegotiator):
    def __init__(self, inverter_cls, aspiration_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._asp = PolyAspiration(1.0, aspiration_type)
        self._inv_cls = inverter_cls
        self._inv = None
        self._min = None
        self._max = None
        self._best = None

    def _target_utility(self, relative_time: float) -> float:
        return (self._max - self._min) * self._asp.utility_at(relative_time) + self._min

    def _offer_near_target(self, target: float):
        span = max(float(self._max - self._min), 1e-12)
        for frac in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
            half_width = span * frac / 2.0
            offer = self._inv.one_in(
                (
                    max(self._min, target - half_width),
                    min(self._max, target + half_width),
                ),
                False,
            )
            if offer is not None:
                return offer
        return self._best

    def on_preferences_changed(self, changes):
        changes = [c for c in changes if c.type not in (PreferencesChangeType.Scale,)]
        if not changes:
            return
        self._inv = self._inv_cls(self.ufun)
        self._inv.init()
        worst, self._best = self.ufun.extreme_outcomes()
        self._min, self._max = self.ufun(worst), self.ufun(self._best)
        super().on_preferences_changed(changes)

    def propose(self, state, dest=None):
        return self._offer_near_target(self._target_utility(state.relative_time))

    def respond(self, state, source=None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        u = float(self.ufun(offer))
        a = self._target_utility(state.relative_time)
        return ResponseType.ACCEPT_OFFER if u >= a - 1e-6 else ResponseType.REJECT_OFFER


INVERTERS = [
    PresortingInverseUtilityFunction,
    PresortingLegacyInverseUtilityFunction,
    BruteForceInverseUtilityFunction,
    SamplingInverseUtilityFunction,
    BIDSInverseUtilityFunction,
    AdaptiveInverseUtilityFunction,
    AttributePlanningInverseUtilityFunction,
    MCTSInverseUtilityFunction,
]


def _run_session(inv_cls, u1, u2, issues):
    random.seed(20260716)
    session = SAOMechanism(issues=issues, n_steps=60)
    n1 = _AspirationInverterNegotiator(inv_cls, "boulware", name="B")
    n2 = _AspirationInverterNegotiator(inv_cls, "linear", name="L")
    assert session.add(n1, ufun=u1)
    assert session.add(n2, ufun=u2)
    state = session.run()
    return state


def test_all_inverters_reach_reasonable_similar_outcomes():
    """Checks that all inverters produce sensible and broadly similar outcomes
    on the same generated scenario (without plotting)."""
    random.seed(20260716)
    issues = [make_issue(5, name=f"i{k}") for k in range(5)]  # 3125 outcomes
    os = make_os(issues)
    u1, u2 = generate_ufuns_for(
        os,
        n_ufuns=2,
        pareto_generators=("piecewise_linear",),
        reserved_values=0.2,
        linear=True,
        guarantee_rational=True,
    )

    utils_1: list[float] = []
    utils_2: list[float] = []
    for inv in INVERTERS:
        state = _run_session(inv, u1, u2, issues)
        assert state.ended
        assert not state.broken
        assert state.agreement is not None
        utils_1.append(float(u1(state.agreement)))
        utils_2.append(float(u2(state.agreement)))

    med_1 = median(utils_1)
    med_2 = median(utils_2)
    # "Kind of similar": most inverters should cluster around the median
    # utility, while allowing a small number of outliers for approximate/search
    # methods.
    d1 = [abs(u - med_1) for u in utils_1]
    d2 = [abs(u - med_2) for u in utils_2]

    assert sum(d > 0.25 for d in d1) <= 3
    assert sum(d > 0.25 for d in d2) <= 3
    assert sum(d > 0.35 for d in d1) <= 1
    assert sum(d > 0.35 for d in d2) <= 1
