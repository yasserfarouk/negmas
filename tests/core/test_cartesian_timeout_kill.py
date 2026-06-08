"""Integration test: a CPU-bound (infinite-loop) negotiator must not hang a
cartesian tournament when ``external_timeout`` is set -- neither serially
(njobs=-1, one isolated worker) nor in parallel (njobs=2). The hung
negotiations must be killed and recorded as errors; the rest must complete.

The hog negotiator is a module-level class so cloudpickle ships it to spawned
worker processes by reference (the workers re-import this module), matching how
``run_isolated_tasks`` is exercised elsewhere.
"""

from __future__ import annotations

import time

import pytest

from negmas.gb.common import ResponseType
from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao.common import SAOResponse
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator
from negmas.tournaments.neg import cartesian_tournament


class CpuHogNegotiator(AspirationNegotiator):
    """Spins forever in a pure-Python loop on its first call."""

    def __call__(self, state, dest=None) -> SAOResponse:  # type: ignore[override]
        x = 0
        while True:
            x += 1
        return SAOResponse(ResponseType.END_NEGOTIATION, None)


def _scenarios(n=1):
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    return [
        Scenario(
            outcome_space=make_os(issues, name=f"S{i}"),
            ufuns=(
                U.random(issues=issues, reserved_value=0.0, normalized=False),
                U.random(issues=issues, reserved_value=0.0, normalized=False),
            ),
            mechanism_type=SAOMechanism,
            mechanism_params=dict(),
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("njobs", [-1, 2])
def test_cpu_hog_negotiator_is_killed_and_tournament_completes(tmp_path, njobs):
    strt = time.perf_counter()
    results = cartesian_tournament(
        competitors=[CpuHogNegotiator, AspirationNegotiator],
        scenarios=_scenarios(1),
        n_steps=20,
        n_repetitions=1,
        njobs=njobs,
        external_timeout=12,
        path=tmp_path / f"t_{njobs}",
        verbosity=0,
        rotate_ufuns=False,
        self_play=True,
    )
    elapsed = time.perf_counter() - strt

    # Must not hang: a handful of negotiations, each capped at ~12s, plus
    # worker spawn overhead. 180s is a generous ceiling that still fails on a
    # true hang (without the fix this never returns).
    assert elapsed < 180, f"tournament took {elapsed:.0f}s -- likely hung"

    details = results.details
    assert len(details) == 4  # self-play: 2x2 orderings of the two competitors
    # The Aspiration-vs-Aspiration negotiation must complete cleanly...
    assert bool((~details["has_error"]).any()), "no negotiation completed successfully"
    # ...and the three CpuHog negotiations must be recorded as failed (killed).
    assert int(details["has_error"].sum()) == 3, (
        "expected exactly the 3 CpuHog negotiations to be flagged as timed-out"
    )
