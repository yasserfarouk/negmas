"""Integration test: a CPU-bound (infinite-loop) negotiator must not hang a
cartesian tournament when ``external_timeout`` is set -- neither serially
(njobs=-1, one isolated worker) nor in parallel (njobs=2). The hung
negotiations must be killed and recorded as *timeouts* (a no-agreement
outcome where each negotiator keeps its reserved value), NOT as exceptions;
the rest must complete.

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
        plot_fraction=0.0,  # avoid the Chrome/Kaleido dependency in CI
        save_scenario_figs=False,
    )
    elapsed = time.perf_counter() - strt

    # Must not hang: a handful of negotiations, each capped at ~12s, plus
    # worker spawn overhead. 180s is a generous ceiling that still fails on a
    # true hang (without the fix this never returns).
    assert elapsed < 180, f"tournament took {elapsed:.0f}s -- likely hung"

    details = results.details
    assert len(details) == 4  # self-play: 2x2 orderings of the two competitors
    # The three CpuHog negotiations can never finish, so they must be killed and
    # recorded as *timeouts*. (The Aspiration-vs-Aspiration negotiation normally
    # completes, but may also time out when the spinning hog starves the CPU on a
    # loaded/low-core machine -- so we assert >= 3 rather than exactly 3.)
    assert int(details["timedout"].sum()) >= 3, (
        "expected the CpuHog negotiations to be flagged as timed-out"
    )
    # The actual contract under test, and load-independent: a timeout is NOT an
    # exception, so no negotiation may be flagged has_error.
    assert int(details["has_error"].sum()) == 0, (
        "timeouts must not be recorded as mechanism/negotiator exceptions"
    )
    # Each timed-out negotiation must end without agreement, so every negotiator
    # keeps its reserved value (reserved_value=0.0 here) -- the no-agreement
    # outcome rather than an error with null utilities.
    killed = details[details["timedout"]]
    assert killed["agreement"].isna().all(), "a timed-out negotiation cannot have an agreement"
