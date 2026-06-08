"""Regression test for the ``TimeoutCaller`` thread-reclaim fix.

A negotiator that enters a pure-Python infinite loop used to leak the worker
thread that ran it: ``concurrent.futures`` cannot cancel an already-running
task, so the thread kept burning CPU forever after the call timed out. Because
those leaked threads can never be reused by the shared ``ThreadPoolExecutor``,
running several such negotiations made the live-thread count grow without
bound (and the program could not exit cleanly).

The fix injects an exception into the timed-out worker thread so it unwinds and
returns to the pool. This test runs several infinite-loop negotiations back to
back and asserts the live-thread count stays bounded (threads are reclaimed and
reused) instead of growing per negotiation.
"""

from __future__ import annotations

import threading
import time

from negmas.gb.common import ResponseType
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao.common import SAOResponse
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator


class _PureLoopNegotiator(AspirationNegotiator):
    """Spins in a pure-Python loop forever on first call."""

    def __call__(self, state, dest=None) -> SAOResponse:  # type: ignore[override]
        x = 0
        while True:
            x += 1
        return SAOResponse(ResponseType.END_NEGOTIATION, None)


def _run_one(hidden_time_limit: float) -> SAOMechanism:
    issues = (make_issue([f"v{i}" for i in range(5)], "issue"),)
    os_ = make_os(issues)
    m = SAOMechanism(
        outcome_space=os_,
        n_steps=50,
        hidden_time_limit=hidden_time_limit,
        step_time_limit=float("inf"),
        ignore_negotiator_exceptions=True,
    )
    for _ in range(2):
        m.add(
            _PureLoopNegotiator(
                preferences=U.random(
                    issues=issues, reserved_value=0.0, normalized=False
                )
            )
        )
    m.run()
    return m


def test_infinite_loop_negotiator_threads_are_reclaimed():
    # Warm the shared thread pool once so the baseline reflects steady state.
    _run_one(hidden_time_limit=0.2)
    time.sleep(0.3)
    baseline = threading.active_count()

    n = 5
    for _ in range(n):
        m = _run_one(hidden_time_limit=0.2)
        assert m.state.timedout
        assert m.state.agreement is None

    # Give injected exceptions a moment to unwind the worker threads.
    time.sleep(0.5)
    grew_by = threading.active_count() - baseline
    # With the leak, each negotiation would strand ~2 threads (one per
    # negotiator call that timed out) -> growth on the order of 2 * n. With the
    # fix the pool reuses a handful of threads regardless of n.
    assert grew_by <= 3, (
        f"thread count grew by {grew_by} after {n} infinite-loop negotiations "
        f"(baseline={baseline}); timed-out worker threads are leaking"
    )
