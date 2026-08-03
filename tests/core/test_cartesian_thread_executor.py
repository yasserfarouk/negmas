"""Tests for ``cartesian_tournament(executor="thread")``.

Covers:
    - thread-safety of per-run ``rep``/annotation (regression guard for the
      ``_make_mechanism``/shared ``mechanism_params`` race fixed alongside
      this feature)
    - structural parity between ``executor="process"`` and
      ``executor="thread"`` for the same seeded tournament
    - an unpicklable negotiator succeeding under ``executor="thread"``
    - ``process_isolation=True`` + ``executor="thread"`` raising ``ValueError``
    - a finite ``external_timeout`` + ``executor="thread"`` emitting
      ``NegmasInfiniteNegotiationWarning``
"""

from __future__ import annotations

import threading

import pytest

from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator, NaiveTitForTatNegotiator
from negmas.tournaments.neg import cartesian_tournament
from negmas.tournaments.neg.simple.cartesian import _make_mechanism
from negmas.warnings import NegmasInfiniteNegotiationWarning


def _scenarios(n=1, seed_offset=0):
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


def test_make_mechanism_does_not_mutate_shared_mechanism_params():
    """Direct, deterministic regression guard for the Phase A fix: calling
    ``_make_mechanism`` must not mutate the caller's ``mechanism_params``
    dict (it writes ``["name"]``/``["verbosity"]``/``["annotation"]``
    in-place before the fix), and each call from a different thread must see
    its own ``rep`` in the constructed mechanism's annotation regardless of
    what other threads are doing concurrently with the *same* shared dict --
    both independent of timing/scheduling luck."""
    issues = (make_issue([f"q{i}" for i in range(3)], "quantity"),)
    scenario = Scenario(
        outcome_space=make_os(issues, name="S0"),
        ufuns=(
            U.random(issues=issues, reserved_value=0.0, normalized=False),
            U.random(issues=issues, reserved_value=0.0, normalized=False),
        ),
        mechanism_type=SAOMechanism,
        mechanism_params=dict(),
    )
    shared_mechanism_params = dict(n_steps=10)
    # Snapshot before any call: none of these keys must appear afterwards.
    assert "name" not in shared_mechanism_params
    assert "verbosity" not in shared_mechanism_params
    assert "annotation" not in shared_mechanism_params

    n_threads = 8
    reps = list(range(n_threads))
    observed_reps: list[int | None] = [None] * n_threads
    barrier = threading.Barrier(n_threads)

    def _call(idx: int, rep: int):
        barrier.wait()  # maximize the chance of overlapping mutation
        m, _, _, _ = _make_mechanism(
            s=scenario,
            partners=(AspirationNegotiator, AspirationNegotiator),
            rep=rep,
            mechanism_type=SAOMechanism,
            mechanism_params=shared_mechanism_params,
            run_id=f"run-{rep}",
        )
        observed_reps[idx] = m._internal_nmi.annotation.get("rep")

    threads = [
        threading.Thread(target=_call, args=(i, reps[i])) for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each call must have observed exactly its own rep -- a shared-dict race
    # would let a concurrently running call's ["annotation"] leak in.
    assert observed_reps == reps

    # The caller's dict must be completely untouched -- this is the
    # deterministic invariant (no race/timing needed to observe it).
    assert shared_mechanism_params == dict(n_steps=10)
    assert "name" not in shared_mechanism_params
    assert "verbosity" not in shared_mechanism_params
    assert "annotation" not in shared_mechanism_params


def test_thread_executor_repetitions_recorded_correctly(tmp_path):
    """Every run's ``rep``/annotation must match its own run, not a
    concurrently-running one, under the thread executor. This specifically
    catches a regression if the ``_make_mechanism``/shared ``mechanism_params``
    race fix is missing or wrong."""
    results = cartesian_tournament(
        competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
        scenarios=_scenarios(1),
        n_steps=20,
        n_repetitions=3,
        njobs=8,
        executor="thread",
        path=tmp_path / "thread_reps",
        verbosity=0,
        rotate_ufuns=False,
        self_play=True,
        plot_fraction=0.0,
        save_scenario_figs=False,
    )
    details = results.details
    # self-play: 2x2 orderings x 3 repetitions
    assert len(details) == 4 * 3
    # "rep" must range exactly over 0..n_repetitions-1 for each
    # scenario/partner-order combination (a race would duplicate/skip reps).
    for (_scenario, partners), group in details.groupby(
        ["scenario", details["partners"].astype(str)]
    ):
        assert sorted(group["rep"].tolist()) == [0, 1, 2]


def test_process_and_thread_executor_structural_parity(tmp_path):
    """Same seeded tournament under executor='process' vs executor='thread'
    must produce the same set of (scenario, partners, rep) rows and the same
    columns -- not necessarily identical negotiation outcomes, but the same
    structure."""
    scenarios = _scenarios(1)

    def _run(path, executor):
        return cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=scenarios,
            n_steps=20,
            n_repetitions=2,
            njobs=4,
            executor=executor,
            path=path,
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
        )

    process_results = _run(tmp_path / "process_run", "process")
    thread_results = _run(tmp_path / "thread_run", "thread")

    def _key_set(details):
        return set(
            zip(details["scenario"], details["partners"].astype(str), details["rep"])
        )

    assert _key_set(process_results.details) == _key_set(thread_results.details)
    assert set(process_results.details.columns) == set(thread_results.details.columns)


class _LockHoldingNegotiator(AspirationNegotiator):
    """Accepts an unpicklable object (e.g. a ``threading.Lock``) as a
    constructor parameter, so it is unpicklable *before* the negotiator is
    ever instantiated -- i.e. the negotiator class + its ``competitor_params``
    (which is exactly what a process-pool task payload serializes) already
    fails to pickle, not just a runtime attribute set inside ``__init__``."""

    def __init__(self, *args, unpicklable_lock=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.unpicklable_lock = unpicklable_lock


def test_thread_executor_supports_unpicklable_negotiator(tmp_path):
    """A negotiator constructed with an unpicklable parameter (a
    ``threading.Lock`` passed via ``competitor_params``) succeeds under
    executor="thread" (no serialization step at all) but fails to even start
    under executor="process" with ``allow_inline_fallback=False`` (proving
    the payload really is unpicklable, not just structurally similar).

    ``path=None`` here: a raw ``threading.Lock`` in ``competitor_params`` also
    cannot round-trip through the tournament's own ``config.yaml``
    serialization (a separate, unrelated limitation of ``serialize()``/YAML
    dumping) -- irrelevant to what this test targets, so results are not
    persisted to disk.
    """
    competitor_params = [{"unpicklable_lock": threading.Lock()}, {}]

    thread_results = cartesian_tournament(
        competitors=[_LockHoldingNegotiator, AspirationNegotiator],
        competitor_params=competitor_params,
        scenarios=_scenarios(1),
        n_steps=20,
        n_repetitions=1,
        njobs=2,
        executor="thread",
        path=None,
        verbosity=0,
        rotate_ufuns=False,
        self_play=True,
        plot_fraction=0.0,
        save_scenario_figs=False,
    )
    assert len(thread_results.details) > 0
    assert int(thread_results.details["has_error"].sum()) == 0

    # Counter-check: the same competitor_params genuinely cannot be
    # serialized for process isolation (with inline fallback disabled, so a
    # failure to pickle cannot be silently masked by running in-process).
    process_results = cartesian_tournament(
        competitors=[_LockHoldingNegotiator, AspirationNegotiator],
        competitor_params=competitor_params,
        scenarios=_scenarios(1),
        n_steps=20,
        n_repetitions=1,
        njobs=2,
        executor="process",
        allow_inline_fallback=False,
        path=None,
        verbosity=0,
        rotate_ufuns=False,
        self_play=True,
        plot_fraction=0.0,
        save_scenario_figs=False,
    )
    assert int(process_results.details["has_error"].sum()) > 0


def test_thread_executor_rejects_process_isolation(tmp_path):
    with pytest.raises(ValueError):
        cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=_scenarios(1),
            n_steps=10,
            n_repetitions=1,
            njobs=2,
            executor="thread",
            process_isolation=True,
            path=tmp_path / "should_not_run",
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
        )


def test_thread_executor_warns_on_finite_external_timeout(tmp_path):
    with pytest.warns(NegmasInfiniteNegotiationWarning):
        cartesian_tournament(
            competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
            scenarios=_scenarios(1),
            n_steps=10,
            n_repetitions=1,
            njobs=2,
            executor="thread",
            external_timeout=5,
            path=tmp_path / "warns_timeout",
            verbosity=0,
            rotate_ufuns=False,
            self_play=True,
            plot_fraction=0.0,
            save_scenario_figs=False,
        )
