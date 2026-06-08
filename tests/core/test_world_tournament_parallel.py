"""Wiring test for the ``Tournament``-based (world) runner after routing its
non-dask parallel path through ``run_isolated_tasks``.

A full-stack world tournament currently can't be asserted end-to-end here (the
world result-saving path has a pre-existing pyarrow serialization issue that
also affects serial runs), so we test the wiring directly: that
``_run_parallel_isolated`` builds one isolated task per (not-yet-run) world set,
computes the per-world timeout, and routes results/timeouts to
``save_run_results`` / the progress callback. The timeout/kill behavior itself
is covered by tests/core/test_isolated_tasks.py.
"""

from __future__ import annotations

import negmas.tournaments.tournaments as T
from negmas.outcomes import make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.sao import AspirationNegotiator, NaiveTitForTatNegotiator
from negmas.situated.neg import Condition
from negmas.tournaments.neg import neg_tournament, scenarios_from_list


def test_world_tournament_completes_and_saves(tmp_path):
    """End-to-end: a small world tournament runs and persists results without a
    pyarrow error. This exercises the real save path (negotiation/world stats ->
    parquet), which previously failed because records stored Agent objects (e.g.
    the ``caller`` column) -- now fixed by storing ids and by the hardened
    parquet sanitizers.
    """
    issues = (make_issue(5, "quantity"), make_issue(5, "price"))
    competitors = [AspirationNegotiator, NaiveTitForTatNegotiator]
    ufuns = (
        U.random(issues=issues, reserved_value=0.0, normalized=False),
        U.random(issues=issues, reserved_value=0.0, normalized=False),
    )
    scenarios = [
        Condition(
            name="d0", issues=issues, ufuns=ufuns, partner_types=(partner,), index=index
        )
        for index in range(2)
        for partner in competitors
    ]
    results = neg_tournament(
        n_configs=len(scenarios),
        scenarios=scenarios_from_list(scenarios),
        competitors=competitors,
        n_steps=1,
        neg_n_steps=10,
        neg_time_limit=None,
        parallelism="serial",
        tournament_path=tmp_path / "wt",
        verbose=False,
    )
    assert results is not None
    assert len(results.scores) > 0


def test_run_parallel_isolated_wiring(monkeypatch):
    captured: dict = {}

    def fake_run_isolated(
        tasks,
        *,
        max_workers,
        timeout,
        total_timeout,
        on_result,
        on_timeout,
        on_error,
        track,
        description,
    ):
        tasks = list(tasks)
        captured.update(
            tasks=tasks,
            timeout=timeout,
            total_timeout=total_timeout,
            max_workers=max_workers,
            on_result=on_result,
            on_timeout=on_timeout,
            on_error=on_error,
        )
        return len(tasks)

    saved: list = []
    progress: list = []
    monkeypatch.setattr(T, "run_isolated_tasks", fake_run_isolated)
    monkeypatch.setattr(T, "save_run_results", lambda *a, **k: saved.append(a))

    # Two world-sets with different per-world time limits.
    assigned = [
        [{"world_params": {"time_limit": 5.0}}],
        [{"world_params": {"time_limit": 3.0}}],
    ]

    T._run_parallel_isolated(
        parallelism="parallel",
        verbose=False,
        assigned=assigned,
        world_generator=lambda **k: [],
        tournament_progress_callback=lambda *a: progress.append(a),
        world_progress_callback=None,
        name="t",
        score_calculator=lambda *a: None,
        run_ids=set(),
        print_exceptions=False,
        override_ran_worlds=False,
        attempts_path=None,
        total_timeout=None,
        max_attempts=1,
    )

    # one isolated task per world-set, each running _run_worlds
    assert len(captured["tasks"]) == 2
    info, fn, args, kwargs = captured["tasks"][0]
    assert fn is T._run_worlds
    assert "run_id" in info
    # per-world timeout = largest world time_limit x TIMEOUT_EXTRA
    assert abs(captured["timeout"] - 5.0 * T.TIMEOUT_EXTRA) < 1e-9

    # on_result unpacks the (run_id, world_paths, score, *stats) tuple -> save
    captured["on_result"]({"run_id": "x"}, ("rid", [], None, None, None, None), 0, 2)
    assert len(saved) == 1
    # on_timeout reports progress as a non-result
    captured["on_timeout"]({"run_id": "x"}, 1, 2)
    assert progress[-1] == (None, 1, 2)


def test_run_parallel_isolated_skips_already_run(monkeypatch):
    captured: dict = {}

    def fake_run_isolated(tasks, **kwargs):
        captured["tasks"] = list(tasks)
        return len(captured["tasks"])

    monkeypatch.setattr(T, "run_isolated_tasks", fake_run_isolated)
    monkeypatch.setattr(T, "save_run_results", lambda *a, **k: None)

    assigned = [
        [{"world_params": {"time_limit": 5.0}}],
        [{"world_params": {"time_limit": 3.0}}],
    ]
    already = {T._hash(assigned[0])}  # mark the first as already run

    T._run_parallel_isolated(
        parallelism="parallel",
        verbose=False,
        assigned=assigned,
        world_generator=lambda **k: [],
        tournament_progress_callback=None,
        world_progress_callback=None,
        name="t",
        score_calculator=lambda *a: None,
        run_ids=already,
        print_exceptions=False,
        override_ran_worlds=False,
        attempts_path=None,
        total_timeout=None,
        max_attempts=1,
    )
    # only the not-yet-run world-set is scheduled
    assert len(captured["tasks"]) == 1
