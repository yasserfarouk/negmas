"""Tests for error-aware resume: retry_failed_on_resume/retry_timedout_on_resume.

Uses storage_optimization="speed" so results/*.json survive a successful run
(the default "space"/"balanced" levels delete results/ once details/scores are
saved), then hand-edits one saved record to simulate a previously-failed or
previously-timed-out negotiation before resuming.

Notes on test configuration (both pre-existing, unrelated to
retry_failed_on_resume/retry_timedout_on_resume):

- ``njobs=2`` (not ``njobs=-1``): the serial/in-process dispatch branch
  recomputes each run's ``run_id`` a second time via ``get_run_id(info)`` on
  an ``info`` that already carries a ``run_id`` key, which changes the
  hashed string and yields a different id than the one
  ``_is_run_completed``/the saved file used. That pre-existing mismatch means
  ``path_exists="continue"`` cannot actually recognize completed runs in
  serial mode even on unmodified negmas (verified against the pre-change
  code); using the process-isolated dispatch path (``njobs>=0``) avoids it.
- ``normalize_ufuns=False``: the default ``normalize_ufuns=True`` calls
  ``ufun.normalize()`` on every ``cartesian_tournament`` call, and
  ``normalize()`` assigns the returned ufun a *new random id* every time it
  is called (confirmed: calling it twice on the same object yields two
  different ids). That makes each run's serialized/hashed content -- and
  thus its stable ``run_id`` -- different across separate top-level calls to
  ``cartesian_tournament``, even for byte-identical input scenarios. Setting
  ``normalize_ufuns=False`` keeps the passed-in ufuns (and their ids) as-is
  so run_ids are stable across the two calls these tests make.
- ``verbosity`` is also embedded into each run's ``mechanism_params``
  (``verbosity - 1``) and therefore into its hashed ``run_id``; tests that
  need a non-default ``verbosity`` on the resumed call (e.g. to capture a
  printed message) must use the same ``verbosity`` on the initial call too.
- When a resume call finds that *every* run is already accounted for (no run
  needs (re-)execution -- e.g. nothing was selected for retry),
  ``cartesian_tournament`` takes an early-return shortcut that loads the
  tournament-level cached ``details``/``all_scores`` files (written after the
  first, un-tampered run) via ``SimpleTournamentResults.load`` instead of
  reconstructing from the per-run ``results/*.json`` files. Since these tests
  hand-edit an individual ``results/*.json`` record but never re-save that
  stale top-level cache, a resume call that doesn't trigger any real
  re-execution would otherwise silently return the pre-edit cached data. Use
  ``_clear_cached_tournament_dataframes`` before such calls so the loader
  falls back to (re-)reading ``results/*.json``, which does reflect the
  edits.
"""

from __future__ import annotations

from negmas.helpers.inout import dump, load
from negmas.inout import Scenario
from negmas.outcomes import make_issue
from negmas.outcomes.outcome_space import make_os
from negmas.preferences import LinearAdditiveUtilityFunction as U
from negmas.preferences.value_fun import TableFun
from negmas.sao.mechanism import SAOMechanism
from negmas.sao.negotiators import AspirationNegotiator, NaiveTitForTatNegotiator
from negmas.tournaments.neg import cartesian_tournament
from negmas.tournaments.neg.simple.cartesian import RESULTS_DIR_NAME


def _scenarios(n: int = 1):
    """Deterministic, clearly-opposed (guaranteed-ZOPA) single-issue scenarios.

    Randomly drawn ufuns (the previous approach) can genuinely fail to reach
    agreement within ``n_steps=20`` between AspirationNegotiator and
    NaiveTitForTatNegotiator on an unlucky draw, making
    ``test_timedout_record_needs_its_own_flag``'s "retrying makes the
    timedout flag go away" assertion intermittently -- and spuriously --
    fail. Fixed, complementary linear preferences over a small outcome space
    converge quickly and deterministically every run.
    """
    issues = (make_issue([f"q{i}" for i in range(5)], "quantity"),)
    scenarios = []
    for i in range(n):
        os_ = make_os(issues, name=f"S{i}")
        scenarios.append(
            Scenario(
                outcome_space=os_,
                ufuns=(
                    U(
                        values=[TableFun({f"q{j}": j / 4 for j in range(5)})],
                        outcome_space=os_,
                        reserved_value=0.0,
                    ),
                    U(
                        values=[TableFun({f"q{j}": (4 - j) / 4 for j in range(5)})],
                        outcome_space=os_,
                        reserved_value=0.0,
                    ),
                ),
                mechanism_type=SAOMechanism,
                mechanism_params=dict(),
            )
        )
    return scenarios


def _run_small_tournament(path, scenarios, n_repetitions=1, **kwargs):
    params = dict(
        competitors=[AspirationNegotiator, NaiveTitForTatNegotiator],
        scenarios=scenarios,
        n_steps=20,
        n_repetitions=n_repetitions,
        njobs=2,  # process-isolated dispatch; see module docstring
        path=path,
        verbosity=0,
        rotate_ufuns=False,
        self_play=False,
        plot_fraction=0.0,
        save_scenario_figs=False,
        storage_optimization="speed",
        normalize_ufuns=False,  # keep ufun ids (and thus run_ids) stable; see module docstring
    )
    params.update(kwargs)
    return cartesian_tournament(**params)


def _result_files(path):
    return sorted((path / RESULTS_DIR_NAME).glob("*.json"))


def _clear_cached_tournament_dataframes(path):
    """See the module docstring's note on the early-return shortcut."""
    for pattern in ("details.*", "all_scores.*"):
        for f in path.glob(pattern):
            f.unlink()


def _force_clean_baseline(path):
    """Force every saved result's has_error/timedout to False.

    A short (n_steps=20) negotiation between AspirationNegotiator and
    NaiveTitForTatNegotiator can genuinely end without agreement (a real,
    non-faulty ``timedout``/no-agreement outcome), which would make an
    "only the hand-tampered record should show has_error/timedout" assertion
    flaky. Normalizing every record to a clean baseline first makes the
    hand-edited record the only one exhibiting the flag under test.
    """
    for f in path.glob(f"{RESULTS_DIR_NAME}/*.json"):
        record = load(f)
        record["has_error"] = False
        record["timedout"] = False
        dump(record, f)


def test_retry_failed_on_resume_reruns_and_deduplicates(tmp_path):
    path = tmp_path / "t"
    scenarios = _scenarios(1)
    results = _run_small_tournament(path, scenarios)
    n_rows_before = len(results.details)

    files = _result_files(path)
    assert len(files) == n_rows_before
    _force_clean_baseline(path)

    target = files[0]
    record = load(target)
    record["has_error"] = True
    record["error_details"] = "simulated failure for test"
    dump(record, target)

    resumed = _run_small_tournament(
        path, scenarios, path_exists="continue", retry_failed_on_resume=True
    )

    # Exactly one row for that run must appear -- not a duplicate from
    # failing to delete the stale file before re-running.
    assert len(resumed.details) == n_rows_before
    assert int(resumed.details["has_error"].sum()) == 0


def test_retry_failed_on_resume_false_skips_by_default(tmp_path, capsys):
    path = tmp_path / "t"
    scenarios = _scenarios(1)
    # verbosity is embedded into each run's mechanism_params (as
    # verbosity - 1) and therefore into its hashed run_id; it must be kept
    # identical across the initial and resumed calls or every run_id changes
    # and resume-matching breaks. Use verbosity=1 on both calls so we can
    # still capture the discoverability message on resume.
    results = _run_small_tournament(path, scenarios, verbosity=1)
    n_rows_before = len(results.details)

    files = _result_files(path)
    _force_clean_baseline(path)
    target = files[0]
    record = load(target)
    record["has_error"] = True
    record["error_details"] = "simulated failure for test"
    dump(record, target)
    _clear_cached_tournament_dataframes(path)

    resumed = _run_small_tournament(
        path,
        scenarios,
        path_exists="continue",
        retry_failed_on_resume=False,
        verbosity=1,
    )
    captured = capsys.readouterr()

    # The tampered record must still be reported as-is (skipped, not re-run).
    assert len(resumed.details) == n_rows_before
    assert int(resumed.details["has_error"].sum()) == 1
    assert "retry_failed_on_resume=True" in captured.out


def test_timedout_record_needs_its_own_flag(tmp_path):
    path = tmp_path / "t"
    scenarios = _scenarios(1)
    _run_small_tournament(path, scenarios)

    files = _result_files(path)
    _force_clean_baseline(path)
    target = files[0]
    record = load(target)
    record["has_error"] = False
    record["timedout"] = True
    record["error_details"] = "simulated timeout for test"
    dump(record, target)
    _clear_cached_tournament_dataframes(path)

    # retry_failed_on_resume alone must NOT retry a timedout-only record.
    not_retried = _run_small_tournament(
        path, scenarios, path_exists="continue", retry_failed_on_resume=True
    )
    assert int(not_retried.details["timedout"].sum()) == 1

    # re-tamper (the previous call did not touch it) and now retry with the
    # dedicated flag.
    record = load(target)
    assert record.get("timedout") is True
    retried = _run_small_tournament(
        path, scenarios, path_exists="continue", retry_timedout_on_resume=True
    )
    assert int(retried.details["timedout"].sum()) == 0


def test_only_the_tampered_repetition_is_rerun(tmp_path):
    path = tmp_path / "t"
    scenarios = _scenarios(1)
    results = _run_small_tournament(path, scenarios, n_repetitions=3)
    n_rows_before = len(results.details)

    files = _result_files(path)
    _force_clean_baseline(path)
    records = {f: load(f) for f in files}
    rep1_files = [f for f, r in records.items() if r.get("rep") == 1]
    assert len(rep1_files) >= 1
    target = rep1_files[0]

    other_files = [f for f in files if f != target]
    other_contents_before = {f: f.read_bytes() for f in other_files}

    record = load(target)
    record["has_error"] = True
    record["error_details"] = "simulated failure for test"
    dump(record, target)

    resumed = _run_small_tournament(
        path,
        scenarios,
        n_repetitions=3,
        path_exists="continue",
        retry_failed_on_resume=True,
    )

    assert len(resumed.details) == n_rows_before
    assert int(resumed.details["has_error"].sum()) == 0

    # Other repetitions' result files must be untouched.
    for f, content_before in other_contents_before.items():
        assert f.read_bytes() == content_before
