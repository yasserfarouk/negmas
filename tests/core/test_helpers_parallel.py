"""Tests for negmas.helpers.parallel."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import pytest

from negmas.helpers.parallel import (
    MAX_TASKS_PER_CHILD,
    TERMINATION_WAIT_TIME,
    make_process_executor,
    parse_parallelism,
    resolve_cpus,
    run_parallel_tasks,
    run_serial_tasks,
)


# -- module-level callables (must be picklable for ProcessPoolExecutor) --


def _add(a, b):
    return a + b


def _sleep_then_return(seconds, value):
    time.sleep(seconds)
    return value


def _raise(msg):
    raise RuntimeError(msg)


def _slow_for_timeout():
    time.sleep(30.0)
    return "should not get here"


# -- defaults --


def test_module_defaults():
    assert MAX_TASKS_PER_CHILD is None
    assert TERMINATION_WAIT_TIME == 10.0


# -- resolve_cpus --


def test_resolve_cpus_none_or_zero_returns_cpu_count():
    n = cpu_count() or 4
    assert resolve_cpus(None) == n
    assert resolve_cpus(0) == n


def test_resolve_cpus_positive_capped_at_cpu_count():
    n = cpu_count() or 4
    assert resolve_cpus(1) == 1
    assert resolve_cpus(n * 10) == n


def test_resolve_cpus_negative_returns_one():
    # callers branch to serial before calling, but the value must still be safe
    assert resolve_cpus(-1) == 1


# -- parse_parallelism --


def test_parse_parallelism_kind_only():
    kind, mw = parse_parallelism("parallel")
    assert kind == "parallel"
    assert mw is None


def test_parse_parallelism_with_fraction():
    n = cpu_count() or 1
    kind, mw = parse_parallelism("parallel:1.0")
    assert kind == "parallel"
    assert mw == max(1, int(1.0 * n))


def test_parse_parallelism_small_fraction_floors_to_one():
    kind, mw = parse_parallelism("parallel:0.0001")
    assert kind == "parallel"
    assert mw == 1


def test_parse_parallelism_dask_kind():
    kind, mw = parse_parallelism("dask")
    assert kind == "dask"
    assert mw is None


# -- make_process_executor --


def test_make_process_executor_returns_executor():
    ex = make_process_executor(max_workers=2)
    try:
        assert isinstance(ex, ProcessPoolExecutor)
        fut = ex.submit(_add, 1, 2)
        assert fut.result(timeout=15) == 3
    finally:
        ex.shutdown()


def test_make_process_executor_default_max_tasks_per_child_is_none():
    # On Py >= 3.11, max_tasks_per_child=None means no recycling. We can't
    # observe the kwarg directly after construction, but we can verify that
    # the executor still runs more tasks than any "small" recycle limit.
    ex = make_process_executor(max_workers=2)
    try:
        results = [ex.submit(_add, i, 1).result(timeout=15) for i in range(20)]
    finally:
        ex.shutdown()
    assert results == [i + 1 for i in range(20)]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="max_tasks_per_child added in 3.11"
)
def test_make_process_executor_honours_explicit_max_tasks_per_child():
    # Just make sure passing an integer doesn't blow up.
    ex = make_process_executor(max_workers=1, max_tasks_per_child=3)
    try:
        for i in range(5):
            assert ex.submit(_add, i, 0).result(timeout=15) == i
    finally:
        ex.shutdown()


# -- run_serial_tasks --


def test_run_serial_tasks_collects_results_in_order():
    tasks = [(i, _add, (i, 10), {}) for i in range(5)]
    results = []

    def on_result(info, result, i, n):
        results.append((info, result, i, n))

    n = run_serial_tasks(tasks, on_result=on_result)
    assert n == 5
    assert [r[0] for r in results] == list(range(5))
    assert [r[1] for r in results] == [i + 10 for i in range(5)]
    assert all(r[3] == 5 for r in results)


def test_run_serial_tasks_routes_errors_to_on_error():
    tasks = [
        ("ok", _add, (1, 2), {}),
        ("bad", _raise, ("boom",), {}),
        ("ok2", _add, (3, 4), {}),
    ]
    results = []
    errors = []

    n = run_serial_tasks(
        tasks,
        on_result=lambda info, res, i, total: results.append((info, res)),
        on_error=lambda exc, info, i, total: errors.append((info, str(exc))),
    )
    assert n == 3
    assert results == [("ok", 3), ("ok2", 7)]
    assert errors == [("bad", "boom")]


def test_run_serial_tasks_respects_total_timeout():
    tasks = [(i, _sleep_then_return, (0.2, i), {}) for i in range(10)]
    results = []
    t0 = time.perf_counter()
    run_serial_tasks(
        tasks,
        on_result=lambda info, res, i, n: results.append(info),
        total_timeout=0.5,
    )
    elapsed = time.perf_counter() - t0
    # we should have stopped well before all 10 finished (~2s)
    assert elapsed < 1.5
    assert len(results) < 10


# -- run_parallel_tasks --


def test_run_parallel_tasks_success_path():
    tasks = [(i, _add, (i, 100), {}) for i in range(8)]
    results = {}
    with make_process_executor(max_workers=2) as pool:
        n = run_parallel_tasks(
            tasks,
            executor=pool,
            on_result=lambda info, res, i, total: results.update({info: res}),
        )
    assert n == 8
    assert results == {i: i + 100 for i in range(8)}


def test_run_parallel_tasks_routes_errors_to_on_error():
    tasks = [
        ("ok1", _add, (1, 1), {}),
        ("bad", _raise, ("kaboom",), {}),
        ("ok2", _add, (2, 2), {}),
    ]
    oks = {}
    errs = {}
    with make_process_executor(max_workers=2) as pool:
        run_parallel_tasks(
            tasks,
            executor=pool,
            on_result=lambda info, res, i, n: oks.update({info: res}),
            on_error=lambda exc, info, i, n: errs.update({info: type(exc).__name__}),
        )
    assert oks == {"ok1": 2, "ok2": 4}
    assert errs == {"bad": "RuntimeError"}


def test_run_parallel_tasks_routes_timeout_to_on_timeout():
    # ``concurrent.futures.as_completed`` only yields completed futures, so
    # ``f.result(timeout=X)`` never triggers ``TimeoutError`` in the normal
    # path. To exercise the on_timeout branch we feed the helper an
    # ``as_completed`` shim that yields the futures *before* they finish.
    tasks = [
        ("fast", _add, (1, 1), {}),
        ("slow", _slow_for_timeout, (), {}),
    ]

    def yield_immediately(fs):
        # yield in submit order, not completion order
        for f in fs:
            yield f

    oks = {}
    timeouts = []
    with make_process_executor(max_workers=2) as pool:
        run_parallel_tasks(
            tasks,
            executor=pool,
            timeout=0.3,
            as_completed_fn=yield_immediately,
            on_result=lambda info, res, i, n: oks.update({info: res}),
            on_timeout=lambda info, fut, i, n: timeouts.append(info),
            on_error=lambda exc, info, i, n: timeouts.append(("err", info)),
        )
        # forcibly kill the slow worker so shutdown doesn't hang
        for p in list(pool._processes.values()):  # type: ignore[attr-defined]
            try:
                p.terminate()
            except Exception:
                pass
    # 'fast' may have completed by the time we read it; 'slow' must time out
    assert "slow" in timeouts


def test_run_parallel_tasks_passes_info_through_unchanged():
    sentinel = object()
    tasks = [(sentinel, _add, (5, 6), {})]
    seen = []
    with make_process_executor(max_workers=1) as pool:
        run_parallel_tasks(
            tasks,
            executor=pool,
            on_result=lambda info, res, i, n: seen.append((info, res)),
        )
    assert seen == [(sentinel, 11)]


def test_run_parallel_tasks_track_wrapper_is_called():
    tasks = [(i, _add, (i, 1), {}) for i in range(3)]
    seen_total = []

    def fake_track(iterator, total=None, description=None):
        seen_total.append((total, description))
        return iterator

    with make_process_executor(max_workers=1) as pool:
        run_parallel_tasks(
            tasks,
            executor=pool,
            track=fake_track,
            description="hello",
        )
    assert seen_total == [(3, "hello")]


def test_run_parallel_tasks_respects_total_timeout():
    # 16 tasks of 0.3s with 2 workers would take ~2.4s; total_timeout=0.3
    # should bail after the first completion or two, well short of 16.
    tasks = [(i, _sleep_then_return, (0.3, i), {}) for i in range(16)]
    results = []
    with make_process_executor(max_workers=2) as pool:
        run_parallel_tasks(
            tasks,
            executor=pool,
            total_timeout=0.3,
            on_result=lambda info, res, i, n: results.append(info),
        )
    assert len(results) < 16
