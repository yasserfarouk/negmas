"""Unit tests for ``negmas.helpers.parallel.run_isolated_tasks``.

This is the centralized process-isolation primitive used by both the Cartesian
and the ``Tournament``-based tournament runners. It must:

- run picklable tasks in worker processes and collect their results
- enforce a per-task timeout, killing a hung (even CPU-bound) task while the
  remaining tasks keep running
- recycle workers after ``max_tasks`` tasks without losing results
- surface task exceptions via ``on_error``
- fall back to in-process execution for payloads that cannot be serialized
"""

from __future__ import annotations

import threading

from negmas.helpers.parallel import run_isolated_tasks

# --- module-level task functions (must be importable in spawned workers) ---


def _square(x):
    return x * x


def _boom(x):
    raise ValueError(f"boom-{x}")


def _forever(x):
    y = 0
    while True:
        y += 1


def _ignore_arg(_obj, value):
    return value


def _collectors():
    results: dict = {}
    timeouts: list = []
    errors: list = []

    def on_result(info, result, i, n):
        results[info] = result

    def on_timeout(info, i, n):
        timeouts.append(info)

    def on_error(exc, info, i, n):
        errors.append((info, type(exc).__name__))

    return results, timeouts, errors, on_result, on_timeout, on_error


def test_all_picklable_tasks_run_and_results_collected():
    results, timeouts, errors, on_result, on_timeout, on_error = _collectors()
    tasks = [(i, _square, (i,), {}) for i in range(10)]
    n = run_isolated_tasks(
        tasks,
        max_workers=3,
        timeout=20,
        max_tasks=4,  # force a couple of worker recycles
        on_result=on_result,
        on_timeout=on_timeout,
        on_error=on_error,
    )
    assert n == 10
    assert results == {i: i * i for i in range(10)}
    assert timeouts == []
    assert errors == []


def test_hung_task_is_killed_and_others_continue():
    results, timeouts, errors, on_result, on_timeout, on_error = _collectors()
    # task 2 hangs in a pure CPU loop; everything else must still complete
    tasks = []
    for i in range(6):
        fn = _forever if i == 2 else _square
        tasks.append((i, fn, (i,), {}))
    run_isolated_tasks(
        tasks,
        max_workers=2,
        timeout=2,
        max_tasks=0,
        on_result=on_result,
        on_timeout=on_timeout,
        on_error=on_error,
    )
    assert timeouts == [2]
    assert set(results.keys()) == {0, 1, 3, 4, 5}
    assert results[5] == 25
    assert errors == []


def test_task_exception_routed_to_on_error():
    results, timeouts, errors, on_result, on_timeout, on_error = _collectors()
    tasks = [(0, _square, (3,), {}), (1, _boom, (1,), {}), (2, _square, (4,), {})]
    run_isolated_tasks(
        tasks,
        max_workers=2,
        timeout=20,
        on_result=on_result,
        on_timeout=on_timeout,
        on_error=on_error,
    )
    assert results == {0: 9, 2: 16}
    assert timeouts == []
    assert errors == [(1, "ValueError")]


def test_unpicklable_payload_falls_back_in_process():
    results, timeouts, errors, on_result, on_timeout, on_error = _collectors()
    # a threading.Lock cannot be (cloud)pickled -> this task must fall back to
    # in-process execution instead of crashing the run.
    lock = threading.Lock()
    tasks = [
        (0, _square, (5,), {}),
        (1, _ignore_arg, (lock, "fallback-ran"), {}),
        (2, _square, (6,), {}),
    ]
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        run_isolated_tasks(
            tasks,
            max_workers=2,
            timeout=20,
            on_result=on_result,
            on_timeout=on_timeout,
            on_error=on_error,
        )
    assert results.get(1) == "fallback-ran"  # ran despite being unpicklable
    assert results.get(0) == 25
    assert results.get(2) == 36
    assert errors == []


def test_empty_tasks_is_noop():
    results, timeouts, errors, on_result, on_timeout, on_error = _collectors()
    n = run_isolated_tasks(
        [], max_workers=2, timeout=5, on_result=on_result, on_error=on_error
    )
    assert n == 0
    assert results == {}
