"""Helpers for running work serially or in a process pool.

Centralizes the ``ProcessPoolExecutor`` configuration used by tournaments
(``negmas.tournaments.tournaments``) and Cartesian negotiation tournaments
(``negmas.tournaments.neg.simple.cartesian``) so both go through the same
code path. ``MAX_TASKS_PER_CHILD`` is kept ``None`` to avoid a Python 3.11
``ProcessPoolExecutor`` deadlock that hits when a worker is recycled while
the main thread is blocked in ``as_completed``.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import cpu_count
from typing import Any, Callable, Iterable

__all__ = [
    "MAX_TASKS_PER_CHILD",
    "TERMINATION_WAIT_TIME",
    "resolve_cpus",
    "parse_parallelism",
    "make_process_executor",
    "kill_future_process",
    "run_parallel_tasks",
    "run_serial_tasks",
]

MAX_TASKS_PER_CHILD: int | None = None
TERMINATION_WAIT_TIME: float = 10.0


def _n_cores(default: int = 4) -> int:
    n = cpu_count()
    return n if n else default


def resolve_cpus(njobs: int | None) -> int:
    """Convert an ``njobs`` value to a concrete worker count.

    Semantics match the existing call sites in ``cartesian.py``:

    - ``njobs is None`` or ``njobs == 0`` -> use all available cores
    - ``njobs > 0`` -> ``min(cpu_count(), njobs)``
    - ``njobs < 0`` -> caller is expected to branch to serial; we still
      return ``1`` so the value is safe to feed to an executor.
    """
    n_cores = _n_cores()
    if njobs is None or njobs == 0:
        return n_cores
    if njobs < 0:
        return 1
    return min(n_cores, njobs)


def parse_parallelism(method: str) -> tuple[str, int | None]:
    """Parse a parallelism specifier of the form ``"kind[:fraction]"``.

    Returns ``(kind, max_workers)``. When no fraction is given the returned
    ``max_workers`` is ``None`` and the executor will pick a default.
    """
    parts = method.split(":")
    kind = parts[0]
    if len(parts) == 1:
        return kind, None
    fraction = float(parts[-1])
    return kind, max(1, int(fraction * _n_cores(default=1)))


def _supports_max_tasks_per_child() -> bool:
    v = sys.version_info
    return v.major > 3 or (v.major == 3 and v.minor > 10)


def make_process_executor(
    *,
    max_workers: int | None = None,
    max_tasks_per_child: int | None = MAX_TASKS_PER_CHILD,
) -> ProcessPoolExecutor:
    """Create a ``ProcessPoolExecutor`` with safe defaults for negmas.

    On Python <= 3.10 ``max_tasks_per_child`` is silently dropped (the
    parameter does not exist). On newer Pythons it is forwarded as-is;
    the default ``None`` disables worker recycling, which is required to
    avoid the 3.11 ``ProcessPoolExecutor`` deadlock during worker
    replacement.
    """
    kwargs: dict[str, Any] = {"max_workers": max_workers}
    if _supports_max_tasks_per_child():
        kwargs["max_tasks_per_child"] = max_tasks_per_child
    return ProcessPoolExecutor(**kwargs)


def kill_future_process(
    future: futures.Future,
    pool: ProcessPoolExecutor,
    wait_time: float = TERMINATION_WAIT_TIME,
) -> bool:
    """Best-effort termination of the worker process running ``future``.

    Returns ``True`` if the kill sequence completed without raising. On
    POSIX a ``SIGTERM`` is sent first, then ``SIGKILL`` if the process is
    still alive after ``wait_time`` seconds. On Windows the worker is
    terminated directly via the multiprocessing handle.
    """
    try:
        pid = getattr(future, "_process_ident", None)
        if pid is None:
            return False
        if os.name == "nt":
            pool._processes[pid].terminate()  # type: ignore[attr-defined]
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(wait_time)
            proc = pool._processes.get(pid)  # type: ignore[attr-defined]
            if proc is not None and not proc.is_alive():
                os.kill(pid, signal.SIGKILL)
        return True
    except Exception:
        return False


def run_serial_tasks(
    tasks: Iterable[tuple[Any, Callable[..., Any], tuple, dict]],
    *,
    on_result: Callable[[Any, Any, int, int], None] | None = None,
    on_error: Callable[[BaseException, Any, int, int], None] | None = None,
    total_timeout: float | None = None,
    track: Callable[..., Iterable] | None = None,
    description: str = "Running",
) -> int:
    """Run ``tasks`` serially, mirroring :func:`run_parallel_tasks`.

    Each task is ``(info, fn, args, kwargs)``. ``info`` is opaque and
    passed back to the callbacks unchanged. Returns the number of tasks
    that were submitted (regardless of how many completed).
    """
    materialized = list(tasks)
    n = len(materialized)
    iterator: Iterable = enumerate(materialized)
    if track is not None:
        iterator = track(iterator, total=n, description=description)
    strt = time.perf_counter()
    for i, (info, fn, args, kwargs) in iterator:
        if total_timeout is not None and time.perf_counter() - strt > total_timeout:
            break
        try:
            result = fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - surface to user callback
            if on_error is not None:
                on_error(e, info, i, n)
            continue
        if on_result is not None:
            on_result(info, result, i, n)
    return n


def run_parallel_tasks(
    tasks: Iterable[tuple[Any, Callable[..., Any], tuple, dict]],
    *,
    executor: ProcessPoolExecutor,
    timeout: float | None = None,
    total_timeout: float | None = None,
    as_completed_fn: Callable[..., Iterable[futures.Future]] | None = None,
    on_result: Callable[[Any, Any, int, int], None] | None = None,
    on_timeout: Callable[[Any, futures.Future, int, int], None] | None = None,
    on_broken_pool: Callable[[BaseException, int, int], bool] | None = None,
    on_error: Callable[[BaseException, Any, int, int], None] | None = None,
    track: Callable[..., Iterable] | None = None,
    description: str = "Running",
) -> int:
    """Submit ``tasks`` to ``executor`` and drive the ``as_completed`` loop.

    Each task is ``(info, fn, args, kwargs)`` -- ``info`` is opaque
    metadata round-tripped to the callbacks. The driver:

    - submits every task to ``executor``
    - iterates results via ``as_completed_fn`` (defaults to
      :func:`concurrent.futures.as_completed`)
    - applies ``timeout`` as a per-future result timeout
    - breaks out of the loop if ``total_timeout`` elapses
    - dispatches outcomes to the callbacks; for ``BrokenProcessPool`` the
      driver breaks the loop iff ``on_broken_pool`` returns truthy

    The driver does **not** shut the executor down; the caller is
    responsible for that (typically via a ``with`` block).
    """
    if as_completed_fn is None:
        as_completed_fn = futures.as_completed
    future_to_info: dict[futures.Future, Any] = {}
    for info, fn, args, kwargs in tasks:
        future_to_info[executor.submit(fn, *args, **kwargs)] = info
    n = len(future_to_info)
    iterator: Iterable = as_completed_fn(list(future_to_info.keys()))
    if track is not None:
        iterator = track(iterator, total=n, description=description)
    strt = time.perf_counter()
    for i, f in enumerate(iterator):
        if total_timeout is not None and time.perf_counter() - strt > total_timeout:
            break
        info = future_to_info.get(f)
        try:
            result = f.result(timeout=timeout)
        except FuturesTimeoutError:
            if on_timeout is not None:
                on_timeout(info, f, i, n)
            continue
        except BrokenProcessPool as e:
            if on_broken_pool is not None and on_broken_pool(e, i, n):
                break
            continue
        except Exception as e:  # noqa: BLE001 - surface to user callback
            if on_error is not None:
                on_error(e, info, i, n)
            continue
        if on_result is not None:
            on_result(info, result, i, n)
    return n
