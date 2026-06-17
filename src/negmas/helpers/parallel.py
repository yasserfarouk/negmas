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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures import wait as futures_wait
from concurrent.futures.process import BrokenProcessPool
from math import isinf
from multiprocessing import cpu_count
from typing import Any, Callable, Iterable

__all__ = [
    "MAX_TASKS_PER_CHILD",
    "TERMINATION_WAIT_TIME",
    "DEFAULT_MAX_TASKS_PER_WORKER",
    "resolve_cpus",
    "parse_parallelism",
    "make_process_executor",
    "kill_future_process",
    "run_parallel_tasks",
    "run_serial_tasks",
    "run_isolated_tasks",
]

MAX_TASKS_PER_CHILD: int | None = None
TERMINATION_WAIT_TIME: float = 10.0
# Recycle each worker process after this many negotiations by default. Bounds
# memory growth (a worker that runs thousands of negotiations accumulates
# state/leaks) while amortizing the process-spawn cost over many tasks: the
# per-task overhead is roughly ``spawn_cost / DEFAULT_MAX_TASKS_PER_WORKER``.
DEFAULT_MAX_TASKS_PER_WORKER: int = 50


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


def _cloud_runner(payload: bytes):
    """Worker entry point: deserialize a cloudpickled ``(fn, args, kwargs)`` and run it.

    Payloads are serialized with cloudpickle (not stdlib pickle) so locally
    defined negotiators and closure/lambda ufuns survive the trip to the worker
    process. Defined at module level so it is importable in spawned workers.
    """
    import cloudpickle

    fn, args, kwargs = cloudpickle.loads(payload)
    return fn(*args, **kwargs)


class UnserializableTaskError(Exception):
    """A task could not be serialized for process isolation.

    Raised/surfaced when a task's payload (negotiators, ufuns, scenario, ...)
    cannot be ``cloudpickle``-serialized, so it cannot run in an isolated,
    killable worker process. When the in-process fallback is disabled the task
    is failed with this error instead of running unprotected in-process.
    """


def _task_label(info: Any, i: int) -> str:
    """Best-effort human label for a task, preferring negotiator class names.

    Used in the unserializable-task message so the offending agents are named
    explicitly rather than shown as an opaque index."""
    if isinstance(info, dict):
        for key in ("partner_names", "partners"):
            vals = info.get(key)
            if vals:
                names = [
                    getattr(p, "__name__", None) or getattr(p, "name", None) or str(p)
                    for p in vals
                ]
                return ", ".join(str(_) for _ in names)
        run_id = info.get("run_id")
        if run_id:
            return f"run {run_id}"
    return f"task {i}"


def _warn_unserializable(message: str) -> None:
    """Surface an unserializable-task event unconditionally.

    Printed straight to ``stderr`` (flushed) *and* raised through negmas'
    warning system, so it is always visible even when Python's warning filters
    would otherwise dedupe or suppress ``warnings.warn``."""
    from negmas import warnings as _warnings

    try:
        print(f"NEGMAS WARNING: {message}", file=sys.stderr, flush=True)
    except Exception:
        pass
    try:
        _warnings.warn(message, _warnings.NegmasUnexpectedValueWarning)
    except Exception:
        pass


def run_isolated_tasks(
    tasks: Iterable[tuple[Any, Callable[..., Any], tuple, dict]],
    *,
    max_workers: int,
    timeout: float | None = None,
    total_timeout: float | None = None,
    max_tasks: int | None = DEFAULT_MAX_TASKS_PER_WORKER,
    window: int | None = None,
    on_result: Callable[[Any, Any, int, int], None] | None = None,
    on_timeout: Callable[[Any, int, int], None] | None = None,
    on_error: Callable[[BaseException, Any, int, int], None] | None = None,
    track: Callable[..., Iterable] | None = None,
    description: str = "Running",
    allow_inline_fallback: bool = True,
) -> int:
    """Run ``tasks`` in isolated worker processes with a real per-task timeout.

    Each task is ``(info, fn, args, kwargs)`` where ``info`` is opaque metadata
    round-tripped to the callbacks. Unlike :func:`run_parallel_tasks` (which
    drives a shared ``ProcessPoolExecutor`` via ``as_completed`` and can neither
    detect a hung task nor survive killing one), this driver uses a
    ``pebble.ProcessPool``:

    - **Per-task timeout that actually fires.** ``timeout`` is enforced by pebble
      from the moment a task starts executing (queued tasks are not penalized).
      On timeout pebble kills the worker process running the task and replaces
      it transparently, so a CPU-bound negotiator stuck in an infinite loop
      (even inside a C extension) is terminated and the remaining tasks keep
      running.
    - **Bounded memory.** Each worker is recycled after ``max_tasks`` tasks
      (default :data:`DEFAULT_MAX_TASKS_PER_WORKER`), so per-negotiation state
      cannot accumulate without bound. Pass ``max_tasks=0`` to reuse workers
      forever (fastest, but unbounded memory).
    - **Optional whole-run budget.** ``total_timeout`` caps the total wall-clock
      time across all tasks; once exceeded the pool is stopped (running workers
      terminated) and the remaining tasks are abandoned.
    - **Bounded scheduling window.** At most ``window`` payloads are serialized
      and in flight at once (default ``2 * max_workers``), so a tournament with
      very many negotiations does not serialize them all up front.
    - **Local ufuns/negotiators.** Payloads are serialized with cloudpickle, so
      closures, lambdas and locally defined classes work.

    Tasks whose payload cannot be serialized at all cannot run in an isolated,
    killable worker. Behaviour is controlled by ``allow_inline_fallback``:

    - ``True`` (default): such a task is run **in-process** as a best-effort
      fallback. A warning naming the offending negotiators and the exact
      serialization error is always emitted. Pure-Python infinite loops can
      still be interrupted via the thread-injection path, but a C-extension /
      CPU-bound hang in an unpicklable task cannot be isolated and will block.
    - ``False``: such a task is **not** run in-process. A warning is emitted
      and the task is failed via ``on_error`` with an
      :class:`UnserializableTaskError`, so the caller records it as a failed
      run (e.g. a no-agreement / reserved-value outcome) instead of risking an
      unkillable in-process hang.

    Callbacks:
        - ``on_result(info, result, i, n)`` on success
        - ``on_timeout(info, i, n)`` when the task exceeded ``timeout``
        - ``on_error(exc, info, i, n)`` on any other failure (including an
          unserializable task when ``allow_inline_fallback`` is ``False``)

    Args:
        allow_inline_fallback: When ``True`` (default), tasks that cannot be
            serialized run in-process without a hard timeout guarantee. When
            ``False``, they are failed via ``on_error`` instead so no negotiation
            can run unprotected by the per-task timeout.

    Returns the number of tasks submitted.
    """
    from pebble import ProcessExpired, ProcessPool

    from negmas import warnings as _warnings

    materialized = list(tasks)
    n = len(materialized)
    if n == 0:
        return 0
    if max_workers is None or max_workers < 1:
        max_workers = 1
    if window is None:
        window = max(2, max_workers * 2)
    if timeout is not None and (isinf(timeout) or timeout <= 0):
        timeout = None
    if total_timeout is not None and (isinf(total_timeout) or total_timeout <= 0):
        total_timeout = None
    if timeout is None:
        _warnings.warn(
            "run_isolated_tasks was given no finite per-task timeout. Worker "
            "processes still bound memory, but a negotiation that never returns "
            "(e.g. an infinite loop) will block the run until it finishes. Pass "
            "a finite timeout (e.g. external_timeout) to guarantee termination.",
            _warnings.NegmasInfiniteNegotiationWarning,
        )

    # pebble uses 0 to mean "no recycling"
    pebble_max_tasks = max_tasks if max_tasks and max_tasks > 0 else 0

    def _drive():
        import cloudpickle

        task_iter = iter(enumerate(materialized))
        inflight: dict[Any, tuple[int, Any]] = {}
        inline: list[tuple[int, Any, Callable, tuple, dict]] = []
        # Tasks that could not be serialized while inline fallback is disabled;
        # failed via on_error after the pool drains so each becomes a recorded
        # failed run (reserved-value outcome) rather than an unkillable hang.
        unserializable: list[tuple[int, Any, BaseException]] = []

        with ProcessPool(max_workers=max_workers, max_tasks=pebble_max_tasks) as pool:

            def submit_next() -> bool:
                """Schedule the next serializable task. Unpicklable tasks are
                either queued for in-process fallback or recorded as failures
                (depending on ``allow_inline_fallback``); either way we keep
                consuming. Returns True if a pool task was scheduled, False when
                the input is exhausted."""
                while True:
                    try:
                        i, (info, fn, args, kwargs) = next(task_iter)
                    except StopIteration:
                        return False
                    try:
                        payload = cloudpickle.dumps((fn, args, kwargs))
                    except Exception as exc:
                        label = _task_label(info, i)
                        cause = f"{type(exc).__name__}: {exc}"
                        if allow_inline_fallback:
                            _warn_unserializable(
                                f"Cannot serialize negotiation task for process "
                                f"isolation [{label}]: {cause}. Running it "
                                "in-process: the per-task timeout cannot kill a "
                                "C-extension/CPU-bound hang in it, so it may "
                                "block. Make the negotiators/ufuns picklable to "
                                "enable process isolation."
                            )
                            inline.append((i, info, fn, args, kwargs))
                        else:
                            _warn_unserializable(
                                f"Cannot serialize negotiation task for process "
                                f"isolation [{label}]: {cause}. Inline fallback "
                                "is disabled; failing this negotiation so every "
                                "negotiator receives its reserved value."
                            )
                            unserializable.append(
                                (
                                    i,
                                    info,
                                    UnserializableTaskError(
                                        f"Task [{label}] could not be serialized "
                                        f"for process isolation: {cause}"
                                    ),
                                )
                            )
                        continue
                    future = pool.schedule(
                        _cloud_runner, args=(payload,), timeout=timeout
                    )
                    inflight[future] = (i, info)
                    return True

            for _ in range(window):
                if not submit_next():
                    break

            start = time.perf_counter()
            while inflight:
                if (
                    total_timeout is not None
                    and time.perf_counter() - start > total_timeout
                ):
                    # Whole-run budget exhausted: stop the pool (terminates any
                    # running workers) and abandon the rest.
                    pool.stop()
                    break
                done, _ = futures_wait(
                    list(inflight.keys()),
                    timeout=1.0 if total_timeout is not None else None,
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    i, info = inflight.pop(future)
                    try:
                        result = future.result()
                    except FuturesTimeoutError:
                        if on_timeout is not None:
                            on_timeout(info, i, n)
                    except ProcessExpired as e:
                        if on_error is not None:
                            on_error(e, info, i, n)
                    except Exception as e:  # noqa: BLE001 - surface to callback
                        if on_error is not None:
                            on_error(e, info, i, n)
                    else:
                        if on_result is not None:
                            on_result(info, result, i, n)
                    yield
                    submit_next()

        # Unserializable tasks with inline fallback disabled: surface each as a
        # failure so the caller records a failed run (reserved-value outcome).
        for i, info, exc in unserializable:
            if on_error is not None:
                on_error(exc, info, i, n)
            yield

        # In-process fallback for tasks that could not be serialized. Best-effort
        # timeout via the shared thread pool (pure-Python loops only).
        if inline:
            import functools

            from negmas.helpers.timeout import TimeoutCaller, TimeoutError

            for i, info, fn, args, kwargs in inline:
                call = functools.partial(fn, *args, **kwargs)
                try:
                    if timeout is None:
                        result = call()
                    else:
                        result = TimeoutCaller.run(call, timeout=timeout)
                except TimeoutError:
                    if on_timeout is not None:
                        on_timeout(info, i, n)
                except Exception as e:  # noqa: BLE001 - surface to callback
                    if on_error is not None:
                        on_error(e, info, i, n)
                else:
                    if on_result is not None:
                        on_result(info, result, i, n)
                yield

    if track is not None:
        for _ in track(_drive(), total=n, description=description):
            pass
    else:
        for _ in _drive():
            pass
    return n
