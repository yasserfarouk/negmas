#!/usr/bin/env python
"""
Utilities for handling timeout and async calls
"""

from __future__ import annotations

import atexit
import ctypes
import threading
from concurrent.futures import TimeoutError
from concurrent.futures import thread as thread
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import contextmanager

from negmas import warnings

__all__ = ["TimeoutError", "TimeoutCaller", "force_single_thread", "single_thread"]
SINGLE_THREAD_FORCED = False


class _NegmasTimeoutInterrupt(BaseException):
    """Injected into a worker thread when its call exceeds the timeout.

    Subclasses ``BaseException`` (not ``Exception``) so that ordinary
    ``except Exception`` handlers inside a negotiator do not accidentally
    swallow it and keep looping.
    """


def force_single_thread(on: bool = True):
    """
    Forces negmas to use a single thread for all internal calls.

    Remarks:
        - This will have the effect of not enforcing time-limits on calls.
        - Only use this with caution and for debugging.
    """
    global SINGLE_THREAD_FORCED
    SINGLE_THREAD_FORCED = on


@contextmanager
def single_thread():
    """Context manager that temporarily forces single-threaded execution."""
    force_single_thread(True)
    yield None
    force_single_thread(False)


def is_single_thread() -> bool:
    """Returns True if negmas is configured to use single-threaded execution."""
    return SINGLE_THREAD_FORCED


class TimeoutCaller:
    """Executes callables with timeout support using a shared thread pool."""

    pool = None

    @classmethod
    def get_pool(cls):
        """Returns the shared thread pool, creating it if necessary."""
        if cls.pool is None:
            cls.pool = ThreadPoolExecutor()
        return cls.pool

    @staticmethod
    def _async_raise(ident: int, exc: type[BaseException]) -> None:
        """Best-effort: asynchronously raise ``exc`` in the thread ``ident``.

        Uses ``PyThreadState_SetAsyncExc`` so a worker thread stuck in a
        *pure-Python* loop unwinds instead of leaking and burning CPU forever.

        Remarks:
            - This only takes effect at a Python bytecode boundary. A thread
              blocked inside a C extension (numpy, an optimizer, ``time.sleep``,
              blocking I/O) will NOT be interrupted by this. The only robust way
              to stop such code is to run it in a separate process and kill the
              process (see the tournament ``external_timeout`` path).
        """
        try:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(ident), ctypes.py_object(exc)
            )
            if res > 1:
                # Should never affect more than one thread; undo if it did.
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(ident), None)
        except Exception:
            pass

    @classmethod
    def run(cls, to_run, timeout: float):
        """
        Executes a callable with a timeout limit.

        Args:
            to_run: A callable to execute in the thread pool.
            timeout: Maximum seconds to wait before raising TimeoutError.

        Remarks:
            - On timeout the worker thread is asked to stop by injecting an
              exception into it (best-effort, pure-Python loops only). This
              prevents the thread from leaking and burning CPU after the call
              is abandoned. CPU-bound C-extension code cannot be interrupted
              this way; use process isolation for a hard guarantee.
        """
        if is_single_thread():
            return to_run()
        pool = cls.get_pool()
        ident_holder: dict[str, int] = {}

        def _wrapped():
            ident_holder["ident"] = threading.get_ident()
            return to_run()

        future = pool.submit(_wrapped)  # type: ignore (Probably ok)
        try:
            result = future.result(timeout)
            return result
        except TimeoutError as s:
            future.cancel()
            # The abandoned worker thread keeps running; ask it to unwind so we
            # do not leak it (best-effort; see _async_raise remarks).
            ident = ident_holder.get("ident")
            if ident is not None and not future.done():
                cls._async_raise(ident, _NegmasTimeoutInterrupt)
            raise s

    @classmethod
    def cleanup(cls):
        """Shuts down the thread pool and cleans up any pending threads."""
        if cls.pool is not None:
            try:
                cls.pool.shutdown(wait=False)
                for t in cls.pool._threads:
                    # we are using an undocumented private value here. DANGEROUS
                    del thread._threads_queues[t]  # type: ignore
            except Exception:
                warnings.warn(
                    "NegMAS have finished processing but there may be some "
                    "threads still hanging there!! If your program does "
                    "not die by itself. Please press Ctrl-c to kill it",
                    warnings.NegmasShutdownWarning,
                )


def cleanup():
    """
    Used to cleanup at normal program exit
    """
    TimeoutCaller.cleanup()


TimeoutCaller.get_pool()
atexit.register(cleanup)
