#!/usr/bin/env python
"""
Utilitites for handling timeout and async calls
"""
from __future__ import annotations

import atexit
import concurrent
import warnings
from concurrent.futures import TimeoutError
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import contextmanager

__all__ = [
    "TimeoutError",
    "TimeoutCaller",
    "force_single_thread",
    "single_thread",
]
SINGLE_THREAD_FORCED = False


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
    force_single_thread(True)
    yield None
    force_single_thread(False)


def is_single_thread() -> bool:
    return SINGLE_THREAD_FORCED


class TimeoutCaller:
    pool = None

    @classmethod
    def run(cls, to_run, timeout: float):
        if is_single_thread():
            return to_run()
        pool = cls.get_pool()
        future = pool.submit(to_run)
        try:
            result = future.result(timeout)
            return result
        except TimeoutError as s:
            future.cancel()
            raise s

    @classmethod
    def get_pool(cls):
        if cls.pool is None:
            cls.pool = ThreadPoolExecutor()
        return cls.pool

    @classmethod
    def cleanup(cls):
        if cls.pool is not None:
            try:
                cls.pool.shutdown(wait=False)
                for thread in cls.pool._threads:
                    del concurrent.futures.thread._threads_queues[thread]
            except:
                warnings.warn(
                    "NegMAS have finished processing but there are some "
                    "threads still hanging there!! If your program does "
                    "not die by itself. Please press Ctrl-c to kill it"
                )


atexit.register(TimeoutCaller.cleanup)
