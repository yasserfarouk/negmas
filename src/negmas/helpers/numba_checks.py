"""Numba JIT compilation utilities with graceful fallback when unavailable."""

from __future__ import annotations

import functools

DISABLE_NUMBA = False
try:
    from numba import jit  # type: ignore

    NUMBA_OK = not DISABLE_NUMBA
except Exception:
    NUMBA_OK = False

    def jit(nopython=True):
        """Jit.

        Args:
            nopython: Nopython.
        """

        def jit_decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                """Wrapper.

                Args:
                    *args: Additional positional arguments.
                    **kwargs: Additional keyword arguments.
                """
                return f(*args, **kwargs)

            return wrapper

        return jit_decorator


__all__ = ["jit", "NUMBA_OK"]
