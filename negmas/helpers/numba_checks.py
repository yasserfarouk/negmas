from __future__ import annotations

import functools

DISABLE_NUMBA = False
try:
    from numba import jit  # type: ignore

    NUMBA_OK = not DISABLE_NUMBA
except:
    NUMBA_OK = False

    def jit(nopython=True):
        def jit_decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

            return wrapper

        return jit_decorator


__all__ = ["jit", "NUMBA_OK"]
