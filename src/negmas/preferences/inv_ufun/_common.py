"""Shared, internal helper functions used by the different inverse-utility-function
implementations in this package.

These are intentionally not part of the public API (not re-exported from
`negmas.preferences.inv_ufun`) since they are low level array-searching primitives.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__: list[str] = []

EPS = 1e-12


def _nearest_around(
    x: float, a: NDArray, i: int, mn: float, mx: float, n: int = 1, eps: float = EPS
) -> int | None:
    """Finds the nearest value in ``a`` to ``x`` around index ``i`` subject to being
    between ``mn`` and ``mx``.

    Remarks:
        - This is the historical clamping behavior relied upon by the presorting
          inverters: when no value strictly within ``[mn, mx]`` is found near ``i`` but
          ``i`` is itself the best-effort clamped index (e.g. the requested utility
          range lies entirely above/below all currently available -- possibly
          discounted -- utilities), the clamped index ``i`` is returned instead of
          ``None`` so callers get the nearest achievable outcome rather than failing.
    """
    n = len(a) - 1
    best, best_diff = i, abs(a[i] - x)
    for j in range(i - n, i + n + 1):
        if j < 0 or j > n:
            continue
        if not (mn <= a[j] <= mx):
            continue
        d = abs(a[j] - x)
        if d < best_diff:
            best, best_diff = j, d
    if best is None:
        return i
    if abs(a[best] - x) > eps and best != i:
        return None
    return best


def index_above_or_equal(a: NDArray, x: Any, lo: int = 0, hi: int | None = None) -> int:
    """Locates the smallest index ``i`` in ``[lo, hi]`` with ``a[i] >= x``.

    ``a[lo : hi + 1]`` **must** be sorted ascendingly. If every value in the given
    window is smaller than ``x``, ``hi`` is returned (best effort, callers must check
    ``a[hi] >= x`` themselves if they need a strict guarantee).
    """
    n = len(a)
    if n == 0:
        raise ValueError("Cannot search an empty array")
    if hi is None:
        hi = n - 1
    lo = max(0, lo)
    hi = min(n - 1, hi)
    if lo > hi:
        return hi
    i = int(np.searchsorted(a[lo : hi + 1], x, side="left")) + lo
    return max(lo, min(hi, i))


def index_below_or_equal(a: NDArray, x: Any, lo: int = 0, hi: int | None = None) -> int:
    """Locates the greatest index ``i`` in ``[lo, hi]`` with ``a[i] <= x``.

    ``a[lo : hi + 1]`` **must** be sorted ascendingly. If every value in the given
    window is larger than ``x``, ``lo`` is returned (best effort, callers must check
    ``a[lo] <= x`` themselves if they need a strict guarantee).
    """
    n = len(a)
    if n == 0:
        raise ValueError("Cannot search an empty array")
    if hi is None:
        hi = n - 1
    lo = max(0, lo)
    hi = min(n - 1, hi)
    if lo > hi:
        return lo
    # bisect_right(x) - 1 gives the rightmost index with a[i] <= x.
    i = int(np.searchsorted(a[lo : hi + 1], x, side="right")) + lo - 1
    return max(lo, min(hi, i))
