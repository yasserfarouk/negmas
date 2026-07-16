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
