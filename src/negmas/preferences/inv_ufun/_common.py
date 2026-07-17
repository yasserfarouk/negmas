"""Shared, internal helper functions used by the different inverse-utility-function
implementations in this package.

These are intentionally not part of the public API (not re-exported from
`negmas.preferences.inv_ufun`) since they are low level array-searching primitives.

Range handling
==============

Every inverter method accepts a ``rng`` argument that is either a scalar utility
value or a ``(lo, hi)`` tuple, plus a ``normalized`` flag indicating whether the
values are in ``[0, 1]`` (normalised) or in the ufun's raw utility range. Before
any search, the inverter must:

1. **Widen** a scalar to a (very small) range ``(v - EPS, v + EPS)``.
2. **Validate** the range for inversion: if ``lo > hi`` and the gap is within
   :data:`INVERTED_RANGE_TOLERANCE`, silently swap the bounds (treat as a
   numerical inaccuracy). If the gap is larger, return ``None`` / ``[]`` as if no
   outcome was found (treat as a caller bug — see :ref:`inverted-ranges`).
3. **Rescale** the bounds from ``[0, 1]`` to the ufun's raw range when
   ``normalized=True`` (using the ufun's ``min``/``max``), with a degenerate-range
   fallback (``min == max``).

The functions :func:`_normalize_rng`, :func:`_resolve_rng`, and
:func:`_un_normalize_range` below implement these three steps. Each inverter
calls one of them (depending on whether it stores ``u_min``/``u_max`` or
``_min``/``_max``) so the validation logic lives in one place.

.. note::

    Two **independent** tolerances are at play:

    * **Range-argument tolerance** (``eps`` / ``rel_eps`` parameters of
      :func:`_normalize_rng`): controls only the scalar-widening step — how much
      a scalar ``v`` is expanded into ``(v - tol, v + tol)``. Defaults are
      :data:`EPS` = ``1e-6`` and :data:`REL_EPS` = ``1e-3``.

    * **Inverter-internal tolerance** (e.g.
      ``SamplingInverseUtilityFunction.eps`` / ``.rel_eps``, or
      ``PresortingInverseUtilityFunction.clamp_tolerance``): controls how far
      *outside* the requested range an inverter is willing to accept an outcome
      during its search. This is set on the inverter *instance*, not on the
      range argument, and is completely independent of the range-argument
      tolerance above.

.. _inverted-ranges:

Inverted ranges
---------------

A range is *inverted* when ``lo > hi``. There are two cases:

* **Small inversion** (``lo - hi <= INVERTED_RANGE_TOLERANCE``): treated as a
  numerical inaccuracy and silently corrected by swapping the bounds. This
  happens, for example, when a caller computes a range around a target utility
  and floating-point rounding puts the lower bound slightly above the upper
  bound.

* **Large inversion** (``lo - hi > INVERTED_RANGE_TOLERANCE``): treated as a
  caller bug. Rather than raising (which would break negotiation flows) or
  silently swapping (which would mask the bug), the inverter returns
  ``None`` / ``[]`` — exactly as if no outcome's utility fell inside the
  requested range. This lets the caller's fallback logic kick in naturally.

The default tolerance is :data:`INVERTED_RANGE_TOLERANCE` = ``0.1``. Set it to
``float("inf")`` to disable the safety check and always swap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__: list[str] = []

EPS = 1e-6
REL_EPS = 1e-3

#: Minimum non-degenerate utility range. Below this, the ufun is treated as
#: constant and normalisation maps everything to a single point.
_MIN_RANGE = 1e-12

#: Maximum difference (in utility units) between the lower and upper bounds of a
#: range that is allowed to be silently swapped when the range is inverted
#: (``lo > hi``). Inversions larger than this are treated as a caller bug: the
#: inverter returns ``None`` / ``[]`` (as if no outcome was found) rather than
#: silently swapping the bounds. Set to ``float("inf")`` to disable this
#: safety check and always swap.
INVERTED_RANGE_TOLERANCE = 0.1


# ---------------------------------------------------------------------------
# Step 1+2: scalar widening + inverted-range validation
# ---------------------------------------------------------------------------


def _normalize_rng(
    rng: float | tuple[float, float], eps=EPS, rel_eps=REL_EPS
) -> tuple[float, float, bool]:
    """Widen a scalar to a pair and validate a range for inversion.

    Args:
        rng: The utility range — a scalar value or a ``(lo, hi)`` tuple.
        eps: Absolute tolerance used when widening a scalar to a pair. The
            scalar ``v`` is widened to ``(v - tol, v + tol)`` where
            ``tol = max(rel_eps * v, eps)``. Defaults to :data:`EPS`.
        rel_eps: Relative tolerance used when widening a scalar. See ``eps``
            above. Defaults to :data:`REL_EPS`.

    .. note::

        ``eps`` and ``rel_eps`` here control only the **scalar-widening** step.
        They are independent of (and unrelated to) the tolerance that each
        inverter class uses internally for accepting outcomes near the bounds
        of a range (e.g. ``SamplingInverseUtilityFunction.eps`` /
        ``SamplingInverseUtilityFunction.rel_eps``, or
        ``PresortingInverseUtilityFunction.clamp_tolerance``). The latter is
        set on the inverter instance; the former is a property of the range
        argument itself.

    Returns:
        ``(lo, hi, ok)`` where:

        * If *rng* is a scalar, it is widened to
          ``(rng - tol, rng + tol)`` with ``tol = max(rel_eps * rng, eps)``
          and ``ok`` is ``True``.
        * If ``lo > hi`` and the difference ``lo - hi`` is within
          :data:`INVERTED_RANGE_TOLERANCE`, the bounds are silently swapped and
          ``ok`` is ``True``.
        * If ``lo > hi`` and the difference exceeds the tolerance, ``ok`` is
          ``False`` so the caller can return ``None`` / ``[]`` (as if no outcome
          was found) rather than silently swapping the bounds.

    See :ref:`inverted-ranges` for the rationale.
    """
    if isinstance(rng, (int, float)):
        tol = max(rel_eps * rng, eps)
        lo = float(rng) - tol
        hi = float(rng) + tol
        return lo, hi, True
    lo, hi = float(rng[0]), float(rng[1])
    if lo > hi:
        if lo - hi > INVERTED_RANGE_TOLERANCE:
            return lo, hi, False
        lo, hi = hi, lo
    return lo, hi, True


# ---------------------------------------------------------------------------
# Step 3a: rescaling for inverters that store u_min / u_max (BIDS, MCTS, AttrPlan)
# ---------------------------------------------------------------------------


def _norm_to_raw(t: float, u_min: float, u_max: float) -> float:
    """Convert a single normalised utility (0=worst, 1=best) to a raw utility.

    Args:
        t: Normalised utility in ``[0, 1]``.
        u_min: Minimum raw utility of the ufun.
        u_max: Maximum raw utility of the ufun.

    Returns:
        The raw utility ``u_min + t * (u_max - u_min)``.
    """
    return u_min + t * (u_max - u_min)


def _raw_to_norm(u: float, u_min: float, u_max: float) -> float:
    """Convert a single raw utility to normalised ``[0, 1]``.

    Args:
        u: Raw utility.
        u_min: Minimum raw utility of the ufun.
        u_max: Maximum raw utility of the ufun.

    Returns:
        ``(u - u_min) / (u_max - u_min)``, or ``0.0`` if the range is
        degenerate (``u_max - u_min < _MIN_RANGE``).
    """
    r = u_max - u_min
    if r < _MIN_RANGE:
        return 0.0
    return (u - u_min) / r


def _resolve_rng(
    rng: float | tuple[float, float], normalized: bool, u_min: float, u_max: float
) -> tuple[float, float] | None:
    """Resolve a range argument to ``(mn_raw, mx_raw)`` or ``None`` if inverted.

    This is the shared implementation behind the per-inverter ``_resolve_rng``
    methods used by `BIDSInverseUtilityFunction`,
    `MCTSInverseUtilityFunction`, and
    `AttributePlanningInverseUtilityFunction` (all of which store ``u_min`` /
    ``u_max``).

    Steps:
      1. Widen a scalar to a pair via :func:`_normalize_rng`.
      2. If the range is inverted beyond :data:`INVERTED_RANGE_TOLERANCE`,
         return ``None`` (caller returns ``None`` / ``[]``).
      3. If ``normalized`` is ``True``, rescale the bounds from ``[0, 1]`` to
         the raw range ``[u_min, u_max]`` via :func:`_norm_to_raw`.

    Args:
        rng: The utility range (scalar or ``(lo, hi)``).
        normalized: If ``True``, *rng* is in ``[0, 1]`` and is rescaled to raw.
        u_min: Minimum raw utility of the ufun.
        u_max: Maximum raw utility of the ufun.

    Returns:
        ``(mn_raw, mx_raw)`` with ``mn_raw <= mx_raw``, or ``None`` if the range
        is inverted beyond the tolerance.
    """
    lo, hi, ok = _normalize_rng(rng)
    if not ok:
        return None
    if normalized:
        lo = _norm_to_raw(lo, u_min, u_max)
        hi = _norm_to_raw(hi, u_min, u_max)
        if lo > hi:
            lo, hi = hi, lo
    return (lo, hi)


# ---------------------------------------------------------------------------
# Step 3b: rescaling for inverters that store _min / _max (presorting family)
# ---------------------------------------------------------------------------


def _un_normalize_range(
    rng: float | tuple[float, float],
    normalized: bool,
    u_min: float,
    u_max: float,
    for_best: bool,
) -> tuple[float, float] | None:
    """Resolve a range argument to ``(mn_raw, mx_raw)`` or ``None`` if inverted.

    This is the shared implementation behind the per-inverter
    ``_un_normalize_range`` methods used by
    `PresortingInverseUtilityFunction`,
    `PresortingLegacyInverseUtilityFunction`, and
    `BruteForceInverseUtilityFunction` (all of which store ``_min`` / ``_max``).

    The difference from :func:`_resolve_rng` is the degenerate-range fallback:
    when ``u_max - u_min < EPS`` (constant ufun), this returns a single point
    ``(0.0, 0.0)`` (for worst-oriented queries) or ``(1.0, 1.0)`` (for
    best-oriented queries) instead of dividing by zero.

    Steps:
      1. Widen a scalar to a pair via :func:`_normalize_rng`.
      2. If the range is inverted beyond :data:`INVERTED_RANGE_TOLERANCE`,
         return ``None`` (caller returns ``None`` / ``[]``).
      3. If ``normalized`` is ``True``, rescale the bounds from ``[0, 1]`` to
         the raw range ``[u_min, u_max]``. If the raw range is degenerate
         (``< EPS``), return the degenerate fallback instead.

    Args:
        rng: The utility range (scalar or ``(lo, hi)``).
        normalized: If ``True``, *rng* is in ``[0, 1]`` and is rescaled to raw.
        u_min: Minimum raw utility of the ufun (``self._min``).
        u_max: Maximum raw utility of the ufun (``self._max``).
        for_best: If ``True``, the degenerate fallback is ``(1.0, 1.0)``;
            otherwise ``(0.0, 0.0)``. Used only when ``u_max - u_min < EPS``.

    Returns:
        ``(mn_raw, mx_raw)`` with ``mn_raw <= mx_raw``, or ``None`` if the range
        is inverted beyond the tolerance.
    """
    lo, hi, ok = _normalize_rng(rng)
    if not ok:
        return None
    if not normalized:
        return (lo, hi)
    d = u_max - u_min
    if d < EPS:
        v = 0.0 if not for_best else 1.0
        return (v, v)
    lo = lo * d + u_min
    hi = hi * d + u_min
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


# ---------------------------------------------------------------------------
# Array-search primitives (unchanged)
# ---------------------------------------------------------------------------


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
