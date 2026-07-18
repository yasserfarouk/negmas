"""Utility function inversion via BIDS (Bidding using Diversified Search).

BIDS exploits the additive structure of `LinearAdditiveUtilityFunction` to answer
utility-lookup and utility-sampling queries over arbitrarily large outcome spaces
through a one-time O(|I|·|V|·10^p) dynamic-programming table construction, after
which each query is answered in O(1) (lookup) or O(n_samples) (Sampling-BIDS).

Reference
---------
Koça, T., de Jonge, D., & Baarslag, T. (2024).
*Search algorithms for automated negotiation in large domains.*
Algorithms 17(5), 200.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np

from negmas.outcomes import Outcome
from negmas.warnings import warn_if_slow

from ..protocols import InverseUFun
from ._common import EPS, _norm_to_raw, _raw_to_norm, _resolve_rng

if TYPE_CHECKING:
    from ..base_ufun import BaseUtilityFunction

__all__ = ["BIDSInverseUtilityFunction"]

# minimum u_range below which we treat the ufun as effectively constant
_MIN_RANGE = 1e-12


class BIDSInverseUtilityFunction(InverseUFun):
    """Utility function inverter using the BIDS dynamic-programming algorithm.

    BIDS exploits the additive structure of `LinearAdditiveUtilityFunction` to
    answer utility-lookup queries over very large outcome spaces efficiently:

    * **Offline phase** (``init()``): builds a DP table of size *O(|I|·10^p)* in
      time *O(|I|·|V|·10^p)*, where *|I|* is the number of issues, *|V|* is the
      maximum number of values per issue, and *p* is the ``precision`` parameter.
    * **Online phase** (``one_in``, ``best_in``, etc.): each utility-lookup query
      is answered in *O(1)* (single table look-up), and *n* utility-sampling
      queries (``some``) take *O(n)* via Sampling-BIDS.

    The approximation error for a single call is at most *|I| · 10^-p* in
    normalised utility. Increasing ``precision`` reduces this error at the cost of
    more memory and initialisation time.

    **Important**: BIDS only works with `LinearAdditiveUtilityFunction` (or its
    subclass `LinearUtilityFunction`, which is also additive). ``init()`` will
    raise ``TypeError`` for any other ufun type.

    Args:
        ufun: The utility function to invert (must be a
            `LinearAdditiveUtilityFunction` or `LinearUtilityFunction`).
        precision: Grid precision *p* (number of decimal places). The table uses
            *10^p + 1* grid points over [0, 1]. Default: 3 (1001 points, error
            at most *|I|·0.001*). Increase for higher accuracy at the cost of
            more memory/init time.
        n_samples: Number of targets sampled when ``some()`` is called without an
            explicit *n* argument, and when ``best_in``/``worst_in`` search for
            the best/worst outcome in a range.

    Remarks:
        - ``min``, ``max``, ``best``, ``worst`` and ``minmax`` are exact: they
          delegate to the wrapped ufun's own ``minmax``/``extreme_outcomes``
          (which are closed-form for additive ufuns).
        - ``within_fractions`` and ``within_indices`` return approximate results
          via dense Sampling-BIDS over the requested range.
        - For maximum accuracy over small or medium outcome spaces, prefer
          `PresortingInverseUtilityFunction` (which is exact). Use BIDS when the
          outcome space is too large to enumerate.
        - This is a **clamping** inverter (see module docs). ``worst_in``/
          ``best_in`` expand the range upward and fall back to the nearest
          boundary outcome (best if the range is in the upper half of the
          utility range, worst if in the lower half) rather than returning
          ``None``.
    """

    def __init__(
        self, ufun: BaseUtilityFunction, precision: int = 3, n_samples: int = 50
    ) -> None:
        self._ufun = ufun
        self._precision = precision
        self._n_samples = n_samples
        self._initialized = False

        # filled in by init()
        self._issues: list[Any] = []
        self._val_list: list[list[Any]] = []
        self._nc_list: list[np.ndarray] = []
        self._table_val_idx: np.ndarray | None = None
        self._table_sum: np.ndarray | None = None
        self._n_pts: int = 0
        self._u_min: float = 0.0
        self._u_max: float = 1.0
        self._u_range: float = 1.0

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def ufun(self) -> BaseUtilityFunction:
        return self._ufun

    @property
    def initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Build the BIDS dynamic-programming table.

        Raises:
            TypeError: if ``ufun`` is not a ``LinearAdditiveUtilityFunction``
                (or its subclass ``LinearUtilityFunction``).
        """
        from ..crisp.linear import LinearAdditiveUtilityFunction

        # Unwrap any discounted/scaled wrapper to the base additive ufun
        base = self._ufun
        while hasattr(base, "ufun") and not isinstance(
            base, LinearAdditiveUtilityFunction
        ):
            base = base.ufun  # type: ignore[attr-defined]

        if not isinstance(base, LinearAdditiveUtilityFunction):
            raise TypeError(
                f"BIDSInverseUtilityFunction requires a LinearAdditiveUtilityFunction "
                f"(or LinearUtilityFunction) but got {type(self._ufun).__name__}. "
                f"Use PresortingInverseUtilityFunction for non-additive ufuns, or "
                f"AdaptiveInverseUtilityFunction to select automatically."
            )

        if not base.issues:
            raise ValueError(
                "BIDSInverseUtilityFunction requires a ufun with an outcome space "
                "(no issues found)."
            )

        self._issues = list(base.issues)
        n = len(self._issues)
        p = self._precision
        n_pts = 10**p + 1
        self._n_pts = n_pts

        warn_if_slow(
            n * n_pts,
            f"BIDSInverseUtilityFunction: building DP table for {n} issues "
            f"and precision={p} ({n_pts} grid points)",
        )

        # ------------------------------------------------------------------
        # Build per-issue raw contributions and normalise to [0, 1]
        # ------------------------------------------------------------------
        raw_contrib_list: list[np.ndarray] = []
        val_list: list[list[Any]] = []

        for issue, vfun, w in zip(self._issues, base.values, base.weights):
            vals = list(issue.all)
            val_list.append(vals)
            contribs = np.array([w * float(vfun(v)) for v in vals], dtype=np.float64)
            raw_contrib_list.append(contribs)

        C_min = float(sum(c.min() for c in raw_contrib_list))
        C_max = float(sum(c.max() for c in raw_contrib_list))
        u_range = C_max - C_min
        if u_range < _MIN_RANGE:
            u_range = _MIN_RANGE  # degenerate ufun: all outcomes identical

        self._u_min = base._bias + C_min
        self._u_max = base._bias + C_max
        self._u_range = u_range

        # nc_list[i][j] = normalised contribution of value j for issue i
        # sum_i nc_i(v_i) == (raw_util - u_min) / u_range == normalised utility
        nc_list: list[np.ndarray] = []
        for contribs in raw_contrib_list:
            nc = (contribs - contribs.min()) / u_range
            nc_list.append(nc)

        self._val_list = val_list
        self._nc_list = nc_list

        # ------------------------------------------------------------------
        # Build DP table using numpy vectorisation
        # ------------------------------------------------------------------
        # table_val_idx[i][t] = index into val_list[i] of the best value
        #                        for issue i when the normalised partial-sum
        #                        target is t / (n_pts - 1)
        # table_sum[i][t]     = achieved normalised partial sum for issues 0..i
        # ------------------------------------------------------------------
        table_val_idx = np.zeros((n, n_pts), dtype=np.int32)
        table_sum = np.zeros((n, n_pts), dtype=np.float64)

        grid = np.linspace(0.0, 1.0, n_pts)  # (n_pts,) target values on [0,1]

        # --- Base case: issue 0 -------------------------------------------
        nc0 = nc_list[0]  # (|V0|,)
        # For each grid point t, choose the value minimising |nc0[j] - t|
        diff0 = np.abs(nc0[:, np.newaxis] - grid[np.newaxis, :])  # (|V0|, n_pts)
        best0 = np.argmin(diff0, axis=0)  # (n_pts,)
        table_val_idx[0] = best0
        table_sum[0] = nc0[best0]

        # --- Inductive step: issues 1 .. n-1 ---------------------------------
        for i in range(1, n):
            nc_i = nc_list[i]  # (|Vi|,)
            nc_i_col = nc_i[:, np.newaxis]  # (|Vi|, 1)
            grid_row = grid[np.newaxis, :]  # (1, n_pts)

            # remaining normalised target for issues 0..i-1
            remaining = grid_row - nc_i_col  # (|Vi|, n_pts)

            # discretise: clamp and round to grid index
            r_idx = np.clip(
                np.round(remaining * (n_pts - 1)).astype(np.int64), 0, n_pts - 1
            )  # (|Vi|, n_pts)

            # achieved partial sum if we pick value v and look up table[i-1][r_idx]
            prev_sum = table_sum[i - 1][r_idx]  # (|Vi|, n_pts)
            candidate_sum = prev_sum + nc_i_col  # (|Vi|, n_pts)

            # error from target
            error = np.abs(candidate_sum - grid_row)  # (|Vi|, n_pts)

            # for each grid point, pick the value with minimum error
            best_v = np.argmin(error, axis=0)  # (n_pts,)
            table_val_idx[i] = best_v
            table_sum[i] = candidate_sum[best_v, np.arange(n_pts)]

        self._table_val_idx = table_val_idx
        self._table_sum = table_sum
        self._initialized = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _norm_to_raw(self, t: float) -> float:
        """Convert a normalised utility (0=worst, 1=best) to a raw utility."""
        return _norm_to_raw(t, self._u_min, self._u_max)

    def _raw_to_norm(self, u: float) -> float:
        """Convert a raw utility to normalised [0,1]."""
        return _raw_to_norm(u, self._u_min, self._u_max)

    def _resolve_rng(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> tuple[float, float] | None:
        """Return ``(mn_raw, mx_raw)`` from *rng* in either normalised or raw form.

        Returns ``None`` if the range is inverted beyond
        :data:`INVERTED_RANGE_TOLERANCE` (see :func:`negmas.preferences.inv_ufun._common._resolve_rng`).
        """
        return _resolve_rng(rng, normalized, self._u_min, self._u_max)

    def _target_to_idx(self, target_raw: float) -> int:
        """Convert a raw utility target to a grid index."""
        t_norm = (target_raw - self._u_min) / self._u_range
        t_norm = float(np.clip(t_norm, 0.0, 1.0))
        return int(round(t_norm * (self._n_pts - 1)))

    def _lookup(self, t_idx: int) -> Outcome:
        """Reconstruct the full outcome from grid index *t_idx* in O(n_issues)."""
        n = len(self._issues)
        n_pts = self._n_pts
        outcome: list[Any] = [None] * n

        t = t_idx
        for i in range(n - 1, -1, -1):
            val_idx = int(self._table_val_idx[i, t])  # type: ignore[index]
            outcome[i] = self._val_list[i][val_idx]

            if i > 0:
                nc_i_val = float(self._nc_list[i][val_idx])
                # Recompute the grid index used for table[i-1]:
                # r_idx = clip(round((grid_t - nc_i_val) * (n_pts-1)), 0, n_pts-1)
                grid_t = t / (n_pts - 1)
                remaining = grid_t - nc_i_val
                t = int(np.clip(round(remaining * (n_pts - 1)), 0, n_pts - 1))

        return tuple(outcome)

    def _bids(self, target_raw: float) -> Outcome:
        """Single BIDS query: return outcome with utility closest to *target_raw*."""
        return self._lookup(self._target_to_idx(target_raw))

    def _sampling_bids(
        self, mn_raw: float, mx_raw: float, n: int, rng_seed: int | None = None
    ) -> list[Outcome]:
        """Sampling-BIDS: sample *n* utility targets from [mn_raw, mx_raw]
        uniformly and apply BIDS to each."""
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw
        targets = [random.uniform(mn_raw, mx_raw) for _ in range(n)]
        return [self._bids(t) for t in targets]

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "BIDSInverseUtilityFunction.init() has not been called. "
                "Call init() before using the inverter."
            )

    # ------------------------------------------------------------------
    # InverseUFun protocol
    # ------------------------------------------------------------------

    def closest(self, target: float, normalized: bool = False) -> Outcome | None:
        """Return the outcome with utility closest to ``target`` via a single
        O(1) BIDS table lookup.

        Args:
            target: The target utility value.
            normalized: if ``True``, ``target`` is in normalised [0,1] space.
        """
        self._check_initialized()
        target_raw = self._norm_to_raw(target) if normalized else float(target)
        return self._bids(target_raw)

    def some(
        self, rng: float | tuple[float, float], normalized: bool, n: int | None = None
    ) -> list[Outcome]:
        """Return a diverse list of outcomes with utilities near the given range.

        Uses Sampling-BIDS: samples *n* utility targets uniformly from *rng* and
        applies BIDS to each, returning up to *n* distinct outcomes.

        Args:
            rng: Utility range (scalar = exact target; tuple = interval).
            normalized: if ``True``, *rng* is in normalised [0,1] space.
            n: Number of samples. Defaults to ``self.n_samples``.
        """
        self._check_initialized()
        k = n if n is not None else self._n_samples
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return []
        mn_raw, mx_raw = resolved
        outcomes = self._sampling_bids(mn_raw, mx_raw, k)
        # deduplicate while preserving order
        seen: set[Outcome] = set()
        result = []
        for o in outcomes:
            if o not in seen:
                seen.add(o)
                result.append(o)
        return result

    def one_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        """Return one outcome with utility in *rng*, or a fallback.

        Uses a single BIDS query targeting the midpoint of *rng*, then checks
        whether the result falls within the range. If not (due to discretisation
        approximation), falls back as requested.

        Args:
            rng: Utility range.
            normalized: if ``True``, *rng* is in normalised [0,1] space.
            fallback_to_higher: if ``True`` and nothing found strictly in *rng*,
                return the closest outcome above the requested minimum.
            fallback_to_best: if ``True`` and still nothing found, return the
                outcome with the best overall utility.
        """
        self._check_initialized()
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return None
        mn_raw, mx_raw = resolved
        # Try midpoint as target
        target = (mn_raw + mx_raw) / 2.0
        outcome = self._bids(target)
        u = float(self._ufun(outcome))  # type: ignore[arg-type]
        if mn_raw - EPS <= u <= mx_raw + EPS:
            return outcome

        # Fallback: try sampling a few targets
        for t in np.linspace(mn_raw, mx_raw, min(self._n_samples, 20)):
            o = self._bids(float(t))
            u2 = float(self._ufun(o))  # type: ignore[arg-type]
            if mn_raw - EPS <= u2 <= mx_raw + EPS:
                return o

        if fallback_to_higher:
            o = self._bids(mn_raw)
            u3 = float(self._ufun(o))  # type: ignore[arg-type]
            if u3 >= mn_raw - EPS:
                return o

        if fallback_to_best:
            return self.best()

        return None

    def best_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_best: bool = True,
    ) -> Outcome | None:
        """Return the outcome with the **highest** utility in *rng*.

        Uses Sampling-BIDS with ``n_samples`` targets and returns the one with
        the highest utility that actually lies within the range.

        Fallback behavior (clamping inverter — see module docs):

        * If no in-range outcome is found and ``fallback_to_higher`` is ``True``,
          the range is expanded upward to ``[rng[0], max]`` and the search is
          retried.
        * If still nothing is found and ``fallback_to_best`` is ``True``, the
          best outcome overall is returned.
        * Otherwise ``None`` is returned.
        """
        self._check_initialized()
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return None
        mn_raw, mx_raw = resolved
        candidates = self._sampling_bids(mn_raw, mx_raw, self._n_samples)
        best_o: Outcome | None = None
        best_u = float("-inf")
        for o in candidates:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            if mn_raw - EPS <= u <= mx_raw + EPS and u > best_u:
                best_u = u
                best_o = o
        if best_o is not None:
            return best_o
        if fallback_to_higher and (not normalized or mx_raw < 1 - EPS):
            new_rng = (mn_raw, float(self.max()))
            return self.best_in(
                new_rng,
                normalized,
                fallback_to_higher=False,
                fallback_to_best=fallback_to_best,
            )
        if fallback_to_best:
            return self.best()
        return None

    def worst_in(
        self,
        rng: float | tuple[float, float],
        normalized: bool,
        fallback_to_higher: bool = True,
        fallback_to_worst: bool = True,
    ) -> Outcome | None:
        """Return the outcome with the **lowest** utility in *rng*.

        Uses Sampling-BIDS with ``n_samples`` targets and returns the one with
        the lowest utility that actually lies within the range.

        Fallback behavior (clamping inverter — see module docs):

        * If no in-range outcome is found and ``fallback_to_higher`` is ``True``,
          the range is expanded upward to ``[rng[0], max]`` and the search is
          retried.
        * If still nothing is found and ``fallback_to_worst`` is ``True``, the
          **nearest boundary** outcome is returned: the best outcome if the
          range lies above the maximum utility, the worst outcome if it lies
          below the minimum.
        * Otherwise ``None`` is returned.
        """
        self._check_initialized()
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return None
        mn_raw, mx_raw = resolved
        candidates = self._sampling_bids(mn_raw, mx_raw, self._n_samples)
        worst_o: Outcome | None = None
        worst_u = float("inf")
        for o in candidates:
            u = float(self._ufun(o))  # type: ignore[arg-type]
            if mn_raw - EPS <= u <= mx_raw + EPS and u < worst_u:
                worst_u = u
                worst_o = o
        if worst_o is not None:
            return worst_o
        if fallback_to_higher and (not normalized or mx_raw < 1 - EPS):
            new_rng = (mn_raw, float(self.max()))
            return self.worst_in(
                new_rng,
                normalized,
                fallback_to_higher=False,
                fallback_to_worst=fallback_to_worst,
            )
        if fallback_to_worst:
            # Return the nearest boundary outcome. If the requested range is
            # in the upper half of the utility range, the best outcome is
            # closer; otherwise the worst is closer.
            u_min, u_max = self.minmax()
            mid = (u_min + u_max) / 2.0
            if mn_raw >= mid:
                return self.best()
            return self.worst()
        return None

    def within_fractions(self, rng: tuple[float, float]) -> list[Outcome]:
        """Return a diverse sample of outcomes within the given normalised fraction range.

        Args:
            rng: ``(lo, hi)`` in normalised [0,1] space (0=worst, 1=best).
        """
        self._check_initialized()
        mn_raw = self._norm_to_raw(rng[0])
        mx_raw = self._norm_to_raw(rng[1])
        return self._sampling_bids(mn_raw, mx_raw, self._n_samples)

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        """Return outcomes approximately corresponding to the given sorted-rank range.

        Maps the index range to a normalised utility range (0=index 0 = best,
        high index = worst) and applies Sampling-BIDS. Results are approximate.

        Args:
            rng: ``(lo_idx, hi_idx)`` into a hypothetical sorted-by-utility list
                 of all outcomes (index 0 = best, largest = worst).
        """
        self._check_initialized()
        os = self._ufun.outcome_space
        total = os.cardinality if os is not None else None
        if total is None or total == 0:
            return []
        lo_frac = 1.0 - rng[1] / total  # high-index = low utility
        hi_frac = 1.0 - rng[0] / total
        lo_frac = max(0.0, min(1.0, lo_frac))
        hi_frac = max(0.0, min(1.0, hi_frac))
        return self.within_fractions((lo_frac, hi_frac))

    def min(self) -> float:
        """Minimum utility (exact, from the wrapped ufun)."""
        return float(self._ufun.minmax()[0])

    def max(self) -> float:
        """Maximum utility (exact, from the wrapped ufun)."""
        return float(self._ufun.minmax()[1])

    def worst(self) -> Outcome:
        """Outcome with the lowest utility (exact, from the wrapped ufun)."""
        return self._ufun.extreme_outcomes()[0]

    def best(self) -> Outcome:
        """Outcome with the highest utility (exact, from the wrapped ufun)."""
        return self._ufun.extreme_outcomes()[1]

    def minmax(self) -> tuple[float, float]:
        """(min, max) utilities (exact, from the wrapped ufun)."""
        return self._ufun.minmax()  # type: ignore[return-value]

    def extreme_outcomes(self) -> tuple[Outcome, Outcome]:
        """(worst, best) outcomes (exact, from the wrapped ufun)."""
        return self._ufun.extreme_outcomes()  # type: ignore[return-value]

    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        """Calling an inverse ufun directly is equivalent to calling ``one_in()``."""
        return self.one_in(rng, normalized)
