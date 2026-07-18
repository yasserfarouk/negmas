"""Utility function inversion via Attribute Planning.

Attribute Planning independently picks the per-issue value whose normalised
per-issue utility is closest to the normalised global target, then combines
them into an outcome.  The algorithm requires no pre-built lookup table: it
runs in O(|I|·|V|) time per query and uses O(|I|·|V|) space.

Reference
---------
Jonker, C. M., & Treur, J. (2001).
*An agent architecture for multi-attribute negotiation.*
Proceedings of IJCAI 2001.

Description from:
Koça, T., de Jonge, D., & Baarslag, T. (2024).
*Search algorithms for automated negotiation in large domains.*
Algorithms 17(5), 200.  Table 2.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np

from negmas.outcomes import Outcome

from ..protocols import InverseUFun
from ._common import EPS, _norm_to_raw, _raw_to_norm, _resolve_rng

if TYPE_CHECKING:
    from ..base_ufun import BaseUtilityFunction

__all__ = ["AttributePlanningInverseUtilityFunction"]

_MIN_RANGE = 1e-12


class AttributePlanningInverseUtilityFunction(InverseUFun):
    """Utility function inverter using the Attribute Planning algorithm.

    Attribute Planning exploits the additive structure of
    ``LinearAdditiveUtilityFunction`` to answer single utility-lookup queries in
    *O(|I|·|V|)* time with *O(|I|·|V|)* space.  For each query with target
    utility *t*, the algorithm independently selects the per-issue value whose
    normalised per-issue utility is closest to *t*, then combines these into an
    outcome.

    **Important**: only works with ``LinearAdditiveUtilityFunction`` (or its
    subclass ``LinearUtilityFunction``).  ``init()`` raises ``TypeError`` for any
    other ufun type.

    This is a **clamping** inverter (see module docs). ``worst_in``/``best_in``
    expand the range upward and fall back to the nearest boundary outcome
    (best if the range is in the upper half of the utility range, worst if in
    the lower half) rather than returning ``None``.

    Args:
        ufun: The utility function to invert.
        n_samples: Number of random targets sampled for ``some()``,
            ``best_in()``, and ``worst_in()``.  Default: 50.
    """

    def __init__(self, ufun: BaseUtilityFunction, n_samples: int = 50) -> None:
        self._ufun = ufun
        self._n_samples = n_samples
        self._initialized = False

        # filled by init()
        self._issues: list[Any] = []
        self._val_list: list[list[Any]] = []
        # per-issue normalised utilities: _nc_list[i][j] = norm util of value j of issue i
        self._nc_list: list[np.ndarray] = []
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
        """Precompute per-issue normalised utilities.

        Raises:
            TypeError: if ``ufun`` is not a ``LinearAdditiveUtilityFunction``.
        """
        from ..crisp.linear import LinearAdditiveUtilityFunction

        base = self._ufun
        while hasattr(base, "ufun") and not isinstance(
            base, LinearAdditiveUtilityFunction
        ):
            base = base.ufun  # type: ignore[attr-defined]

        if not isinstance(base, LinearAdditiveUtilityFunction):
            raise TypeError(
                f"AttributePlanningInverseUtilityFunction requires a "
                f"LinearAdditiveUtilityFunction but got {type(self._ufun).__name__}. "
                f"Use PresortingInverseUtilityFunction for non-additive ufuns."
            )

        if not base.issues:
            raise ValueError(
                "AttributePlanningInverseUtilityFunction requires a ufun with an "
                "outcome space (no issues found)."
            )

        self._issues = list(base.issues)
        val_list: list[list[Any]] = []
        nc_list: list[np.ndarray] = []

        # Collect raw weighted contributions per issue
        raw_contrib_list: list[np.ndarray] = []
        for issue, vfun, w in zip(self._issues, base.values, base.weights):
            vals = list(issue.all)
            val_list.append(vals)
            raw = np.array([w * float(vfun(v)) for v in vals], dtype=np.float64)
            raw_contrib_list.append(raw)

        # Global utility bounds
        c_min = float(sum(c.min() for c in raw_contrib_list))
        c_max = float(sum(c.max() for c in raw_contrib_list))
        u_range = c_max - c_min
        if u_range < _MIN_RANGE:
            u_range = _MIN_RANGE

        self._u_min = base._bias + c_min
        self._u_max = base._bias + c_max
        self._u_range = u_range

        # Per-issue normalised utilities: nc_i(v) = (raw_i(v) - min_i) / u_range
        # Note: divide by the GLOBAL range so that sum_i nc_i(v_i) == normalised utility.
        for raw in raw_contrib_list:
            nc = (raw - raw.min()) / u_range
            nc_list.append(nc)

        self._val_list = val_list
        self._nc_list = nc_list
        self._initialized = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "AttributePlanningInverseUtilityFunction.init() has not been called."
            )

    def _norm_to_raw(self, t: float) -> float:
        return _norm_to_raw(t, self._u_min, self._u_max)

    def _raw_to_norm(self, u: float) -> float:
        return _raw_to_norm(u, self._u_min, self._u_max)

    def _resolve_rng(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> tuple[float, float] | None:
        return _resolve_rng(rng, normalized, self._u_min, self._u_max)

    def _query(self, target_raw: float) -> Outcome:
        """Attribute-Planning query: split the global target into a per-issue target
        and, for each issue, pick the value whose per-issue utility is closest to it.

        The per-issue contributions ``nc_i`` are normalised by the *global* utility
        range, so ``sum_i nc_i.max() == 1``.  A value ``nc_i`` therefore lives in
        ``[0, nc_i.max()]`` -- a fraction of ``[0, 1]`` -- and cannot be compared
        directly against the global target ``t_norm`` which lives in ``[0, 1]``.
        Following Jonker & Treur, the global target is distributed across issues by
        giving each issue a target proportional to its share of the total range
        (``t_norm * nc_i.max()``).  In the continuous case this makes the achieved
        utility equal to ``t_norm``; in the discrete case each issue rounds to its
        nearest value.
        """
        t_norm = float(np.clip((target_raw - self._u_min) / self._u_range, 0.0, 1.0))
        outcome: list[Any] = []
        for nc, vals in zip(self._nc_list, self._val_list):
            per_issue_target = t_norm * float(nc.max())
            idx = int(np.argmin(np.abs(nc - per_issue_target)))
            outcome.append(vals[idx])
        return tuple(outcome)

    # ------------------------------------------------------------------
    # InverseUFun protocol
    # ------------------------------------------------------------------

    def closest(self, target: float, normalized: bool = False) -> Outcome | None:
        """Return the outcome built by Attribute Planning targeting ``target``.

        Args:
            target: The target utility value.
            normalized: if ``True``, ``target`` is in normalised [0,1] space.
        """
        self._check_initialized()
        target_raw = self._norm_to_raw(target) if normalized else float(target)
        return self._query(target_raw)

    def some(
        self, rng: float | tuple[float, float], normalized: bool, n: int | None = None
    ) -> list[Outcome]:
        self._check_initialized()
        k = n if n is not None else self._n_samples
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return []
        mn_raw, mx_raw = resolved
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw
        targets = [random.uniform(mn_raw, mx_raw) for _ in range(k)]
        seen: set[Outcome] = set()
        result: list[Outcome] = []
        for t in targets:
            o = self._query(t)
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
        self._check_initialized()
        resolved = self._resolve_rng(rng, normalized)
        if resolved is None:
            return None
        mn_raw, mx_raw = resolved
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw
        # Random target for diversity
        target = random.uniform(mn_raw, mx_raw)
        outcome = self._query(target)
        u = float(self._ufun(outcome))  # type: ignore[arg-type]
        if mn_raw - EPS <= u <= mx_raw + EPS:
            return outcome

        # Try a few more targets uniformly spaced
        for t in np.linspace(mn_raw, mx_raw, min(self._n_samples, 20)):
            o = self._query(float(t))
            u2 = float(self._ufun(o))  # type: ignore[arg-type]
            if mn_raw - EPS <= u2 <= mx_raw + EPS:
                return o

        if fallback_to_higher:
            o = self._query(mn_raw)
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
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw
        best_o: Outcome | None = None
        best_u = float("-inf")
        for t in np.linspace(mn_raw, mx_raw, self._n_samples):
            o = self._query(float(t))
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
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw
        worst_o: Outcome | None = None
        worst_u = float("inf")
        for t in np.linspace(mn_raw, mx_raw, self._n_samples):
            o = self._query(float(t))
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
        self._check_initialized()
        mn_raw = self._norm_to_raw(rng[0])
        mx_raw = self._norm_to_raw(rng[1])
        if mn_raw > mx_raw:
            mn_raw, mx_raw = mx_raw, mn_raw
        targets = [random.uniform(mn_raw, mx_raw) for _ in range(self._n_samples)]
        seen: set[Outcome] = set()
        result: list[Outcome] = []
        for t in targets:
            o = self._query(t)
            if o not in seen:
                seen.add(o)
                result.append(o)
        return result

    def within_indices(self, rng: tuple[int, int]) -> list[Outcome]:
        self._check_initialized()
        os = self._ufun.outcome_space
        total = os.cardinality if os is not None else None
        if total is None or total == 0:
            return []
        lo_frac = 1.0 - rng[1] / total
        hi_frac = 1.0 - rng[0] / total
        lo_frac = max(0.0, min(1.0, lo_frac))
        hi_frac = max(0.0, min(1.0, hi_frac))
        return self.within_fractions((lo_frac, hi_frac))

    def min(self) -> float:
        return float(self._ufun.minmax()[0])

    def max(self) -> float:
        return float(self._ufun.minmax()[1])

    def worst(self) -> Outcome:
        return self._ufun.extreme_outcomes()[0]

    def best(self) -> Outcome:
        return self._ufun.extreme_outcomes()[1]

    def minmax(self) -> tuple[float, float]:
        return self._ufun.minmax()  # type: ignore[return-value]

    def extreme_outcomes(self) -> tuple[Outcome, Outcome]:
        return self._ufun.extreme_outcomes()  # type: ignore[return-value]

    def __call__(
        self, rng: float | tuple[float, float], normalized: bool
    ) -> Outcome | None:
        return self.one_in(rng, normalized)
