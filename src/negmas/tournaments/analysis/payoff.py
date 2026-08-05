"""Construction of the empirical payoff matrix shared by every post-tournament
analysis method in this package (EGTA, sequential elimination ranking,
replicator dynamics, equilibrium computation).

This mirrors the "common data structure" used in both the EGTA literature
(the Heuristic Payoff Table, restricted here to two-player/bilateral games)
and de Jonge's "Introduction to Automated Negotiation" (chapter 6): a square
matrix ``M`` where ``M[i, j]`` is the average payoff (utility/advantage/...)
strategy ``i`` obtains when negotiating against strategy ``j``, averaged over
scenarios, repetitions and negotiator roles.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from attrs import define

__all__ = ["PayoffTable", "build_payoff_table", "iterated_elimination"]


_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "min": np.min,
    "max": np.max,
}


@define
class PayoffTable:
    """A square empirical payoff matrix over a set of strategies.

    ``matrix[i, j]`` is the aggregated payoff of ``strategies[i]`` when
    negotiating against ``strategies[j]``. Note this is generally **not**
    symmetric as a matrix (``matrix[i, j] != matrix[j, i]`` in general) even
    though it represents a symmetric game: every strategy can play either
    role, and the payoff to the row player is looked up as ``matrix[row, col]``
    regardless of who plays first.
    """

    strategies: list[str]
    matrix: np.ndarray
    counts: np.ndarray
    metric: str
    stat: str
    samples: dict[tuple[str, str], np.ndarray] | None = None

    @property
    def n(self) -> int:
        return len(self.strategies)

    def index(self, strategy: str) -> int:
        return self.strategies.index(strategy)

    def has_missing(self) -> bool:
        return bool(np.isnan(self.matrix).any())

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.strategies, columns=self.strategies)

    def restrict(self, strategies: list[str]) -> "PayoffTable":
        """Returns a new :class:`PayoffTable` restricted to a subset of strategies."""
        idx = [self.index(s) for s in strategies]
        samples = None
        if self.samples is not None:
            samples = {
                (a, b): v
                for (a, b), v in self.samples.items()
                if a in strategies and b in strategies
            }
        return PayoffTable(
            strategies=list(strategies),
            matrix=self.matrix[np.ix_(idx, idx)].copy(),
            counts=self.counts[np.ix_(idx, idx)].copy(),
            metric=self.metric,
            stat=self.stat,
            samples=samples,
        )


def build_payoff_table(
    records: pd.DataFrame,
    metric: str = "advantage",
    stat: str = "mean",
    strategies: list[str] | None = None,
    keep_samples: bool = False,
    require_self_play: bool = True,
) -> PayoffTable:
    """Builds a :class:`PayoffTable` from the common analysis DataFrame format
    (see :mod:`negmas.tournaments.analysis.loaders`).

    Args:
        records: DataFrame with (at least) columns ``strategy``, ``partner``
            and ``metric``, e.g. as returned by
            :func:`negmas.tournaments.analysis.loaders.load_records`.
        metric: Which column to aggregate into the payoff matrix (typically
            ``"advantage"`` or ``"utility"``).
        stat: Aggregation statistic: one of ``"mean"``, ``"median"``, ``"std"``,
            ``"min"``, ``"max"``.
        strategies: Optional explicit strategy ordering/subset. Defaults to
            all strategies appearing in ``records``, sorted by name.
        keep_samples: If True, keeps the raw per-cell samples (needed e.g. for
            bootstrap confidence intervals) in ``PayoffTable.samples``.
        require_self_play: If True (default), raises a ``ValueError`` if any
            strategy is missing self-play data (``M[i, i]``). Both EGTA (pure
            symmetric Nash) and Sequential Elimination Ranking need the
            diagonal to be well defined.
    """
    if metric not in records.columns:
        raise ValueError(f"metric {metric!r} is not a column of records")
    if stat not in _STATS:
        raise ValueError(f"stat must be one of {sorted(_STATS)}, got {stat!r}")
    agg = _STATS[stat]

    if strategies is None:
        strategies = sorted(set(records["strategy"]) | set(records["partner"]))
    n = len(strategies)
    pos = {s: i for i, s in enumerate(strategies)}

    matrix = np.full((n, n), np.nan, dtype=float)
    counts = np.zeros((n, n), dtype=int)
    samples: dict[tuple[str, str], np.ndarray] | None = {} if keep_samples else None

    grouped = records.groupby(["strategy", "partner"])[metric]
    for (s, p), values in grouped:
        if s not in pos or p not in pos:
            continue
        vals = values.to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        i, j = pos[s], pos[p]
        matrix[i, j] = agg(vals)
        counts[i, j] = vals.size
        if samples is not None:
            samples[(s, p)] = vals

    if require_self_play:
        missing_self = [strategies[i] for i in range(n) if np.isnan(matrix[i, i])]
        if missing_self:
            raise ValueError(
                "Missing self-play data for strategies "
                f"{missing_self}: EGTA and Sequential Elimination Ranking "
                "both require the payoff matrix diagonal to be defined "
                "(run the tournament with self_play=True), or pass "
                "require_self_play=False to proceed with NaNs on the diagonal."
            )

    return PayoffTable(
        strategies=strategies,
        matrix=matrix,
        counts=counts,
        metric=metric,
        stat=stat,
        samples=samples,
    )


def iterated_elimination(payoff: PayoffTable) -> PayoffTable:
    """Iteratively removes strictly dominated strategies from a symmetric-game
    payoff table.

    A strategy ``i`` is strictly dominated by ``i'`` if, for every remaining
    opponent strategy ``j``, ``M[i', j] > M[i, j]``. Dominated strategies are
    removed one at a time until no further removal is possible (fixpoint),
    mirroring the IESDS preprocessing step commonly used before solving for
    Nash equilibria.
    """
    remaining = list(range(payoff.n))
    M = payoff.matrix
    changed = True
    while changed and len(remaining) > 1:
        changed = False
        for i in list(remaining):
            row_i = M[i, remaining]
            for k in remaining:
                if k == i:
                    continue
                row_k = M[k, remaining]
                mask = ~(np.isnan(row_i) | np.isnan(row_k))
                if mask.sum() == 0:
                    continue
                if np.all(row_k[mask] > row_i[mask]):
                    remaining.remove(i)
                    changed = True
                    break
    surviving = [payoff.strategies[i] for i in remaining]
    return payoff.restrict(surviving)
