"""Equilibrium computation over an empirical payoff matrix
(:class:`negmas.tournaments.analysis.payoff.PayoffTable`).

Implements:
    - Pure symmetric Nash equilibria and the "EGTA based evaluation" method
      of de Jonge's "Introduction to Automated Negotiation", chapter 6.2.
    - Regret and Nash-averaging/NE-regret ranking (Jordan et al. 2007;
      Balduzzi et al. 2018), as covered in the EGTA survey (Wellman, Tuyls &
      Greenwald 2025), section 3.4.
    - Optional mixed-strategy Nash equilibria via ``nashpy``/``pygambit``
      (extra ``negmas[egta]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from .payoff import PayoffTable

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "PureNashResult",
    "pure_symmetric_nash_equilibria",
    "egta_evaluation",
    "regret",
    "nash_averaging_ranking",
    "mixed_nash_equilibria",
]


@define
class PureNashResult:
    """Result of :func:`pure_symmetric_nash_equilibria`/:func:`egta_evaluation`."""

    candidates: list[str]
    """All strategies that are a pure symmetric Nash equilibrium."""
    best: str | None
    """The candidate with the highest diagonal payoff, or None if there are none."""
    unique: bool
    """Whether ``best`` is the unique best pure symmetric Nash equilibrium (no tie)."""


def pure_symmetric_nash_equilibria(payoff: PayoffTable, tol: float = 1e-9) -> list[str]:
    """Finds all pure symmetric Nash equilibria of the payoff matrix.

    Strategy ``k`` is a pure symmetric Nash equilibrium iff it is a best
    response to itself: ``M[k, k] == max_i M[i, k]`` (de Jonge, eq. 6.2/6.3).

    Since the payoff matrix is built from sample averages (not exact
    values), an exact ``==`` comparison would reject every strategy whose
    diagonal is a hair below the true best response due to sampling noise.
    ``tol`` (default ``1e-9``, i.e. effectively exact) widens the comparison
    to ``M[k, k] >= max_i M[i, k] - tol``; increase it if the payoff matrix
    was built from few repetitions and near-ties should be treated as NE.
    """
    M = payoff.matrix
    n = payoff.n
    candidates = []
    for k in range(n):
        col = M[:, k]
        if np.isnan(col).all():
            continue
        best_value = np.nanmax(col)
        if not np.isnan(col[k]) and col[k] >= best_value - tol:
            candidates.append(payoff.strategies[k])
    return candidates


def egta_evaluation(payoff: PayoffTable, tol: float = 1e-9) -> PureNashResult:
    """The "EGTA based evaluation" method of de Jonge, chapter 6.2: identifies
    the best pure symmetric Nash equilibrium of the tournament's payoff
    matrix, i.e. the single "recommended" strategy.

    Note this method only ever recommends a *single* best strategy (or
    reports that none exists among pure equilibria) — it does not produce a
    full ranking of all strategies. For a full ranking, see
    :func:`negmas.tournaments.analysis.ranking.sequential_elimination_ranking`.

    See :func:`pure_symmetric_nash_equilibria` for the meaning of ``tol``.
    """
    candidates = pure_symmetric_nash_equilibria(payoff, tol=tol)
    if not candidates:
        return PureNashResult(candidates=[], best=None, unique=False)
    diag = {s: payoff.matrix[payoff.index(s), payoff.index(s)] for s in candidates}
    best_value = max(diag.values())
    tied = sorted(s for s, v in diag.items() if v >= best_value)
    return PureNashResult(candidates=candidates, best=tied[0], unique=len(tied) == 1)


def regret(payoff: PayoffTable, profile: dict[str, float]) -> float:
    """Game regret of a (possibly mixed) symmetric strategy profile against
    itself: the largest gain a single deviator could obtain by switching to
    some other pure strategy while the rest of the population keeps playing
    ``profile`` (Wellman, Tuyls & Greenwald 2025, eq. 1, specialized to
    symmetric single-population games).

    Args:
        profile: mapping from strategy name to probability (must sum to 1).
    """
    M = payoff.matrix
    probs = np.zeros(payoff.n)
    for s, p in profile.items():
        probs[payoff.index(s)] = p
    if not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"profile probabilities must sum to 1, got {probs.sum()}")
    payoffs_vs_profile = np.nansum(M * probs[None, :], axis=1)
    current = float(np.nansum(payoffs_vs_profile * probs))
    return float(np.nanmax(payoffs_vs_profile) - current)


def nash_averaging_ranking(
    payoff: PayoffTable, equilibrium: dict[str, float] | None = None
) -> pd.DataFrame:
    """Ranks strategies by their deviation regret against a reference
    equilibrium profile (NE-regret ranking / Nash-averaging, Jordan et al.
    2007; Balduzzi et al. 2018): lower regret is better.

    If ``equilibrium`` is not given, the best pure symmetric Nash equilibrium
    from :func:`egta_evaluation` is used as the reference profile (a
    degenerate one-hot mixed strategy); ties are broken alphabetically by
    strategy name (see :func:`egta_evaluation`).
    """
    import pandas as pd

    if equilibrium is None:
        best = egta_evaluation(payoff).best
        if best is None:
            raise ValueError(
                "No pure symmetric Nash equilibrium found; pass an explicit "
                "`equilibrium` mixed profile (e.g. from mixed_nash_equilibria)."
            )
        equilibrium = {best: 1.0}

    M = payoff.matrix
    probs = np.zeros(payoff.n)
    for s, p in equilibrium.items():
        probs[payoff.index(s)] = p
    payoffs_vs_profile = np.nansum(M * probs[None, :], axis=1)
    best_value = float(np.nanmax(payoffs_vs_profile))
    rows = [
        dict(
            strategy=s,
            deviation_payoff=float(payoffs_vs_profile[i]),
            regret=best_value - float(payoffs_vs_profile[i]),
        )
        for i, s in enumerate(payoff.strategies)
    ]
    df = pd.DataFrame(rows).sort_values("regret", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def mixed_nash_equilibria(
    payoff: PayoffTable, method: str = "support_enumeration"
) -> list[dict[str, float]]:
    """Computes mixed-strategy Nash equilibria of the symmetric game defined by
    ``payoff`` using ``nashpy`` (requires the optional ``negmas[egta]`` extra).

    Args:
        payoff: the empirical payoff matrix (NaN cells are treated as 0, with
            a warning, since ``nashpy`` requires a complete matrix).
        method: one of nashpy's ``Game`` equilibria methods: ``"support_enumeration"``,
            ``"vertex_enumeration"`` or ``"lemke_howson_enumeration"``.

    Returns:
        A list of equilibria, each a mapping from strategy name to
        probability under the row player's equilibrium mixture.
    """
    try:
        import nashpy as npy
    except ImportError as e:
        raise ImportError(
            "mixed_nash_equilibria() requires the optional 'nashpy' dependency. "
            "Install it with: pip install negmas[egta]"
        ) from e

    M = payoff.matrix
    if np.isnan(M).any():
        import warnings

        warnings.warn(
            "Payoff matrix has missing cells; treating them as 0 for mixed "
            "Nash equilibrium computation. Consider restricting to strategies "
            "with complete pairwise data (see PayoffTable.restrict).",
            stacklevel=2,
        )
        M = np.nan_to_num(M, nan=0.0)

    game = npy.Game(M, M.T)
    if method == "support_enumeration":
        equilibria = game.support_enumeration()
    elif method == "vertex_enumeration":
        equilibria = game.vertex_enumeration()
    elif method == "lemke_howson_enumeration":
        equilibria = game.lemke_howson_enumeration()
    else:
        raise ValueError(
            "method must be one of 'support_enumeration', 'vertex_enumeration', "
            f"'lemke_howson_enumeration', got {method!r}"
        )

    results = []
    for row_mix, _ in equilibria:
        results.append({s: float(row_mix[i]) for i, s in enumerate(payoff.strategies)})
    return results
