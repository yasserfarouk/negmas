"""Pairwise statistical-significance testing and normality diagnostics over
the common analysis DataFrame format (see
:mod:`negmas.tournaments.analysis.loaders`).

Unlike the rest of this package, these functions operate directly on the
per-negotiation records (not on a :class:`~negmas.tournaments.analysis.payoff.PayoffTable`):
significance/normality are properties of a strategy's *distribution* of
scores, not of the pairwise payoff matrix.

Supports both unpaired (independent-samples) and paired (matched-by-scenario)
variants of the parametric t-test and the non-parametric rank-sum test, plus
the two-sample Kolmogorov-Smirnov test (unpaired only), and the three
standard multiple-comparison corrections (Bonferroni, Holm-Bonferroni,
Benjamini-Hochberg/FDR).
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "pairwise_tests",
    "apply_corrections",
    "normality_tests",
    "significance_matrix",
    "plot_significance_marks",
    "plot_significance_heatmap",
]

_TESTS = ("ttest", "ranksum", "ks")
_CORRECTIONS = ("raw", "bonferroni", "holm", "bh")
_P_COLUMNS = {
    "raw": "p_value",
    "bonferroni": "p_bonferroni",
    "holm": "p_holm",
    "bh": "p_bh",
}


def _paired_samples(
    records: pd.DataFrame, a: str, b: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Aligns strategy ``a``'s and ``b``'s per-scenario mean ``metric`` on the
    scenarios both were evaluated on (the natural pairing for negotiation
    tournaments: it controls for scenario difficulty)."""
    ga = records.loc[records["strategy"] == a].groupby("scenario")[metric].mean()
    gb = records.loc[records["strategy"] == b].groupby("scenario")[metric].mean()
    common = ga.index.intersection(gb.index)
    return ga.loc[common].to_numpy(dtype=float), gb.loc[common].to_numpy(dtype=float)


def _run_test(
    test: str, xa: np.ndarray, xb: np.ndarray, paired: bool
) -> tuple[float, float]:
    if test == "ttest":
        res = stats.ttest_rel(xa, xb) if paired else stats.ttest_ind(xa, xb)
    elif test == "ranksum":
        res = (
            stats.wilcoxon(xa, xb)
            if paired
            else stats.mannwhitneyu(xa, xb, alternative="two-sided")
        )
    elif test == "ks":
        if paired:
            raise ValueError(
                "The Kolmogorov-Smirnov test does not support paired samples "
                "(it compares two distributions, not matched observations)."
            )
        res = stats.ks_2samp(xa, xb)
    else:
        raise ValueError(f"test must be one of {_TESTS}, got {test!r}")
    return float(res.statistic), float(res.pvalue)  # type: ignore[union-attr]


def pairwise_tests(
    records: pd.DataFrame,
    metric: str = "advantage",
    strategies: list[str] | None = None,
    test: str = "ttest",
    paired: bool = False,
) -> pd.DataFrame:
    """Runs a pairwise statistical test between every pair of strategies.

    Args:
        records: common analysis DataFrame format (see
            :func:`negmas.tournaments.analysis.loaders.load_records`).
        metric: which column to compare (typically ``"advantage"`` or ``"utility"``).
        strategies: optional explicit subset/ordering; defaults to all
            strategies appearing in ``records``, sorted by name.
        test: one of ``"ttest"`` (parametric), ``"ranksum"`` (Mann-Whitney U
            unpaired / Wilcoxon signed-rank paired), or ``"ks"``
            (Kolmogorov-Smirnov, unpaired only).
        paired: if True, matches each pair of strategies' samples by the
            scenario they were both evaluated on (averaging over repetitions
            within a scenario first); if False, compares the raw independent
            sample of all scores each strategy obtained.

    Returns:
        One row per unordered pair of strategies, with columns
        ``strategy_a``, ``strategy_b``, ``statistic``, ``p_value``, ``n_a``,
        ``n_b`` (sample sizes actually used, i.e. post-pairing for
        ``paired=True``). Pairs with fewer than 2 usable samples on either
        side get ``NaN`` statistic/p_value rather than raising.
    """
    if strategies is None:
        strategies = sorted(records["strategy"].unique())
    rows = []
    for a, b in combinations(strategies, 2):
        if paired:
            xa, xb = _paired_samples(records, a, b, metric)
        else:
            xa = records.loc[records["strategy"] == a, metric].dropna().to_numpy()
            xb = records.loc[records["strategy"] == b, metric].dropna().to_numpy()
        if len(xa) < 2 or len(xb) < 2:
            rows.append(
                dict(
                    strategy_a=a,
                    strategy_b=b,
                    statistic=float("nan"),
                    p_value=float("nan"),
                    n_a=len(xa),
                    n_b=len(xb),
                )
            )
            continue
        statistic, p_value = _run_test(test, xa, xb, paired)
        rows.append(
            dict(
                strategy_a=a,
                strategy_b=b,
                statistic=statistic,
                p_value=p_value,
                n_a=len(xa),
                n_b=len(xb),
            )
        )
    return pd.DataFrame(
        rows, columns=["strategy_a", "strategy_b", "statistic", "p_value", "n_a", "n_b"]
    )


def _holm_correction(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(m)
    running_max = 0.0
    for i, pv in enumerate(ranked):
        running_max = max(running_max, (m - i) * pv)
        adjusted[i] = min(running_max, 1.0)
    out = np.empty(m)
    out[order] = adjusted
    return out


def _bh_correction(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * m / (np.arange(1, m + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def apply_corrections(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Adds Bonferroni, Holm-Bonferroni and Benjamini-Hochberg corrected
    p-values (and the corresponding ``significant_*`` boolean columns at the
    given ``alpha``) to the output of :func:`pairwise_tests`.

    Rows with a ``NaN`` p-value (not enough samples) are left as ``NaN`` in
    every corrected column and ``False`` in every ``significant_*`` column,
    and are excluded from ``m`` (the number of comparisons the corrections
    are computed over).
    """
    df = df.copy()
    p = df["p_value"].to_numpy(dtype=float)
    valid = ~np.isnan(p)
    m = int(valid.sum())

    df["p_bonferroni"] = float("nan")
    df["p_holm"] = float("nan")
    df["p_bh"] = float("nan")
    if m:
        df.loc[valid, "p_bonferroni"] = np.clip(p[valid] * m, 0.0, 1.0)
        df.loc[valid, "p_holm"] = _holm_correction(p[valid])
        df.loc[valid, "p_bh"] = _bh_correction(p[valid])

    for correction, col in _P_COLUMNS.items():
        df[f"significant_{correction}"] = df[col] < alpha
    return df


def normality_tests(
    records: pd.DataFrame,
    metric: str = "advantage",
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Per-strategy normality diagnostics: skewness, excess kurtosis,
    Shapiro-Wilk and D'Agostino K² tests.

    Diagnoses the normality assumption behind the *unpaired* t-test (the
    paired test instead assumes normality of the per-scenario differences,
    not covered here). At the large sample sizes a tournament produces these
    tests reject normality on negligible departures, so skewness (~0 is
    symmetric) and excess kurtosis (~0 is normal-tailed) are usually more
    informative than the p-values.

    Returns one row per strategy with columns ``strategy``, ``n``,
    ``skewness``, ``kurtosis``, ``shapiro_w``, ``shapiro_p``, ``dagostino_p``.
    Entries that cannot be computed (Shapiro-Wilk needs ``n >= 3``,
    D'Agostino K² needs ``n >= 8``) are ``NaN``.
    """
    if strategies is None:
        strategies = sorted(records["strategy"].unique())
    rows = []
    for s in strategies:
        x = records.loc[records["strategy"] == s, metric].dropna().to_numpy()
        n = len(x)
        skewness = kurtosis = shapiro_w = shapiro_p = dagostino_p = float("nan")
        if n >= 3:
            skewness = float(stats.skew(x))
            kurtosis = float(stats.kurtosis(x, fisher=True))
            shapiro_w, shapiro_p = (float(v) for v in stats.shapiro(x))
        if n >= 8:
            dagostino_p = float(stats.normaltest(x).pvalue)
        rows.append(
            dict(
                strategy=s,
                n=n,
                skewness=skewness,
                kurtosis=kurtosis,
                shapiro_w=shapiro_w,
                shapiro_p=shapiro_p,
                dagostino_p=dagostino_p,
            )
        )
    return pd.DataFrame(rows)


def significance_matrix(
    df: pd.DataFrame, strategies: list[str] | None = None, value_col: str = "p_value"
) -> pd.DataFrame:
    """Converts the long-format output of :func:`pairwise_tests`/
    :func:`apply_corrections` into a symmetric strategy x strategy matrix of
    ``value_col`` (e.g. ``"p_value"``, ``"p_bonferroni"``, ``"significant_holm"``),
    with ``NaN`` on the diagonal and for pairs not present in ``df``.
    """
    if strategies is None:
        strategies = sorted(set(df["strategy_a"]) | set(df["strategy_b"]))
    n = len(strategies)
    pos = {s: i for i, s in enumerate(strategies)}
    M = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        i, j = pos[row["strategy_a"]], pos[row["strategy_b"]]
        M[i, j] = M[j, i] = row[value_col]
    return pd.DataFrame(M, index=strategies, columns=strategies)


def _get_ax(ax: Axes | None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_significance_heatmap(
    df: pd.DataFrame,
    strategies: list[str] | None = None,
    correction: str = "bonferroni",
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of pairwise p-values (one of the corrections in
    :func:`apply_corrections`'s output, or ``"raw"`` for the uncorrected
    p-value), mirroring the "Heatmap" display mode of the tournament website's
    pairwise significance matrix.
    """
    if correction not in _CORRECTIONS:
        raise ValueError(
            f"correction must be one of {_CORRECTIONS}, got {correction!r}"
        )
    value_col = _P_COLUMNS[correction]
    matrix = significance_matrix(df, strategies=strategies, value_col=value_col)
    ax = _get_ax(ax)
    im = ax.imshow(matrix.to_numpy(), cmap="viridis_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(f"Pairwise significance ({correction} p-value)")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p-value")
    return ax


def plot_significance_marks(
    df: pd.DataFrame,
    strategies: list[str] | None = None,
    alpha: float = 0.05,
    correction: str = "bonferroni",
    ax: Axes | None = None,
) -> Axes:
    """Grid of check marks: a checkmark at cell ``(i, j)`` means strategies
    ``i`` and ``j`` differ significantly at ``alpha`` under the given
    correction, mirroring the "Marks" display mode of the tournament
    website's pairwise significance matrix.
    """
    if correction not in _CORRECTIONS:
        raise ValueError(
            f"correction must be one of {_CORRECTIONS}, got {correction!r}"
        )
    sig_df = (
        df
        if f"significant_{correction}" in df.columns
        else apply_corrections(df, alpha=alpha)
    )
    matrix = significance_matrix(
        sig_df, strategies=strategies, value_col=f"significant_{correction}"
    )
    ax = _get_ax(ax)
    ax.imshow(
        np.zeros_like(matrix.to_numpy(), dtype=float), cmap="Greys", vmin=0, vmax=1
    )
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            if i == j:
                continue
            v = matrix.iloc[i, j]
            if pd.isna(v):
                mark, color = "?", "gray"
            elif bool(v):
                mark, color = "✓", "green"
            else:
                mark, color = "", "black"
            if mark:
                ax.text(j, i, mark, ha="center", va="center", color=color, fontsize=12)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(
        f"Pairwise significance marks (✓ = significant, {correction}, α={alpha})"
    )
    return ax
