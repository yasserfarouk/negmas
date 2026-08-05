"""Adapters that convert tournament results (in memory or on disk) into a
common long-format DataFrame consumed by the analysis core functions in this
package (:mod:`negmas.tournaments.analysis.payoff`, ``.ranking``, ``.equilibria``,
``.dynamics``).

The common format has one row per *negotiator side* of a bilateral negotiation:

======  ===========================================================
Column  Meaning
======  ===========================================================
strategy   Name of the negotiator/strategy being scored
partner    Name of the opposing negotiator/strategy
scenario   Name of the scenario/domain the negotiation was run on
utility    Utility (or reserved value if no agreement) obtained
reserved_value  Reservation value of ``strategy``
advantage  Normalized utility gain, see :func:`negmas.tournaments.neg.simple.cartesian.make_scores`
======  ===========================================================

Only bilateral (two-negotiator) records are kept: the game-theoretic methods
implemented here (pure/mixed Nash equilibria, sequential elimination ranking,
replicator dynamics, ...) all operate on a pairwise payoff matrix and are not
meaningfully defined for negotiations with more than two parties.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from negmas.tournaments.neg.simple.cartesian import SimpleTournamentResults
    from negmas.tournaments.tournaments import TournamentResults

__all__ = [
    "COMMON_COLUMNS",
    "records_from_simple_results",
    "records_from_path",
    "records_from_situated",
    "load_records",
]

COMMON_COLUMNS = [
    "strategy",
    "partner",
    "scenario",
    "utility",
    "reserved_value",
    "advantage",
]


def _as_partner_list(value: Any) -> list[Any]:
    """Normalizes the ``partners`` field of a scores row to a list.

    Handles both the in-memory representation (a bare string for bilateral
    negotiations, a tuple otherwise) and the on-disk representation after a
    CSV/parquet round-trip, where a tuple can come back as its ``repr`` string.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("(") or s.startswith("["):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except (ValueError, SyntaxError):
                pass
        return [value]
    return [value]


def _normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)
    partner_lists = df["partners"].apply(_as_partner_list)
    bilateral = partner_lists.apply(len) == 1
    n_dropped = int((~bilateral).sum())
    if n_dropped:
        import warnings

        warnings.warn(
            f"Dropped {n_dropped} non-bilateral negotiation records while "
            "building the common analysis format: EGTA/ranking/replicator "
            "dynamics methods only support pairwise (two-negotiator) games.",
            stacklevel=2,
        )
    out = df.loc[bilateral, :].copy()
    out["partner"] = partner_lists.loc[bilateral].apply(lambda x: x[0])
    for col in ("reserved_value", "advantage"):
        if col not in out.columns:
            out[col] = float("nan")
    return out[COMMON_COLUMNS].reset_index(drop=True)


def records_from_simple_results(results: SimpleTournamentResults) -> pd.DataFrame:
    """Converts in-memory :class:`SimpleTournamentResults` (from
    :func:`negmas.tournaments.neg.simple.cartesian.cartesian_tournament`) into
    the common analysis DataFrame format."""
    return _normalize_scores(results.scores)


def records_from_path(path: str | Path) -> pd.DataFrame:
    """Loads a tournament saved to ``path`` (as produced by
    ``cartesian_tournament(..., path=...)``) and converts it to the common
    analysis DataFrame format."""
    from negmas.tournaments.neg.simple.cartesian import SimpleTournamentResults

    results = SimpleTournamentResults.load(Path(path))
    return records_from_simple_results(results)


def records_from_situated(results: TournamentResults) -> pd.DataFrame:
    """Best-effort adapter for situated/world-based :class:`TournamentResults`.

    Situated tournaments run worlds with an arbitrary number of participating
    agent types, some of which may be ``non_competitors`` (present in the
    world but not scored/competed against). What determines whether a world
    is bilateral is the number of *competitors* (``n_competitors_per_world``
    in the tournament's parameters), not the raw number of distinct agent
    types in the world's score rows -- non-competitors are excluded before
    counting.

    Only worlds with exactly two distinct *competitor* agent types are
    converted (each becomes one bilateral record per side); worlds with more
    than two have no well-defined pairwise decomposition (crediting a type's
    world score to each of several co-participants would fabricate identical
    payoff-matrix cells that carry no real pairwise information) and are
    dropped, with a warning reporting how many were skipped. If the
    tournament's own ``n_competitors_per_world`` parameter is present and is
    not 2, every world is skipped (the tournament as a whole is not
    bilateral) and an empty DataFrame is returned.
    """
    df = results.scores
    if df.empty or "agent_type" not in df.columns:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    params = results.params or {}
    n_competitors_per_world = params.get("n_competitors_per_world")
    if n_competitors_per_world is not None and n_competitors_per_world != 2:
        import warnings

        warnings.warn(
            f"Situated tournament has n_competitors_per_world="
            f"{n_competitors_per_world} (!= 2): it is not a bilateral "
            "tournament, so no pairwise records can be built from it.",
            stacklevel=2,
        )
        return pd.DataFrame(columns=COMMON_COLUMNS)

    non_competitors = set(params.get("non_competitors") or ())

    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for world, group in df.groupby("world"):
        competitors = group.loc[~group["agent_type"].isin(non_competitors)]
        types: list[dict[str, Any]] = competitors[["agent_type", "score"]].to_dict(
            orient="records"
        )
        distinct = {t["agent_type"] for t in types}
        if len(distinct) != 2:
            n_skipped += 1
            continue
        for i, a in enumerate(types):
            for j, b in enumerate(types):
                if i == j:
                    continue
                rows.append(
                    dict(
                        strategy=a["agent_type"],
                        partner=b["agent_type"],
                        scenario=str(world),
                        utility=a["score"],
                        reserved_value=float("nan"),
                        advantage=float("nan"),
                    )
                )
    if n_skipped:
        import warnings

        warnings.warn(
            f"Skipped {n_skipped} world(s) with a number of distinct "
            "*competitor* agent types other than 2 while building the "
            "common analysis format: pairwise payoff matrices are only "
            "well defined for worlds with exactly two competing agent types "
            "(non_competitors are excluded from this count).",
            stacklevel=2,
        )
    result = pd.DataFrame(rows, columns=list(COMMON_COLUMNS))
    return result


def load_records(
    source: "SimpleTournamentResults | TournamentResults | pd.DataFrame | str | Path",
) -> pd.DataFrame:
    """Dispatches to the right adapter based on the type of ``source`` and
    returns the common analysis DataFrame format.

    Accepts:
        - an in-memory ``SimpleTournamentResults`` (from ``cartesian_tournament``)
        - an in-memory situated ``TournamentResults``
        - a path (``str``/``Path``) to a saved cartesian tournament
        - an already-normalized DataFrame (passed through after validation)
    """
    from negmas.tournaments.neg.simple.cartesian import SimpleTournamentResults
    from negmas.tournaments.tournaments import TournamentResults

    if isinstance(source, (str, Path)):
        return records_from_path(source)
    if isinstance(source, SimpleTournamentResults):
        return records_from_simple_results(source)
    if isinstance(source, TournamentResults):
        return records_from_situated(source)
    if isinstance(source, pd.DataFrame):
        missing = set(COMMON_COLUMNS) - set(source.columns)
        if missing:
            raise ValueError(
                f"DataFrame passed to load_records() is missing required "
                f"columns: {sorted(missing)}. Expected columns: {COMMON_COLUMNS}"
            )
        return source[COMMON_COLUMNS].reset_index(drop=True)
    raise TypeError(
        f"Cannot load tournament records from object of type {type(source)}"
    )
