"""Full-ranking methods over an empirical payoff matrix
(:class:`negmas.tournaments.analysis.payoff.PayoffTable`).

Implements Sequential Elimination Ranking (de Jonge's "Introduction to
Automated Negotiation", chapter 6.3) and plain Tournament Evaluation (chapter
6.1) as a baseline for comparison against it and against
:func:`negmas.tournaments.analysis.equilibria.egta_evaluation`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .payoff import PayoffTable

__all__ = ["tournament_evaluation_ranking", "sequential_elimination_ranking"]


def tournament_evaluation_ranking(payoff: PayoffTable) -> pd.DataFrame:
    """Baseline "Tournament Evaluation" ranking (de Jonge, chapter 6.1): each
    strategy's score is simply the average of its payoff row over all
    opponents (including itself). Ranks strategies by descending score.
    """
    scores = np.nanmean(payoff.matrix, axis=1)
    df = pd.DataFrame({"strategy": payoff.strategies, "score": scores})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def sequential_elimination_ranking(payoff: PayoffTable) -> pd.DataFrame:
    """Sequential Elimination Ranking (de Jonge, chapter 6.3).

    Produces a full ranking of all strategies by repeatedly eliminating the
    currently-worst strategy, where "worst" is judged only against the
    strategies still remaining (unlike Tournament Evaluation, which always
    averages over the *entire* original strategy set):

    At each round, with a remaining set ``Ag_i`` of size ``i``, every
    remaining strategy's score is::

        U^i(ag) = (1 / i) * sum(U(ag, ag') for ag' in Ag_i)

    (including ``ag`` itself, i.e. self-play is part of the average). The
    remaining strategy with the *lowest* score is eliminated and assigned
    rank ``i`` (worse ranks are higher numbers); the process repeats on the
    shrinking set until one strategy remains, which is assigned rank 1.

    This models a population that only knows, at any time, which of the
    *currently available* strategies is weakest -- unlike
    :func:`tournament_evaluation_ranking`, injecting one uniformly weak
    "filler" strategy does not change the relative ranking of the others,
    since it gets eliminated in the first round and is excluded from every
    subsequent average.

    Ties are broken deterministically by alphabetical strategy name.
    """
    remaining = list(payoff.strategies)
    n = len(remaining)
    order: list[str] = []  # worst-first elimination order
    M = payoff.to_frame()
    for _ in range(n):
        sub = M.loc[remaining, remaining]
        scores = sub.mean(axis=1, skipna=True)
        worst_score = scores.min()
        worst = sorted(s for s in remaining if scores[s] <= worst_score)[0]
        order.append(worst)
        remaining.remove(worst)

    # `order` is worst -> best; rank n (worst) down to rank 1 (best).
    ranks = {s: n - i for i, s in enumerate(order)}
    df = pd.DataFrame(
        {"strategy": order, "rank": [ranks[s] for s in order]}
    ).sort_values("rank")
    return df.reset_index(drop=True)
