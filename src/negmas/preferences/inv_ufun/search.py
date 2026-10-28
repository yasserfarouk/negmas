"""An inverter for outcome spaces that cannot be enumerated (e.g. with list issues).

`SearchInverseUtilityFunction` answers queries like
`PresortingInverseUtilityFunction`, but over a searched pool of outcomes: the
extreme outcomes, random outcomes, and outcomes found by mutation
hill-climbing toward evenly spaced utility targets. It never enumerates.
"""

from __future__ import annotations

import math
import random
from typing import Any

from negmas.outcomes import Outcome

from ..base_ufun import BaseUtilityFunction
from .presorting import PresortingInverseUtilityFunction

# Finite spaces with at most this many outcomes are enumerated (exactly).
SMALL_SPACE = 100_000

__all__ = [
    "SearchInverseUtilityFunction",
    "mutate_outcome",
    "needs_search",
    "search_pool",
]


def mutate_outcome(outcome_space, outcome: Outcome) -> Outcome:
    """A random neighbour of `outcome`: one issue's value mutated or resampled."""
    issues = outcome_space.issues
    i = random.randrange(len(issues))
    issue = issues[i]
    mutate = getattr(issue, "mutate", None)
    v = mutate(outcome[i]) if mutate is not None else issue.rand()
    return tuple(outcome[:i]) + (v,) + tuple(outcome[i + 1 :])


def needs_search(outcome_space) -> bool:
    """True if the space has an issue that must be searched, not enumerated.

    These are list issues (unless the space is small and finite), and infinite
    issues that are neither discrete nor
    continuous ranges (e.g. strings).
    """
    from negmas.outcomes.list_issue import ListIssue

    issues = getattr(outcome_space, "issues", None)
    if not issues:
        return False
    if outcome_space.is_discrete() and outcome_space.cardinality <= SMALL_SPACE:
        return False  # small enough to enumerate exactly
    return any(
        isinstance(_, ListIssue) or (not _.is_discrete() and not _.is_continuous())
        for _ in issues
    )


class SearchInverseUtilityFunction(PresortingInverseUtilityFunction):
    """Presorting over a searched pool of outcomes; never enumerates.

    Args:
        ufun: The utility function to invert.
        n_samples: Random outcomes in the pool.
        n_targets: Evenly spaced utility targets to climb toward.
        n_steps: Mutation steps per target.
        rational_only, eps, rel_eps, clamp_tolerance: As in
            `PresortingInverseUtilityFunction`.

    Remarks:
        The extreme outcomes come from `ufun.extreme_outcomes()`, which is
        exact (greedy) for list value functions.
    """

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        n_samples: int = 1000,
        n_targets: int = 50,
        n_steps: int = 40,
        rational_only: bool = False,
        eps: float = 1e-12,
        rel_eps: float = 1e-6,
        clamp_tolerance: float = float("inf"),
    ):
        super().__init__(
            ufun,
            rational_only=rational_only,
            eps=eps,
            rel_eps=rel_eps,
            clamp_tolerance=clamp_tolerance,
        )
        self.n_samples = n_samples
        self.n_targets = n_targets
        self.n_steps = n_steps

    def persistable_config(self) -> dict[str, Any] | None:  # type: ignore[override]
        """Not persisted: the pool is random."""
        return None

    def _candidate_outcomes(self, outcome_space) -> list[Outcome]:
        return search_pool(
            self._ufun,
            outcome_space,
            (self._worst, self._best),
            self.n_samples,
            self.n_targets,
            self.n_steps,
        )


def search_pool(
    ufun,
    outcome_space,
    extremes: tuple[Outcome | None, Outcome | None] = (None, None),
    n_samples: int = 1000,
    n_targets: int = 50,
    n_steps: int = 40,
) -> list[Outcome]:
    """Outcomes spread over the utility range, found without enumerating.

    The extremes, `n_samples` random outcomes, and mutation hill-climbs toward
    `n_targets` evenly spaced utility targets.
    """
    pool: dict[Outcome, float] = {}

    def add(o):
        if o is None or o in pool or not outcome_space.is_valid(o):
            return
        if not outcome_space.satisfies_constraints(o):
            return
        x = float(ufun.eval(o))
        if not math.isnan(x):
            pool[o] = x

    for o in extremes:
        add(o)
    for _ in range(n_samples):
        try:
            add(outcome_space.random_outcome())
        except ValueError:
            break
    if not pool:
        return []
    lo, hi = min(pool.values()), max(pool.values())
    n = max(n_targets, 1)
    for k in range(n):
        t = lo + (hi - lo) * (k + 0.5) / n
        o = min(pool, key=lambda _: abs(pool[_] - t))
        d = abs(pool[o] - t)
        for _ in range(n_steps):
            c = mutate_outcome(outcome_space, o)
            add(c)
            if c in pool and abs(pool[c] - t) < d:
                o, d = c, abs(pool[c] - t)
    return list(pool)
