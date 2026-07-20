"""Utility-aware ("balanced") discretizers.

These discretizers select a *subset* of outcomes from a dense candidate grid so
that the selection is balanced with respect to one or more utility functions.
They all return a :class:`~negmas.outcomes.SubsetCartesianOutcomeSpace` (issue
structure preserved, only the selected outcomes valid).

Two axes distinguish the four public discretizers:

- **Bin definition**
    - *variance* (:class:`BalancedUFunVarianceDiscretizer`,
      :class:`BalancedUFunsVarianceDiscretizer`): **quantile** (equal-frequency)
      utility bins.  Bin boundaries adapt to the density of utilities, so each bin
      holds roughly the same share of candidates — this balances the within-bin
      utility variance.
    - *outcome counts* (:class:`BalancedOutcomeCountsInUFunBinsDiscretizer`,
      :class:`BalancedOutcomeCountsInUFunsBinsDiscretizer`): **equal-width**
      utility bins; an equal number of outcomes is selected from each bin, which
      flattens the utility distribution of the selection.
- **Number of ufuns**
    - single-ufun variants bin the (scalar) utility directly.
    - multi-ufun variants (``...UFuns...``) bin along **each** ufun independently
      into ``n_bins`` levels, forming a joint ``n_bins``-per-ufun grid of cells and
      balancing across that joint grid (capturing trade-offs between the ufuns).

Candidate generation
---------------------
For continuous/infinite spaces a *dense* internal grid is enumerated first
(independent of ``min_levels``, capped at ``max_candidates``), and the balanced
selection then picks ``max_outcomes`` from it.  For already-discrete spaces the
outcomes are enumerated (or sampled down to ``max_candidates``).

Budget
------
``max_outcomes`` is the total selection budget, distributed as evenly as possible
across the (non-empty) bins/cells.  If a sparse bin cannot absorb its fair share
the shortfall is *water-filled* into bins that still have capacity, so the budget
is met whenever enough candidates exist (even for skewed utility distributions).
When ``max_outcomes`` is ``None`` the largest *balanced* selection is returned: an
equal count taken from every non-empty bin, limited by the least-populated bin.

Numeric illustration
--------------------
One issue with 9 outcomes whose utilities are ``[0,1,2,3,4,50,60,70,80]`` (a dense
low cluster + a sparse high cluster), ``n_bins=3``, ``max_outcomes=3`` (one per
bin):

- ``balanced_ufun_variance`` (quantile / equal-frequency) selects utilities
  ``[0, 3, 60]`` — representation follows *density*, so the crowded low region
  contributes two of the three picks.
- ``balanced_outcome_counts_in_ufun_bins`` (equal-width) selects utilities
  ``[0, 50, 60]`` — bins split the utility *range* evenly, so the sparse high
  region gets relatively more representation.

Edge cases: ``n_bins=1`` → a single bin (all candidates compete for the budget);
a degenerate ufun (all-equal utilities) → one bin; ``max_outcomes`` larger than
the candidate pool → every candidate is kept.

Hole-free (full-grid) output
----------------------------
By default these discretizers *select outcomes*, producing a
:class:`~negmas.outcomes.SubsetCartesianOutcomeSpace` that can have "holes"
(valid issue-value combinations that are not selected). Passing ``full_grid=True``
instead returns an ordinary hole-free
:class:`~negmas.outcomes.DiscreteCartesianOutcomeSpace`: the number of values kept
per issue is fixed by ``min_levels``/``max_outcomes`` (a balanced allocation), and
an optimizer chooses *which* values per issue so the full Cartesian grid's utility
histogram over **fixed** reference bins (computed once from the dense pool) is as
uniform as possible.

The optimizer is selected with ``grid_optimizer``:

- ``"coordinate"`` (default): deterministic coordinate descent — swaps one chosen
  value at a time, keeping strictly-improving moves until convergence.
- ``"scipy"``: :func:`scipy.optimize.differential_evolution` over integer value
  indices (stochastic but seeded; a heavier alternative backend).

On a skewed ufun the optimized grid is far more balanced than a naive even grid
(e.g. chi-square of bin counts ``2`` vs ``70`` in internal tests).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from negmas.outcomes.categorical_issue import CategoricalIssue
from negmas.outcomes.outcome_space import (
    DiscreteCartesianOutcomeSpace,
    SubsetCartesianOutcomeSpace,
    _balanced_levels,
    _grid_indices,
)

from ._protocol import BaseDiscretizer
from .grid import GridBasedDiscretizer

if TYPE_CHECKING:
    from negmas.outcomes.protocols import DiscreteOutcomeSpace, Outcome, OutcomeSpace
    from negmas.preferences.base_ufun import BaseUtilityFunction

__all__ = [
    "BalancedUFunVarianceDiscretizer",
    "BalancedUFunsVarianceDiscretizer",
    "BalancedOutcomeCountsInUFunBinsDiscretizer",
    "BalancedOutcomeCountsInUFunsBinsDiscretizer",
]

#: Default size of the dense candidate grid used before selection.
DEFAULT_MAX_CANDIDATES: int = 10_000
#: Default per-continuous-issue density of the internal candidate grid.
DEFAULT_DENSE_LEVELS: int = 100


class BaseBalancedDiscretizer(BaseDiscretizer):
    """Shared machinery for the utility-aware balanced discretizers.

    Subclasses only choose the binning strategy via :attr:`_binning`
    (``"quantile"`` or ``"equal_width"``); everything else — dense candidate
    generation, joint multi-dimensional binning, balanced budget allocation and
    representative selection — is shared.

    Args:
        ufuns: One or more utility functions used to bin candidates.
        n_bins: Number of bins per ufun dimension.
        max_outcomes: Total selection budget (``None`` == largest balanced set).
        min_levels: In subset mode, kept for the :class:`Discretizer` construction
            contract (does not size the dense candidate grid). In ``full_grid``
            mode it caps the number of values kept per issue.
        max_candidates: Size cap for the dense candidate grid.
        full_grid: If True, return a hole-free
            :class:`~negmas.outcomes.DiscreteCartesianOutcomeSpace` whose per-issue
            values are optimized for balance, instead of a
            :class:`~negmas.outcomes.SubsetCartesianOutcomeSpace` of selected
            outcomes.
        grid_optimizer: Backend for ``full_grid`` optimization — ``"coordinate"``
            (default, deterministic coordinate descent) or ``"scipy"``
            (:func:`scipy.optimize.differential_evolution`).
        max_opt_iters: Iteration cap for the ``full_grid`` optimizer.
    """

    #: Binning strategy: ``"quantile"`` (equal-frequency) or ``"equal_width"``.
    _binning: str = "quantile"

    def __init__(
        self,
        ufuns: Sequence[BaseUtilityFunction],
        n_bins: int,
        max_outcomes: int | None = None,
        min_levels: int | None = None,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        full_grid: bool = False,
        grid_optimizer: str = "coordinate",
        max_opt_iters: int = 50,
    ) -> None:
        super().__init__(max_outcomes=max_outcomes, min_levels=min_levels)
        self.ufuns = list(ufuns)
        if not self.ufuns:
            raise ValueError("At least one utility function is required")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1 (got {n_bins})")
        if grid_optimizer not in ("coordinate", "scipy"):
            raise ValueError(
                f"grid_optimizer must be 'coordinate' or 'scipy' (got {grid_optimizer!r})"
            )
        self.n_bins = int(n_bins)
        self.max_candidates = int(max_candidates)
        #: When True, return a hole-free ``DiscreteCartesianOutcomeSpace`` (a full
        #: Cartesian grid) whose per-issue values are optimized for balance,
        #: instead of a ``SubsetCartesianOutcomeSpace`` of selected outcomes.
        self.full_grid = bool(full_grid)
        self.grid_optimizer = grid_optimizer
        self.max_opt_iters = int(max_opt_iters)

    # -- candidate generation ------------------------------------------------ #
    def _candidates(self, outcome_space: OutcomeSpace) -> tuple[tuple, list[Outcome]]:
        """Returns ``(discrete_issues, candidate_outcomes)`` for ``outcome_space``."""
        os = outcome_space
        if os.is_discrete():
            dos: DiscreteOutcomeSpace = os  # type: ignore[assignment]
        else:
            dos = GridBasedDiscretizer(
                max_outcomes=self.max_candidates, min_levels=DEFAULT_DENSE_LEVELS
            )(os)
        # Shrink an over-large pool *deterministically* (via limit_cardinality)
        # rather than random sampling, so discretization is stable across calls.
        if dos.cardinality > self.max_candidates:
            dos = dos.limit_cardinality(self.max_candidates)
        issues = tuple(dos.issues)  # type: ignore[attr-defined]
        outcomes = list(dos.enumerate())
        return issues, outcomes

    # -- binning ------------------------------------------------------------- #
    def _bin_edges(self, pool_values: np.ndarray) -> np.ndarray | None:
        """Computes the (fixed) bin edges from a reference pool of utilities.

        Returns ``None`` for a single bin or a degenerate (all-equal) pool. These
        edges must be computed **once** from the reference pool and reused for all
        subsequent :meth:`_digitize` calls — recomputing them from a candidate
        grid's own utilities would make every grid trivially "balanced".
        """
        n = self.n_bins
        if n == 1:
            return None
        vmin, vmax = float(pool_values.min()), float(pool_values.max())
        if vmax <= vmin:
            return None
        if self._binning == "equal_width":
            return np.linspace(vmin, vmax, n + 1)
        if self._binning == "quantile":
            edges = np.unique(np.quantile(pool_values, np.linspace(0.0, 1.0, n + 1)))
            return edges if len(edges) >= 2 else None
        raise ValueError(f"Unknown binning strategy: {self._binning}")

    @staticmethod
    def _digitize(values: np.ndarray, edges: np.ndarray | None) -> np.ndarray:
        """Assigns each value to a bin index in ``[0, len(edges)-1)`` using ``edges``."""
        if edges is None:
            return np.zeros(len(values), dtype=int)
        # Interior edges only; clip so the maximum value lands in the last bin.
        return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)

    def _assign_bins(self, values: np.ndarray) -> np.ndarray:
        """Bins ``values`` using edges derived from ``values`` themselves.

        Used by the subset path, where the reference pool *is* the candidate set.
        """
        return self._digitize(values, self._bin_edges(values))

    # -- selection ----------------------------------------------------------- #
    def _select(
        self, candidates: list[Outcome], utilities: np.ndarray
    ) -> list[Outcome]:
        """Selects a balanced subset of ``candidates`` from their ``utilities``."""
        n_dims = utilities.shape[1]
        # Per-dimension bin index, then group by the joint (multi-dim) cell.
        bin_ids = np.stack(
            [self._assign_bins(utilities[:, d]) for d in range(n_dims)], axis=1
        )
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, key in enumerate(map(tuple, bin_ids.tolist())):
            groups[key].append(i)

        # A scalar utility per candidate (mean across ufuns) for even in-bin picks.
        scalar = utilities.mean(axis=1)

        group_items = list(groups.values())
        sizes = [len(g) for g in group_items]
        budget = self.max_outcomes

        if budget is None:
            # Largest *balanced* set: an equal count from every non-empty bin,
            # limited by the least-populated bin.
            per_group = [min(sizes)] * len(group_items)
        else:
            per_group = _allocate_budget(sizes, budget)

        selected: list[Outcome] = []
        for members, k in zip(group_items, per_group):
            members_sorted = sorted(members, key=lambda i: (scalar[i], i))
            for i in _pick_evenly(len(members_sorted), k):
                selected.append(candidates[members_sorted[i]])
        return selected

    def __call__(  # type: ignore[override]
        self, outcome_space: OutcomeSpace
    ) -> DiscreteCartesianOutcomeSpace:
        issues, candidates = self._candidates(outcome_space)
        name = getattr(outcome_space, "name", None)
        if not candidates:
            return SubsetCartesianOutcomeSpace(issues, outcomes=())
        # Utilities of the dense candidate pool — the fixed reference distribution.
        utilities = np.array(
            [[float(f(o)) for f in self.ufuns] for o in candidates], dtype=float
        )
        if self.full_grid:
            return self._optimize_grid(issues, utilities, name)
        selected = self._select(candidates, utilities)
        return SubsetCartesianOutcomeSpace(issues, outcomes=tuple(selected), name=name)

    # -- hole-free grid optimization ----------------------------------------- #
    def _optimize_grid(
        self, issues: tuple, ref_utilities: np.ndarray, name: str | None
    ) -> DiscreteCartesianOutcomeSpace:
        """Chooses per-issue values so the full Cartesian grid is balanced.

        The number of values kept per issue is fixed by ``min_levels`` /
        ``max_outcomes`` (via :func:`_balanced_levels`); the optimizer then chooses
        *which* values so the grid's utility histogram over the fixed reference
        bins is as uniform as possible. Returns a hole-free
        :class:`DiscreteCartesianOutcomeSpace`.
        """
        values = [list(iss.all) for iss in issues]  # candidate values per issue
        caps = [len(v) for v in values]
        n_issues = len(values)

        # Per-issue output counts (never exceed the candidate count or min_levels;
        # product bounded by max_outcomes, balanced across issues).
        lvl = self.min_levels
        level_caps = [c if lvl is None else min(c, int(lvl)) for c in caps]
        level_caps = [max(1, c) for c in level_caps]
        if self.max_outcomes is None:
            n_per = level_caps
        else:
            n_per = _balanced_levels(level_caps, int(self.max_outcomes))

        n_dims = ref_utilities.shape[1]
        # FIXED reference bin edges from the dense pool (computed once).
        edges = [self._bin_edges(ref_utilities[:, d]) for d in range(n_dims)]
        n_cells = 1
        for e in edges:
            n_cells *= 1 if e is None else len(e) - 1

        # Cache ufun evaluations of grid outcomes.
        cache: dict[Outcome, tuple[float, ...]] = {}

        def util_vec(outcome: Outcome) -> tuple[float, ...]:
            u = cache.get(outcome)
            if u is None:
                u = tuple(float(f(outcome)) for f in self.ufuns)
                cache[outcome] = u
            return u

        def objective(chosen_idx: list[list[int]]) -> float:
            sel = [[values[i][j] for j in chosen_idx[i]] for i in range(n_issues)]
            grid = list(itertools.product(*sel))
            u = np.array([util_vec(o) for o in grid], dtype=float)
            flat = np.zeros(len(grid), dtype=int)
            mult = 1
            for d in range(n_dims):
                e = edges[d]
                nb = 1 if e is None else len(e) - 1
                flat += self._digitize(u[:, d], e) * mult
                mult *= nb
            counts = np.bincount(flat, minlength=n_cells).astype(float)
            target = len(grid) / n_cells
            return float(np.sum((counts - target) ** 2))

        # Initial selection: evenly-spaced indices per issue.
        chosen = [_grid_indices(caps[i], n_per[i]) for i in range(n_issues)]
        if n_cells > 1:  # nothing to balance for a single bin
            if self.grid_optimizer == "scipy":
                chosen = _scipy_grid_search(caps, n_per, objective, self.max_opt_iters)
            else:
                chosen = _coordinate_descent(
                    caps, chosen, objective, self.max_opt_iters
                )

        out_issues = tuple(
            CategoricalIssue([values[i][j] for j in sorted(chosen[i])], name=iss.name)
            for i, iss in enumerate(issues)
        )
        return DiscreteCartesianOutcomeSpace(out_issues, name=name)


def _pick_evenly(n: int, k: int) -> list[int]:
    """Returns up to ``k`` evenly-spaced indices from ``range(n)`` (sorted, unique)."""
    if k <= 0 or n <= 0:
        return []
    if k >= n:
        return list(range(n))
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return sorted(set(int(i) for i in idx))


def _allocate_budget(sizes: list[int], budget: int) -> list[int]:
    """Distributes ``budget`` slots across bins as evenly as possible.

    Each bin gets at most its population (``sizes[i]``). The total allocation
    equals ``min(budget, sum(sizes))`` — any shortfall from bins that cannot
    absorb their fair share is *water-filled* into bins that still have capacity,
    so a caller asking for ``budget`` outcomes gets them whenever enough
    candidates exist (even when the utility distribution is skewed).
    """
    n = len(sizes)
    alloc = [0] * n
    remaining = min(budget, sum(sizes))
    # Bins that still have spare capacity, smallest-first so tight bins fill up
    # and release capacity fairly to the rest.
    active = sorted(range(n), key=lambda i: (sizes[i], i))
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for i in list(active):
            if remaining <= 0:
                break
            give = min(share, sizes[i] - alloc[i], remaining)
            if give <= 0:
                continue
            alloc[i] += give
            remaining -= give
            progressed = True
            if alloc[i] >= sizes[i]:
                active.remove(i)
        if not progressed:
            break
    return alloc


def _coordinate_descent(
    caps: list[int],
    chosen: list[list[int]],
    objective: Callable[[list[list[int]]], float],
    max_iters: int,
) -> list[list[int]]:
    """Greedy per-issue value swapping until no single swap improves ``objective``.

    Deterministic: for each issue and each chosen slot, tries every unused
    candidate value and accepts the first strictly-improving swap; repeats sweeps
    until a full sweep makes no improvement (or ``max_iters`` is reached).
    """
    chosen = [list(c) for c in chosen]
    best = objective(chosen)
    for _ in range(max_iters):
        improved = False
        for i in range(len(chosen)):
            in_use = set(chosen[i])
            for pos in range(len(chosen[i])):
                for cand in range(caps[i]):
                    if cand in in_use:
                        continue
                    trial = list(chosen[i])
                    trial[pos] = cand
                    trial_all = chosen[:i] + [trial] + chosen[i + 1 :]
                    score = objective(trial_all)
                    if score < best - 1e-12:
                        best = score
                        chosen[i] = trial
                        in_use = set(trial)
                        improved = True
                        break
        if not improved:
            break
    return chosen


def _scipy_grid_search(
    caps: list[int],
    n_per: list[int],
    objective: Callable[[list[list[int]]], float],
    max_iters: int,
) -> list[list[int]]:
    """Optimizes the per-issue value choice with ``scipy.optimize.differential_evolution``.

    Each chosen slot is an integer variable in ``[0, cap_i)``; duplicate indices
    within an issue are penalized so the decoder still yields distinct values.
    Provided as an optional backend; :func:`_coordinate_descent` is the default.
    """
    from scipy.optimize import differential_evolution

    bounds: list[tuple[int, int]] = []
    slices: list[tuple[int, int]] = []
    start = 0
    for i in range(len(caps)):
        for _ in range(n_per[i]):
            bounds.append((0, caps[i] - 1))
        slices.append((start, start + n_per[i]))
        start += n_per[i]

    def decode(x) -> list[list[int]]:
        return [sorted({int(round(v)) for v in x[s:e]}) for (s, e) in slices]

    def penalized(x) -> float:
        idx = decode(x)
        # Penalize collapsed (duplicate) selections so each slot stays distinct.
        pen = sum(n_per[i] - len(idx[i]) for i in range(len(idx))) * 1e9
        return objective(idx) + pen

    result = differential_evolution(
        penalized,
        bounds,
        integrality=[True] * len(bounds),
        rng=0,
        maxiter=max_iters,
        polish=False,
    )
    return decode(result.x)


class BalancedUFunVarianceDiscretizer(BaseBalancedDiscretizer):
    """Balanced discretizer using **quantile** (equal-frequency) bins of one ufun.

    Constructed with a single utility function and ``n_bins``.  Quantile bins make
    each bin span an equal share of candidates (balancing within-bin utility
    variance); an equal count is then selected from each bin.

    Args:
        ufun: The utility function to bin.
        n_bins: Number of quantile bins.
        max_outcomes: Total selection budget (``None`` == largest balanced set).
        min_levels: Construction-contract argument (does not size the dense grid).
        max_candidates: Size cap for the dense candidate grid.

    Also accepts ``full_grid`` / ``grid_optimizer`` / ``max_opt_iters`` — see
    :class:`BaseBalancedDiscretizer` for the hole-free full-grid output.
    """

    _binning = "quantile"

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        n_bins: int,
        max_outcomes: int | None = None,
        min_levels: int | None = None,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        full_grid: bool = False,
        grid_optimizer: str = "coordinate",
        max_opt_iters: int = 50,
    ) -> None:
        super().__init__(
            [ufun],
            n_bins,
            max_outcomes=max_outcomes,
            min_levels=min_levels,
            max_candidates=max_candidates,
            full_grid=full_grid,
            grid_optimizer=grid_optimizer,
            max_opt_iters=max_opt_iters,
        )


class BalancedUFunsVarianceDiscretizer(BaseBalancedDiscretizer):
    """Balanced discretizer using joint **quantile** bins of multiple ufuns.

    Each ufun is binned into ``n_bins`` quantile levels; candidates are grouped by
    the joint (multi-dimensional) cell, and an equal count is selected per
    non-empty cell — balancing variance jointly across all ufuns.

    Args:
        ufuns: The utility functions to bin.
        n_bins: Number of quantile bins per ufun.
        max_outcomes: Total selection budget (``None`` == largest balanced set).
        min_levels: Construction-contract argument (does not size the dense grid).
        max_candidates: Size cap for the dense candidate grid.

    Also accepts ``full_grid`` / ``grid_optimizer`` / ``max_opt_iters`` — see
    :class:`BaseBalancedDiscretizer` for the hole-free full-grid output.
    """

    _binning = "quantile"


class BalancedOutcomeCountsInUFunBinsDiscretizer(BaseBalancedDiscretizer):
    """Balanced discretizer using **equal-width** utility bins of one ufun.

    The ufun's utility range is split into ``n_bins`` equal-width bins and an equal
    number of outcomes is selected from each bin, flattening the utility
    distribution of the selection.

    Args:
        ufun: The utility function to bin.
        n_bins: Number of equal-width bins.
        max_outcomes: Total selection budget (``None`` == largest balanced set).
        min_levels: Construction-contract argument (does not size the dense grid).
        max_candidates: Size cap for the dense candidate grid.

    Also accepts ``full_grid`` / ``grid_optimizer`` / ``max_opt_iters`` — see
    :class:`BaseBalancedDiscretizer` for the hole-free full-grid output.
    """

    _binning = "equal_width"

    def __init__(
        self,
        ufun: BaseUtilityFunction,
        n_bins: int,
        max_outcomes: int | None = None,
        min_levels: int | None = None,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        full_grid: bool = False,
        grid_optimizer: str = "coordinate",
        max_opt_iters: int = 50,
    ) -> None:
        super().__init__(
            [ufun],
            n_bins,
            max_outcomes=max_outcomes,
            min_levels=min_levels,
            max_candidates=max_candidates,
            full_grid=full_grid,
            grid_optimizer=grid_optimizer,
            max_opt_iters=max_opt_iters,
        )


class BalancedOutcomeCountsInUFunsBinsDiscretizer(BaseBalancedDiscretizer):
    """Balanced discretizer using joint **equal-width** bins of multiple ufuns.

    Each ufun's range is split into ``n_bins`` equal-width bins; candidates are
    grouped by the joint (multi-dimensional) cell and an equal number of outcomes
    is selected per non-empty cell.

    Args:
        ufuns: The utility functions to bin.
        n_bins: Number of equal-width bins per ufun.
        max_outcomes: Total selection budget (``None`` == largest balanced set).
        min_levels: Construction-contract argument (does not size the dense grid).
        max_candidates: Size cap for the dense candidate grid.

    Also accepts ``full_grid`` / ``grid_optimizer`` / ``max_opt_iters`` — see
    :class:`BaseBalancedDiscretizer` for the hole-free full-grid output.
    """

    _binning = "equal_width"
