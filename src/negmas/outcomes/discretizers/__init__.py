"""Discretizers: callables that turn an ``OutcomeSpace`` into a finite one.

A :class:`Discretizer` is *constructed* with two bounds — ``max_outcomes`` (a
cardinality cap, ``None`` == no cap) and ``min_levels`` (levels per continuous
issue, ``None`` == :data:`DEFAULT_LEVELS`) — and then *called* with an
:class:`~negmas.outcomes.OutcomeSpace` to produce a
:class:`~negmas.outcomes.DiscreteOutcomeSpace`::

    from negmas.outcomes.discretizers import GridBasedDiscretizer

    discretize = GridBasedDiscretizer(max_outcomes=1000, min_levels=10)
    discrete_os = discretize(outcome_space)

See :class:`BaseDiscretizer` for the full construction contract (including how
``max_outcomes`` and ``min_levels`` interact when they conflict).

Relationship to existing negmas discretization
-----------------------------------------------
negmas already ships several discretization primitives; discretizers wrap and
generalize them behind a single, configurable, reusable callable:

- :meth:`OutcomeSpace.to_discrete(levels, max_cardinality)
  <negmas.outcomes.OutcomeSpace.to_discrete>` — samples ``levels`` grid values
  from each *continuous* issue (discrete issues kept as-is), raising if the
  result would exceed ``max_cardinality`` (it does **not** step levels down).
- :meth:`OutcomeSpace.to_largest_discrete(levels, max_cardinality)
  <negmas.outcomes.OutcomeSpace.to_largest_discrete>` — picks the largest level
  count that fits under the cap by stepping levels down; this is exactly what
  ``GridBasedDiscretizer`` does (``min_levels=levels``, ``max_outcomes=max_cardinality``).
- :meth:`DiscreteOutcomeSpace.limit_cardinality(max_cardinality, levels)
  <negmas.outcomes.DiscreteOutcomeSpace.limit_cardinality>` — the complementary
  tool that *shrinks an already-discrete space* (dropping values, balanced across
  issues); ``GridBasedDiscretizer`` does not do this (it keeps discrete issues and
  raises if they exceed the cap).
- :meth:`Issue.to_discrete <negmas.outcomes.Issue.to_discrete>` — the per-issue
  grid sampler that the outcome-space methods call for continuous issues.

Available discretizers
-----------------------
- :class:`GridBasedDiscretizer` (``grid_based``): even per-issue grid; the
  faithful, ufun-agnostic default matching the built-in ``to_discrete``.
- :class:`BalancedUFunVarianceDiscretizer` (``balanced_ufun_variance``):
  quantile (equal-frequency) utility bins of one ufun.
- :class:`BalancedUFunsVarianceDiscretizer` (``balanced_ufuns_variance``):
  joint quantile bins of multiple ufuns.
- :class:`BalancedOutcomeCountsInUFunBinsDiscretizer`
  (``balanced_outcome_counts_in_ufun_bins``): equal-width utility bins of one
  ufun with an equal outcome count selected per bin.
- :class:`BalancedOutcomeCountsInUFunsBinsDiscretizer`
  (``balanced_outcome_counts_in_ufuns_bins``): joint equal-width bins of multiple
  ufuns.

The utility-aware ("balanced") discretizers select a subset of outcomes and
therefore return a :class:`~negmas.outcomes.SubsetCartesianOutcomeSpace`.
"""

from __future__ import annotations

from ._protocol import DEFAULT_LEVELS, BaseDiscretizer, Discretizer
from .balanced import (
    BalancedOutcomeCountsInUFunBinsDiscretizer,
    BalancedOutcomeCountsInUFunsBinsDiscretizer,
    BalancedUFunsVarianceDiscretizer,
    BalancedUFunVarianceDiscretizer,
    BaseBalancedDiscretizer,
)
from .grid import GridBasedDiscretizer

#: The default discretizer used across the library when none is specified.
DefaultDiscretizer = GridBasedDiscretizer

#: Registry mapping a short name to a :class:`Discretizer` class. Used by
#: :meth:`OutcomeSpace.to_discrete`'s ``method=`` argument. Extend this when adding
#: a new discretizer so it becomes addressable by name.
DISCRETIZERS: dict[str, type] = {
    "grid_based": GridBasedDiscretizer,
    "balanced_ufun_variance": BalancedUFunVarianceDiscretizer,
    "balanced_ufuns_variance": BalancedUFunsVarianceDiscretizer,
    "balanced_outcome_counts_in_ufun_bins": BalancedOutcomeCountsInUFunBinsDiscretizer,
    "balanced_outcome_counts_in_ufuns_bins": BalancedOutcomeCountsInUFunsBinsDiscretizer,
}


def get_discretizer(name: str) -> type:
    """Returns the :class:`Discretizer` class registered under ``name``.

    Raises:
        ValueError: If ``name`` is not a known discretizer.
    """
    try:
        return DISCRETIZERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown discretizer {name!r}. Known names: {sorted(DISCRETIZERS)}. "
            "Alternatively pass a Discretizer instance or class."
        ) from None


__all__ = [
    "Discretizer",
    "BaseDiscretizer",
    "DEFAULT_LEVELS",
    "GridBasedDiscretizer",
    "DefaultDiscretizer",
    "BaseBalancedDiscretizer",
    "BalancedUFunVarianceDiscretizer",
    "BalancedUFunsVarianceDiscretizer",
    "BalancedOutcomeCountsInUFunBinsDiscretizer",
    "BalancedOutcomeCountsInUFunsBinsDiscretizer",
    "DISCRETIZERS",
    "get_discretizer",
]
