"""Grid-based discretization.

:class:`GridBasedDiscretizer` samples every continuous issue on an even grid of
``min_levels`` values (endpoints included), keeps discrete issues unchanged, and
steps the number of levels down when necessary so the total never exceeds
``max_outcomes``.

It is the same operation as :meth:`OutcomeSpace.to_largest_discrete
<negmas.outcomes.OutcomeSpace.to_largest_discrete>` (both step levels down to fit
the cap), and it equals :meth:`OutcomeSpace.to_discrete
<negmas.outcomes.OutcomeSpace.to_discrete>` in the common case where the
``min_levels`` grid already fits under ``max_outcomes`` (``to_discrete`` does not
step down — it raises instead).

Numeric summary (two continuous issues ``a, b``):

===================================  ================  =========================
call                                 result            note
===================================  ================  =========================
``min_levels=5,  max_outcomes=100``  ``5 x 5 = 25``    grid fits as-is
``min_levels=5,  max_outcomes=20``   ``4 x 4 = 16``    stepped down (25 > 20)
``min_levels=None (=> 10)``          ``10 x 10 = 100``  default levels, no cap
===================================  ================  =========================
"""

from __future__ import annotations

from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

from negmas.outcomes.outcome_space import DiscreteCartesianOutcomeSpace

from ._protocol import BaseDiscretizer

if TYPE_CHECKING:
    from negmas.outcomes.protocols import DiscreteOutcomeSpace, OutcomeSpace

__all__ = ["GridBasedDiscretizer"]


class GridBasedDiscretizer(BaseDiscretizer):
    """Discretizes by sampling an even grid of values from each continuous issue.

    This is the faithful, ufun-agnostic default discretizer:

    - **Continuous issues** are discretized to an even grid of ``min_levels``
      values (endpoints included). If the grid would exceed ``max_outcomes``, the
      number of levels is stepped down (toward 1) until it fits.
    - **Discrete issues** are kept unchanged (their cardinality is never increased
      to reach ``min_levels`` and never decreased to meet ``max_outcomes``). Use
      :meth:`DiscreteOutcomeSpace.limit_cardinality` to shrink an already-discrete
      space.
    - An already-discrete space is returned unchanged if it fits under
      ``max_outcomes``; otherwise a :class:`ValueError` is raised (grid
      discretization does not drop values from discrete issues).
    - A :class:`ValueError` is also raised if even a single level per continuous
      issue still exceeds ``max_outcomes`` (e.g. a large discrete issue alongside a
      continuous one).

    Edge cases (numeric):

    - ``min_levels=None`` → ``DEFAULT_LEVELS`` (10) values per continuous issue.
    - ``max_outcomes=None`` → no cardinality cap (levels never stepped down).
    - discrete ``3 x 4`` with ``max_outcomes=100`` → returned unchanged (12 ≤ 100).
    - discrete ``5 x 5`` with ``max_outcomes=10`` → ``ValueError`` (25 > 10; use
      ``limit_cardinality``).
    - continuous ``a`` + discrete ``d`` of size 100, ``max_outcomes=50`` →
      ``ValueError`` (``1 x 100 = 100 > 50`` even at one level).

    Examples:
        >>> from negmas.outcomes import make_issue, make_os
        >>> from negmas.outcomes.discretizers import GridBasedDiscretizer
        >>> os = make_os([make_issue((0.0, 1.0), "a"), make_issue((0.0, 1.0), "b")])
        >>> discretize = GridBasedDiscretizer(max_outcomes=100, min_levels=5)
        >>> discretize(os).cardinality
        25
        >>> # the cap steps the number of levels down to fit (4 * 4 = 16 <= 20)
        >>> GridBasedDiscretizer(max_outcomes=20, min_levels=5)(os).cardinality
        16
    """

    def __call__(self, outcome_space: OutcomeSpace) -> DiscreteOutcomeSpace:
        os = outcome_space
        max_card = self.max_cardinality

        if os.is_discrete():
            dos: DiscreteOutcomeSpace = os  # type: ignore[assignment]
            if dos.cardinality <= max_card:
                return dos
            raise ValueError(
                f"Cannot discretize the already-discrete {os} to at most "
                f"{self.max_outcomes} outcomes (it has {dos.cardinality}). Grid "
                f"discretization does not shrink discrete issues; use "
                f"`limit_cardinality` instead."
            )

        # Step levels down until the continuous grid fits under the cap.
        for level in range(self.levels, 0, -1):
            issues = tuple(
                issue.to_discrete(level, compact=False, grid=True, endpoints=True)
                if issue.is_continuous()
                else issue
                for issue in os.issues  # type: ignore[attr-defined]
            )
            cardinality = reduce(mul, [i.cardinality for i in issues], 1)
            if cardinality <= max_card:
                return DiscreteCartesianOutcomeSpace(
                    issues, name=getattr(os, "name", None)
                )

        # Even one level per continuous issue is too many (large discrete issues).
        one_level = reduce(
            mul,
            [
                (1 if i.is_continuous() else i.cardinality)
                for i in os.issues  # type: ignore[attr-defined]
            ],
            1,
        )
        raise ValueError(
            f"Cannot discretize {os} to at most {self.max_outcomes} outcomes: even "
            f"one level per continuous issue yields {one_level} outcomes."
        )
