"""The :class:`Discretizer` protocol and a shared base implementation.

A *discretizer* is a callable that turns any :class:`~negmas.outcomes.OutcomeSpace`
into a finite :class:`~negmas.outcomes.DiscreteOutcomeSpace`.  It is *constructed*
with two knobs that bound the result and then *called* with the outcome space to
discretize::

    discretize = GridBasedDiscretizer(max_outcomes=1000, min_levels=10)
    discrete_os = discretize(outcome_space)

Construction contract
----------------------
Every discretizer is built with (at least) two optional integers:

``max_outcomes``
    The maximum number of outcomes allowed in the result (maps to the
    ``max_cardinality`` argument of :meth:`OutcomeSpace.to_discrete`).  ``None``
    means *no cap* (i.e. ``float("inf")``).

``min_levels``
    The desired number of discretization levels for each continuous issue (maps
    to the ``levels`` argument of :meth:`OutcomeSpace.to_discrete`).  ``None``
    means *use the default* (:data:`DEFAULT_LEVELS`).

When the two conflict — i.e. ``min_levels`` per continuous issue would produce
more than ``max_outcomes`` outcomes — the cap wins: implementations reduce the
number of levels (or select a subset of outcomes) so the result never exceeds
``max_outcomes``.  This mirrors the behaviour of
:meth:`OutcomeSpace.to_largest_discrete`.

Because :class:`typing.Protocol` cannot usefully describe a constructor (and
``runtime_checkable`` only checks method *presence*), the protocol itself only
declares ``__call__``.  The construction contract is provided by
:class:`BaseDiscretizer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from negmas.outcomes.protocols import DiscreteOutcomeSpace, OutcomeSpace

__all__ = ["Discretizer", "BaseDiscretizer", "DEFAULT_LEVELS"]

#: The default number of discretization levels used for a continuous issue when
#: ``min_levels`` is not given.  Matches the default of
#: :meth:`negmas.outcomes.CartesianOutcomeSpace.to_discrete`.
DEFAULT_LEVELS: int = 10


@runtime_checkable
class Discretizer(Protocol):
    """A callable that discretizes an :class:`~negmas.outcomes.OutcomeSpace`.

    Implementations are *constructed* with (at least) ``max_outcomes`` and
    ``min_levels`` (both ``int | None``) and are then *called* with the outcome
    space to discretize, returning a finite
    :class:`~negmas.outcomes.DiscreteOutcomeSpace`.

    See :class:`BaseDiscretizer` for the construction contract and shared
    helpers.
    """

    def __call__(self, outcome_space: OutcomeSpace) -> DiscreteOutcomeSpace:
        """Discretizes ``outcome_space`` respecting ``max_outcomes``/``min_levels``."""
        ...


class BaseDiscretizer(ABC):
    """Base class implementing the :class:`Discretizer` construction contract.

    Args:
        max_outcomes: Maximum number of outcomes in the result. ``None`` means no
            cap.
        min_levels: Desired number of levels per continuous issue. ``None`` means
            use :data:`DEFAULT_LEVELS`.
    """

    def __init__(
        self, max_outcomes: int | None = None, min_levels: int | None = None
    ) -> None:
        self.max_outcomes = max_outcomes
        self.min_levels = min_levels

    @property
    def max_cardinality(self) -> int | float:
        """``max_outcomes`` as a cardinality (``float("inf")`` when unset)."""
        return float("inf") if self.max_outcomes is None else self.max_outcomes

    @property
    def levels(self) -> int:
        """``min_levels`` with the default applied when unset."""
        return DEFAULT_LEVELS if self.min_levels is None else self.min_levels

    @abstractmethod
    def __call__(self, outcome_space: OutcomeSpace) -> DiscreteOutcomeSpace:
        """Discretizes ``outcome_space`` (see :class:`Discretizer`)."""
        ...

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"max_outcomes={self.max_outcomes}, min_levels={self.min_levels})"
        )
