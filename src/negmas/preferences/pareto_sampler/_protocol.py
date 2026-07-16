"""Protocol definition for Pareto sampler components.

A ``ParetoSampler`` is similar to an ``InverseUFun`` but is designed for
*trade-off queries*: given an estimate of the opponent's utility function it
searches for outcomes that are approximately Pareto-optimal with respect to
both the agent's own utility and the opponent's utility.

Access to the opponent's utility estimate is expected via
``Negotiator.opponent_ufun`` (which returns ``None`` when no estimate is
available, e.g. at the start of a negotiation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from negmas.outcomes import Outcome
    from negmas.preferences.base_ufun import BaseUtilityFunction

__all__ = ["ParetoSampler"]


@runtime_checkable
class ParetoSampler(Protocol):
    """Protocol for components that sample from the approximate Pareto frontier.

    A ``ParetoSampler`` wraps the agent's own utility function and, optionally,
    an estimate of the opponent's utility function to answer trade-off queries:

    * **Pareto outcomes** – outcomes that cannot be improved for one party
      without harming the other.
    * **Best for opponent** – the Pareto-optimal outcome that maximises the
      opponent's utility subject to the agent's own utility being at least
      some minimum threshold.

    The opponent's utility function may be ``None`` (e.g. at the start of a
    negotiation before any modelling has been done); implementations should
    handle this gracefully (e.g. by returning ``None`` or an empty list).
    """

    ufun: BaseUtilityFunction
    """The agent's own utility function."""

    initialized: bool
    """Whether ``init()`` has been called."""

    def init(self) -> None:
        """One-time (offline) initialisation.

        Any computationally expensive setup (e.g. building a DP table or
        computing the Pareto frontier) should be done here rather than in
        ``__init__``.
        """
        ...

    def pareto_outcomes(
        self,
        n: int | None = None,
        *,
        min_util: float = 0.0,
        normalized: bool = False,
        opponent_ufun: BaseUtilityFunction | None = None,
    ) -> list[Outcome]:
        """Return a list of approximately Pareto-optimal outcomes.

        Args:
            n: Maximum number of outcomes to return (``None`` = return all
               found on the approximate frontier).
            min_util: Only return outcomes with own utility ≥ this value.
            normalized: if ``True``, *min_util* is in normalised [0,1] space
               (0 = worst, 1 = best for own ufun).
            opponent_ufun: Opponent utility function to use for this call. If
               ``None``, the instance's stored opponent ufun is used.

        Returns:
            A list of outcomes on (or close to) the Pareto frontier,
            filtered by *min_util*.  May be empty if no opponent ufun is
            available or if no rational outcome exists.
        """
        ...

    def best_for_opponent(
        self,
        *,
        min_util: float,
        normalized: bool = False,
        opponent_ufun: BaseUtilityFunction | None = None,
    ) -> Outcome | None:
        """Return the outcome that maximises the opponent's utility subject to
        own utility ≥ *min_util*.

        This corresponds to the *trade-off query* from the BIDS paper
        (Koça et al., 2024):

        .. code-block:: none

            argmax  u'(ω)
             ω ∈ Ω
            subject to  u(ω) ≥ min_util

        Args:
            min_util: Minimum required own utility.
            normalized: if ``True``, *min_util* is in normalised [0,1] space.
            opponent_ufun: Opponent utility function.  Defaults to the stored
               opponent ufun.

        Returns:
            The best trade-off outcome, or ``None`` if no opponent ufun is
            available or no outcome satisfies the constraint.
        """
        ...
