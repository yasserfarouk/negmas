"""Hybrid negotiator combining multiple strategies."""

from __future__ import annotations

from ..components.acceptance import ACNext
from ..components.offering import HybridOfferingPolicy
from .modular.mapneg import MAPNegotiator

__all__ = ["HybridNegotiator"]


class HybridNegotiator(MAPNegotiator):
    """
    A negotiator mixing a time-based (Bezier) concession curve with a
    behaviour-based reaction to the opponent, accepting via ACnext.

    Args:
         name: Negotiator name
         parent: Parent controller if any
         preferences: The preferences of the negotiator
         ufun: The ufun of the negotiator (overrides prefrences)
         owner: The `Agent` that owns the negotiator.
         alpha: ACnext utility scale (see `ACNext`).
         beta: ACnext utility offset (see `ACNext`).
         initial_utility: Bezier control point at ``t=0``. ``NaN`` means *auto*
             (see `HybridOfferingPolicy`).
         concession_ratio: Middle Bezier control point. ``NaN`` means *auto*.
         final_utility: Bezier control point at ``t=1``. ``NaN`` means *auto*
             (derived from the domain size).
         empathy_score: How strongly the opponent's concession moves our target.
             ``NaN`` means *auto*.
         auto_initial_utility: ``initial_utility`` used in auto mode.
         auto_concession_ratio: ``concession_ratio`` used in auto mode.
         auto_empathy_score: ``empathy_score`` used in auto mode.
         domain_size_cap: Domain cardinality is clipped to this before consulting
             ``final_utility_ladder``.
         final_utility_ladder: Ordered ``(max_domain_size, final_utility)`` pairs
             used to derive ``final_utility`` in auto mode.
         final_utility_floor: ``final_utility`` in auto mode for domains larger
             than every ladder threshold.
         behavior_min_offers: Offers to receive before mixing in the
             behaviour-based component.
         enumeration_levels: Discretization levels for continuous outcome spaces.
         enumeration_max_cardinality: Max outcomes enumerated for continuous
             outcome spaces.
         frac_time_based: Window weights over the opponent's recent utility
             differences.
         above_only: Only consider outcomes at or above the target utility.
         offering: A ready `HybridOfferingPolicy` to use instead of building one
             from the arguments above (which are then ignored).
         acceptance: A ready `AcceptancePolicy` to use instead of `ACNext`.

    Remarks:
        - Every hyperparameter of `HybridOfferingPolicy` is reachable here, so
          the negotiator can be tuned without constructing components by hand.
    """

    def __init__(
        self,
        *args,
        alpha: float = 1.0,
        beta: float = 0.0,
        initial_utility: float = float("nan"),
        concession_ratio: float = float("nan"),
        final_utility: float = float("nan"),
        empathy_score: float = float("nan"),
        auto_initial_utility: float = 1.0,
        auto_concession_ratio: float = 0.75,
        auto_empathy_score: float = 0.5,
        domain_size_cap: int = 100_000,
        final_utility_ladder: tuple[tuple[float, float], ...] = (
            (450, 0.80),
            (1500, 0.775),
            (4500, 0.75),
            (18000, 0.725),
            (33000, 0.70),
        ),
        final_utility_floor: float = 0.675,
        behavior_min_offers: int = 2,
        enumeration_levels: int = 10,
        enumeration_max_cardinality: int = 1_000_000,
        frac_time_based: dict[int, tuple[float, ...]] | None = None,
        above_only: bool = False,
        **kwargs,
    ):
        """Initializes the instance."""
        offering = kwargs.pop("offering", None)
        if offering is None:
            offering_kwargs: dict = dict(
                initial_utility=initial_utility,
                concession_ratio=concession_ratio,
                final_utility=final_utility,
                empathy_score=empathy_score,
                auto_initial_utility=auto_initial_utility,
                auto_concession_ratio=auto_concession_ratio,
                auto_empathy_score=auto_empathy_score,
                domain_size_cap=domain_size_cap,
                final_utility_ladder=final_utility_ladder,
                final_utility_floor=final_utility_floor,
                behavior_min_offers=behavior_min_offers,
                enumeration_levels=enumeration_levels,
                enumeration_max_cardinality=enumeration_max_cardinality,
                above_only=above_only,
            )
            if frac_time_based is not None:
                offering_kwargs["frac_time_based"] = frac_time_based
            offering = HybridOfferingPolicy(**offering_kwargs)
        kwargs["offering"] = offering
        kwargs.setdefault("acceptance", ACNext(offering, alpha=alpha, beta=beta))
        super().__init__(*args, **kwargs)
