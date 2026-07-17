"""Tit-for-tat negotiator implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from negmas.gb.components.concession import KindConcessionRecommender

from ..components import TFTAcceptancePolicy, TFTOfferingPolicy, ZeroSumModel
from .modular import MAPNegotiator

if TYPE_CHECKING:
    pass


__all__ = ["NaiveTitForTatNegotiator", "SimpleTitForTatNegotiator"]


class NaiveTitForTatNegotiator(MAPNegotiator):
    """
    Implements a naive tit-for-tat strategy that does not depend on the
    availability of an opponent model.

    The negotiator mirrors the opponent's concession: if the opponent's last
    offer was better for this negotiator than the one before it, the negotiator
    considers that the opponent has conceded by the difference and concedes a
    matching amount (adjusted by ``kindness``). This implicitly assumes a
    zero-sum situation (no opponent model is kept).

    Args:
        name: Negotiator name.
        preferences: Negotiator preferences (deprecated; use ``ufun``).
        ufun: Negotiator utility function (overrides ``preferences``).
        parent: A controller that manages this negotiator.
        kindness (float): How 'kind' the agent is. ``0.0`` is standard
            tit-for-tat. Positive values make the negotiator concede faster;
            negative values make it concede slower. Defaults to ``0.0``.
        stochastic (bool): If ``True``, offers are randomized within the band
            determined by the current concession (which reflects the
            opponent's concession). If ``False`` (default), the worst outcome
            in the band is proposed. Defaults to ``False``.
        punish (bool): If ``True``, the agent punishes a partner who does not
            concede by requiring higher utilities. Defaults to ``False``.
        initial_concession (float | str): How much the agent should concede at
            the beginning, in utility units. Can be a non-negative float or the
            string ``"min"`` (treated as ``0.0`` — minimum concession).
            Defaults to ``"min"``.
        rank_only (bool): If ``True``, only the relative ranks of outcomes (not
            their actual utilities) are used for inversion. Defaults to ``False``.
        **kwargs: Forwarded to `MAPNegotiator`.

    Remarks:
        - This negotiator does not keep an opponent model. It thinks only in
          terms of changes in its own utility. If the opponent's last offer
          was better for the negotiator compared with the one before it, it
          considers that the opponent has conceded by the difference. This
          means that it implicitly assumes a zero-sum situation.
    """

    def __init__(
        self,
        *args,
        kindness=0.0,
        punish=False,
        initial_concession: float | Literal["min"] = "min",
        rank_only: bool = False,
        stochastic: bool = False,
        **kwargs,
    ):
        """Initialize the instance.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        partner_model = ZeroSumModel(rank_only=rank_only, above_reserve=False)
        if isinstance(initial_concession, str):
            initial_concession = 0
        recommender = KindConcessionRecommender(
            initial_concession=initial_concession, kindness=kindness, punish=punish
        )
        acceptance = TFTAcceptancePolicy(
            recommender=recommender, partner_ufun=partner_model
        )
        offering = TFTOfferingPolicy(
            recommender=recommender, partner_ufun=partner_model, stochastic=stochastic
        )
        super().__init__(
            *args,
            models=[partner_model, recommender],
            model_names=["partner-model", "concession-recommender"],
            acceptance=acceptance,
            offering=offering,
            **kwargs,
        )


SimpleTitForTatNegotiator = NaiveTitForTatNegotiator
"""A simple tit-for-tat negotiator based on the MAP architecture"""
