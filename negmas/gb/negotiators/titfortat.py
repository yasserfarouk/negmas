from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from negmas.gb.components.concession import KindConcessionRecommender

from ..components import TFTAcceptancePolicy, TFTOfferingPolicy, ZeroSumModel
from .modular import MAPNegotiator

if TYPE_CHECKING:
    pass


__all__ = [
    "NaiveTitForTatNegotiator",
    "SimpleTitForTatNegotiator",
]


class NaiveTitForTatNegotiator(MAPNegotiator):
    """
    Implements a naive tit-for-tat strategy that does not depend on the availability of an opponent model.

    Args:
        name: Negotiator name
        preferences: negotiator preferences
        ufun: negotiator ufun (overrides preferences)
        parent: A controller
        kindness: How 'kind' is the agent. A value of zero is standard tit-for-tat. Positive values makes the negotiator
                  concede faster and negative values slower.
        stochastic: If `True`, the offers will be randomized above the level determined by the current concession
                        which in turn reflects the opponent's concession.
        punish: If `True` the agent punish a partner who does not seem to conede by requiring higher utilities
        initial_concession: How much should the agent concede in the beginning in terms of utility. Should be a number
                            or the special string value 'min' for minimum concession

    Remarks:
        - This negotiator does not keep an opponent model. It thinks only in terms of changes in its own utility.
          If the opponent's last offer was better for the negotiator compared with the one before it, it considers
          that the opponent has conceded by the difference. This means that it implicitly assumes a zero-sum
          situation.
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
