"""Tit-for-tat negotiator implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from negmas.gb.components.acceptance import ACCombi
from negmas.gb.components.concession import KindConcessionRecommender
from negmas.gb.components.models.ufun import FrequencyLinearUFunModel, UFunModel
from negmas.gb.components.offering import NiceTitForTatOfferingPolicy

from ..components import TFTAcceptancePolicy, TFTOfferingPolicy, ZeroSumModel
from .modular import MAPNegotiator

if TYPE_CHECKING:
    pass


__all__ = [
    "NaiveTitForTatNegotiator",
    "SimpleTitForTatNegotiator",
    "NiceTitForTatNegotiator",
]


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


class NiceTitForTatNegotiator(MAPNegotiator):
    """
    The Nice Tit for Tat agent (Baarslag, Hindriks & Jonker, 2013).

    A MAP negotiator combining the `NiceTitForTatOfferingPolicy` bidding
    strategy (reciprocate in the agent's own utility while aiming for a
    bargaining-solution point, and make offers attractive to the opponent) with
    the `ACCombi` acceptance condition (accept when the opponent's offer beats
    our next planned offer, or when time is running out).

    The opponent model is the one piece the bidding strategy depends on. It is
    accessed through the ``opponent_model`` property (read by the offering
    policy via ``self.negotiator.opponent_model``). You can either:

    - pass a ready opponent model as ``opponent_model`` (any `UFunModel`,
      e.g. a learned `FrequencyLinearUFunModel`, an oracle
      `PeekingOpponentModel` for tests, or a `ZeroSumModel`), or
    - pass an opponent-model *type* as ``opponent_model_type`` (a `UFunModel`
      subclass), constructed with no required args, or
    - pass neither and use the default: `FrequencyLinearUFunModel` — a
      frequency-based learner assuming a linear-additive opponent ufun, the
      same assumption as the Bayesian opponent model of Hindriks & Tykhonov
      (2008) used in the paper. Use `FrequencyUFunModel` instead when the
      opponent's ufun is not known to be linear-additive.

    Args:
        opponent_model: A ready `UFunModel` to use as the opponent model. Takes
            precedence over ``opponent_model_type``.
        opponent_model_type: A `UFunModel` subclass to instantiate as the
            default opponent model. Defaults to `FrequencyLinearUFunModel`.
        target: Bargaining solution the offering strategy aims for — one of
            ``"nash"`` (default), ``"kalai"``, ``"kalai_smorodinsky"``/``"ks"``,
            ``"max_welfare"``, ``"max_relative_welfare"``. Forwarded to
            `NiceTitForTatOfferingPolicy`.
        sample_size, max_cardinality, nash_refresh, stochastic: Forwarded to
            `NiceTitForTatOfferingPolicy`.
        pareto_sampler_type: The `ParetoSampler` implementation used by the
            offering policy for the opponent-attractive trade-off query.
            ``None`` (default) uses the offering policy's own default
            (`BruteForceParetoSampler`, exact). Pass `IPSParetoSampler` (or
            another additive sampler) for very large additive domains.
        a, b, t: ACcombi parameters (forwarded to `ACCombi`).

    Remarks:
        - Exposes ``opponent_model`` (the `UFunModel` in use) as a property.
        - The offering policy degrades to naive tit-for-tat if the opponent
          model is unavailable or uninformative.

    *AI Generated (Nice Tit for Tat MAP negotiator, after Baarslag et al. 2013).*
    """

    def __init__(
        self,
        *args,
        opponent_model: UFunModel | None = None,
        opponent_model_type: type | None = None,
        target: str = "nash",
        sample_size: int = 100,
        max_cardinality: int = 10000,
        nash_refresh: int = 1,
        stochastic: bool = False,
        pareto_sampler_type: type | None = None,
        a: float = 1.0,
        b: float = 0.0,
        t: float = 0.98,
        **kwargs,
    ):
        """Initialize the Nice Tit for Tat negotiator.

        Args:
            *args: Forwarded to `MAPNegotiator`.
            **kwargs: Forwarded to `MAPNegotiator`.
        """
        model: UFunModel
        if opponent_model is None:
            model = (opponent_model_type or FrequencyLinearUFunModel)()
        else:
            model = opponent_model
        self._opponent_model = model
        offering_kwargs: dict = dict(
            sample_size=sample_size,
            max_cardinality=max_cardinality,
            nash_refresh=nash_refresh,
            stochastic=stochastic,
            target=target,
        )
        if pareto_sampler_type is not None:
            offering_kwargs["pareto_sampler_type"] = pareto_sampler_type
        offering = NiceTitForTatOfferingPolicy(**offering_kwargs)
        acceptance = ACCombi(offering_strategy=offering, a=a, b=b, t=t)
        super().__init__(
            *args,
            models=[model],
            model_names=["opponent-model"],
            acceptance=acceptance,
            offering=offering,
            **kwargs,
        )

    @property
    def opponent_model(self) -> UFunModel | None:
        """The opponent utility-function model used by the offering strategy."""
        return self._opponent_model
