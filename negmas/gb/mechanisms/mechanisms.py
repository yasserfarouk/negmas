from __future__ import annotations

from negmas import check_one_and_only, ensure_os
from negmas.gb.constraints import RepeatFinalOfferOnly
from negmas.gb.evaluators import (
    INFINITE,
    GAOEvaluationStrategy,
    TAUEvaluationStrategy,
    any_accept,
)
from negmas.gb.mechanisms import GBMechanism, SerialGBMechanism
from negmas.outcomes import Issue, Outcome, OutcomeSpace
from negmas.plots.util import Colorizer, opacity_colorizer

__all__ = ["GAOMechanism", "TAUMechanism"]


class GAOMechanism(GBMechanism):
    def __init__(self, *args, **kwargs):
        kwargs["local_evaluator_type"] = GAOEvaluationStrategy
        kwargs["response_combiner"] = any_accept
        super().__init__(*args, **kwargs)

    def plot(self, *args, colorizer: Colorizer = opacity_colorizer, **kwargs):
        return super().plot(*args, colorizer=colorizer, **kwargs)


class TAUMechanism(SerialGBMechanism):
    def __init__(
        self,
        *args,
        cardinality=INFINITE,
        min_unique=0,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | int | None = None,
        **kwargs,
    ):
        check_one_and_only(outcome_space, issues, outcomes)
        outcome_space = ensure_os(outcome_space, issues, outcomes)
        kwargs["evaluator_type"] = TAUEvaluationStrategy
        kwargs["evaluator_params"] = dict(
            cardinality=cardinality, n_outcomes=outcome_space.cardinality
        )
        kwargs["local_constraint_type"] = RepeatFinalOfferOnly
        kwargs["local_constraint_params"] = dict(n=min_unique)
        super().__init__(
            *args,
            outcome_space=outcome_space,
            **kwargs,
        )
