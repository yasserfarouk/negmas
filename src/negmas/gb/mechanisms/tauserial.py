"""
Implements the TAU protocol as in the paper.
"""

from __future__ import annotations


from negmas.outcomes import Outcome, Issue, OutcomeSpace, ensure_os, check_one_and_only
from negmas.gb.constraints import RepeatFinalOfferOnly
from negmas.gb.evaluators import INFINITE, TAUEvaluationStrategy
from negmas.gb.mechanisms import SerialGBMechanism


__all__ = ["SerialTAUMechanism"]


class SerialTAUMechanism(SerialGBMechanism):
    """Implements the TAU protocol using the SerialGBMechanism construct in NegMAS"""

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
        # implementing the filtering rule
        kwargs["local_constraint_type"] = RepeatFinalOfferOnly
        kwargs["local_constraint_params"] = dict(n=min_unique)
        super().__init__(*args, outcome_space=outcome_space, **kwargs)
