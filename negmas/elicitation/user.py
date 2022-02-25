from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from negmas import warnings
from negmas.preferences.preferences import Preferences
from negmas.types.rational import Rational

from ..outcomes import Outcome
from ..preferences import MappingUtilityFunction, UtilityFunction
from .queries import Constraint, CostEvaluator, QResponse, Query

np.seterr(all="raise")  # setting numpy to raise exceptions in case of errors

__all__ = ["User", "ElicitationRecord"]


@dataclass
class ElicitationRecord:
    cost: float
    query: Query
    answer_index: int
    step: int | None = None

    def __str__(self):
        if self.step is None:
            return f"{self.query}: {self.query.answers[self.answer_index]} @{self.cost}"
        return f"[{self.step}] {self.query}: {self.query.answers[self.answer_index]} @{self.cost}"

    __repr__ = __str__


class User(Rational):
    """Abstract base class for all representations of users used for elicitation

    Args:
        preferences: The real utility function of the user (pass either ufun or preferences).
        ufun: The real utility function of the user (pass either ufun or preferences).
        cost: A cost to be added for every question asked to the user.
        nmi: [Optional] The `AgentMechanismInterface` representing *the*
             negotiation session engaged in by this user using this `ufun`.

    """

    def __init__(self, cost: float = 0.0, nmi=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost = cost
        self.total_cost = 0.0
        self._elicited_queries: list[ElicitationRecord] = []
        self.nmi = nmi

    def set(self, preferences: Preferences | None = None, cost: float = None):
        """
        Sets the ufun and/or cost for this user        <`0:desc`>

        Args:
            preferences: The ufun if given
            cost: The cost if given
        """

        if preferences is not None:
            self._preferences = preferences
        if cost is not None:
            self.cost = cost

    @property
    def ufun(self) -> UtilityFunction:
        """Gets a `UtilityFunction` representing the real utility_function of the user"""
        return (
            self._preferences
            if self._preferences is not None
            else MappingUtilityFunction(lambda x: None)
        )

    def ask(self, q: Query | None) -> QResponse:
        """Query the user and get a response."""
        if q is None:
            return QResponse(answer=None, indx=-1, cost=0.0)
        self.total_cost += self.cost + q.cost
        for i, reply in enumerate(q.answers):
            if reply.constraint.is_satisfied(self.ufun, reply.outcomes):
                self.total_cost += reply.cost
                self._elicited_queries.append(
                    ElicitationRecord(
                        query=q,
                        cost=self.cost + q.cost + reply.cost,
                        answer_index=i,
                        step=self.nmi.state.step if self.nmi else None,
                    )
                )
                return QResponse(
                    answer=reply, indx=i, cost=CostEvaluator(self.cost)(q, reply)
                )
        warnings.warn(
            f"No response for {q} (ufun={self.ufun})", warnings.NegmasNoResponseWarning
        )
        return QResponse(answer=None, indx=-1, cost=q.cost)

    def cost_of_asking(
        self, q: Query | None = None, answer_id: int = -1, estimate_answer_cost=True
    ) -> float:
        """
        Returns the cost of asking the given `Quers`.

        Args:
            q: The query
            answer_id: If >= 0, the answer expected. Used to add the specific
                      cost of the answer if `estimate_answer_cost` is `True`
            estimate_answer_cost: If `True` and `answer_id` >= 0, the specific
                                  cost of getting this answer is added.
        """
        if q is None:
            return self.cost
        cost = self.cost + q.cost
        if not estimate_answer_cost:
            return cost
        if answer_id <= 0:
            return cost + q.answers[answer_id].cost
        return cost + sum(a.cost for a in q.answers) / len(q.answers)

    def is_satisfied(self, constraint: Constraint, outcomes=List[Outcome]) -> bool:
        """Checks if the given consgtraint is satisfied for the user
        utility fun and the given outocmes.

        Args:
            constraint: The `Constraint`
            outcome: A list of `Outcome`s to be passed to the constraint along
                     with the user's ufun.
        """
        return constraint.is_satisfied(self.ufun, outcomes=outcomes)

    def elicited_queries(self) -> list[ElicitationRecord]:
        """Returns a list of elicited queries.

        For each elicited query, the following dataclass is returned:
        ElicitationRecord(query, cost, answer_index, step)
        """
        return self._elicited_queries
