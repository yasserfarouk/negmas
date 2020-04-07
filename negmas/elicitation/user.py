from dataclasses import dataclass

import numpy as np
from typing import List, Optional

from ..outcomes import Outcome
from ..utilities import UtilityFunction
from .queries import Query, QResponse, Constraint, CostEvaluator

np.seterr(all="raise")  # setting numpy to raise exceptions in case of errors

__all__ = ["User", "ElicitationRecord"]


@dataclass
class ElicitationRecord:
    cost: float
    query: Query
    answer_index: int

    def __str__(self):
        return f"{self.query} --> {self.query.answers[self.answer_index]} ({self.cost})"

    __repr__ = __str__


class User:
    """Abstract base class for all representations of users used for elicitation"""

    def __init__(self, ufun: Optional[UtilityFunction] = None, cost: float = 0.0):
        super().__init__()
        self.utility_function = ufun
        self.cost = cost
        self.total_cost = 0.0
        self._elicited_queries: List[ElicitationRecord] = []

    def set(self, ufun: Optional[UtilityFunction] = None, cost: float = None):
        if ufun is not None:
            self.utility_function = ufun
        if cost is not None:
            self.cost = cost

    @property
    def ufun(self) -> UtilityFunction:
        """Gets a `UtilityFunction` representing the real utility_function of the user"""
        return (
            self.utility_function
            if self.utility_function is not None
            else lambda x: None
        )

    def ask(self, q: Optional[Query]) -> QResponse:
        """Query the user and get a response."""
        if q is None:
            return QResponse(answer=None, indx=-1, cost=0.0)
        self.total_cost += self.cost + q.cost
        for i, reply in enumerate(q.answers):
            if reply.constraint.is_satisfied(self.ufun, reply.outcomes):
                self.total_cost += reply.cost
                self._elicited_queries.append(
                    ElicitationRecord(
                        query=q, cost=self.cost + q.cost + reply.cost, answer_index=i
                    )
                )
                return QResponse(
                    answer=reply, indx=i, cost=CostEvaluator(self.cost)(q, reply)
                )
        print(f"No response for {q} (ufun={self.ufun})")
        return QResponse(answer=None, indx=-1, cost=q.cost)

    def cost_of_asking(
        self, q: Optional[Query] = None, answer_id: int = -1, estimate_answer_cost=True
    ) -> float:
        if q is None:
            return self.cost
        cost = self.cost + q.cost
        if not estimate_answer_cost:
            return cost
        if answer_id <= 0:
            return cost + q.answers[answer_id].cost
        return cost + sum(a.cost for a in q.answers) / len(q.answers)

    def is_satisfied(self, constraint: Constraint, outcomes=List[Outcome]) -> bool:
        """Query the user"""
        return constraint.is_satisfied(self.ufun, outcomes=outcomes)

    def elicited_queries(self):
        """Returns a list of elicited queries.

        For each elicited ask, the following tuple is returned:
        ElicitationRecord(query, cost, answer_index)

        scale == 0 -> an exact Negotiator is returned

        """
        return self._elicited_queries
