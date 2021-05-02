from dataclasses import dataclass
import warnings

import numpy as np
from typing import List, Optional

from ..outcomes import Outcome
from ..utilities import UtilityFunction, MappingUtilityFunction
from .queries import Query, QResponse, Constraint, CostEvaluator

np.seterr(all="raise")  # setting numpy to raise exceptions in case of errors

__all__ = ["User", "ElicitationRecord"]


@dataclass
class ElicitationRecord:
    cost: float
    query: Query
    answer_index: int
    step: Optional[int] = None

    def __str__(self):
        if self.step is None:
            return f"{self.query}: {self.query.answers[self.answer_index]} @{self.cost}"
        return f"[{self.step}] {self.query}: {self.query.answers[self.answer_index]} @{self.cost}"

    __repr__ = __str__


class User:
    """Abstract base class for all representations of users used for elicitation

    Args:
        ufun: The real utility function of the user.
        cost: A cost to be added for every question asked to the user.
        ami: [Optional] The `AgentMechanismInterface` representing *the*
             negotiation session engaged in by this user using this `ufun`.

    """

    def __init__(
        self, ufun: Optional[UtilityFunction] = None, cost: float = 0.0, ami=None
    ):
        super().__init__()
        self.utility_function = ufun
        self.cost = cost
        self.total_cost = 0.0
        self._elicited_queries: List[ElicitationRecord] = []
        self.ami = ami

    def set(self, ufun: Optional[UtilityFunction] = None, cost: float = None):
        """
        Sets the ufun and/or cost for this user        <`0:desc`>

        Args:
            ufun: The ufun if given
            cost: The cost if given
        """

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
            else MappingUtilityFunction(lambda x: None)
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
                        query=q,
                        cost=self.cost + q.cost + reply.cost,
                        answer_index=i,
                        step=self.ami.state.step if self.ami else None,
                    )
                )
                return QResponse(
                    answer=reply, indx=i, cost=CostEvaluator(self.cost)(q, reply)
                )
        warnings.warn(f"No response for {q} (ufun={self.ufun})")
        return QResponse(answer=None, indx=-1, cost=q.cost)

    def cost_of_asking(
        self, q: Optional[Query] = None, answer_id: int = -1, estimate_answer_cost=True
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

    def elicited_queries(self) -> List[ElicitationRecord]:
        """Returns a list of elicited queries.

        For each elicited query, the following dataclass is returned:
        ElicitationRecord(query, cost, answer_index, step)
        """
        return self._elicited_queries
