import copy
import operator
import pprint
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from .common import _loc, _upper
from ..common import AgentMechanismInterface
from ..outcomes import Outcome
from ..utilities import (
    UtilityDistribution,
    UtilityFunction,
    UtilityValue,
)

__all__ = [
    "Constraint",
    "MarginalNeutralConstraint",
    "RankConstraint",
    "ComparisonConstraint",
    "RangeConstraint",
    "Answer",
    "Query",
    "QResponse",
    "next_query",
    "possible_queries",
    "CostEvaluator",
]


class Constraint(ABC):
    """Some constraint on allowable utility values for given outcomes."""

    def __init__(
        self,
        full_range: Union[Sequence[Tuple[float, float]], Tuple[float, float]] = (
            0.0,
            1.0,
        ),
        outcomes: List[Outcome] = None,
    ):
        super().__init__()
        self.outcomes = outcomes
        self.index = None
        if outcomes is not None:
            self.index = dict(zip(outcomes, range(len(outcomes))))
            if not isinstance(full_range, tuple):
                full_range = [full_range] * len(outcomes)
        self.full_range = full_range

    @abstractmethod
    def is_satisfied(
        self, ufun: UtilityFunction, outcomes: Optional[Iterable[Outcome]] = None
    ) -> bool:
        """
        Whether or not the constraint is satisfied.
        """

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def __repr__(self):
        return self.__dict__.__repr__()

    @abstractmethod
    def marginals(
        self, outcomes: Iterable[Outcome] = None
    ) -> List[UtilityDistribution]:
        ...

    @abstractmethod
    def marginal(self, outcome: "Outcome") -> UtilityDistribution:
        ...


class MarginalNeutralConstraint(Constraint):
    """Constraints that do not affect the marginals of any outcomes. These constraints may only affect the joint
    distribution."""

    def marginals(
        self, outcomes: Iterable[Outcome] = None
    ) -> List[UtilityDistribution]:
        if outcomes is None:
            outcomes = self.outcomes
        # this works only for real-valued outcomes.
        return [
            UtilityDistribution(
                dtype="uniform",
                loc=self.full_range[_][0],
                scale=self.full_range[_][1] - self.full_range[_][0],
            )
            for _ in range(len(outcomes))
        ]

    def marginal(self, outcome: "Outcome") -> UtilityDistribution:
        # this works only for real-valued outcomes.
        if self.outcomes is None:
            return UtilityDistribution(
                dtype="uniform",
                loc=self.full_range[0],
                scale=self.full_range[1] - self.full_range[0],
            )
        indx = self.index[outcome]
        return UtilityDistribution(
            dtype="uniform",
            loc=self.full_range[indx][0],
            scale=self.full_range[indx][1] - self.full_range[indx][0],
        )


class RankConstraint(MarginalNeutralConstraint):
    """Constraints the utilities of given outcomes to be in ascending order
    """

    def __init__(
        self,
        rankings: List[int],
        full_range: Union[Sequence[Tuple[float, float]], Tuple[float, float]] = (
            0.0,
            1.0,
        ),
        outcomes: List[Outcome] = None,
    ):
        super().__init__(full_range=full_range, outcomes=outcomes)
        self.rankings = rankings

    def is_satisfied(
        self, ufun: UtilityFunction, outcomes: Optional[Iterable[Outcome]] = None
    ) -> bool:
        if outcomes is None:
            outcomes = self.outcomes
        if outcomes is None:
            raise ValueError("No outcomes are  given in construction or to the call")
        u = [(ufun(o), i) for i, o in enumerate(outcomes)]
        ranking = sorted(u, key=lambda x: x[0])
        return ranking == self.rankings


class ComparisonConstraint(MarginalNeutralConstraint):
    """Constraints the utility of given two outcomes (must be exactly two) to satisfy the given operation (e.g. >, <)"""

    def __init__(
        self,
        op: Union[str, Callable[[UtilityValue, UtilityValue], bool]],
        full_range: Union[Sequence[Tuple[float, float]], Tuple[float, float]] = (
            0.0,
            1.0,
        ),
        outcomes: List[Outcome] = None,
    ):
        super().__init__(full_range=full_range, outcomes=outcomes)
        if outcomes is not None and len(outcomes) != 2:
            raise ValueError(
                f"{len(outcomes)} outcomes were given to {self.__class__.__name__}"
            )
        self.op_name = op
        if isinstance(op, str):
            if op in ("less", "l", "<"):
                op = operator.lt
            elif op in ("greater", "g", ">"):
                op = operator.gt
            elif op in ("equal", "=", "=="):
                op = operator.eq
            elif op in ("le", "<="):
                op = operator.le
            elif op in ("ge", ">="):
                op = operator.ge
            else:
                raise ValueError(f"Unknown operation {op}")
        self.op = op

    def is_satisfied(
        self, ufun: UtilityFunction, outcomes: Optional[Iterable[Outcome]] = None
    ) -> bool:
        if outcomes is None:
            outcomes = self.outcomes
        if outcomes is None:
            raise ValueError("No outcomes are  given in construction or to the call")
        if len(outcomes) != 2:
            raise ValueError(
                f"{len(outcomes)} outcomes were given to {self.__class__.__name__}"
            )
        u = [(ufun(o), i) for i, o in enumerate(outcomes)]
        return self.op(u[0], u[1])

    def __str__(self):
        return f"{self.outcomes[0]} {self.op_name} {self.outcomes[0]}"

    __repr__ = __str__


class RangeConstraint(Constraint):
    """Constraints the utility of each of the given outcomes to lie within the given range"""

    def __init__(
        self,
        rng: Tuple = (None, None),
        full_range: Union[Sequence[Tuple[float, float]], Tuple[float, float]] = (
            0.0,
            1.0,
        ),
        outcomes: List[Outcome] = None,
        eps=1e-5,
    ):
        super().__init__(full_range=full_range, outcomes=outcomes)

        if outcomes is not None:
            self.index = dict(zip(outcomes, range(len(outcomes))))
            if not isinstance(rng, tuple):
                rng = [rng] * len(outcomes)
        self.range = rng
        self.eps = eps
        if outcomes is None:
            self.effective_range = (
                rng[0] if rng[0] is not None else self.full_range[0],
                rng[1] if rng[1] is not None else self.full_range[1],
            )
        else:
            self.effective_range = [
                (r[0] if r[0] is not None else f[0], r[1] if r[1] is not None else f[1])
                for r, f in zip(self.range, self.full_range)
            ]

    def is_satisfied(
        self, ufun: UtilityFunction, outcomes: Optional[Iterable[Outcome]] = None
    ) -> bool:
        if outcomes is None:
            outcomes = self.outcomes
        if outcomes is None:
            raise ValueError("No outcomes are  given in construction or to the call")
        us = [ufun(o) for o in outcomes]
        mn, mx = self.range
        if mn is not None:
            for u in us:
                if u < mn - self.eps:
                    return False
        if mx is not None:
            for u in us:
                if u > mx + self.eps:
                    return False
        return True

    def marginals(
        self, outcomes: Iterable[Outcome] = None
    ) -> List[UtilityDistribution]:
        if outcomes is None:
            outcomes = self.outcomes
        # this works only for real-valued outcomes.
        return [
            UtilityDistribution(
                dtype="uniform",
                loc=self.effective_range[_][0],
                scale=self.effective_range[_][1] - self.effective_range[_][0],
            )
            for _ in range(len(outcomes))
        ]

    def marginal(self, outcome: "Outcome") -> UtilityDistribution:
        # this works only for real-valued outcomes.
        if self.outcomes is None:
            return UtilityDistribution(
                dtype="uniform",
                loc=self.effective_range[0],
                scale=self.effective_range[1] - self.effective_range[0],
            )
        indx = self.index[outcome]
        return UtilityDistribution(
            dtype="uniform",
            loc=self.effective_range[indx][0],
            scale=self.effective_range[indx][1] - self.effective_range[indx][0],
        )

    def __str__(self):
        result = f"{self.range}"
        if self.outcomes is not None and len(self.outcomes) > 0:
            result += f"{self.outcomes}"
        return result

    __repr__ = __str__


@dataclass
class Answer:
    outcomes: List[Outcome]
    constraint: Constraint
    cost: float = 0.0
    name: str = ""

    def __str__(self):
        if len(self.name) > 0:
            return self.name + f"{self.constraint}"
        else:
            output = f"{self.constraint}"
            if self.cost > 1e-7:
                output += f"(cost:{self.cost})"
            if len(self.outcomes) > 0:
                output += f"(outcomes:{self.outcomes})"
            return output

    __repr__ = __str__


@dataclass
class Query:
    answers: List[Answer]
    probs: List[float]
    cost: float = 0.0
    name: str = ""

    def __str__(self):
        if len(self.name) > 0:
            return self.name
        else:
            if self.cost < 1e-7:
                return f"answers: {self.answers}"
            else:
                return f"answers: {self.answers} (cost:{self.cost})"

    __repr__ = __str__


@dataclass
class QResponse:
    answer: Optional[Answer]
    indx: int
    cost: float


def possible_queries(
    ami: AgentMechanismInterface,
    strategy: "EStrategy",
    user: "User",
    outcome: "Outcome" = None,
) -> List[Tuple[Outcome, List["UtilityDistribution"], float]]:
    """Gets all queries that could be asked for that outcome until an exact value of ufun is found.

    For each ask,  the following tuple is returned:
    (outcome, query, cost)

    """
    user = copy.deepcopy(user)
    strategy = copy.deepcopy(strategy)

    def _possible_queries(outcome, strategy=strategy, ami=ami):
        queries_before = user.elicited_queries()
        utility_before = strategy.utility_estimate(outcome)
        lower_before, upper_before = _loc(utility_before), _upper(utility_before)
        n_before = len(queries_before)
        while True:
            u, _ = strategy.apply(user=user, outcome=outcome)
            if isinstance(u, float):
                break
        _qs = user.elicited_queries()[n_before:]

        # update costs
        s = 0.0
        qs = []
        for i, q in enumerate(_qs):
            qs.append((outcome, q.query, q.cost + s - user.cost))
            s += q.cost

        # # add possible other answers
        # for old_indx, _ in enumerate(qs):
        #     if strategy.strategy == 'exact':
        #         qs[old_indx] = (_[0], [_[1]], _[3], _[4], _[5])
        #         continue
        #     others = []
        #     if (_[1] - lower_before) > epsilon:
        #         others.append(UtilityDistribution(dtype='uniform', loc=lower_before, scale=_[1] - lower_before))
        #     others.append(UtilityDistribution(dtype='uniform', loc=_[1], scale=_[2]) if _[2] > 0 else _[1])
        #     end = (_[1] + _[2])
        #     if (upper_before - end) > epsilon:
        #         others.append(UtilityDistribution(dtype='uniform', loc=end, scale=upper_before - end))
        #     qs[old_indx] = (_[0], others, _[3], _[4], _[5])
        return qs

    if outcome is None:
        queries = []
        for outcome in ami.outcomes:
            queries += _possible_queries(outcome)
    else:
        queries = _possible_queries(outcome)
    return queries


def next_query(
    strategy: "EStrategy", user: "User", outcome: "Outcome" = None
) -> List[Tuple[Outcome, Query, float]]:
    """Gets the possible outcomes for the next ask with its cost.

    The following tuple is returned:
    (outcome, query, cost)

    """

    def _next_query(outcome, strategy=strategy):
        return outcome, strategy.next_query(outcome), user.cost_of_asking()

    if outcome is None:
        queries = []
        for outcome in strategy.outcomes:
            queries.append(_next_query(outcome))
    else:
        queries = [_next_query(outcome)]
    return queries


class CostEvaluator:
    def __init__(self, cost: float):
        self.cost = cost

    def __call__(self, query: Query, answer: Answer):
        return self.cost + query.cost + (answer.cost if answer.cost else 0.0)
