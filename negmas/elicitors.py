""""The interface to all agents capable of eliciting user preferences before, during negotiations.
"""
import copy
import functools
import logging
import operator
import pprint
from abc import ABC, abstractmethod
from collections import defaultdict
from heapq import *
from math import sqrt
from typing import Union, Iterable, Callable, Tuple, List, Sequence
from typing import Dict
from negmas.inout import load_genius_domain_from_folder
from negmas import UncertainOpponentModel, GeniusNegotiator
from negmas.sao import *
from negmas.helpers import create_loggers
import math
import random
import time
from typing import Optional, Any
import pandas as pd
import numpy as np
import scipy.optimize as opt

try:
    from blist import sortedlist
except:
    print(
        "blist is not found. VOI based elicitation methods will not work. YOu can install"
        " blist by running:"
        ""
        ">> pip install blist"
        ""
        "or "
        ""
        ">> pip install negmas[elicitation]"
    )
from dataclasses import dataclass

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
from negmas.common import *
from negmas.negotiators import AspirationMixin
from negmas.modeling import DiscreteAcceptanceModel, AdaptiveDiscreteAcceptanceModel
from negmas.sao import AspirationNegotiator, SAONegotiator
from negmas.utilities import (
    IPUtilityFunction,
    UtilityFunction,
    UtilityDistribution,
    UtilityValue,
)
from negmas.utilities import MappingUtilityFunction
from negmas.outcomes import Outcome, ResponseType

np.seterr(all="raise")  # setting numpy to raise exceptions in case of errors


__all__ = [
    "BasePandoraElicitor",  # An agent capable of eliciting utilities through a proxy
    "EStrategy",  # A base class for objects representing the elcitation strategy
    "FullElicitor",
    "FullKnowledgeElicitor",
    "OptimisticElicitor",
    "PessimisticElicitor",
    "PandoraElicitor",
    "FastElicitor",
    "MeanElicitor",
    "DummyElicitor",
    "BalancedElicitor",
    "OptimalIncrementalElicitor",
    "RandomElicitor",
    "VOIElicitor",
    "VOIFastElicitor",
    "VOIOptimalElicitor",
    "VOINoUncertaintyElicitor",
    "possible_queries",
    "next_query",
    "Constraint",
    "RankConstraint",
    "RangeConstraint",
    "ComparisonConstraint",
    "User",
    "Query",
    "Answer",
    "SAOElicitingMechanism",
]


def _loc(u: UtilityValue):
    """Returns the lower bound of a UtilityValue"""
    return u if isinstance(u, float) else u.loc


def _locs(us: Iterable[UtilityValue]):
    """Returns the lower bound of an iterable of UtilityValue(s)"""
    return [u if isinstance(u, float) else u.loc for u in us]


def _scale(u: UtilityValue):
    """Returns the difference between the upper and lower bounds of a UtilityValue"""
    return 0.0 if isinstance(u, float) else u.scale


def _upper(u: UtilityValue):
    """Returns the upper bound of a UtilityValue"""
    return u if isinstance(u, float) else (u.loc + u.scale)


def _uppers(us: Iterable[UtilityValue]):
    """Returns the upper bounds of an Iterble of UtilityValues"""
    return [u if isinstance(u, float) else (u.loc + u.scale) for u in us]


def argmax(iterable: Iterable[Any]):
    """Returns the index of the maximum"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def argsort(iterable: Iterable[Any]):
    """Returns a list of indices that would sort the iterable"""
    return [_[0] for _ in sorted(enumerate(iterable), key=lambda x: x[1])]


def argmin(iterable):
    """Returns the index of the minimum"""
    return min(enumerate(iterable), key=lambda x: x[1])[0]


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
        ...

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
    def marginal(self, outcome: Outcome) -> UtilityDistribution:
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

    def marginal(self, outcome: Outcome) -> UtilityDistribution:
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

    def marginal(self, outcome: Outcome) -> UtilityDistribution:
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


class CostEvaluator:
    def __init__(self, cost: float):
        self.cost = cost

    def __call__(self, query: Query, answer: Answer):
        return self.cost + query.cost + (answer.cost if answer.cost else 0.0)


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


class EStrategy(object):
    """A proxy for a user that have some true utilities which can be elicited.

    Args:

        strategy: a string specifying the elicitation strategy or a callable.


    Remarks:

        - Supported string elicitation_strategies can be found using the `supported_strategies` class method
        - If a callable is passed then it must receive four `float` numbers indicating the lower and upper
          boundaries of the current Negotiator distribution, the true Negotiator and a threshold (resolution).
          It must return a new lower and upper values. To stop
          eliciting and return an exact number, the callable should set lower to the same value as upper

    """

    def __init__(
        self, strategy: str, resolution=1e-4, stop_at_cost: bool = True
    ) -> None:
        super().__init__()
        self.lower = None
        self.upper = None
        self.outcomes = None
        self.indices = None
        self.strategy = strategy
        self.resolution = resolution
        self.stop_at_cost = stop_at_cost

    @classmethod
    def supported_strategies(cls):
        return [
            "exact",
            "titration{f}",
            "titration-{f}",
            "dtitration{f}",
            "dtitration-{f}",
            "bisection",
            "pingpong-{f}",
            "pingpong{f}",
            "dpingpong-{f}",
            "dpingpong{f}",
        ]

    def apply(
        self, user: User, outcome: Outcome
    ) -> Tuple[Optional[UtilityValue], Optional[QResponse]]:
        """Do the elicitation and incur the cost.

        Remarks:

            - This function returns a uniform distribution whenever it returns a distribution
            - Can return `None` which indicates that elicitation failed
            - If it could find an exact value, it will return a `float` not a `UtilityDistribution`

        """

        lower, upper, outcomes = self.lower, self.upper, self.outcomes
        index = self.indices[outcome]
        lower, upper = lower[index], upper[index]
        epsilon = self.resolution

        if abs(upper - lower) < epsilon:
            return (upper + lower) / 2, None

        if self.stop_at_cost and abs(upper - lower) < 2 * user.cost:
            return (upper + lower) / 2, None

        reply = None
        query = self.next_query(outcome=outcome)
        if query is not None:
            reply = user.ask(query)
            if reply is None or reply.answer is None:
                return (
                    UtilityDistribution(
                        dtype="uniform", loc=lower, scale=upper - lower
                    ),
                    None,
                )
            lower_new, upper_new = (
                reply.answer.constraint.range[0],
                reply.answer.constraint.range[1],
            )
            if abs(upper_new - lower_new) >= abs(upper - lower):
                upper_new = lower_new = (upper_new + lower_new) / 2
            self.lower[index], self.upper[index] = lower_new, upper_new
            lower, upper = lower_new, upper_new
        if self.strategy == "exact":
            u = user.ufun(outcome)
        elif abs(upper - lower) < epsilon or query is None:
            u = (upper + lower) / 2
        else:
            u = UtilityDistribution(dtype="uniform", loc=lower, scale=upper - lower)
        return u, reply

    def next_query(self, outcome: Outcome) -> Optional[Query]:
        lower, upper, outcomes = self.lower, self.upper, self.outcomes
        index = self.indices[outcome]
        lower, upper = lower[index], upper[index]

        if abs(upper - lower) < self.resolution:
            return None

        if self.strategy is None:
            return None
        elif self.strategy == "exact":
            return None
        else:
            if self.strategy == "bisection":
                middle = 0.5 * (lower + upper)
                _range = upper - lower
                query = Query(
                    answers=[
                        Answer([outcome], RangeConstraint((lower, middle)), name="yes"),
                        Answer([outcome], RangeConstraint((middle, upper)), name=f"no"),
                    ],
                    probs=[0.5, 0.5],
                    name=f"{outcome}<{middle}",
                )
            elif "pingpong" in self.strategy:
                nstrt = len("pingpong") + (self.strategy.startswith("d"))
                step = (
                    float(self.strategy[nstrt:])
                    if len(self.strategy) > nstrt
                    else self.resolution
                )
                if self.strategy.startswith("dpingpong") and (upper - lower) < step:
                    step = min(step, self.resolution)
                if step == 0.0:
                    raise ValueError(f"Cannot do pingpong with a zero step")
                if abs(step) >= (upper - lower):
                    return None
                if not hasattr(self, "_pingpong_up"):
                    self._pingpong_up = False
                self._pingpong_up = not self._pingpong_up
                if self._pingpong_up:
                    lower_new = lower + step
                    _range = upper - lower
                    query = Query(
                        answers=[
                            Answer(
                                [outcome],
                                RangeConstraint((lower, lower_new)),
                                name="yes",
                            ),
                            Answer(
                                [outcome],
                                RangeConstraint((lower_new, upper)),
                                name="no",
                            ),
                        ],
                        probs=[step / _range, (upper - lower_new) / _range],
                        name=f"{outcome}<{lower_new}",
                    )
                    lower = lower_new
                else:
                    upper_new = upper - step
                    _range = upper - lower
                    query = Query(
                        answers=[
                            Answer(
                                [outcome],
                                RangeConstraint((lower, upper_new)),
                                name="no",
                            ),
                            Answer(
                                [outcome],
                                RangeConstraint((upper_new, upper)),
                                name="yes",
                            ),
                        ],
                        probs=[(upper_new - lower) / _range, step / _range],
                        name=f"{outcome}>{upper_new}",
                    )
                    upper = upper_new
            else:
                if "titration" in self.strategy:
                    nstrt = len("titration") + (self.strategy.startswith("d"))
                    try:
                        step = (
                            float(self.strategy[nstrt:])
                            if len(self.strategy) > nstrt
                            else self.resolution
                        )
                    except:
                        step = self.resolution

                    if "down" in self.strategy:
                        step = -abs(step)
                    elif "up" in self.strategy:
                        step = abs(step)
                    if (
                        self.strategy.startswith("dtitration")
                        and (upper - lower) < step
                    ):
                        step = min(self.resolution, step)
                    if step == 0.0:
                        raise ValueError(f"Cannot do titration with a zero step")
                    if abs(step) >= (upper - lower):
                        return None
                    up = step > 0.0
                    if up:
                        lower_new = lower + step
                        _range = upper - lower
                        query = Query(
                            answers=[
                                Answer(
                                    [outcome],
                                    RangeConstraint((lower, lower_new)),
                                    name="yes",
                                ),
                                Answer(
                                    [outcome],
                                    RangeConstraint((lower_new, upper)),
                                    name="no",
                                ),
                            ],
                            probs=[step / _range, (upper - lower_new) / _range],
                            name=f"{outcome}<{lower_new}",
                        )
                        lower = lower_new
                    else:
                        upper_new = upper + step
                        _range = upper - lower
                        query = Query(
                            answers=[
                                Answer(
                                    [outcome],
                                    RangeConstraint((lower, upper_new)),
                                    name="no",
                                ),
                                Answer(
                                    [outcome],
                                    RangeConstraint((upper_new, upper)),
                                    name="yes",
                                ),
                            ],
                            probs=[(upper_new - lower) / _range, -step / _range],
                            name=f"{outcome}>{upper_new}",
                        )
                        upper = upper_new
                else:
                    raise ValueError(f"Unknown elicitation strategy: {self.strategy}")

        return query

    def utility_estimate(self, outcome: Outcome) -> UtilityValue:
        """Gets a probability distribution of the Negotiator for this outcome without elicitation. Costs nothing"""
        indx = self.indices[outcome]
        scale = self.upper[indx] - self.lower[indx]
        if scale < self.resolution:
            return self.lower[indx]
        return UtilityDistribution(dtype="uniform", loc=self.lower[indx], scale=scale)

    def until(
        self,
        outcome: Outcome,
        user: User,
        dist: Union[List[UtilityValue], UtilityValue],
    ) -> UtilityValue:
        if isinstance(dist, list):
            targets = [
                (_ - self.resolution, _ + self.resolution)
                if isinstance(_, float)
                else (_.loc, _.loc + _.scale)
                for _ in dist
            ]
        else:
            targets = (
                [(dist - self.resolution, dist + self.resolution)]
                if isinstance(dist, float)
                else [(dist.loc, dist.loc + dist.scale)]
            )

        u = self.utility_estimate(outcome)

        def within_a_target(u, targets=targets):
            for lower, upper in targets:
                if (_loc(u) >= (lower - self.resolution)) and (
                    (_upper(u)) <= upper + self.resolution
                ):
                    return True
            return False

        while not within_a_target(u):
            u, _ = self.apply(user=user, outcome=outcome)
            if isinstance(u, float):
                break
        return u

    def on_enter(
        self, ami: AgentMechanismInterface, ufun: IPUtilityFunction = None
    ) -> None:
        self.lower = [0.0] * ami.n_outcomes
        self.upper = [1.0] * ami.n_outcomes
        self.indices = dict(zip(ami.outcomes, range(ami.n_outcomes)))
        if ufun is not None:
            distributions = list(ufun.distributions.values())
            for i, dist in enumerate(distributions):
                self.lower[i] = _loc(dist)
                self.upper[i] = _upper(dist)
        self.outcomes = ami.outcomes
        self._total_cost = 0.0
        self._elicited_queries = []


def possible_queries(
    ami: AgentMechanismInterface,
    strategy: EStrategy,
    user: User,
    outcome: Outcome = None,
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
    strategy: EStrategy, user: User, outcome: Outcome = None
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


class Expector(ABC):
    def __init__(self, ami: Optional[AgentMechanismInterface] = None):
        self.ami = ami

    @abstractmethod
    def is_dependent_on_negotiation_info(self) -> bool:
        ...

    @abstractmethod
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        ...


class StaticExpector(Expector):
    def is_dependent_on_negotiation_info(self) -> bool:
        return False

    @abstractmethod
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        ...


class MeanExpector(StaticExpector):
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else float(u)


class MaxExpector(StaticExpector):
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else u.loc + u.scale


class MinExpector(StaticExpector):
    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        return u if isinstance(u, float) else u.loc


class BalancedExpector(Expector):
    def is_dependent_on_negotiation_info(self) -> bool:
        return True

    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        if state is None:
            state = self.ami.state
        if isinstance(u, float):
            return u
        else:
            return state.relative_time * u.loc + (1.0 - state.relative_time) * (
                u.loc + u.scale
            )


class AspiringExpector(Expector, AspirationMixin):
    def __init__(
        self,
        ami: Optional[AgentMechanismInterface] = None,
        max_aspiration=1.0,
        aspiration_type: Union[str, int, float] = "linear",
    ) -> bool:
        Expector.__init__(self, ami=ami)
        self.aspiration_init(
            max_aspiration=max_aspiration,
            aspiration_type=aspiration_type,
            above_reserved_value=False,
        )

    def is_dependent_on_negotiation_info(self) -> bool:
        return True

    def __call__(self, u: UtilityValue, state: MechanismState = None) -> float:
        if state is None:
            state = self.ami.state
        if isinstance(u, float):
            return u
        else:
            alpha = self.aspiration(state.relative_time)
            return alpha * u.loc + (1.0 - alpha) * (u.loc + u.scale)


class BaseElicitor(SAONegotiator):
    def __init__(
        self,
        user: User,
        *,
        strategy: Optional[EStrategy] = None,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        expector_factory: Union[Expector, Callable[[], Expector]] = MeanExpector,
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        true_utility_on_zero_cost=False,
    ) -> None:
        """

        Args:
            user:
            strategy:
            base_negotiator:
            opponent_model_factory:
            expector_factory:
            single_elicitation_per_round:
            continue_eliciting_past_reserved_val:
            epsilon:
            true_utility_on_zero_cost:
        """
        super().__init__()
        self.add_capabilities(
            {
                "propose": True,
                "respond": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self.strategy = strategy
        self.opponent_model_factory = opponent_model_factory
        self.expector_factory = expector_factory
        self.single_elicitation = single_elicitation_per_round
        self.continue_eliciting_past_reserved_val = continue_eliciting_past_reserved_val
        self.epsilon = epsilon
        self.true_utility_on_zero_cost = true_utility_on_zero_cost
        self.elicitation_history = []
        self.opponent_model = None
        self._elicitation_time = None
        self.asking_time = 0.0
        self.offerable_outcomes = (
            []
        )  # will contain outcomes with known or at least elicited utilities
        self.indices = None
        self.initial_utility_priors = None
        self.user = user
        self.acc_limit = self.accuracy_limit(self.user.cost_of_asking())
        self.base_negotiator = base_negotiator
        self.expect = None
        if strategy is not None:
            strategy.resolution = max(self.acc_limit, strategy.resolution)

    def join(
        self,
        ami: AgentMechanismInterface,
        state: MechanismState,
        *,
        ufun: Optional["UtilityFunction"] = None,
        role: str = "agent",
        **kwargs,
    ) -> bool:
        if ufun is None:
            ufun = IPUtilityFunction(outcomes=ami.outcomes, reserved_value=0.0)
        if not super().join(ami=ami, state=state, ufun=ufun, role=role):
            return False
        self.expect = self.expector_factory(self._ami)
        self.init_elicitation(ufun=ufun, **kwargs)
        self.base_negotiator.join(
            ami,
            state,
            ufun=MappingUtilityFunction(
                mapping=lambda x: self.expect(self.utility_function(x), state=state),
                reserved_value=self.reserved_value,
            ),
        )
        return True

    def on_negotiation_start(self, state: MechanismState):
        self.base_negotiator.on_negotiation_start(state=state)

    def utility_distributions(self):
        if self.utility_function is None:
            return [None] * len(self._ami.outcomes)
        if self.utility_function.base_type == "ip":
            return list(self.utility_function.distributions.values())
        else:
            return [self.utility_function(o) for o in self._ami.outcomes]

    @property
    def ufun(self):
        return (
            lambda x: self.user.ufun(x) - self.user.total_cost
            if x is not None
            else self.user.ufun(x)
        )

    @property
    def elicitation_cost(self):
        return self.user.total_cost

    @property
    def elicitation_time(self):
        return self._elicitation_time

    def maximum_attainable_utility(self):
        return max(_uppers(self.utility_distributions()))

    def minimum_guaranteed_utility(self):
        return min(_locs(self.utility_distributions()))

    def on_partner_proposal(
        self, state: MechanismState, agent_id: str, offer: "Outcome"
    ):
        self.base_negotiator.on_partner_proposal(
            agent_id=agent_id, offer=offer, state=state
        )
        old_prob = self.opponent_model.probability_of_acceptance(offer)
        self.opponent_model.update_offered(offer)
        new_prob = self.opponent_model.probability_of_acceptance(offer)
        self.on_opponent_model_updated([offer], old=[old_prob], new=[new_prob])
        return

    def on_partner_response(
        self,
        state: MechanismState,
        agent_id: str,
        outcome: Outcome,
        response: "ResponseType",
    ):
        self.base_negotiator.on_partner_response(
            state=state, agent_id=agent_id, outcome=outcome, response=response
        )
        if response == ResponseType.REJECT_OFFER:
            old_probs = [self.opponent_model.probability_of_acceptance(outcome)]
            self.opponent_model.update_rejected(outcome)
            new_probs = [self.opponent_model.probability_of_acceptance(outcome)]
            self.on_opponent_model_updated([outcome], old=old_probs, new=new_probs)
        elif response == ResponseType.ACCEPT_OFFER:
            old_probs = [self.opponent_model.probability_of_acceptance(outcome)]
            self.opponent_model.update_accepted(outcome)
            new_probs = [self.opponent_model.probability_of_acceptance(outcome)]
            self.on_opponent_model_updated([outcome], old=old_probs, new=new_probs)

    def respond_(self, state: MechanismState, offer: Outcome) -> ResponseType:
        ami = self._ami
        my_offer, meu = self.best_offer(state=state)
        if my_offer is None:
            return self.base_negotiator.respond_(state=state, offer=offer)
        if self.offerable_outcomes is not None and offer not in self.offerable_outcomes:
            self.strategy.apply(user=self.user, outcome=offer)
        offered_utility = self.utility_function(offer)
        if offered_utility is None:
            return self.base_negotiator.respond_(state=state, offer=offer)
        offered_utility = self.expect(offered_utility, state=state)
        if (
            self.maximum_attainable_utility() - self.user.total_cost
            < self.reserved_value
        ):
            return ResponseType.END_NEGOTIATION
        if meu < offered_utility:
            return ResponseType.ACCEPT_OFFER
        else:
            return self.base_negotiator.respond_(state=state, offer=offer)

    def propose(self, state: MechanismState) -> Outcome:
        if self.can_elicit():
            self.elicit(state=state)
        return self.base_negotiator.propose(state=state)

    def elicit(self, state: MechanismState) -> None:
        if (
            self.maximum_attainable_utility() - self.elicitation_cost
            <= self.reserved_value
        ):
            return
        start = time.perf_counter()
        self.before_eliciting()
        if self.single_elicitation:
            self.elicit_single(state=state)
        else:
            while self.elicit_single(state=state):
                if (
                    self.maximum_attainable_utility() - self.elicitation_cost
                    <= self.reserved_value
                    or state.relative_time >= 1
                ):
                    break
        elapsed = time.perf_counter() - start
        self._elicitation_time += elapsed
        self.asking_time += elapsed

    def accuracy_limit(self, cost: float):
        return 0.5 * max(self.epsilon, cost)

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        **kwargs,
    ) -> None:
        ami = self._ami
        self.elicitation_history = []
        self.indices = dict(zip(ami.outcomes, range(ami.n_outcomes)))
        self.offerable_outcomes = []
        self._elicitation_time = 0.0
        if self.opponent_model_factory is None:
            self.opponent_model = None
        else:
            self.opponent_model = self.opponent_model_factory(ami)
            self.base_negotiator.opponent_model = self.opponent_model
        outcomes = ami.outcomes
        if ufun is None:
            ufun = [
                UtilityDistribution(dtype="uniform", loc=0.0, scale=1.0)
                for _ in outcomes
            ]
            ufun = IPUtilityFunction(
                outcomes=outcomes, distributions=ufun, reserved_value=0.0
            )
        elif isinstance(ufun, UtilityDistribution):
            ufun = [copy.copy(ufun) for _ in outcomes]
            ufun = IPUtilityFunction(
                outcomes=outcomes, distributions=ufun, reserved_value=0.0
            )
        elif (
            isinstance(ufun, list)
            and len(ufun) > 0
            and isinstance(ufun[0], UtilityDistribution)
        ):
            ufun = IPUtilityFunction(
                outcomes=outcomes, distributions=ufun, reserved_value=0.0
            )
        self.utility_function = ufun
        self.initial_utility_priors = copy.copy(ufun)

    def offering_utility(self, outcome, state) -> UtilityValue:
        if self.opponent_model is None:
            return self.utility_function(outcome)
        u = self.utility_function(outcome)
        p = self.opponent_model.probability_of_acceptance(outcome)
        return p * u + (1 - p) * self.utility_on_rejection(outcome, state=state)

    def offering_utilities(self, state) -> np.ndarray:
        us = np.asarray(self.utility_distributions())
        ps = np.asarray(self.opponent_model.acceptance_probabilities())
        return ps * us + (1 - ps) * np.asarray(self.utilities_on_rejection(state=state))

    def utility_on_acceptance(self, outcome: Outcome) -> UtilityValue:
        return self.utility_function(outcome)

    def best_offer(self, state) -> Tuple[Optional[Outcome], float]:
        """Maximum Expected Utility at a given aspiration level (alpha)

        Args:
            state:
        """
        if len(self.offerable_outcomes) == 0:
            self.elicit(state=state)
        if len(self.offerable_outcomes) == 0:
            return None, self.reserved_value
        best, best_utility, bsf = None, self.reserved_value, self.reserved_value
        for i, outcome in enumerate(self.offerable_outcomes):
            if outcome is None:
                continue
            utilitiy = self.offering_utility(outcome, state=state)
            expected_utility = self.expect(utilitiy, state=state)
            if expected_utility >= bsf:
                best, best_utility, bsf = outcome, utilitiy, expected_utility
        return best, self.expect(best_utility, state=state)

    def utility_on_rejection(
        self, outcome: Outcome, state: MechanismState
    ) -> UtilityValue:
        """Expected Negotiator if this outcome is given and rejected

        Args:
            outcome
            state:
        """
        raise NotImplementedError(
            f"Must override utility_on_rejection in {self.__class__.__name__}"
        )

    def utilities_on_rejection(self, state: MechanismState) -> List[UtilityValue]:
        return [
            self.utility_on_rejection(outcome=outcome, state=state)
            for outcome in self._ami.outcomes
        ]

    def on_opponent_model_updated(
        self, outcomes: List[Outcome], old: List[float], new: List[float]
    ) -> None:
        """Called whenever an opponents model is updated."""
        pass

    def __str__(self):
        return f"{self.name}"

    def before_eliciting(self) -> None:
        """Called by apply just before continuously calling elicit_single"""
        pass

    def can_elicit(self) -> bool:
        """Wheather we have something to apply yet"""
        raise NotImplementedError()

    def elicit_single(self, state: MechanismState):
        """Does a single elicitation act

        Args:
            state:
        """
        raise NotImplementedError()

    def __getattr__(self, item):
        return getattr(self.base_negotiator, item)


class DummyElicitor(BaseElicitor):
    def utility_on_rejection(
        self, outcome: Outcome, state: MechanismState
    ) -> UtilityValue:
        return self.reserved_value

    def can_elicit(self) -> bool:
        return True

    def elicit_single(self, state: MechanismState):
        return False

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        **kwargs,
    ):
        super().init_elicitation(ufun=ufun, **kwargs)
        strt_time = time.perf_counter()
        self.offerable_outcomes = self._ami.outcomes
        self._elicitation_time += time.perf_counter() - strt_time


class FullKnowledgeElicitor(BaseElicitor):
    def utility_on_rejection(
        self, outcome: Outcome, state: MechanismState
    ) -> UtilityValue:
        return self.reserved_value

    def can_elicit(self) -> bool:
        return True

    def elicit_single(self, state: MechanismState):
        return False

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        **kwargs,
    ):
        super().init_elicitation(ufun=self.user.ufun)
        strt_time = time.perf_counter()
        self.offerable_outcomes = self._ami.outcomes
        self._elicitation_time += time.perf_counter() - strt_time


class BasePandoraElicitor(BaseElicitor, AspirationMixin):
    def __init__(
        self,
        user: User,
        strategy: EStrategy,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        expector_factory: Union[Expector, Callable[[], Expector]] = MeanExpector,
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        true_utility_on_zero_cost=False,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
        incremental=True,
        max_aspiration=0.99,
        aspiration_type="boulware",
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            opponent_model_factory=opponent_model_factory,
            expector_factory=expector_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            base_negotiator=base_negotiator,
        )
        self.aspiration_init(
            max_aspiration=max_aspiration,
            aspiration_type=aspiration_type,
            above_reserved_value=True,
        )
        self.add_capabilities(
            {
                "propose": True,
                "respond": True,
                "propose-with-value": False,
                "max-proposals": None,  # indicates infinity
            }
        )
        self.my_last_proposals: Optional[Outcome] = None
        self.deep_elicitation = deep_elicitation
        self.elicitation_history = []
        self.cutoff_utility = None
        self.opponent_model = None
        self._elicitation_time = None
        self.offerable_outcomes = (
            []
        )  # will contain outcomes with known or at least elicited utilities
        self.cutoff_utility = None
        self.unknown = None
        self.assume_uniform = assume_uniform
        self.user_model_in_index = user_model_in_index
        self.precalculated_index = precalculated_index
        self.incremental = incremental

    def utility_on_rejection(
        self, outcome: Outcome, state: MechanismState
    ) -> UtilityValue:
        return self.aspiration(state.relative_time)

    def update_cutoff_utility(self) -> None:
        self.cutoff_utility = self.reserved_value
        expected_utilities = [
            float(self.ufun(outcome)) for outcome in self.offerable_outcomes
        ]
        if len(expected_utilities) > 0:
            self.cutoff_utility = max(expected_utilities)

    def elicit_single(self, state: MechanismState):
        z, best_index = self.offer_to_elicit()
        if z < self.cutoff_utility:
            return False
        if best_index is None:
            return self.continue_eliciting_past_reserved_val
        outcome = self._ami.outcomes[best_index]
        u = self.do_elicit(outcome, None)
        self.utility_function.distributions[outcome] = u
        expected_value = self.offering_utility(outcome, state=state)
        self.offerable_outcomes.append(outcome)
        if isinstance(u, float):
            self.remove_best_offer_from_unknown_list()
        else:
            self.update_best_offer_utility(outcome, u)
        self.cutoff_utility = max(
            (self.cutoff_utility, self.expect(expected_value, state=state))
        )
        self.elicitation_history.append((outcome, u, state.step))
        return True

    def do_elicit(self, outcome: Outcome, state: MechanismState) -> UtilityValue:
        # @todo replace input with negotiation instead of ami
        if not self.deep_elicitation:
            return self.strategy.apply(user=self.user, outcome=outcome)[0]

        while True:
            u, _ = self.strategy.apply(user=self.user, outcome=outcome)
            if isinstance(u, float):
                break
            if u.scale < self.acc_limit:
                u = float(u)
                break

        # we do that after the normal elicitation so that the elicitation time
        # is recorded correctly
        if self.user.cost_of_asking() == 0.0 and self.true_utility_on_zero_cost:
            return self.user.ufun(outcome)
        return u

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        **kwargs,
    ):
        super().init_elicitation(ufun=ufun, **kwargs)
        strt_time = time.perf_counter()
        self.cutoff_utility = self.reserved_value
        self.unknown = None  # needed as init_unknowns uses unknown
        self.init_unknowns()
        self._elicitation_time += time.perf_counter() - strt_time

    def before_eliciting(self):
        self.update_cutoff_utility()

    def offer_to_elicit(self) -> Tuple[float, Optional["int"]]:
        unknowns = self.unknown
        if len(unknowns) > 0:
            return -unknowns[0][0], unknowns[0][1]
        return 0.0, None

    def update_best_offer_utility(self, outcome: Outcome, u: UtilityValue):
        self.unknown[0] = (
            -weitzman_index_uniform(_loc(u), _scale(u), self.user.cost_of_asking()),
            self.unknown[0][1],
        )
        heapify(self.unknown)

    def init_unknowns(self):
        self.unknown = self.z_index(updated_outcomes=None)

    def remove_best_offer_from_unknown_list(self) -> Tuple[int, float]:
        return heappop(self.unknown)

    def can_elicit(self) -> bool:
        return len(self.unknown) != 0

    def z_index(self, updated_outcomes: Optional[List[Outcome]] = None):
        """Update the internal z-index or create it if needed.
        """
        n_outcomes = self._ami.n_outcomes
        outcomes = self._ami.outcomes
        unknown = self.unknown
        if unknown is None:
            unknown = [(-self.reserved_value, _) for _ in range(n_outcomes)]
        else:
            unknown = [_ for _ in unknown if _[1] is not None]

        xw = list(self.utility_function.distributions.values())
        if len(unknown) == 0:
            return
        unknown_indices = [_[1] for _ in unknown if _[1] is not None]
        if updated_outcomes is None:
            updated_outcomes = unknown_indices
        else:
            updated_outcomes = [self.indices[_] for _ in updated_outcomes]
            updated_outcomes = list(set(unknown_indices).intersection(updated_outcomes))
        if len(updated_outcomes) == 0:
            return unknown
        z = unknown
        if self.assume_uniform:
            for j, (u, i) in enumerate(unknown):
                if i is None or i not in updated_outcomes:
                    continue
                loc = xw[i].loc if not isinstance(xw[i], float) else xw[i]
                scale = xw[i].scale if not isinstance(xw[i], float) else 0.0
                if self.user_model_in_index:
                    p = self.opponent_model.probability_of_acceptance(outcomes[i])
                    current_loc = loc
                    loc = p * loc + (1 - p) * self.reserved_value
                    scale = (
                        p * (current_loc + scale) + (1 - p) * self.reserved_value - loc
                    )
                cost = self.user.cost_of_asking()
                z[j] = (-weitzman_index_uniform(loc, scale, cost=cost), i)
        else:

            def qualityfun(z, distribution, cost):
                c_estimate = distribution.expect(lambda x: x - z, lb=z, ub=1.0)
                if self.user_model_in_index:
                    p = self.opponent_model.probability_of_acceptance(outcomes[i])
                    c_estimate = p * c_estimate + (1 - p) * self.reserved_value
                return sqrt(c_estimate - cost)

            for j, (u, i) in enumerate(unknown):
                if i is None or i not in updated_outcomes:
                    continue
                cost = self.user.cost_of_asking()
                f = functools.partial(qualityfun, distribution=xw[i], cost=cost)
                z[j] = (
                    -opt.minimize(
                        f, x0=np.asarray([u]), bounds=[(0.0, 1.0)], method="L-BFGS-B"
                    ).x[0],
                    i,
                )
                # we always push the reserved value for the outcome None representing breaking
        heapify(z)
        return z

    def on_opponent_model_updated(
        self, outcomes: List[Outcome], old: List[float], new: List[float]
    ) -> None:
        """Callback when an opponents model is updated"""
        if not self.precalculated_index:
            self.unknown = self.z_index(
                updated_outcomes=outcomes if self.incremental else None
            )


###########################################################################
#     BASELINE ELICITORS                                                  #
###########################################################################


class FullElicitor(BasePandoraElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        epsilon=0.001,
        true_utility_on_zero_cost=False,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=True,
            epsilon=epsilon,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            base_negotiator=base_negotiator,
        )
        self.elicited = {}

    def update_best_offer_utility(self, outcome: Outcome, u: UtilityValue):
        pass

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        **kwargs,
    ):
        super().init_elicitation(ufun=ufun)
        strt_time = time.perf_counter()
        self.elicited = False
        self._elicitation_time += time.perf_counter() - strt_time

    def elicit(self, state: MechanismState):
        if not self.elicited:
            outcomes = self._ami.outcomes
            utilities = [
                self.expect(self.do_elicit(outcome, None), state=state)
                for outcome in self._ami.outcomes
            ]
            self.offerable_outcomes = list(outcomes)
            self.elicitation_history = [zip(outcomes, utilities)]
            self.elicited = True

    def init_unknowns(self) -> List[Tuple[float, int]]:
        self.unknown = []


class RandomElicitor(BasePandoraElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        deep_elicitation=True,
        true_utility_on_zero_cost=False,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        single_elicitation_per_round=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            epsilon=0.001,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            opponent_model_factory=opponent_model_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            base_negotiator=base_negotiator,
        )

    def init_unknowns(self) -> None:
        n = self._ami.n_outcomes
        z: List[Tuple[float, Optional[int]]] = list(
            zip((-random.random() for _ in range(n + 1)), range(n + 1))
        )
        z[-1] = (z[-1][0], None)
        heapify(z)
        self.unknown = z

    def update_best_offer_utility(self, outcome: Outcome, u: UtilityValue):
        pass


def weitzman_index_uniform(
    loc: float, scale: float, cost: float, time_discount: float = 1.0
):
    """Implements Weitzman's 1979 Bandora's Box index calculation."""
    # assume zi < l
    end = loc + scale
    z = time_discount * (loc + end) / 2.0 - cost

    b = -2 * (end + scale * (1.0 - time_discount) / time_discount)
    c = end * end - 2 * scale * cost / time_discount

    d = b * b - 4 * c
    if d < 0:
        z1 = z2 = -1.0
    else:
        d = sqrt(d)
        z1 = (d - b) / 2.0
        z2 = -(d + b) / 2.0

    if z <= loc and not loc < z1 <= end and not loc < z2 <= end:
        return z
    if z > loc and loc < z1 <= end and not loc < z2 <= end:
        return z1
    if z > loc and not loc < z1 <= end and loc < z2 <= end:
        return z2

    if z <= loc or (z - loc) < 1e-5:
        return z
    elif loc < z1 <= end:
        return z1
    elif loc < z2 <= end:
        return z2
    for _ in (z1, z2):
        if abs(_ - loc) < 1e-5 or abs(_ - end) < 1e-3:
            return _
    print(
        "No solutions are found for (l={}, s={}, c={}, time_discount={}) [{}, {}, {}]".format(
            loc, scale, cost, time_discount, z, z1, z2
        )
    )
    return 0.0


class PandoraElicitor(BasePandoraElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool = True,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        expector_factory: Union[Expector, Callable[[], Expector]] = MeanExpector,
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
        incremental=True,
        true_utility_on_zero_cost=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            opponent_model_factory=opponent_model_factory,
            expector_factory=expector_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            assume_uniform=assume_uniform,
            user_model_in_index=user_model_in_index,
            precalculated_index=precalculated_index,
            incremental=incremental,
            base_negotiator=base_negotiator,
        )


class FastElicitor(PandoraElicitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_elicitation = False

    def update_best_offer_utility(self, outcome: Outcome, u: UtilityValue):
        """We need not do anything here as we will remove the outcome anyway to the known list"""
        pass

    def do_elicit(self, outcome: Outcome, state: MechanismState):
        return self.expect(super().do_elicit(outcome, None), state=state)


class OptimalIncrementalElicitor(PandoraElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool,
        expector_factory: Union[Expector, Callable[[], Expector]] = MeanExpector,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            opponent_model_factory=opponent_model_factory,
            expector_factory=expector_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            base_negotiator=base_negotiator,
            assume_uniform=assume_uniform,
            user_model_in_index=user_model_in_index,
            precalculated_index=precalculated_index,
            incremental=True,
        )


class MeanElicitor(OptimalIncrementalElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool = False,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            opponent_model_factory=opponent_model_factory,
            expector_factory=MeanExpector,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            base_negotiator=base_negotiator,
            assume_uniform=assume_uniform,
            user_model_in_index=user_model_in_index,
            precalculated_index=precalculated_index,
        )


class BalancedElicitor(OptimalIncrementalElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool = False,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            opponent_model_factory=opponent_model_factory,
            expector_factory=BalancedExpector,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            base_negotiator=base_negotiator,
            assume_uniform=assume_uniform,
            user_model_in_index=user_model_in_index,
            precalculated_index=precalculated_index,
        )


class AspiringElicitor(OptimalIncrementalElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        max_aspiration: float = 1.0,
        aspiration_type: Union[float, str] = "linear",
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool = False,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            opponent_model_factory=opponent_model_factory,
            expector_factory=lambda: AspiringExpector(
                max_aspiration=max_aspiration,
                aspiration_type=aspiration_type,
                ami=self._ami,
            ),
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            base_negotiator=base_negotiator,
            assume_uniform=assume_uniform,
            user_model_in_index=user_model_in_index,
            precalculated_index=precalculated_index,
        )


class PessimisticElicitor(OptimalIncrementalElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool = False,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            opponent_model_factory=opponent_model_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            base_negotiator=base_negotiator,
            assume_uniform=assume_uniform,
            user_model_in_index=user_model_in_index,
            precalculated_index=precalculated_index,
            expector_factory=MinExpector,
        )


class OptimisticElicitor(OptimalIncrementalElicitor):
    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        deep_elicitation: bool = False,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        assume_uniform=True,
        user_model_in_index=True,
        precalculated_index=False,
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            deep_elicitation=deep_elicitation,
            opponent_model_factory=opponent_model_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            base_negotiator=base_negotiator,
            assume_uniform=assume_uniform,
            user_model_in_index=user_model_in_index,
            precalculated_index=precalculated_index,
            expector_factory=MaxExpector,
        )


class BaseVOIElicitor(BaseElicitor):
    """Base class for all VOI elicitation algorithms
    """

    def __init__(
        self,
        strategy: EStrategy,
        user: User,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        dynamic_query_set=False,
        queries=None,
        adaptive_answer_probabilities=True,
        expector_factory: Union[Expector, Callable[[], Expector]] = MeanExpector,
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        true_utility_on_zero_cost=False,
        each_outcome_once=False,
        update_related_queries=True,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
    ) -> None:
        super().__init__(
            strategy=strategy,
            user=user,
            opponent_model_factory=opponent_model_factory,
            expector_factory=expector_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            base_negotiator=base_negotiator,
        )
        # todo confirm that I need this. aspiration mixin. I think I do not.
        self.aspiration_init(max_aspiration=1.0, aspiration_type="boulware")
        self.eu_policy = None
        self.eeu_query = None
        self.query_index_of_outcome = None
        self.dynamic_query_set = dynamic_query_set
        self.adaptive_answer_probabilities = adaptive_answer_probabilities
        self.current_eeu = None
        self.eus = None
        self.queries = queries if queries is not None else []
        self.outcome_in_policy = None
        self.each_outcome_once = each_outcome_once
        self.queries_of_outcome = None
        self.update_related_queries = update_related_queries
        self.total_voi = 0.0

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        queries: Optional[List[Query]] = None,
    ) -> None:
        super().init_elicitation(ufun=ufun)
        strt_time = time.perf_counter()
        ami = self._ami
        self.eus = np.array([_.mean() for _ in self.utility_distributions()])
        self.offerable_outcomes = ami.outcomes
        if self.dynamic_query_set and not isinstance(self.strategy, EStrategy):
            raise ValueError("The strategy must be a EStrategy for VOIElicitor")
        if not self.dynamic_query_set and self.strategy is not None:
            raise ValueError(
                "If you are not using a dynamic query set, then you cannot pass a strategy. It will not be used"
            )
        if not self.dynamic_query_set and self.queries is None and queries is None:
            raise ValueError(
                "If you are not using a dynamic query set then you must pass a set of queries"
            )
        if self.dynamic_query_set and queries is not None:
            raise ValueError(
                "You cannot pass a set of queries if you use dynamic ask sets"
            )
        if not self.dynamic_query_set and queries is not None:
            self.queries += queries
        self.init_optimal_policy()
        if self.dynamic_query_set:
            self.queries = [
                (outcome, self.strategy.next_query(outcome), 0.0)
                for outcome in ami.outcomes
            ]
        else:
            if self.update_related_queries:
                queries_of_outcome = defaultdict(list)
                for i, (_o, _q, _c) in enumerate(self.queries):
                    queries_of_outcome[_o].append(i)
                self.queries_of_outcome = queries_of_outcome
            pass
        self.init_query_eeus()
        self._elicitation_time += time.perf_counter() - strt_time

    def best_offer(self, state: MechanismState) -> Tuple[Optional[Outcome], float]:
        """Maximum Expected Utility at a given aspiration level (alpha)

        Args:
            state:
        """
        if self.each_outcome_once:
            # todo this needs correction. When I opp from the eu_policy, all eeu_query become wrong
            if len(self.eu_policy) < 1:
                self.init_optimal_policy()
            _, outcome_index = self.eu_policy.pop()
        else:
            outcome_index = self.eu_policy[0][1]
        if self.eus[outcome_index] < self.reserved_value:
            return None, self.reserved_value
        return (
            self._ami.outcomes[outcome_index],
            self.expect(
                self.utility_function(self._ami.outcomes[outcome_index]), state=state
            ),
        )

    def can_elicit(self) -> bool:
        return True

    def best_offers(self, n: int) -> List[Tuple[Optional[Outcome], float]]:
        """Maximum Expected Utility at a given aspiration level (alpha)"""
        return [self.best_offer()] * n

    def before_eliciting(self):
        pass

    def on_opponent_model_updated(
        self, outcomes: List[Outcome], old: List[float], new: List[float]
    ) -> None:
        if any(o != n for o, n in zip(old, new)):
            self.init_optimal_policy()
            self.init_query_eeus()

    def update_optimal_policy(
        self, index: int, outcome: Outcome, oldu: float, newu: float
    ):
        """Updates the optimal policy after a change happens to some utility"""
        if oldu != newu:
            self.init_optimal_policy()

    def elicit_single(self, state: MechanismState):
        if self.eeu_query is not None and len(self.eeu_query) < 1:
            return False
        if not self.can_elicit():
            return False
        eeu, q = heappop(self.eeu_query)
        if q is None or -eeu <= self.current_eeu:
            return False
        if (not self.continue_eliciting_past_reserved_val) and (
            -eeu - (self.user.cost_of_asking() + self.elicitation_cost)
            < self.reserved_value
        ):
            return False
        outcome, query, cost = self.queries[q]
        if query is None:
            return False
        self.queries[q] = (None, None, None)
        oldu = self.utility_function.distributions[outcome]
        if _scale(oldu) < 1e-7:
            return False
        if self.dynamic_query_set:
            newu, u = self.strategy.apply(user=self.user, outcome=outcome)
        else:
            u = self.user.ask(query)
            newu = u.answer.constraint.marginal(outcome)
            if self.queries_of_outcome is not None:
                if _scale(newu) > 1e-7:
                    newu = newu & oldu
                    newmin, newmax = newu.loc, newu.scale + newu.loc
                    good_queries = []
                    for i, qind in enumerate(self.queries_of_outcome.get(outcome, [])):
                        _o, _q, _c = self.queries[qind]
                        if _q is None:
                            continue
                        answers = _q.answers
                        tokeep = []
                        for j, ans in enumerate(answers):
                            rng = ans.constraint.range
                            if newmin == rng[0] and newmax == rng[1]:
                                continue
                            if newmin <= rng[0] <= newmax or rng[0] <= newmin <= rng[1]:
                                tokeep.append(j)
                        if len(tokeep) < 2:
                            self.queries[i] = None, None, None
                            continue
                        good_queries.append(qind)
                        if len(tokeep) < len(answers):
                            ans = _q.answers
                            self.queries[i].answers = [ans[j] for j in tokeep]
                    self.queries_of_outcome[outcome] = good_queries
                else:
                    for i, _ in enumerate(self.queries_of_outcome.get(outcome, [])):
                        self.queries[i] = None, None, None
                        self.queries_of_outcome[outcome] = []
        self.total_voi += -eeu - self.current_eeu
        outcome_index = self.indices[outcome]
        if _scale(newu) < 1e-7:
            self.utility_function.distributions[outcome] = newu
        else:
            self.utility_function.distributions[outcome] = newu & oldu
        eu = float(newu)
        self.eus[outcome_index] = eu
        self.update_optimal_policy(
            index=outcome_index, outcome=outcome, oldu=float(oldu), newu=eu
        )
        if self.dynamic_query_set:
            o, q, c = outcome, self.strategy.next_query(outcome), 0.0
            if not (o is None or q is None):
                self.queries.append((o, q, c))
                qeeu = self._query_eeu(
                    query,
                    len(self.queries) - 1,
                    outcome,
                    cost,
                    outcome_index,
                    self.eu_policy,
                    self.current_eeu,
                )
                self.add_query((qeeu, len(self.queries) - 1))
        self.init_query_eeus()
        self.elicitation_history.append((query, newu, state.step, self.current_eeu))
        return True

    def init_query_eeus(self) -> None:
        """Updates the heap eeu_query which has records of (-EEU, quesion)"""
        queries = self.queries
        eu_policy, eeu = self.eu_policy, self.current_eeu
        eeu_query = []
        for qindex, current in enumerate(queries):
            outcome, query, cost = current
            if query is None or outcome is None:
                continue
            outcome_index = self.indices[outcome]
            qeeu = self._query_eeu(
                query, qindex, outcome, cost, outcome_index, eu_policy, eeu
            )
            eeu_query.append((qeeu, qindex))
        heapify(eeu_query)
        self.eeu_query = eeu_query

    def utility_on_rejection(
        self, outcome: Outcome, state: MechanismState
    ) -> UtilityValue:
        raise ValueError("utility_on_rejection should never be called on VOI Elicitors")

    def add_query(self, qeeu: Tuple[float, int]) -> None:
        heappush(self.eeu_query, qeeu)

    @abstractmethod
    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors. The optimal plicy should be sorted ascendingly
        on -EU or -EU * Acceptance"""

    @abstractmethod
    def _query_eeu(
        self, query, qindex, outcome, cost, outcome_index, eu_policy, eeu
    ) -> float:
        """Find the eeu value associated with this query and return it with the query index. Should return - EEU"""


class VOIElicitor(BaseVOIElicitor):
    """OCA algorithm proposed by Baarslag in IJCAI2017
    """

    def eeu(self, policy: np.ndarray, eus: np.ndarray) -> float:
        """Expected Expected Negotiator for following the policy"""
        p = np.ones((len(policy) + 1))
        m = self.opponent_model.acceptance_probabilities()[policy]
        r = 1 - m
        eup = -eus * m
        p[1:-1] = np.cumprod(r[:-1])
        try:
            result = np.sum(eup * p[:-1])
        except FloatingPointError:
            result = 0.0
            try:
                result = eup[0] * p[0]
                for i in range(1, len(eup)):
                    try:
                        result += eup[0] * p[i]
                    except:
                        break
            except FloatingPointError:
                result[0] = 0.0
        return round(float(result), 6)

    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors"""
        ami = self._ami
        n_outcomes = ami.n_outcomes
        # remaining_steps = ami.remaining_steps if ami.remaining_steps is not None else ami.n_outcomes
        D = n_outcomes
        indices = set(list(range(n_outcomes)))
        p = self.opponent_model.acceptance_probabilities()
        eus = self.eus
        eeus1outcome = eus * p
        best_indx = argmax(eeus1outcome)
        eu_policy = [(-eus[best_indx], best_indx)]
        indices.remove(best_indx)
        D -= 1
        best_eeu = eus[best_indx]
        for _ in range(D):
            if len(indices) < 1:
                break
            candidate_policies = [copy.copy(eu_policy) for _ in indices]
            best_index, best_eeu, eu_policy = None, -10.0, None
            for i, candidate_policy in zip(indices, candidate_policies):
                heappush(candidate_policy, (-eus[i], i))
                # now we have the sorted list of outcomes as a candidate policy
                _policy = np.array([_[1] for _ in candidate_policy])
                _eus = np.array([_[0] for _ in candidate_policy])
                current_eeu = self.eeu(policy=_policy, eus=_eus)
                if (
                    current_eeu > best_eeu
                ):  # all numbers are negative so really that means current_eeu > best_eeu
                    best_eeu, best_index, eu_policy = current_eeu, i, candidate_policy
            if best_index is not None:
                indices.remove(best_index)
        self.outcome_in_policy = {}
        for i, (eu, outcome) in enumerate(eu_policy):
            self.outcome_in_policy[outcome] = i
        heapify(eu_policy)
        self.eu_policy, self.current_eeu = eu_policy, best_eeu

    def _query_eeu(
        self, query, qindex, outcome, cost, outcome_index, eu_policy, eeu
    ) -> float:
        current_util = self.utility_function(outcome)
        answers = query.answers
        answer_probabilities = query.probs
        answer_eeus = []
        for answer in answers:
            self.init_optimal_policy()
            policy_record_index = self.outcome_in_policy[outcome_index]
            eu_policy = copy.deepcopy(self.eu_policy)
            new_util = (
                -float(answer.constraint.marginal(outcome) & current_util),
                outcome_index,
            )
            eu_policy[policy_record_index] = new_util
            heapify(eu_policy)
            _policy = np.array([_[1] for _ in eu_policy])
            _eus = np.array([_[0] for _ in eu_policy])
            answer_eeus.append(self.eeu(policy=_policy, eus=_eus))
        return cost - sum([a * b for a, b in zip(answer_probabilities, answer_eeus)])


class VOIFastElicitor(BaseVOIElicitor):
    """FastVOI algorithm proposed by Mohammad in PRIMA 2014.
    """

    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors"""
        ami = self._ami
        n_outcomes = ami.n_outcomes
        eus = -self.eus
        eu_policy = sortedlist(zip(eus, range(n_outcomes)))
        policy = np.array([_[1] for _ in eu_policy])
        eu = np.array([_[0] for _ in eu_policy])
        p = np.ones((len(policy) + 1))
        ac = self.opponent_model.acceptance_probabilities()[policy]
        eup = -eu * ac
        r = 1 - ac
        p[1:] = np.cumprod(r)
        try:
            s = np.cumsum(eup * p[:-1])
        except FloatingPointError:
            s = np.zeros(len(eup))
            try:
                s[0] = eup[0] * p[0]
            except FloatingPointError:
                s[0] = 0
            for i in range(1, len(eup)):
                try:
                    s[i] = s[i - 1] + eup[0] * p[i]
                except:
                    s[i:] = s[i - 1]
                    break
        self.current_eeu = round(s[-1], 6)
        self.p, self.s = p, s
        self.eu_policy = sortedlist(eu_policy)
        self.outcome_in_policy = {}
        for j, pp in enumerate(self.eu_policy):
            self.outcome_in_policy[pp[1]] = pp

    def _query_eeu(
        self, query, qindex, outcome, cost, outcome_index, eu_policy, eeu
    ) -> float:
        answers = query.answers
        answer_probabilities = query.probs
        answer_eeus = []
        current_util = self.utility_function(outcome)
        old_util = self.outcome_in_policy[outcome_index]
        old_indx = eu_policy.index(old_util)
        eu_policy.remove(old_util)
        for answer in answers:
            reeu = self.current_eeu
            a = self.opponent_model.probability_of_acceptance(outcome)
            eu = float(answer.constraint.marginal(outcome) & current_util)
            if old_util[0] != -eu:
                new_util = (-eu, outcome_index)
                p, s = self.p, self.s
                eu_policy.add(new_util)
                new_indx = eu_policy.index(new_util)
                moved_back = new_indx > old_indx or new_indx == old_indx
                u_old, u_new = -old_util[0], eu
                try:
                    if new_indx == old_indx:
                        reeu = eeu - a * u_old * p[new_indx] + a * u_new * p[new_indx]
                    else:
                        s_before_src = s[old_indx - 1] if old_indx > 0 else 0.0
                        if moved_back:
                            p_after = p[new_indx + 1]
                            if a < 1.0 - 1e-6:
                                reeu = (
                                    s_before_src
                                    + (s[new_indx] - s[old_indx]) / (1 - a)
                                    + a * u_new * p_after / (1 - a)
                                    + eeu
                                    - s[new_indx]
                                )
                            else:
                                reeu = s_before_src + eeu - s[new_indx]
                        else:
                            s_before_dst = s[new_indx - 1] if new_indx > 0 else 0.0
                            if a < 1.0 - 1e-6:
                                reeu = (
                                    s_before_dst
                                    + a * u_new * p[new_indx]
                                    + (s_before_src - s_before_dst) * (1 - a)
                                    + eeu
                                    - s[old_indx]
                                )
                            else:
                                reeu = (
                                    s_before_dst
                                    + a * u_new * p[new_indx]
                                    + eeu
                                    - s[old_indx]
                                )
                except FloatingPointError:
                    pass

                self.eu_policy.remove(new_util)
            answer_eeus.append(reeu)
        self.eu_policy.add(old_util)
        qeeu = cost - sum([a * b for a, b in zip(answer_probabilities, answer_eeus)])
        return qeeu


class VOINoUncertaintyElicitor(BaseVOIElicitor):
    """A dummy VOI Elicitation Agent. It simply assumes no uncertainty in own utility function"""

    def eeu(self, policy: np.ndarray, eup: np.ndarray) -> float:
        """Expected Expected Negotiator for following the policy"""
        p = np.ones((len(policy) + 1))
        r = 1 - self.opponent_model.acceptance_probabilities()[policy]
        p[1:] = np.cumprod(r)
        try:
            result = np.sum(eup * p[:-1])
        except FloatingPointError:
            result = 0.0
            try:
                result = eup[0] * p[0]
                for i in range(1, len(eup)):
                    try:
                        result += eup[0] * p[i]
                    except:
                        break
            except FloatingPointError:
                result[0] = 0.0
        return float(result)  # it was - for a reason I do not undestand (2018.11.16)

    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors"""
        ami = self._ami
        n_outcomes = ami.n_outcomes
        p = self.opponent_model.acceptance_probabilities()
        eus = -self.eus * p
        eu_policy = sortedlist(zip(eus, range(n_outcomes)))
        self.current_eeu = self.eeu(
            policy=np.array([_[1] for _ in eu_policy]),
            eup=np.array([_[0] for _ in eu_policy]),
        )
        self.eu_policy = eu_policy
        self.outcome_in_policy = {}
        for j, (_, indx) in enumerate(eu_policy):
            self.outcome_in_policy[indx] = (_, indx)

    def init_query_eeus(self) -> None:
        pass

    def add_query(self, qeeu: Tuple[float, int]) -> None:
        pass

    def _query_eeu(
        self, query, qindex, outcome, cost, outcome_index, eu_policy, eeu
    ) -> float:
        return -1.0

    def elicit_single(self, state: MechanismState):
        return False


class VOIOptimalElicitor(BaseElicitor):
    """Base class for all VOI elicitation algorithms
    """

    def __init__(
        self,
        user: User,
        *,
        base_negotiator: SAONegotiator = AspirationNegotiator(),
        adaptive_answer_probabilities=True,
        expector_factory: Union[Expector, Callable[[], Expector]] = MeanExpector,
        single_elicitation_per_round=False,
        continue_eliciting_past_reserved_val=False,
        epsilon=0.001,
        resolution=0.025,
        true_utility_on_zero_cost=False,
        each_outcome_once=False,
        update_related_queries=True,
        prune=True,
        opponent_model_factory: Optional[
            Callable[["AgentMechanismInterface"], "DiscreteAcceptanceModel"]
        ] = lambda x: AdaptiveDiscreteAcceptanceModel.from_negotiation(ami=x),
    ) -> None:
        super().__init__(
            strategy=None,
            user=user,
            opponent_model_factory=opponent_model_factory,
            expector_factory=expector_factory,
            single_elicitation_per_round=single_elicitation_per_round,
            continue_eliciting_past_reserved_val=continue_eliciting_past_reserved_val,
            epsilon=epsilon,
            true_utility_on_zero_cost=true_utility_on_zero_cost,
            base_negotiator=base_negotiator,
        )
        # todo confirm that I need this. aspiration mixin. I think I do not.
        self.aspiration_init(max_aspiration=1.0, aspiration_type="boulware")
        self.eu_policy = None
        self.eeu_query = None
        self.query_index_of_outcome = None
        self.adaptive_answer_probabilities = adaptive_answer_probabilities
        self.current_eeu = None
        self.eus = None
        self.outcome_in_policy = None
        self.each_outcome_once = each_outcome_once
        self.queries_of_outcome = None
        self.queries = None
        self.update_related_queries = update_related_queries
        self.total_voi = 0.0
        self.resolution = resolution
        self.prune = prune

    def init_elicitation(
        self,
        ufun: Optional[Union["IPUtilityFunction", "UtilityDistribution"]],
        queries: Optional[List[Query]] = None,
    ) -> None:
        super().init_elicitation(ufun=ufun)
        if queries is not None:
            raise ValueError(
                f"self.__class__.__name__ does not allow the user to specify queries"
            )
        strt_time = time.perf_counter()
        ami = self._ami
        self.eus = np.array([_.mean() for _ in self.utility_distributions()])
        self.offerable_outcomes = ami.outcomes
        self.init_optimal_policy()
        self.init_query_eeus()
        self._elicitation_time += time.perf_counter() - strt_time

    def best_offer(self, state: MechanismState) -> Tuple[Optional[Outcome], float]:
        """Maximum Expected Utility at a given aspiration level (alpha)

        Args:
            state:
        """
        if self.each_outcome_once:
            # todo this needs correction. When I opp from the eu_policy, all eeu_query become wrong
            if len(self.eu_policy) < 1:
                self.init_optimal_policy()
            _, outcome_index = self.eu_policy.pop()
        else:
            outcome_index = self.eu_policy[0][1]
        if self.eus[outcome_index] < self.reserved_value:
            return None, self.reserved_value
        return (
            self._ami.outcomes[outcome_index],
            self.expect(
                self.utility_function(self._ami.outcomes[outcome_index]), state=state
            ),
        )

    def can_elicit(self) -> bool:
        return True

    def best_offers(self, n: int) -> List[Tuple[Optional[Outcome], float]]:
        """Maximum Expected Utility at a given aspiration level (alpha)"""
        return [self.best_offer()] * n

    def before_eliciting(self):
        pass

    def on_opponent_model_updated(
        self, outcomes: List[Outcome], old: List[float], new: List[float]
    ) -> None:
        if any(o != n for o, n in zip(old, new)):
            self.init_optimal_policy()
            self.init_query_eeus()

    def update_optimal_policy(
        self, index: int, outcome: Outcome, oldu: float, newu: float
    ):
        """Updates the optimal policy after a change happens to some utility"""
        if oldu != newu:
            self.init_optimal_policy()

    def elicit_single(self, state: MechanismState):
        if self.eeu_query is not None and len(self.eeu_query) < 1:
            return False
        if not self.can_elicit():
            return False
        eeu, q = heappop(self.eeu_query)
        if q is None or -eeu <= self.current_eeu:
            return False
        if (not self.continue_eliciting_past_reserved_val) and (
            -eeu - (self.user.cost_of_asking() + self.elicitation_cost)
            < self.reserved_value
        ):
            return False
        outcome, query, cost = self.queries[q]
        if query is None:
            return False
        self.queries[q] = (None, None, None)
        oldu = self.utility_function.distributions[outcome]
        if _scale(oldu) < 1e-7:
            return False
        u = self.user.ask(query)
        newu = u.answer.constraint.marginal(outcome)
        if self.queries_of_outcome is not None:
            if _scale(newu) > 1e-7:
                newu = newu & oldu
                newmin, newmax = newu.loc, newu.scale + newu.loc
                good_queries = []
                for i, qind in enumerate(self.queries_of_outcome.get(outcome, [])):
                    _o, _q, _c = self.queries[qind]
                    if _q is None:
                        continue
                    answers = _q.answers
                    tokeep = []
                    for j, ans in enumerate(answers):
                        rng = ans.constraint.range
                        if newmin == rng[0] and newmax == rng[1]:
                            continue
                        if newmin <= rng[0] <= newmax or rng[0] <= newmin <= rng[1]:
                            tokeep.append(j)
                    if len(tokeep) < 2:
                        self.queries[i] = None, None, None
                        continue
                    good_queries.append(qind)
                    if len(tokeep) < len(answers):
                        ans = _q.answers
                        self.queries[i].answers = [ans[j] for j in tokeep]
                self.queries_of_outcome[outcome] = good_queries
            else:
                for i, _ in enumerate(self.queries_of_outcome.get(outcome, [])):
                    self.queries[i] = None, None, None
                    self.queries_of_outcome[outcome] = []
        self.total_voi += -eeu - self.current_eeu
        outcome_index = self.indices[outcome]
        if _scale(newu) < 1e-7:
            self.utility_function.distributions[outcome] = newu
        else:
            self.utility_function.distributions[outcome] = newu & oldu
        eu = float(newu)
        self.eus[outcome_index] = eu
        self.update_optimal_policy(
            index=outcome_index, outcome=outcome, oldu=float(oldu), newu=eu
        )
        self._update_query_eeus(
            k=outcome_index,
            outcome=outcome,
            s=self.s,
            p=self.p,
            n=self._ami.n_outcomes,
            eeu=self.current_eeu,
            eus=[-_[0] for _ in self.eu_policy],
        )
        self.elicitation_history.append((query, newu, state.step, self.current_eeu))
        return True

    def _update_query_eeus(self, k: int, outcome: Outcome, s, p, n, eeu, eus):
        """Updates the best query for a single outcome"""
        this_outcome_solutions = []
        m = self.opponent_model.probability_of_acceptance(outcome)
        m1 = 1.0 - m
        m2 = m / m1 if m1 > 1e-6 else 0.0
        uk = self.utility_function.distributions[outcome]
        beta, alpha = uk.scale + uk.loc, uk.loc
        delta = beta - alpha
        if abs(delta) < max(self.resolution, 1e-6):
            return
        sk1, sk, pk = s[k - 1] if k > 0 else 0.0, s[k], p[k]
        for jp in range(k + 1):
            sjp1, sjp = s[jp - 1] if jp > 0 else 0.0, s[jp]
            if (
                beta < eus[jp]
            ):  # ignore cases where it is impossible to go to this low j
                continue
            for jm in range(k, n):
                if jp == k and jm == k:
                    continue
                if (
                    alpha > eus[jp]
                ):  # ignore cases where it is impossible to go to this large j
                    continue
                try:
                    sjm1, sjm = s[jm - 1] if jm > 0 else 0.0, s[jm]
                    if m1 > 1e-6:
                        y = ((sk1 - sk) + m * (sjm - sk1)) / m1
                    else:
                        y = 0.0
                    z = sk1 - sk + m * (sjp1 - sk1)
                    pjm1, pjp, pjm = p[jm + 1], p[jp], p[jm]
                    if jp < k < jm:  # Problem 1
                        a = (m2 * pjm1 - m * pjp) / (2 * delta)
                        b = (y - z) / delta
                        c = (
                            2 * z * beta
                            + m * pjp * beta * beta
                            - 2 * y * alpha
                            - m2 * pjm1 * alpha * alpha
                        ) / (2 * delta)
                    elif jp < k == jm:  # Problem 2
                        a = m * (pk - pjp) / (2 * delta)
                        b = -(2 * z + m * pk * (beta + alpha)) / (2 * delta)
                        c = (
                            beta
                            * (2 * z + m * pjp * beta + m * pk * alpha)
                            / (2 * delta)
                        )
                    else:  # Problem 3
                        a = (m2 * pjm1 - m * pk) / (2 * delta)
                        b = (2 * y + m * pk * (beta + alpha)) / (2 * delta)
                        c = (
                            -alpha
                            * (2 * y + m * pk * beta + m2 * pjm1 * alpha)
                            / (2 * delta)
                        )
                    if abs(a) < 1e-6:
                        continue
                    x = -b / (2 * a)
                    voi = c - a * x * x
                except FloatingPointError:
                    continue
                if x < alpha or x > beta or voi < self.user.cost_of_asking():
                    if self.prune:
                        break
                    continue  # ignore cases when the optimum is at the limit
                q = Query(
                    answers=[
                        Answer(
                            outcomes=[outcome],
                            constraint=RangeConstraint((x, beta)),
                            name="yes",
                        ),
                        Answer(
                            outcomes=[outcome],
                            constraint=RangeConstraint((alpha, x)),
                            name="no",
                        ),
                    ],
                    probs=[(beta - x) / delta, (x - alpha) / delta],
                    name=f"{outcome}>{x}",
                )
                this_outcome_solutions.append((voi, q))
            if self.prune and len(this_outcome_solutions) > 0:
                break
        if len(this_outcome_solutions) > 0:
            voi, q = max(this_outcome_solutions, key=lambda x: x[0])
            self.queries.append((outcome, q, self.user.cost_of_asking()))
            qindx = len(self.queries) - 1
            heappush(self.eeu_query, (-voi - eeu, qindx))
            self.queries_of_outcome[outcome] = [qindx]

    def init_query_eeus(self) -> None:
        """Updates the heap eeu_query which has records of (-EEU, quesion)"""
        # todo code for creating the optimal queries
        outcomes = self._ami.outcomes
        policy = [_[1] for _ in self.eu_policy]
        eus = [-_[0] for _ in self.eu_policy]
        n = len(outcomes)
        p, s = self.p, self.s
        eeu = self.current_eeu
        self.queries_of_outcome = dict()
        self.queries = []
        self.eeu_query = []
        heapify(self.eeu_query)
        for k, outcome_indx in enumerate(policy):
            self._update_query_eeus(
                k=k, outcome=outcomes[outcome_indx], s=s, p=p, n=n, eeu=eeu, eus=eus
            )

    def utility_on_rejection(
        self, outcome: Outcome, state: MechanismState
    ) -> UtilityValue:
        raise ValueError("utility_on_rejection should never be called on VOI Elicitors")

    def add_query(self, qeeu: Tuple[float, int]) -> None:
        heappush(self.eeu_query, qeeu)

    def init_optimal_policy(self) -> None:
        """Gets the optimal policy given Negotiator utility_priors"""
        ami = self._ami
        n_outcomes = ami.n_outcomes
        eus = -self.eus
        eu_policy = sortedlist(zip(eus, range(n_outcomes)))
        policy = np.array([_[1] for _ in eu_policy])
        eu = np.array([_[0] for _ in eu_policy])
        p = np.ones((len(policy) + 1))
        ac = self.opponent_model.acceptance_probabilities()[policy]
        eup = -eu * ac
        r = 1 - ac
        p[1:] = np.cumprod(r)
        try:
            s = np.cumsum(eup * p[:-1])
        except FloatingPointError:
            s = np.zeros(len(eup))
            try:
                s[0] = eup[0] * p[0]
            except FloatingPointError:
                s[0] = 0
            for i in range(1, len(eup)):
                try:
                    s[i] = s[i - 1] + eup[0] * p[i]
                except:
                    s[i:] = s[i - 1]
                    break
        self.current_eeu = round(s[-1], 6)
        self.p, self.s = p, s
        self.eu_policy = sortedlist(eu_policy)
        self.outcome_in_policy = {}
        for j, pp in enumerate(self.eu_policy):
            self.outcome_in_policy[pp[1]] = pp


def uniform():
    loc = random.random()
    scale = random.random() * (1.0 - loc)
    return UtilityDistribution(dtype="uniform", loc=loc, scale=scale)


def current_aspiration(
    elicitor: "AspirationMixin", outcome: "Outcome", negotiation: "Mechanism"
) -> "UtilityValue":
    return elicitor.aspiration(negotiation.relative_time)


def create_negotiator(
    negotiator_type, ufun, can_propose, outcomes, dynamic_ufun, toughness, **kwargs
):
    if negotiator_type == "limited_outcomes":
        if can_propose:
            negotiator = LimitedOutcomesNegotiator(
                acceptable_outcomes=outcomes,
                acceptance_probabilities=list(ufun.mapping.values()),
                outcomes=outcomes,
                **kwargs,
            )
        else:
            negotiator = LimitedOutcomesAcceptor(
                acceptable_outcomes=outcomes,
                acceptance_probabilities=list(ufun.mapping.values()),
                outcomes=outcomes,
                **kwargs,
            )
    elif negotiator_type == "random":
        negotiator = RandomNegotiator(
            reserved_value=ufun.reserved_value,
            outcomes=outcomes,
            can_propose=can_propose,
        )
    elif negotiator_type == "tough":
        negotiator = ToughNegotiator(dynamic_ufun=dynamic_ufun, can_propose=can_propose)
    elif negotiator_type in ("only_best", "best_only", "best"):
        negotiator = OnlyBestNegotiator(
            dynamic_ufun=dynamic_ufun,
            min_utility=None,
            top_fraction=1.0 - toughness,
            best_first=False,
            can_propose=can_propose,
        )
    elif negotiator_type.startswith("aspiration"):
        asp_kind = negotiator_type[len("aspiration") :]
        if asp_kind.startswith("_"):
            asp_kind = asp_kind[1:]
        try:
            asp_kind = float(asp_kind)
        except:
            pass
        if asp_kind == "":
            if toughness < 0.5:
                toughness *= 2
                toughness = 9.0 * toughness + 1.0
            elif toughness == 0.5:
                toughness = 1.0
            else:
                toughness = 2 * (toughness - 0.5)
                toughness = 1 - 0.9 * toughness
            asp_kind = toughness
        negotiator = AspirationNegotiator(
            aspiration_type=asp_kind,
            dynamic_ufun=dynamic_ufun,
            can_propose=can_propose,
            **kwargs,
        )
    elif negotiator_type.startswith("genius"):
        class_name = negotiator_type[len("genius") :]
        if class_name.startswith("_"):
            class_name = class_name[1:]
        if class_name == "auto" or len(class_name) < 1:
            negotiator = GeniusNegotiator.random_negotiator(
                keep_issue_names=False, keep_value_names=False, can_propose=can_propose
            )
        else:
            negotiator = GeniusNegotiator(
                java_class_name=class_name,
                keep_value_names=False,
                keep_issue_names=False,
                can_propose=can_propose,
            )
        negotiator.utility_function = ufun
    else:
        raise ValueError(f"Unknown opponents type {negotiator_type}")
    return negotiator


def _beg(x):
    if isinstance(x, float):
        return x
    else:
        return x.loc


def _scale(x):
    if isinstance(x, float):
        return 0.0
    else:
        return x.scale


def _end(x):
    if isinstance(x, float):
        return x
    else:
        return x.loc + x.scale


class SAOElicitingMechanism(SAOMechanism):
    @classmethod
    def generate_config(
        cls,
        cost,
        n_outcomes: int = None,
        rand_ufuns=True,
        conflict: float = None,
        conflict_delta: float = None,
        winwin=None,  # only if rand_ufuns is false
        genius_folder: str = None,
        n_steps=None,
        time_limit=None,
        own_utility_uncertainty=0.5,
        own_uncertainty_variablility=0.0,
        own_reserved_value=0.0,
        own_base_agent="aspiration",
        opponent_model_uncertainty=0.5,
        opponent_model_adaptive=False,
        opponent_proposes=True,
        opponent_type="best_only",
        opponent_toughness=0.9,
        opponent_reserved_value=0.0,
    ) -> Dict[str, Any]:
        config = {}
        if n_steps is None and time_limit is None and "aspiration" in opponent_type:
            raise ValueError(
                "Cannot use aspiration negotiators when no step limit or time limit is given"
            )
        if n_outcomes is None and genius_folder is None:
            raise ValueError(
                "Must specify a folder to run from or a number of outcomes"
            )
        if genius_folder is not None:
            domain, agent_info, issues = load_genius_domain_from_folder(
                folder_name=genius_folder,
                force_single_issue=True,
                keep_issue_names=False,
                keep_value_names=False,
                ignore_discount=True,
                ignore_reserved=opponent_reserved_value is not None,
            )
            n_outcomes = domain.ami.n_outcomes
            outcomes = domain.outcomes
            elicitor_indx = 0 + int(random.random() <= 0.5)
            opponent_indx = 1 - elicitor_indx
            ufun = agent_info[elicitor_indx]["ufun"]
            ufun.reserved_value = own_reserved_value
            opp_utility = agent_info[opponent_indx]["ufun"]
            opp_utility.reserved_value = opponent_reserved_value
        else:
            outcomes = [(_,) for _ in range(n_outcomes)]
            if rand_ufuns:
                ufun, opp_utility = UtilityFunction.generate_random_bilateral(
                    outcomes=outcomes
                )
            else:
                ufun, opp_utility = UtilityFunction.generate_bilateral(
                    outcomes=outcomes,
                    conflict_level=opponent_toughness,
                    conflict_delta=conflict_delta,
                    win_win=winwin,
                )
            ufun.reserved_value = own_reserved_value
            domain = SAOMechanism(
                outcomes=outcomes,
                n_steps=n_steps,
                time_limit=time_limit,
                max_n_agents=2,
                dynamic_entry=False,
                keep_issue_names=False,
                cache_outcomes=True,
            )

        true_utilities = list(ufun.mapping.values())
        priors = IPUtilityFunction.from_ufun(
            ufun,
            uncertainty=own_utility_uncertainty,
            variability=own_uncertainty_variablility,
        )

        outcomes = domain.ami.outcomes

        opponent = create_negotiator(
            negotiator_type=opponent_type,
            can_propose=opponent_proposes,
            ufun=opp_utility,
            outcomes=outcomes,
            dynamic_ufun=False,
            toughness=opponent_toughness,
        )
        opponent_model = UncertainOpponentModel(
            outcomes=outcomes,
            uncertainty=opponent_model_uncertainty,
            opponents=opponent,
            adaptive=opponent_model_adaptive,
        )
        config["n_steps"], config["time_limit"] = n_steps, time_limit
        config["priors"] = priors
        config["true_utilities"] = true_utilities
        config["elicitor_reserved_value"] = own_reserved_value
        config["cost"] = cost
        config["opp_utility"] = opp_utility
        config["opponent_model"] = opponent_model
        config["opponent"] = opponent
        config["base_agent"] = own_base_agent
        return config

    def __init__(
        self,
        priors,
        true_utilities,
        elicitor_reserved_value,
        cost,
        opp_utility,
        opponent,
        n_steps,
        time_limit,
        base_agent,
        opponent_model,
        elicitation_strategy="pingpong",
        toughness=0.95,
        elicitor_type="balanced",
        history_file_name: str = None,
        screen_log: bool = False,
        dynamic_queries=True,
        each_outcome_once=False,
        rational_answer_probs=True,
        update_related_queries=True,
        resolution=0.1,
        cost_assuming_titration=False,
        name: Optional[str] = None,
    ):
        self.elicitation_state = {}
        initial_priors = priors
        self.xw_real = priors

        outcomes = list(initial_priors.distributions.keys())

        self.U = true_utilities

        super().__init__(
            issues=None,
            outcomes=outcomes,
            n_steps=n_steps,
            time_limit=time_limit,
            max_n_agents=2,
            dynamic_entry=False,
            name=name,
        )
        if elicitor_reserved_value is None:
            elicitor_reserved_value = 0.0
        self.logger = create_loggers(
            file_name=history_file_name,
            screen_level=logging.DEBUG if screen_log else logging.ERROR,
        )
        user = User(
            ufun=MappingUtilityFunction(
                dict(zip(self.outcomes, self.U)), reserved_value=elicitor_reserved_value
            ),
            cost=cost,
        )
        if resolution is None:
            resolution = max(elicitor_reserved_value / 4, 0.025)
        if "voi" in elicitor_type and "optimal" in elicitor_type:
            strategy = None
        else:
            strategy = EStrategy(strategy=elicitation_strategy, resolution=resolution)
            strategy.on_enter(ami=self.ami, ufun=initial_priors)

        def create_elicitor(type_, strategy=strategy, opponent_model=opponent_model):
            base_negotiator = create_negotiator(
                negotiator_type=base_agent,
                ufun=None,
                can_propose=True,
                outcomes=outcomes,
                dynamic_ufun=True,
                toughness=toughness,
            )
            if type_ == "full":
                return FullElicitor(
                    strategy=strategy, user=user, base_negotiator=base_negotiator
                )

            if type_ == "dummy":
                return DummyElicitor(
                    strategy=strategy, user=user, base_negotiator=base_negotiator
                )

            if type_ == "full_knowledge":
                return FullKnowledgeElicitor(
                    strategy=strategy, user=user, base_negotiator=base_negotiator
                )

            if type_ == "random_deep":
                return RandomElicitor(
                    strategy=strategy,
                    deep_elicitation=True,
                    user=user,
                    base_negotiator=base_negotiator,
                )

            if type_ in ("random_shallow", "random"):
                return RandomElicitor(
                    strategy=strategy,
                    deep_elicitation=False,
                    user=user,
                    base_negotiator=base_negotiator,
                )
            if type_ in (
                "pessimistic",
                "optimistic",
                "balanced",
                "pandora",
                "fast",
                "mean",
            ):
                type_ = type_.title() + "Elicitor"
                return eval(type_)(
                    strategy=strategy,
                    user=user,
                    base_negotiator=base_negotiator,
                    opponent_model_factory=lambda x: opponent_model,
                    single_elicitation_per_round=False,
                    assume_uniform=True,
                    user_model_in_index=True,
                    precalculated_index=False,
                )
            if "voi" in type_:
                expector_factory = MeanExpector
                if "balanced" in type_:
                    expector_factory = BalancedExpector
                elif "optimistic" in type_ or "max" in type_:
                    expector_factory = MaxExpector
                elif "pessimistic" in type_ or "min" in type_:
                    expector_factory = MinExpector

                if "fast" in type_:
                    factory = VOIFastElicitor
                elif "optimal" in type_:
                    prune = "prune" in type_ or "fast" in type_
                    if "no" in type_:
                        no_prune = not prune
                    return VOIOptimalElicitor(
                        user=user,
                        resolution=resolution,
                        opponent_model_factory=lambda x: opponent_model,
                        single_elicitation_per_round=False,
                        base_negotiator=base_negotiator,
                        each_outcome_once=each_outcome_once,
                        expector_factory=expector_factory,
                        update_related_queries=update_related_queries,
                        prune=prune,
                    )
                elif "no_uncertainty" in type_ or "full_knowledge" in type_:
                    factory = VOINoUncertaintyElicitor
                else:
                    factory = VOIElicitor

                if not dynamic_queries and "optimal" not in type_:
                    queries = []
                    for outcome in self.outcomes:
                        u = initial_priors(outcome)
                        scale = _scale(u)
                        if scale < resolution:
                            continue
                        bb, ee = _beg(u), _end(u)
                        n_q = int((ee - bb) / resolution)
                        limits = np.linspace(bb, ee, n_q, endpoint=False)[1:]
                        for i, limit in enumerate(limits):
                            if cost_assuming_titration:
                                qcost = cost * min(i, len(limits) - i - 1)
                            else:
                                qcost = cost
                            answers = [
                                Answer(
                                    outcomes=[outcome],
                                    constraint=RangeConstraint(rng=(0.0, limit)),
                                    name="yes",
                                ),
                                Answer(
                                    outcomes=[outcome],
                                    constraint=RangeConstraint(rng=(limit, 1.0)),
                                    name="no",
                                ),
                            ]
                            probs = (
                                [limit, 1.0 - limit]
                                if rational_answer_probs
                                else [0.5, 0.5]
                            )
                            query = Query(
                                answers=answers,
                                cost=qcost,
                                probs=probs,
                                name=f"{outcome}<{limit}",
                            )
                            queries.append((outcome, query, qcost))
                else:
                    queries = None
                return factory(
                    strategy=strategy if dynamic_queries else None,
                    user=user,
                    opponent_model_factory=lambda x: opponent_model,
                    single_elicitation_per_round=False,
                    dynamic_query_set=dynamic_queries,
                    queries=queries,
                    base_negotiator=base_negotiator,
                    each_outcome_once=each_outcome_once,
                    expector_factory=expector_factory,
                    update_related_queries=update_related_queries,
                )

        elicitor = create_elicitor(elicitor_type)

        if isinstance(opponent, GeniusNegotiator):
            if n_steps is not None and time_limit is not None:
                self.ami.n_steps = None

        self.add(opponent, ufun=opp_utility)
        self.add(elicitor, ufun=initial_priors)
        if len(self.negotiators) != 2:
            raise ValueError(
                f"I could not add the two negotiators {elicitor.__class__.__name__}, {opponent.__class__.__name__}"
            )
        self.total_time = 0.0

    def loginfo(self, s: str) -> None:
        """logs ami-level information

        Args:
            s (str): The string to log

        """
        self.logger.info(s.strip())

    def logdebug(self, s) -> None:
        """logs debug-level information

        Args:
            s (str): The string to log

        """
        self.logger.debug(s.strip())

    def logwarning(self, s) -> None:
        """logs warning-level information

        Args:
            s (str): The string to log

        """
        self.logger.warning(s.strip())

    def logerror(self, s) -> None:
        """logs error-level information

        Args:
            s (str): The string to log

        """
        self.logger.error(s.strip())

    def step(self) -> SAOState:
        start = time.perf_counter()
        _ = super().step()
        self.total_time += time.perf_counter() - start
        self.loginfo(
            f"[{self._step}] {self._current_proposer} offered {self._current_offer}"
        )
        return _

    def on_negotiation_start(self):
        if not super().on_negotiation_start():
            return False
        self.elicitation_state = {}
        self.elicitation_state["steps"] = None
        self.elicitation_state["relative_time"] = None
        self.elicitation_state["broken"] = False
        self.elicitation_state["timedout"] = False
        self.elicitation_state["agreement"] = None
        self.elicitation_state["agreed"] = False
        self.elicitation_state["utils"] = [
            0.0 for a in self.negotiators
        ]  # not even the reserved value
        self.elicitation_state["welfare"] = sum(self.elicitation_state["utils"])
        self.elicitation_state["elicitor"] = self.negotiators[
            1
        ].__class__.__name__.replace("Elicitor", "")
        self.elicitation_state["opponents"] = self.negotiators[
            0
        ].__class__.__name__.replace("Aget", "")
        self.elicitation_state["elicitor_utility"] = self.elicitation_state["utils"][1]
        self.elicitation_state["opponent_utility"] = self.elicitation_state["utils"][0]
        self.elicitation_state["opponent_params"] = str(self.negotiators[0])
        self.elicitation_state["elicitor_params"] = str(self.negotiators[1])
        self.elicitation_state["elicitation_cost"] = None
        self.elicitation_state["total_time"] = None
        self.elicitation_state["pareto"] = None
        self.elicitation_state["pareto_distance"] = None
        self.elicitation_state["_elicitation_time"] = None
        self.elicitation_state["real_asking_time"] = None
        self.elicitation_state["n_queries"] = 0
        return True

    def plot(self, consider_costs=False):
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec

            if len(self.negotiators) > 2:
                print("Cannot visualize negotiations with more than 2 negotiators")
            else:
                # has_front = int(len(self.outcomes[0]) <2)
                has_front = 1
                n_agents = len(self.negotiators)
                history = pd.DataFrame(data=[_[1] for _ in self.history])
                history["time"] = [_[0].time for _ in self.history]
                history["relative_time"] = [_[0].relative_time for _ in self.history]
                history["step"] = [_[0].step for _ in self.history]
                history = history.loc[~history.offer.isnull(), :]
                ufuns = self._get_ufuns(consider_costs=consider_costs)
                elicitor_dist = self.negotiators[1].utility_function
                outcomes = self.outcomes

                utils = [tuple(f(o) for f in ufuns) for o in outcomes]
                agent_names = [
                    a.__class__.__name__ + ":" + a.name for a in self.negotiators
                ]
                history["offer_index"] = [outcomes.index(_) for _ in history.offer]
                frontier, frontier_outcome = self.pareto_frontier(sort_by_welfare=True)
                frontier_outcome_indices = [outcomes.index(_) for _ in frontier_outcome]
                fig_util, fig_outcome = plt.figure(), plt.figure()
                gs_util = gridspec.GridSpec(n_agents, has_front + 1)
                gs_outcome = gridspec.GridSpec(n_agents, has_front + 1)
                axs_util, axs_outcome = [], []

                agent_names_for_legends = [
                    agent_names[a]
                    .split(":")[0]
                    .replace("Negotiator", "")
                    .replace("Elicitor", "")
                    for a in range(n_agents)
                ]
                if agent_names_for_legends[0] == agent_names_for_legends[1]:
                    agent_names_for_legends = [
                        agent_names[a]
                        .split(":")[0]
                        .replace("Negotiator", "")
                        .replace("Elicitor", "")
                        + agent_names[a].split(":")[1]
                        for a in range(n_agents)
                    ]

                for a in range(n_agents):
                    if a == 0:
                        axs_util.append(fig_util.add_subplot(gs_util[a, has_front]))
                        axs_outcome.append(
                            fig_outcome.add_subplot(gs_outcome[a, has_front])
                        )
                    else:
                        axs_util.append(
                            fig_util.add_subplot(
                                gs_util[a, has_front], sharex=axs_util[0]
                            )
                        )
                        axs_outcome.append(
                            fig_outcome.add_subplot(
                                gs_outcome[a, has_front], sharex=axs_outcome[0]
                            )
                        )
                    axs_util[-1].set_ylabel(agent_names_for_legends[a])
                    axs_outcome[-1].set_ylabel(agent_names_for_legends[a])
                for a, (au, ao) in enumerate(zip(axs_util, axs_outcome)):
                    h = history.loc[
                        history.offerer == agent_names[a],
                        ["relative_time", "offer_index", "offer"],
                    ]
                    h["utility"] = h.offer.apply(ufuns[a])
                    ao.plot(h.relative_time, h.offer_index)
                    au.plot(h.relative_time, h.utility)
                    # if a == 1:
                    h["dist"] = h.offer.apply(elicitor_dist)
                    h["beg"] = h.dist.apply(_beg)
                    h["end"] = h.dist.apply(_end)
                    h["p_acceptance"] = h.offer.apply(
                        self.negotiators[1].opponent_model.probability_of_acceptance
                    )
                    au.plot(h.relative_time, h.end, color="r")
                    au.plot(h.relative_time, h.beg, color="r")
                    au.plot(h.relative_time, h.p_acceptance, color="g")
                    au.set_ylim(-0.1, 1.1)

                if has_front:
                    axu = fig_util.add_subplot(gs_util[:, 0])
                    axu.plot([0, 1], [0, 1], "g--")
                    axu.scatter(
                        [_[0] for _ in utils],
                        [_[1] for _ in utils],
                        label="outcomes",
                        color="yellow",
                        marker="s",
                        s=20,
                    )
                    axo = fig_outcome.add_subplot(gs_outcome[:, 0])
                    clrs = ("blue", "green")
                    for a in range(n_agents):
                        h = history.loc[
                            history.offerer == agent_names[a],
                            ["relative_time", "offer_index", "offer"],
                        ]
                        h["u0"] = h.offer.apply(ufuns[0])
                        h["u1"] = h.offer.apply(ufuns[1])

                        axu.scatter(
                            h.u0,
                            h.u1,
                            color=clrs[a],
                            label=f"{agent_names_for_legends[a]}",
                        )
                    steps = sorted(history.step.unique().tolist())
                    aoffers = [[], []]
                    for step in steps[::2]:
                        offrs = []
                        for a in range(n_agents):
                            a_offer = history.loc[
                                (history.offerer == agent_names[a])
                                & ((history.step == step) | (history.step == step + 1)),
                                "offer_index",
                            ]
                            if len(a_offer) > 0:
                                offrs.append(a_offer.values[-1])
                        if len(offrs) == 2:
                            aoffers[0].append(offrs[0])
                            aoffers[1].append(offrs[1])
                    axo.scatter(aoffers[0], aoffers[1], color=clrs[0], label=f"offers")

                    if self.state.agreement is not None:
                        axu.scatter(
                            [ufuns[0](self.state.agreement)],
                            [ufuns[1](self.state.agreement)],
                            color="black",
                            marker="*",
                            s=120,
                            label="SCMLAgreement",
                        )
                        axo.scatter(
                            [outcomes.index(self.state.agreement)],
                            [outcomes.index(self.state.agreement)],
                            color="black",
                            marker="*",
                            s=120,
                            label="SCMLAgreement",
                        )
                    f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
                    axu.scatter(f1, f2, label="frontier", color="red", marker="x")
                    axo.scatter(
                        frontier_outcome_indices,
                        frontier_outcome_indices,
                        color="red",
                        marker="x",
                        label="frontier",
                    )
                    axu.legend()
                    axo.legend()
                    axo.set_xlabel(agent_names_for_legends[0])
                    axo.set_ylabel(agent_names_for_legends[1])

                    axu.set_xlabel(agent_names_for_legends[0] + " utility")
                    axu.set_ylabel(agent_names_for_legends[1] + " utility")
                    if self.agreement is not None:
                        pareto_distance = 1e9
                        cu = (ufuns[0](self.agreement), ufuns[1](self.agreement))
                        for pu in frontier:
                            dist = math.sqrt(
                                (pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2
                            )
                            if dist < pareto_distance:
                                pareto_distance = dist
                        axu.text(
                            0,
                            0.95,
                            f"Pareto-distance={pareto_distance:5.2}",
                            verticalalignment="top",
                            transform=axu.transAxes,
                        )

                fig_util.show()
                fig_outcome.show()
        except:
            pass

    def on_negotiation_end(self):
        super().on_negotiation_end()
        self.elicitation_state = {}
        self.elicitation_state["steps"] = self._step + 1
        self.elicitation_state["relative_time"] = self.relative_time
        self.elicitation_state["broken"] = self.state.broken
        self.elicitation_state["timedout"] = (
            not self.state.broken and self.state.agreement is None
        )
        self.elicitation_state["agreement"] = self.state.agreement
        self.elicitation_state["agreed"] = (
            self.state.agreement is not None and not self.state.broken
        )

        if self.elicitation_state["agreed"]:
            self.elicitation_state["utils"] = [
                float(a.ufun(self.state.agreement)) if a.ufun is not None else 0.0
                for a in self.negotiators
            ]
        else:
            self.elicitation_state["utils"] = [
                a.reserved_value if a.reserved_value is not None else 0.0
                for a in self.negotiators
            ]
        self.elicitation_state["welfare"] = sum(self.elicitation_state["utils"])
        self.elicitation_state["elicitor"] = self.negotiators[
            1
        ].__class__.__name__.replace("Elicitor", "")
        self.elicitation_state["opponents"] = self.negotiators[
            0
        ].__class__.__name__.replace("Aget", "")
        self.elicitation_state["elicitor_utility"] = self.elicitation_state["utils"][1]
        self.elicitation_state["opponent_utility"] = self.elicitation_state["utils"][0]
        self.elicitation_state["opponent_params"] = str(self.negotiators[0])
        self.elicitation_state["elicitor_params"] = str(self.negotiators[1])
        self.elicitation_state["elicitation_cost"] = self.negotiators[
            1
        ].elicitation_cost
        self.elicitation_state["total_time"] = self.total_time
        self.elicitation_state["_elicitation_time"] = self.negotiators[
            1
        ].elicitation_time
        self.elicitation_state["asking_time"] = self.negotiators[1].asking_time
        self.elicitation_state["pareto"], pareto_outcomes = self.pareto_frontier()
        if self.elicitation_state["agreed"]:
            if self.state.agreement in pareto_outcomes:
                min_dist = 0.0
            else:
                min_dist = 1e12
                for p in self.elicitation_state["pareto"]:
                    dist = 0.0
                    for par, real in zip(p, self.elicitation_state["utils"]):
                        dist += (par - real) ** 2
                    dist = math.sqrt(dist)
                    if dist < min_dist:
                        min_dist = dist
            self.elicitation_state["pareto_distance"] = (
                min_dist if min_dist < 1e12 else None
            )
        else:
            self.elicitation_state["pareto_distance"] = None
        try:
            self.elicitation_state["n_queries"] = len(
                self.negotiators[1].user.elicited_queries()
            )
        except:
            self.elicitation_state["n_queries"] = None
        if hasattr(self.negotiators[1], "total_voi"):
            self.elicitation_state["total_voi"] = self.negotiators[1].total_voi
        else:
            self.elicitation_state["total_voi"] = None
