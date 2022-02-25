from __future__ import annotations

from ..common import NegotiatorMechanismInterface, Value
from ..helpers.prob import ScipyDistribution
from ..preferences import IPUtilityFunction
from .common import _loc, _upper
from .queries import Answer, Query, RangeConstraint

__all__ = ["EStrategy"]


class EStrategy:
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

    def next_query(self, outcome: Outcome) -> Query | None:
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

    def apply(
        self, user: User, outcome: Outcome
    ) -> tuple[Value | None, QResponse | None]:
        """Do the elicitation and incur the cost.

        Remarks:

            - This function returns a uniform distribution whenever it returns a distribution
            - Can return `None` which indicates that elicitation failed
            - If it could find an exact value, it will return a `float` not a `Distribution`

        """

        lower, upper, _ = self.lower, self.upper, self.outcomes
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
                    ScipyDistribution(type="uniform", loc=lower, scale=upper - lower),
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
            u = ScipyDistribution(type="uniform", loc=lower, scale=upper - lower)
        return u, reply

    def utility_estimate(self, outcome: Outcome) -> Value:
        """Gets a probability distribution of the Negotiator for this outcome without elicitation. Costs nothing"""
        indx = self.indices[outcome]
        scale = self.upper[indx] - self.lower[indx]
        if scale < self.resolution:
            return self.lower[indx]
        return ScipyDistribution(type="uniform", loc=self.lower[indx], scale=scale)

    def until(
        self,
        outcome: Outcome,
        user: User,
        dist: list[Value] | Value,
    ) -> Value:
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
        self, nmi: NegotiatorMechanismInterface, preferences: IPUtilityFunction = None
    ) -> None:
        self.lower = [0.0] * nmi.n_outcomes
        self.upper = [1.0] * nmi.n_outcomes
        self.indices = dict(zip(nmi.outcomes, range(nmi.n_outcomes)))
        if preferences is not None:
            distributions = list(preferences.distributions.values())
            for i, dist in enumerate(distributions):
                self.lower[i] = _loc(dist)
                self.upper[i] = _upper(dist)
        self.outcomes = nmi.outcomes
        self._total_cost = 0.0
        self._elicited_queries = []
