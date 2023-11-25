from __future__ import annotations

import random
from functools import lru_cache, partial
from typing import Any, Callable, Iterable, Mapping

from negmas import warnings
from negmas.helpers import get_full_type_name
from negmas.helpers.numeric import make_range
from negmas.helpers.prob import EPSILON
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.base_issue import DiscreteIssue
from negmas.outcomes.common import check_one_at_most, os_or_none
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.outcomes.protocols import IndependentIssuesOS, OutcomeSpace
from negmas.preferences.protocols import SingleIssueFun
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from ..crisp_ufun import UtilityFunction
from ..mixins import StationaryMixin
from ..value_fun import IdentityFun, LambdaFun, TableFun

__all__ = [
    "LinearUtilityAggregationFunction",
    "LinearAdditiveUtilityFunction",
    "LinearUtilityFunction",
    "AffineUtilityFunction",
]

NLEVELS = 20


def _rand_mapping(x, r):
    return (r - 0.5) * x


def _rand_mapping_normalized(x, mx, mn, r):
    return r * (x - mn) / (mx - mn)


def _random_mapping(issue: Issue, normalized=False):
    r = random.random()
    if issue.is_numeric():
        return (
            partial(
                _rand_mapping_normalized, mx=issue.max_value, mn=issue.min_value, r=r
            )
            if normalized
            else partial(_rand_mapping, r=r)
        )
    if isinstance(issue, DiscreteIssue):
        return dict(
            zip(
                issue.all,
                [
                    random.random() - (0.5 if not normalized else 0.0)
                    for _ in range(issue.cardinality)
                ],
            )
        )
    return (
        partial(_rand_mapping_normalized, mx=issue.max_value, mn=issue.min_value)
        if normalized
        else partial(_rand_mapping, r=r)
    )


class AffineUtilityFunction(
    StationaryMixin,
    UtilityFunction,
):
    r"""
    An affine utility function for multi-issue negotiations.

    Models a linear utility function using predefined weights.

    Args:
         weights: weights for combining `values`
         bias: The offset added
         name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility value is calculated as:

        .. math::

            u = \alpha + \sum_{i=0}^{n_{outcomes}-1} {\alpha_i * \omega_i}


        where $\alpha$ is the bias term and $\alpha_i$ is the weight of issue $i$,
        and $\omega_i$ is the value of issue $i$ in the input outcome $\omega$.

    Examples:

        >>> from negmas.outcomes import make_issue
        >>> issues = [make_issue((10.0, 20.0), 'price'), make_issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', 'quality: (0, 4)']
        >>> f = AffineUtilityFunction({'price': 1.0, 'quality': 4.0}, issues=issues)
        >>> f((2, 14.0)) -  (2 * 1.0 + 14.0 * 4)
        0.0
        >>> f = LinearUtilityFunction([1.0, 2.0])
        >>> f((2, 14)) - (2 * 1.0 + 14 * 2.0)
        0.0

    Remarks:

        - If an outcome contains combinations of strings and numeric values that have corresponding weights, an
          exception will be raised when its utility is calculated
        - If you pass weights as a dictionary (mapping issue names to wieght
          values), you **must** pass the issues as well ( using `outcome_space`
          or `issues` ) and issue names must match the keys of the weights
          dict exactly (i.e. all issues are
          represented and all keys in the dict are in the issues).
        - If you pass the weights as a tuple, you need not pass the issues (but
          you still should to help with things like scaling and shifting ufun
          values)
    """

    def __init__(
        self,
        weights: dict[str, float] | list[float] | tuple[float, ...] | None = None,
        bias: float = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.outcome_space and not isinstance(
            self.outcome_space, IndependentIssuesOS
        ):
            raise ValueError(
                f"Cannot create {self.type} ufun with an outcomespace without indpendent issues"
                f".\n Given OS: {self.outcome_space} of type {type(self.outcome_space)}\n"
                f"Given args {kwargs}"
            )
        self.issues: list[Issue] | None = (
            list(self.outcome_space.issues) if self.outcome_space else None  # type: ignore
        )
        if weights is None:
            if not self.issues:
                raise ValueError(
                    "Cannot initializes with no weights if you are not specifying issues"
                )
            weights = [1.0 / len(self.issues)] * len(self.issues)

        if isinstance(weights, dict):
            if not self.issues:
                raise ValueError(
                    "Cannot initializes with dict of weights if you are not specifying issues"
                )
            weights = [
                weights[_]
                for _ in [i.name if isinstance(i, Issue) else i for i in self.issues]
            ]

        self._weights: list[float] = list(weights)
        self._bias = bias
        self._values = [IdentityFun() for _ in self.issues] if self.issues else []

    @property
    def bias(self):
        return self._bias

    @property
    def weights(self):
        return self._weights

    @property
    def values(self):
        return self._values

    def eval(self, offer: Outcome | None) -> float | None:
        if offer is None:
            return self.reserved_value
        return self._bias + sum(w * v for w, v in zip(self._weights, offer))

    def xml(self, issues: list[Issue] | None = None) -> str:
        """Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> from negmas.outcomes import make_issue
            >>> issues = [make_issue(values=10, name='i1'), make_issue(values=4, name='i2')]
            >>> f = LinearUtilityFunction(weights=[1.0, 4.0], issues=issues)
            >>> print(f.xml(issues))
            <issue index="1" etype="discrete" type="discrete" vtype="integer" name="i1">
                <item index="1" value="0" evaluation="0.0" />
                <item index="2" value="1" evaluation="1.0" />
                <item index="3" value="2" evaluation="2.0" />
                <item index="4" value="3" evaluation="3.0" />
                <item index="5" value="4" evaluation="4.0" />
                <item index="6" value="5" evaluation="5.0" />
                <item index="7" value="6" evaluation="6.0" />
                <item index="8" value="7" evaluation="7.0" />
                <item index="9" value="8" evaluation="8.0" />
                <item index="10" value="9" evaluation="9.0" />
            </issue>
            <issue index="2" etype="discrete" type="discrete" vtype="integer" name="i2">
                <item index="1" value="0" evaluation="0.0" />
                <item index="2" value="1" evaluation="1.0" />
                <item index="3" value="2" evaluation="2.0" />
                <item index="4" value="3" evaluation="3.0" />
            </issue>
            <weight index="1" value="1.0">
            </weight>
            <weight index="2" value="4.0">
            </weight>
            <BLANKLINE>

        """
        # todo save for continuous issues using evaluator ftype linear offset slope (see from_xml_str)
        output = ""
        if not issues:
            if not self.issues:
                raise ValueError(
                    "Cannot convert the ufn to xml as its outcome-space is unknown"
                )
            issues = list(self.issues)
        for i, (issue, vfun, weight) in enumerate(
            zip(issues, self._values, self._weights)
        ):
            if not issue.is_numeric():
                raise ValueError(
                    f"Issue {issue} is not numeric. Cannot  use a LinearUtilityFunction. Try a LinearAdditiveUtilityFunction"
                )
            bias = self._bias / weight if weight else 0.0
            output += vfun.xml(i, issue, bias)

        for i, w in enumerate(self._weights):
            output += f'<weight index="{i+1}" value="{w}">\n</weight>\n'
        if abs(self._bias) > EPSILON:
            output += f'<weight index="{len(self._weights) + 1}" value="{self._bias}">\n</weight>\n'
        return output

    @classmethod
    def random(
        cls,
        issues: list[Issue] | tuple[Issue, ...],
        reserved_value=(0.0, 1.0),
        normalized=True,
    ):
        # from negmas.preferences.ops import normalize
        for issue in issues:
            if not issue.is_numeric():
                raise ValueError(
                    f"Issue {issue} is not numeric. Cannot  use a LinearUtilityFunction. Try a LinearAdditiveUtilityFunction"
                )

        reserved_value = make_range(reserved_value)
        n_issues = len(issues)
        reserved_value = (
            reserved_value
            if reserved_value is not None
            else tuple([random.random()] * 2)
        )
        if normalized:
            weights = [random.random() for _ in range(n_issues)]
            m = sum(weights)
            if m:
                weights = [_ / m for _ in weights]
            for i, issue in enumerate(issues):
                weights[i] /= issue.max_value  # type: ignore (we know that all numeric issues has a maximum value)
            bias = 0.0
        else:
            weights = [2 * (random.random() - 0.5) for _ in range(n_issues)]
            bias = sum(2 * (random.random() - 0.5) * _ for _ in weights)
        ufun = cls(
            weights=weights,
            bias=bias,
            issues=issues,
            reserved_value=random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0],
        )
        return ufun

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            **d,
            weights=self._weights,
            bias=self._bias,
        )

    @classmethod
    def from_dict(cls, d: dict):
        if isinstance(d, cls):
            return d
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        # d["values"]=deserialize(d["values"]),  # type: ignore (deserialize can return anything but it should be OK)
        d = deserialize(d, deep=True, remove_type_field=True)  # type: ignore
        return cls(**d)  # type: ignore I konw that d will be a dict with string keys

    def shift_by(
        self, offset: float, shift_reserved: bool = True
    ) -> AffineUtilityFunction:
        return AffineUtilityFunction(
            self._weights,
            self._bias + offset,
            outcome_space=self.outcome_space,
            name=self.name,
            reserved_value=self.reserved_value
            if not shift_reserved
            else (self.reserved_value + offset),
        )

    def scale_by(
        self, scale: float, scale_reserved: bool = True
    ) -> AffineUtilityFunction:
        if scale < 0:
            raise ValueError(f"Cannot have a negative scale: {scale}")
        weights = [_ * scale for _ in self._weights]
        return AffineUtilityFunction(
            weights=weights,
            bias=self._bias * scale,
            outcome_space=self.outcome_space,
            name=self.name,
            reserved_value=self.reserved_value
            if not scale_reserved
            else (self.reserved_value * scale),
        )

    def normalize_for(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        outcome_space: OutcomeSpace | None = None,
    ) -> ConstUtilityFunction | AffineUtilityFunction:  # type: ignore
        """
        Creates a new utility function that is normalized based on input conditions.

        Args:
            to: The minimum and maximum value to normalize to. If either is None, it is ignored.
                 This means that passing `(None, 1.0)` will normalize the ufun so that the maximum
                 is `1` but will not guarantee any limit for the minimum and so on.
             outcome_space: the outcome space to normalize within
        """
        epsilon: float = 1e-8
        if outcome_space is None:
            outcome_space = self.outcome_space

        mn, mx = self.minmax(outcome_space)

        if sum(self._weights) < epsilon:
            raise ValueError(
                f"Cannot normalize a ufun with zero weights to have a non-zero range"
            )

        if abs(mx - to[1]) < epsilon and abs(mn - to[0]) < epsilon:
            return self

        if abs(mx - mn) < epsilon:
            if (to[1] - to[0]) < epsilon:
                scale = 1.0
            else:
                from negmas.preferences.crisp.const import ConstUtilityFunction

                return ConstUtilityFunction(
                    to[1],
                    outcome_space=outcome_space,
                    name=self.name,
                )
        else:
            scale = (to[1] - to[0]) / (mx - mn)
        if scale < 0:
            raise ValueError(
                f"Cannot have a negative scale: max, min = ({mx}, {mn}) and rng  = {to}"
            )
        bias = to[1] - scale * mx + self._bias * scale
        weights = [_ * scale for _ in self._weights]
        if abs(bias) < epsilon:
            return LinearUtilityFunction(
                weights, outcome_space=outcome_space, name=self.name
            )
        return AffineUtilityFunction(
            weights, bias, outcome_space=outcome_space, name=self.name
        )

    def normalize(
        self,
        to: tuple[float, float] = (0.0, 1.0),
    ) -> ConstUtilityFunction | AffineUtilityFunction | LinearUtilityFunction:  # type: ignore
        return self.normalize_for(to, self.outcome_space)

    @lru_cache
    def extreme_outcomes(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        max_cardinality=1000,
    ) -> tuple[Outcome, Outcome]:
        """Finds the best and worst outcomes

        Args:
            ufun: The utility function
            issues: list of issues (optional)
            outcomes: A collection of outcomes (optional)
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                            given)
        Returns:
            (worst, best) outcomes

        """
        # The minimum and maximum must be at one of the edges of the outcome space. Just enumerate them
        original_os = outcome_space

        check_one_at_most(outcome_space, issues, outcomes)
        outcome_space = os_or_none(outcome_space, issues, outcomes)
        if outcome_space is None:
            outcome_space = self.outcome_space

        if outcome_space is not None:
            if not isinstance(outcome_space, IndependentIssuesOS):
                return super().extreme_outcomes(
                    original_os, issues, outcomes, max_cardinality
                )
            if outcomes is not None:
                warnings.warn(
                    f"Passing outcomes and issues (or having known issues) to linear ufuns is redundant. The outcomes passed will be used which is much slower than if you do not pass them",
                    warnings.NegmasSpeedWarning,
                )
                return super().extreme_outcomes(
                    outcome_space=original_os,
                    issues=issues,
                    outcomes=outcomes,
                    max_cardinality=max_cardinality,
                )
            uranges = []
            if not self.issues:
                raise ValueError(
                    "Cannot find extreme outcomes of a ufun without knowing its outcome-space"
                )
            issues = self.issues
            for issue in issues:
                mx, mn = float("-inf"), float("inf")
                for v in issue.value_generator(n=max_cardinality):
                    if v >= mx:
                        mx = v
                    if v < mn:
                        mn = v
                uranges.append((mn, mx))

            best_outcome, worst_outcome = [], []
            for w, urng in zip(self._weights, uranges):
                if w > 0:
                    best_outcome.append(urng[1])
                    worst_outcome.append(urng[0])
                else:
                    best_outcome.append(urng[0])
                    worst_outcome.append(urng[1])

            return tuple(worst_outcome), tuple(best_outcome)

        return super().extreme_outcomes(original_os, issues, outcomes, max_cardinality)

    def __str__(self):
        return f"w: {self._weights}, b: {self._bias}"


class LinearUtilityFunction(AffineUtilityFunction):
    r"""
    A special case of the `AffineUtilityFunciton` for which the bias is zero.

    Args:
         weights: weights for combining `values`
         bias: The offset added
         name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility value is calculated as:

        .. math::

            u = \sum_{i=0}^{n_{outcomes}-1} {\alpha_i * \omega_i}

        where $\alpha_i$ is the weight of issue $i$, and $\omega_i$ is the value of issue $i$ in the input outcome $\omega$.
    """

    def __init__(
        self,
        weights: dict[str, float] | list[float] | tuple[float, ...] | None = None,
        *args,
        **kwargs,
    ) -> None:
        kwargs["bias"] = 0
        super().__init__(weights, *args, **kwargs)


class LinearAdditiveUtilityFunction(  # type: ignore
    StationaryMixin,
    UtilityFunction,
):
    r"""A linear aggregation utility function for multi-issue negotiations.

    Models a linear utility function using predefined weights:\.

    Args:
         values: utility functions for individual issues
         weights: weights for combining `values`
         name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility value is calculated as:

        .. math::

            u = \sum_{i=0}^{n_{outcomes}-1} {w_i * u_i(\omega_i)}


    Examples:

        >>> from negmas.outcomes import dict2outcome, make_issue
        >>> issues = [make_issue((10.0, 20.0), 'price'), make_issue(['delivered', 'not delivered'], 'delivery')
        ...           , make_issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: (0, 4)']
        >>> f = LinearAdditiveUtilityFunction(
        ...          {'price': lambda x: 2.0*x
        ...            , 'delivery': {'delivered': 10, 'not delivered': -10}
        ...            , 'quality': lambda x: x-3}
        ...         , weights={'price': 1.0, 'delivery': 2.0, 'quality': 4.0}
        ...         , issues=issues)
        >>> float(f((2, 'delivered', 14.0)))
        68.0

        You can confirm this yourself by calculating the ufun manually:

        >>> 68.0 ==  (1.0 * 2.0 * 2 + 2.0 * 10 + 4.0 * (14.0 - 3))
        True

        Yout can use a dictionary to represent the outcome for more readability:

        >>> float(f(dict2outcome(dict(price=2, quality=14, delivery='delivered'), issues=issues)))
        68.0

        You can use lists instead of dictionaries for defining outcomes, weights
        but that is less readable. The advantage here is that you do not need to pass the issues

        >>> f = LinearAdditiveUtilityFunction([
        ...         lambda x: 2.0*x
        ...          , {'delivered': 10, 'not delivered': -10}
        ...          , LambdaFun(lambda x: x-3)]
        ...         , weights=[1.0, 2.0, 4.0], issues=issues)
        >>> float(f((14.0, 'delivered', 2)))
        44.0

    Remarks:
        The mapping need not use all the issues in the output as the last example show.

    """

    def __init__(
        self,
        values: dict[str, SingleIssueFun]
        | tuple[SingleIssueFun, ...]
        | list[SingleIssueFun],
        weights: Mapping[Any, float] | list[float] | tuple[float, ...] | None = None,
        bias: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._bias = bias
        if self.outcome_space and not isinstance(
            self.outcome_space, IndependentIssuesOS
        ):
            raise ValueError(
                f"Cannot create {self.type} ufun with an outcomespace without indpendent "
                f"issues.\n Given OS: {self.outcome_space} of type "
                f"{type(self.outcome_space)}\nGiven args {kwargs}"
            )
        self.issues: list[Issue] | None = (
            list(self.outcome_space.issues) if self.outcome_space else None  # type: ignore
        )
        if isinstance(values, dict):
            if self.issues is None:
                raise ValueError(
                    "Must specify issues when passing `values` or `weights` is a dict"
                )
            values = [
                values.get(_, IdentityFun())  # type: ignore
                for _ in [i.name if isinstance(i, Issue) else i for i in self.issues]
            ]
        else:
            values = list(values)
        if weights is None:
            weights = [1.0] * len(values)
        if isinstance(weights, dict):
            if self.issues is None:
                raise ValueError(
                    "Must specify issues when passing `values` or `weights` is a dict"
                )
            weights = [
                weights.get(_, 1.0)
                # weights[_]
                for _ in [i.name if isinstance(i, Issue) else i for i in self.issues]
            ]
        self.values = []
        for i, v in enumerate(values):
            if isinstance(v, SingleIssueFun):
                self.values.append(v)
            elif isinstance(v, dict):
                self.values.append(TableFun(v))
            elif isinstance(v, Callable):
                self.values.append(LambdaFun(v))
            elif isinstance(v, Iterable):
                if (
                    not self.issues
                    or len(self.issues) < i + 1
                    or not self.issues[i].is_discrete()
                ):
                    raise TypeError(
                        f"When passing an iterable as the value function for an issue, "
                        f"the issue MUST be discrete"
                    )
                d = dict(zip(self.issues[i].enumerate(), v))  # type: ignore We know the issue is discrete
                self.values.append(TableFun(d))
            else:
                raise TypeError(
                    f"Mapping {v} is not supported: Itis of type ({type(v)}) but we only support SingleIssueFun, Dict or Lambda mappings"
                )

        self._weights = list(weights)  # type: ignore

    @property
    def weights(self):
        return self._weights

    def eval(self, offer: Outcome | None) -> float | None:
        if offer is None:
            return self.reserved_value
        u = self._bias
        for v, w, iu in zip(offer, self.weights, self.values):
            current_utility = iu(v)
            if current_utility is None:
                return None

            try:
                u += w * current_utility
            except FloatingPointError:
                continue
        return u

    def xml(self, issues: list[Issue] | None = None) -> str:
        """Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> from negmas.outcomes import make_issue
            >>> issues = [make_issue(values=10, name='i1'), make_issue(values=['delivered', 'not delivered'], name='i2')
            ...     , make_issue(values=4, name='i3')]
            >>> f = LinearAdditiveUtilityFunction([lambda x: 2.0*x
            ...                          , {'delivered': 10, 'not delivered': -10}
            ...                          , LambdaFun(lambda x: x-3)]
            ...         , weights=[1.0, 2.0, 4.0], issues=issues)
            >>> print(f.xml(issues))
            <issue index="1" etype="discrete" type="discrete" vtype="integer" name="i1">
                <item index="1" value="0" evaluation="0.0" />
                <item index="2" value="1" evaluation="2.0" />
                <item index="3" value="2" evaluation="4.0" />
                <item index="4" value="3" evaluation="6.0" />
                <item index="5" value="4" evaluation="8.0" />
                <item index="6" value="5" evaluation="10.0" />
                <item index="7" value="6" evaluation="12.0" />
                <item index="8" value="7" evaluation="14.0" />
                <item index="9" value="8" evaluation="16.0" />
                <item index="10" value="9" evaluation="18.0" />
            </issue>
            <issue index="2" etype="discrete" type="discrete" vtype="discrete" name="i2">
                <item index="1" value="delivered" evaluation="10.0" />
                <item index="2" value="not delivered" evaluation="-10.0" />
            </issue>
            <issue index="3" etype="discrete" type="discrete" vtype="integer" name="i3">
                <item index="1" value="0" evaluation="-3.0" />
                <item index="2" value="1" evaluation="-2.0" />
                <item index="3" value="2" evaluation="-1.0" />
                <item index="4" value="3" evaluation="0.0" />
            </issue>
            <weight index="1" value="1.0">
            </weight>
            <weight index="2" value="2.0">
            </weight>
            <weight index="3" value="4.0">
            </weight>
            <BLANKLINE>

        """
        output = ""
        if not issues:
            issues = self.issues
        if not issues:
            raise ValueError(
                "Cannot convert a ufun to xml() without konwing its outcome-space"
            )

        # <issue vtype="integer" lowerbound="1" upperbound="17" name="Charging Speed" index="3" etype="integer" type="integer">
        # <evaluator ftype="linear" offset="0.4" slope="0.0375">
        # </evaluator>
        for i, (issue, vfun) in enumerate(zip(issues, self.values)):
            output += vfun.xml(i, issue, 0.0)

        for i, w in enumerate(self.weights):
            output += f'<weight index="{i+1}" value="{w}">\n</weight>\n'
        # if we have a bias, just add one extra issue with a weight equal to the bias (this issue will implicitly be assumed to be numeric with a single value of 1)
        if abs(self._bias) > EPSILON:
            output += f'<weight index="{len(self.weights) + 1}" value="{self._bias}">\n</weight>\n'
        return output

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        d.update(super().to_dict())
        return dict(
            **d,
            weights=self.weights,
            values=serialize(self.values),
        )

    @classmethod
    def from_dict(cls, d: dict):
        if isinstance(d, cls):
            return d
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        # d["values"]=deserialize(d["values"]),  # type: ignore (deserialize can return anything but it should be OK)
        d = deserialize(d, deep=True, remove_type_field=True)  # type: ignore
        return cls(**d)  # type: ignore I konw that d will be a dict with string keys

    @lru_cache
    def extreme_outcomes(
        self,
        outcome_space: OutcomeSpace | None = None,
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        max_cardinality=1000,
    ) -> tuple[Outcome, Outcome]:
        """Finds the best and worst outcomes

        Args:
            ufun: The utility function
            issues: list of issues (optional)
            outcomes: A collection of outcomes (optional)
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                            given)
        Returns:
            (worst, best) outcomes

        """
        # The minimum and maximum must be at one of the edges of the outcome space. Just enumerate them
        original_os = outcome_space

        check_one_at_most(outcome_space, issues, outcomes)
        outcome_space = os_or_none(outcome_space, issues, outcomes)
        if outcome_space is None:
            outcome_space = self.outcome_space
        if outcome_space is None or not isinstance(
            outcome_space, CartesianOutcomeSpace
        ):
            return super().extreme_outcomes(
                original_os, issues, outcomes, max_cardinality
            )
        if outcomes is not None:
            warnings.warn(
                f"Passing outcomes and issues (or having known issues) to linear ufuns is redundant. The outcomes passed will be used which is much slower than if you do not pass them",
                warnings.NegmasSpeedWarning,
            )
            return super().extreme_outcomes(
                original_os,
                issues,
                outcomes,
                max_cardinality,  # type:ignore
            )
        uranges, vranges = [], []
        myissues: list[Issue] = outcome_space.issues  # type: ignore We checked earlier that this is an CartesianOutcomeSpace. It MUST have issues
        for i, issue in enumerate(myissues):
            fn = self.values[i]
            mx, mn = float("-inf"), float("inf")
            mxv, mnv = None, None
            for v in issue.value_generator(n=max_cardinality):
                uval = fn(v)
                if uval >= mx:
                    mx, mxv = uval, v
                if uval < mn:
                    mn, mnv = uval, v
            vranges.append((mnv, mxv))
            uranges.append((mn, mx))

        best_outcome, worst_outcome = [], []
        best_util, worst_util = 0.0, 0.0
        for w, urng, vrng in zip(self.weights, uranges, vranges):
            if w > 0:
                best_util += w * urng[1]
                best_outcome.append(vrng[1])
                worst_util += w * urng[0]
                worst_outcome.append(vrng[0])
            else:
                best_util += w * urng[0]
                best_outcome.append(vrng[0])
                worst_util += w * urng[1]
                worst_outcome.append(vrng[1])

        return tuple(worst_outcome), tuple(best_outcome)

    @classmethod
    def random(
        cls,
        outcome_space: CartesianOutcomeSpace | None = None,
        issues: list[Issue] | tuple[Issue, ...] | None = None,
        reserved_value=(0.0, 1.0),
        normalized=True,
        **kwargs,
    ):
        # from negmas.preferences.ops import normalize
        if not issues and outcome_space:
            issues = outcome_space.issues
        if not issues:
            raise ValueError(f"Cannot generate a random ufun withot knowing the issues")

        reserved_value = make_range(reserved_value)

        n_issues = len(issues)
        # r = reserved_value if reserved_value is not None else random.random()
        rand_weights = [random.random() for _ in range(n_issues)]
        if normalized:
            m = sum(rand_weights)
            if m:
                rand_weights = [_ / m for _ in rand_weights]
        weights = rand_weights
        values = [_random_mapping(issue, normalized) for issue in issues]

        ufun = cls(
            weights=weights,
            values=values,  # type: ignore
            issues=issues,
            reserved_value=random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0],
            **kwargs,
        )
        return ufun

    def shift_by(
        self, offset: float, shift_reserved: bool = True, change_bias_only: bool = False
    ) -> LinearAdditiveUtilityFunction:
        if change_bias_only:
            return LinearAdditiveUtilityFunction(
                values=self.values,  # type: ignore
                weights=self.weights,
                name=self.name,
                bias=self._bias + offset,
                reserved_value=self.reserved_value
                if not shift_reserved
                else (self.reserved_value + offset),
                outcome_space=self.outcome_space,
            )
        values = [v.shift_by(offset) for v in self.values]

        return LinearAdditiveUtilityFunction(
            values=values,
            weights=self.weights,
            name=self.name,
            bias=self._bias,
            reserved_value=self.reserved_value
            if not shift_reserved
            else (self.reserved_value + offset),
            outcome_space=self.outcome_space,
        )

    def scale_by(
        self,
        scale: float,
        scale_reserved: bool = True,
        change_weights_only: bool = False,
        normalize_weights: bool = True,
    ) -> LinearAdditiveUtilityFunction:
        if scale < 0:
            raise ValueError(f"Cannot have a negative scale: {scale}")
        if change_weights_only and normalize_weights:
            raise ValueError(
                f"Cannot normalize weights only and in the same time change weights only"
            )

        wscale = 1.0
        if normalize_weights:
            w = sum(self.weights)
            if w > 1e-6:
                wscale = 1.0 / w
            else:
                wscale = 1.0
        if change_weights_only:
            wscale *= scale
            return LinearAdditiveUtilityFunction(
                values=self.values,  # type: ignore
                weights=[wscale * _ for _ in self.weights],
                outcome_space=self.outcome_space,
                reserved_value=self.reserved_value
                if not scale_reserved
                else (self.reserved_value * wscale),
                name=self.name,
            )
        return LinearAdditiveUtilityFunction(
            values=[_.scale_by(scale / wscale) for _ in self.values],
            weights=[wscale * _ for _ in self.weights],
            outcome_space=self.outcome_space,
            reserved_value=self.reserved_value
            if not scale_reserved
            else (self.reserved_value * wscale * scale),
            name=self.name,
        )

    def __str__(self):
        return f"u: {self.values}\n w: {self.weights}"


LinearUtilityAggregationFunction = LinearAdditiveUtilityFunction
"""An alias for `LinearAdditiveUtilityFunction`"""
