from __future__ import annotations

import random
import warnings
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Union

from negmas.generics import GenericMapping
from negmas.helpers import get_full_type_name, make_range
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.base_issue import DiscreteIssue
from negmas.outcomes.range_issue import RangeIssue
from negmas.preferences.mapping import MappingUtilityFunction
from negmas.preferences.protocols import (
    IndIssues,
    Normalizable,
    Scalable,
    StationaryCrisp,
)
from negmas.protocols import XmlSerializable
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .ufun import UtilityFunction

__all__ = ["LinearUtilityAggregationFunction", "LinearUtilityFunction"]

NLEVELS = 20


def _rand_mapping(x):
    return (random.random() - 0.5) * x


def _rand_mapping_normalized(x):
    return random.random() * x


def _random_mapping(issue: "Issue", normalized=False):
    if issubclass(issue.value_type, float):
        return _rand_mapping_normalized if normalized else _rand_mapping
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
    return _rand_mapping_normalized if normalized else _rand_mapping


class LinearUtilityFunction(
    UtilityFunction, IndIssues, XmlSerializable, Scalable, Normalizable, StationaryCrisp
):
    r"""
    A linear utility function for multi-issue negotiations.

    Models a linear utility function using predefined weights.

    Args:
         weights: weights for combining `issue_utilities`
         name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility value is calculated as:

        .. math::

            u = \sum_{i=0}^{n_{outcomes}-1} {w_i * \omega_i}


    Examples:

        >>> issues = [make_issue((10.0, 20.0), 'price'), make_issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', 'quality: (0, 4)']
        >>> f = LinearUtilityFunction({'price': 1.0, 'quality': 4.0}, issues=issues)
        >>> f((2, 14.0)) -  (2 * 1.0 + 14.0 * 4)
        0.0
        >>> f = LinearUtilityFunction([1.0, 2.0])
        >>> f((2, 14)) - (2 * 1.0 + 14 * 2.0)
        0.0

    Remarks:

        - The mapping need not use all the issues in the output as the first example shows.
        - If an outcome contains combinations of strings and numeric values that have corresponding weights, an
          exception will be raised when its utility is calculated


    """

    def __init__(
        self,
        weights: dict[str, float] | list[float] | None = None,
        bias: float = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
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

        self.weights: list[float] = weights
        self.bias = bias
        self.values = [
            MappingUtilityFunction(lambda x: float(x[0]), issues=[_])
            for _ in self.issues
        ]

    def eval(self, offer: Optional["Outcome"]) -> float | None:
        if offer is None:
            return self.reserved_value
        return self.bias + sum(w * v for w, v in zip(self.weights, offer))

    def xml(self, issues: list[Issue] = None) -> str:
        """Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> issues = [make_issue(values=10, name='i1'), make_issue(values=4, name='i2')]
            >>> f = LinearUtilityFunction(weights=[1.0, 4.0])
            >>> print(f.xml(issues))
            <issue index="1" etype="integer" type="integer" vtype="integer" name="i1">
                <evaluator ftype="linear" offset="0.0" slope="1.0"></evaluator>
            </issue>
            <issue index="2" etype="integer" type="integer" vtype="integer" name="i2">
                <evaluator ftype="linear" offset="0.0" slope="1.0"></evaluator>
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
            issues = self.issues
        for i, issue in enumerate(issues):
            if not issue.is_numeric():
                raise ValueError(
                    f"Issue {issue} is not numeric. Cannot  use a LinearUtilityFunction. Try a LinearUtilityAggregationFunction"
                )
            issue_name = issue.name
            bias = self.bias / self.weights[i]
            if issue.is_continuous():
                output += f'<issue index="{i + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
                output += f'    <evaluator ftype="linear" offset="{bias}" slope="{1.0}"></evaluator>\n'
            elif isinstance(issue, RangeIssue) and issue.is_integer():
                output += f'<issue index="{i + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
                output += f'    <evaluator ftype="linear" offset="{bias}" slope="{1.0}"></evaluator>\n'
            else:
                # dtype = "integer" if issue.is_integer() else "real" if issue.is_float() else "discrete"
                dtype = "discrete"
                vtype = (
                    "integer"
                    if issue.is_integer()
                    else "real"
                    if issue.is_float()
                    else "discrete"
                )
                output += f'<issue index="{i+1}" etype="{dtype}" type="{dtype}" vtype="{vtype}" name="{issue_name}">\n'
                vals = issue.all  # type: ignore
                for indx, u in enumerate(vals):
                    uu = issue.value_type(u + bias)  # type: ignore
                    output += (
                        f'    <item index="{indx+1}" value="{uu}" evaluation="{u}" />\n'
                    )
            output += "</issue>\n"

        for i, w in enumerate(self.weights):
            output += f'<weight index="{i+1}" value="{w}">\n</weight>\n'
        return output

    def __str__(self):
        return f"w: {self.weights}, b: {self.bias}"

    @classmethod
    def random(cls, issues: list["Issue"], reserved_value=(0.0, 1.0), normalized=True):
        # from negmas.preferences.ops import normalize
        for issue in issues:
            if not issue.is_numeric():
                raise ValueError(
                    f"Issue {issue} is not numeric. Cannot  use a LinearUtilityFunction. Try a LinearUtilityAggregationFunction"
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
            bias = sum([2 * (random.random() - 0.5) * _ for _ in weights])
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
        return dict(
            **d,
            weights=self.weights,
            bias=self.bias,
            name=self.name,
            id=self.id,
            reserved_value=self.reserved_value,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        return cls(
            weights=d.get("weights", None),
            bias=d.get("bias", None),
            name=d.get("name", None),
            reserved_value=d.get("reserved_value", None),
            id=d.get("id", None),
        )

    def scale_min(
        self,
        to: float,
    ) -> "LinearUtilityFunction":
        """
        Creates a new ufun with maximum value scaled to the given value

        Args:
            to: The value to scale to
            issues: The outcome space in which to do the scaling. If not given
                    the whole outcmoe space of the ufun is used
            outcomes: A set of outcomes to limit our attention to. If not given,
                      the whole ufun is scaled
        """
        issues = self.issues
        mn, _ = self.utility_range(issues)
        if abs(mn) < 1e-10:
            if abs(to) < 1e-10:
                return self
            raise ValueError(f"Cannot normalize maximum only with zero max")

        scale = to / mn
        if scale < 0:
            raise ValueError(f"Cannot have a negative scale: min = {mn} and to = {to}")
        weights = [_ * scale for _ in self.weights]
        return LinearUtilityFunction(
            weights, self.bias * scale, issues=issues, name=self.name
        )

    def scale_max(
        self,
        to: float,
    ) -> "LinearUtilityFunction":
        """
        Creates a new ufun with maximum value scaled to the given value

        Args:
            to: The value to scale to
            issues: The outcome space in which to do the scaling. If not given
                    the whole outcmoe space of the ufun is used
            outcomes: A set of outcomes to limit our attention to. If not given,
                      the whole ufun is scaled
        """
        issues = self.issues
        _, mx = self.utility_range(issues)
        if abs(mx) < 1e-10:
            if abs(to) < 1e-10:
                return self
            raise ValueError(f"Cannot normalize maximum only with zero max")

        scale = to / mx
        if scale < 0:
            raise ValueError(f"Cannot have a negative scale: max = {mx} and to = {to}")
        weights = [_ * scale for _ in self.weights]
        return LinearUtilityFunction(
            weights, self.bias * scale, issues=issues, name=self.name
        )

    def normalize_for(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
    ) -> "IndIssues":
        """
        Creates a new utility function that is normalized based on input conditions.

        Args:
            outcomes: A set of outcomes to limit our attention to. If not given,
                      the whole ufun is normalized
            rng: The minimum and maximum value to normalize to. If either is None, it is ignored.
                 This means that passing `(None, 1.0)` will normalize the ufun so that the maximum
                 is `1` but will not guarantee any limit for the minimum and so on.
            infeasible_cutoff: outcomes with utility value less than or equal to this value will not
                               be considered during normalization
            epsilon: A small allowed error in normalization
            max_cardinality: Maximum ufun evaluations to conduct
        """
        infeasible_cutoff: float = float("-inf")
        epsilon: float = 1e-6
        max_cardinality: int = 1000
        if not issues:
            issues = self.issues
        if to[0] is None and to[1] is None:
            return self

        max_only = to[0] is None or to[0] == float("-inf")
        min_only = to[1] is None or to[1] == float("inf")
        if max_only and min_only:
            return self
        mn, mx = self.utility_range(
            issues, outcomes, infeasible_cutoff, max_cardinality
        )
        if mn < infeasible_cutoff:
            warnings.warn(
                f"Normalizing a linear ufun with a minimum of {mn} "
                f"and an infeasible_cutoff of {infeasible_cutoff} is"
                f" a nonlinear operation and will lead to a "
                f"Complex*UtilityFunction."
            )
            return super().normalize(issues, outcomes, to, infeasible_cutoff, epsilon, max_cardinality)  # type: ignore

        if sum(self.weights) < epsilon:
            if min_only:
                return LinearUtilityFunction(
                    weights=self.weights,
                    bias=to[0],
                    issues=self.issues,
                    name=self.name,
                )
            if max_only:
                return LinearUtilityFunction(
                    weights=self.weights,
                    bias=to[1],
                    issues=self.issues,
                    name=self.name,
                )
            raise ValueError(
                f"Cannot normalize a ufun with zero weights to have a non-zero range"
            )

        if min_only:
            if abs(to[0]) < epsilon:
                raise ValueError(f"Cannot normalize minimum only with zero min")
            to = (to[0], mx if mx > to[0] else to[0] + mx - mn)
        if max_only:
            if abs(to[1]) < epsilon:
                raise ValueError(f"Cannot normalize maximum only with zero max")
            to = (mn if mn < to[1] else to[1] - mx + mn, to[1])
        if abs(mx - to[1]) < epsilon and abs(mn - to[0]) < epsilon:
            return self

        if max_only:
            # if abs(mx) < epsilon:
            #     return self
            scale = to[1] / mx
        elif min_only:
            # if abs(mn) < epsilon:
            #     return self
            scale = to[0] / mn
        else:
            if abs(mx - mn) < epsilon and (to[1] - to[0]) < epsilon:
                scale = 1.0
            elif abs(mx - mn) < epsilon:
                from negmas.preferences.const import ConstUFun

                return ConstUFun(
                    to[1] if not min_only else to[0], issues=issues, name=self.name
                )
            else:
                scale = (to[1] - to[0]) / (mx - mn)
        if scale < 0:
            raise ValueError(
                f"Cannot have a negative scale: max, min = ({mx}, {mn}) and rng  = {to}"
            )
        bias = to[1] - scale * mx + self.bias * scale
        weights = [_ * scale for _ in self.weights]
        return LinearUtilityFunction(weights, bias, issues=issues, name=self.name)

    def extreme_outcomes(
        self,
        issues: list[Issue] = None,
        outcomes: list[Outcome] = None,
        infeasible_cutoff: float = float("-inf"),
        max_cardinality=1000,
    ) -> tuple[Outcome, Outcome]:
        """Finds the best and worst outcomes

        Args:
            ufun: The utility function
            issues: list of issues (optional)
            outcomes: A collection of outcomes (optional)
            infeasible_cutoff: A value under which any utility is considered infeasible and is not used in calculation
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                            given)
        Returns:
            (worst, best) outcomes

        """
        # The minimum and maximum must be at one of the edges of the outcome space. Just enumerate them

        # TODO test this method and add other methods for utility operations
        if issues is None:
            issue = self.issues
        if issues is not None:
            if outcomes is not None:
                warnings.warn(
                    f"Passing outcomes and issues (or having known issues) to linear ufuns is redundant. The outcomes passed will be used which is much slower than if you do not pass them"
                )
                super().extreme_outcomes(
                    issues, outcomes, infeasible_cutoff, max_cardinality
                )
            uranges = []
            for issue in issues:
                mx, mn = float("-inf"), float("inf")
                for v in issue.value_generator(n=max_cardinality):
                    if v >= mx:
                        mx = v
                    if v < mn:
                        mn = v
                uranges.append((mn, mx))

            best_outcome, worst_outcome = [], []
            for w, urng in zip(self.weights, uranges):
                if w > 0:
                    best_outcome.append(urng[1])
                    worst_outcome.append(urng[0])
                else:
                    best_outcome.append(urng[0])
                    worst_outcome.append(urng[1])

            return tuple(worst_outcome), tuple(best_outcome)

        return super().extreme_outcomes(
            issues, outcomes, infeasible_cutoff, max_cardinality
        )


class LinearUtilityAggregationFunction(UtilityFunction):
    r"""A linear aggregation utility function for multi-issue negotiations.

    Models a linear utility function using predefined weights:\.

    Args:
         issue_utilities: utility functions for individual issues
         weights: weights for combining `issue_utilities`
         name: name of the utility function. If None a random name will be generated.

    Notes:

        The utility value is calculated as:

        .. math::

            u = \sum_{i=0}^{n_{outcomes}-1} {w_i * u_i(\omega_i)}


    Examples:

        >>> from negmas.preferences.nonlinear import MappingUtilityFunction
        >>> from negmas.outcomes import dict2outcome
        >>> issues = [make_issue((10.0, 20.0), 'price'), make_issue(['delivered', 'not delivered'], 'delivery')
        ...           , make_issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: (0, 4)']
        >>> f = LinearUtilityAggregationFunction(
        ...          {'price': lambda x: 2.0*x
        ...            , 'delivery': {'delivered': 10, 'not delivered': -10}
        ...            , 'quality': MappingUtilityFunction(lambda x: x-3)}
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

        >>> f = LinearUtilityAggregationFunction([
        ...         lambda x: 2.0*x
        ...          , {'delivered': 10, 'not delivered': -10}
        ...          , MappingUtilityFunction(lambda x: x-3)]
        ...         , weights=[1.0, 2.0, 4.0])
        >>> float(f((14.0, 'delivered', 2)))
        44.0

    Remarks:
        The mapping need not use all the issues in the output as the last example show.

    """

    def __init__(
        self,
        values: Union[MutableMapping[Any, GenericMapping], Sequence[GenericMapping]],
        weights: Optional[Union[Mapping[Any, float], Sequence[float]]] = None,
        **kwargs,
    ) -> None:
        from negmas.preferences.mapping import MappingUtilityFunction

        super().__init__(**kwargs)
        if weights is None:
            weights = [1.0] * len(values)
        if isinstance(values, dict):
            if self.issues is None:
                raise ValueError(
                    "Must specify issues when passing `issue_utilties` or `weights` is a dict"
                )
            values = [
                values[_]
                for _ in [i.name if isinstance(i, Issue) else i for i in self.issues]
            ]
        if isinstance(weights, dict):
            if self.issues is None:
                raise ValueError(
                    "Must specify issues when passing `issue_utilties` or `weights` is a dict"
                )
            weights = [
                weights[_]
                for _ in [i.name if isinstance(i, Issue) else i for i in self.issues]
            ]
        self.values = [
            MappingUtilityFunction(_, issues=[i])
            if not isinstance(_, UtilityFunction)
            else _
            for _, i in zip(values, self.issues)
        ]
        self.weights = weights

    def eval(self, offer: Optional["Outcome"]) -> float | None:
        if offer is None:
            return self.reserved_value
        u = float(0.0)
        for v, w, iu in zip(offer, self.weights, self.values):
            current_utility = iu(v)
            if current_utility is None:
                return None

            try:
                u += w * current_utility
            except FloatingPointError:
                continue
        return u

    def xml(self, issues: list[Issue] = None) -> str:
        """Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> from negmas.preferences.nonlinear import MappingUtilityFunction
            >>> issues = [make_issue(values=10, name='i1'), make_issue(values=['delivered', 'not delivered'], name='i2')
            ...     , make_issue(values=4, name='i3')]
            >>> f = LinearUtilityAggregationFunction([lambda x: 2.0*x
            ...                          , {'delivered': 10, 'not delivered': -10}
            ...                          , MappingUtilityFunction(lambda x: x-3)]
            ...         , weights=[1.0, 2.0, 4.0])
            >>> print(f.xml(issues))
            <issue index="1" etype="integer" type="integer" vtype="integer" name="i1">
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
                <item index="1" value="delivered" evaluation="10" />
                <item index="2" value="not delivered" evaluation="-10" />
            </issue>
            <issue index="3" etype="integer" type="integer" vtype="integer" name="i3">
                <item index="1" value="0" evaluation="-3" />
                <item index="2" value="1" evaluation="-2" />
                <item index="3" value="2" evaluation="-1" />
                <item index="4" value="3" evaluation="0" />
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
        for i, issue in enumerate(issues):
            issue_name = issue.name
            if issue.is_continuous():
                raise ValueError(
                    f"Cannot save a linear aggregation UtilityFunction for issue {issue} because it is continuous"
                )
            # dtype = "integer" if issue.is_integer() else "real" if issue.is_float() else "discrete"
            dtype = "discrete"
            output += f'<issue index="{i+1}" etype="{dtype}" type="{dtype}" vtype="{dtype}" name="{issue_name}">\n'
            vals = issue.all  # type: ignore (We know that the issue is discrete because we are in the else)
            for indx, value in enumerate(vals):
                u = self.values[i](value)
                output += (
                    f'    <item index="{indx+1}" value="{value}" evaluation="{u}" />\n'
                )
            output += "</issue>\n"

        for i, w in enumerate(self.weights):
            output += f'<weight index="{i+1}" value="{w}">\n</weight>\n'
        return output

    def __str__(self):
        return f"u: {self.values}\n w: {self.weights}"

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        return dict(
            **d,
            weights=self.weights,
            issue_utilities=serialize(self.values),
            name=self.name,
            id=self.id,
            reserved_value=self.reserved_value,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        return cls(
            values=deserialize(d["issue_utilities"]),  # type: ignore (deserialize can return anything but it should be OK)
            weights=d.get("weights", None),
            name=d.get("name", None),
            reserved_value=d.get("reserved_value", None),
            id=d.get("id", None),
        )

    def extreme_outcomes(
        self,
        issues: list[Issue] = None,
        outcomes: list[Outcome] = None,
        infeasible_cutoff: float = float("-inf"),
        max_cardinality=1000,
    ) -> tuple[Outcome, Outcome]:
        """Finds the best and worst outcomes

        Args:
            ufun: The utility function
            issues: list of issues (optional)
            outcomes: A collection of outcomes (optional)
            infeasible_cutoff: A value under which any utility is considered infeasible and is not used in calculation
            max_cardinality: the maximum number of outcomes to try sampling (if sampling is used and outcomes are not
                            given)
        Returns:
            (worst, best) outcomes

        """
        # The minimum and maximum must be at one of the edges of the outcome space. Just enumerate them

        # TODO test this method and add other methods for utility operations
        if issues is None:
            issue = self.issues
        if issues is not None:
            if outcomes is not None:
                warnings.warn(
                    f"Passing outcomes and issues (or having known issues) to linear ufuns is redundant. The outcomes passed will be used which is much slower than if you do not pass them"
                )
                super().extreme_outcomes(
                    issues, outcomes, infeasible_cutoff, max_cardinality
                )
            uranges, vranges = [], []
            for i, issue in enumerate(issues):
                u = self.values[i]
                mx, mn = float("-inf"), float("inf")
                mxv, mnv = None, None
                for v in issue.value_generator(n=max_cardinality):
                    uval = u(v)
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

        return super().extreme_outcomes(
            issues, outcomes, infeasible_cutoff, max_cardinality
        )

    @classmethod
    def random(
        cls, issues: list["Issue"], reserved_value=(0.0, 1.0), normalized=True, **kwargs
    ):
        # from negmas.preferences.ops import normalize

        reserved_value = make_range(reserved_value)

        n_issues = len(issues)
        # r = reserved_value if reserved_value is not None else random.random()
        rand_weights = [random.random() for _ in range(n_issues)]
        if normalized:
            m = sum(rand_weights)
            if m:
                rand_weights = [_ / m for _ in rand_weights]
        weights = rand_weights
        issue_utilities = [_random_mapping(issue, normalized) for issue in issues]

        ufun = cls(
            weights=weights,
            issue_utilities=issue_utilities,
            issues=issues,
            reserved_value=random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0],
            **kwargs,
        )
        return ufun

    def normalize_for(
        self,
        to: tuple[float, float] = (0.0, 1.0),
        issues: list[Issue] | None = None,
        outcomes: list[Outcome] | None = None,
        normalize_weights: bool = True,
        change_weights: bool = False,
    ) -> "LinearUtilityAggregationFunction":
        infeasible_cutoff = float("-inf")
        max_cardinality = 10_000
        epsilon = 1e-6
        if not issues:
            issues = self.issues
        if not isinstance(to, Iterable):
            to = (to, to)
        if to[0] is None and to[1] is None:
            return self
        max_only = to[0] is None or to[0] == float("-inf")
        min_only = to[1] is None or to[1] == float("inf")
        mn, mx = self.utility_range(
            issues, outcomes, infeasible_cutoff, max_cardinality
        )
        if mn < infeasible_cutoff:
            warnings.warn(
                f"Normalizing a linear ufun with a minimum of {mn} and an infeasible_cutoff of {infeasible_cutoff} is a nonlinear operation and will lead to a Complex*UtilityFunction."
            )
            return super().normalize(issues, outcomes, to, infeasible_cutoff, epsilon, max_cardinality)  # type: ignore

        if min_only:
            to = (to[0], mx)
        if max_only:
            to = (mn, to[1])
        if abs(mx - to[1]) < epsilon and abs(mn - to[0]) < epsilon:
            return self

        if max_only:
            scale = to[1] / mx
        elif min_only:
            scale = to[0] / mn
        else:
            scale = (to[1] - to[0]) / (mx - mn)
        wscale = 1.0
        if normalize_weights:
            w = sum(self.weights)
            wscale = 1.0 / w
        if change_weights:
            wscale *= scale
            scale = None
        if scale is None:
            return LinearUtilityAggregationFunction(
                values=self.values,
                weights=[wscale * _ for _ in self.weights],
            )
        return LinearUtilityAggregationFunction(
            values=[(lambda x: scale * _(x)) for _ in self.values],
            weights=[wscale * _ for _ in self.weights],
        )
