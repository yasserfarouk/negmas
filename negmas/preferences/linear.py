from __future__ import annotations

import itertools
import random
from typing import (
    Any,
    Collection,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from negmas.generics import GenericMapping
from negmas.helpers import get_full_type_name, make_range
from negmas.outcomes import Issue, Outcome
from negmas.outcomes.base_issue import DiscreteIssue, RangeIssue
from negmas.serialization import PYTHON_CLASS_IDENTIFIER, deserialize, serialize

from .base import UtilityValue
from .base_crisp import UtilityFunction
from .static import StaticPreferences

__all__ = [
    "LinearUtilityAggregationFunction",
    "LinearUtilityFunction",
]

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


class LinearUtilityFunction(StaticPreferences, UtilityFunction):
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

        >>> issues = [Issue((10.0, 20.0), 'price'), Issue(5, 'quality')]
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
        biases: dict[str, float] | list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if weights is None:
            if not self.issues:
                raise ValueError(
                    "Cannot initializes with no weights if you are not specifying issues or ami"
                )
            weights = [1.0 / len(self.issues)] * len(self.issues)

        if isinstance(weights, dict):
            if not self.issues:
                raise ValueError(
                    "Cannot initializes with dict of weights if you are not specifying issues or ami"
                )
            weights = [
                weights[_]
                for _ in [i.name if isinstance(i, Issue) else i for i in self.issues]
            ]

        if biases is None:
            biases = [0.0] * len(weights)
        if isinstance(biases, dict):
            if not self.issues:
                raise ValueError(
                    "Cannot initializes with dict of biases if you are not specifying issues or ami"
                )
            biases = [
                biases[_]
                for _ in [i.name if isinstance(i, Issue) else i for i in self.issues]
            ]

        self.weights: list[float] = weights
        self.biases: list[float] = biases

    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        if offer is None:
            return self.reserved_value
        return sum(w * (v + b) for w, b, v in zip(self.weights, self.biases, offer))

    def xml(self, issues: List[Issue] = None) -> str:
        """Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> issues = [Issue(values=10, name='i1'), Issue(values=4, name='i2')]
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
            bias = self.biases[i]
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
                vals = issue.all
                for indx, u in enumerate(vals):
                    uu = issue.value_type(u + bias)
                    output += (
                        f'    <item index="{indx+1}" value="{uu}" evaluation="{u}" />\n'
                    )
            output += "</issue>\n"

        for i, w in enumerate(self.weights):
            output += f'<weight index="{i+1}" value="{w}">\n</weight>\n'
        return output

    def __str__(self):
        return f"w: {self.weights}, b: {self.biases}"

    def utility_range(
        self,
        issues: List[Issue] = None,
        outcomes: Collection[Outcome] = None,
        infeasible_cutoff: Optional[float] = None,
        return_outcomes=False,
        max_n_outcomes=1000,
        ami=None,
    ) -> Union[
        Tuple[UtilityValue, UtilityValue],
        Tuple[UtilityValue, UtilityValue, Outcome, Outcome],
    ]:
        # The minimum and maximum must be at one of the edges of the outcome space. Just enumerate them
        if issues is None and ami is not None:
            issues = ami.issues
        if issues is not None:
            ranges = [(i.min_value, i.max_value) for i in issues]
            u = sorted(
                (self(outcome), outcome) for outcome in itertools.product(*ranges)
            )
            if return_outcomes:
                return (
                    u[0][0],
                    u[-1][0],
                    u[0][1],
                    u[-1][1],
                )
            return u[0][0], u[-1][0]
        return super().utility_range(
            issues, outcomes, infeasible_cutoff, return_outcomes, max_n_outcomes
        )

    @classmethod
    def random(
        cls, issues: List["Issue"], reserved_value=(0.0, 1.0), normalized=True, **kwargs
    ):
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
                weights[i] /= issue.max_value
            biases = [0.0] * n_issues
        else:
            weights = [2 * (random.random() - 0.5) for _ in range(n_issues)]
            biases = [2 * (random.random() - 0.5) for _ in range(n_issues)]
        ufun = cls(
            weights=weights,
            biases=biases,
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
            biases=self.biases,
            name=self.name,
            id=self.id,
            reserved_value=self.reserved_value,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        return cls(
            weights=d.get("weights", None),
            biases=d.get("biases", None),
            name=d.get("name", None),
            reserved_value=d.get("reserved_value", None),
            ami=d.get("ami", None),
            id=d.get("id", None),
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
        >>> issues = [Issue((10.0, 20.0), 'price'), Issue(['delivered', 'not delivered'], 'delivery')
        ...           , Issue(5, 'quality')]
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
        issue_utilities: Union[
            MutableMapping[Any, GenericMapping], Sequence[GenericMapping]
        ],
        weights: Optional[Union[Mapping[Any, float], Sequence[float]]] = None,
        **kwargs,
    ) -> None:
        from negmas.preferences.nonlinear import MappingUtilityFunction

        super().__init__(**kwargs)
        if weights is None:
            weights = [1.0] * len(issue_utilities)
        if isinstance(issue_utilities, dict):
            if self.issues is None:
                raise ValueError(
                    "Must specify issues when passing `issue_utilties` or `weights` is a dict"
                )
            issue_utilities = [
                issue_utilities[_]
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
        self.issue_utilities = [
            MappingUtilityFunction(_) if not isinstance(_, UtilityFunction) else _
            for _ in issue_utilities
        ]
        self.weights = weights

    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        if offer is None:
            return self.reserved_value
        u = float(0.0)
        for v, w, iu in zip(offer, self.weights, self.issue_utilities):
            current_utility = iu(v)
            if current_utility is None:
                return None

            try:
                u += w * current_utility
            except FloatingPointError:
                continue
        return u

    def xml(self, issues: List[Issue] = None) -> str:
        """Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> from negmas.preferences.nonlinear import MappingUtilityFunction
            >>> issues = [Issue(values=10, name='i1'), Issue(values=['delivered', 'not delivered'], name='i2')
            ...     , Issue(values=4, name='i3')]
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
            else:
                # dtype = "integer" if issue.is_integer() else "real" if issue.is_float() else "discrete"
                dtype = "discrete"
                output += f'<issue index="{i+1}" etype="{dtype}" type="{dtype}" vtype="{dtype}" name="{issue_name}">\n'
                vals = issue.all
                for indx, value in enumerate(vals):
                    u = self.issue_utilities[i](value)
                    output += f'    <item index="{indx+1}" value="{value}" evaluation="{u}" />\n'
            output += "</issue>\n"

        for i, w in enumerate(self.weights):
            output += f'<weight index="{i+1}" value="{w}">\n</weight>\n'
        return output

    def __str__(self):
        return f"u: {self.issue_utilities}\n w: {self.weights}"

    def to_dict(self):
        d = {PYTHON_CLASS_IDENTIFIER: get_full_type_name(type(self))}
        return dict(
            **d,
            weights=self.weights,
            issue_utilities=serialize(self.issue_utilities),
            name=self.name,
            id=self.id,
            reserved_value=self.reserved_value,
        )

    @classmethod
    def from_dict(cls, d):
        d.pop(PYTHON_CLASS_IDENTIFIER, None)
        return cls(
            issue_utilities=deserialize(d["issue_utilities"]),
            weights=d.get("weights", None),
            name=d.get("name", None),
            reserved_value=d.get("reserved_value", None),
            ami=d.get("ami", None),
            id=d.get("id", None),
        )

    def utility_range(
        self,
        issues: List[Issue] = None,
        outcomes: Collection[Outcome] = None,
        infeasible_cutoff: Optional[float] = None,
        return_outcomes=False,
        max_n_outcomes=1000,
        ami=None,
    ) -> Union[
        Tuple[UtilityValue, UtilityValue],
        Tuple[UtilityValue, UtilityValue, Outcome, Outcome],
    ]:
        # The minimum and maximum must be at one of the edges of the outcome space. Just enumerate them

        # TODO test this method and add other methods for utility operations
        if ami is not None and issues is None:
            issues = ami.issues
        if issues is not None:
            uranges, vranges = [], []
            for i, issue in enumerate(issues):
                u = self.issue_utilities[i]
                mx, mn = float("-inf"), float("inf")
                mxv, mnv = None, None
                for v in issue.alli(n=max_n_outcomes):
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

            if return_outcomes:
                return (worst_util, best_util, worst_outcome, best_outcome)
            return worst_util, best_util
        return super().utility_range(
            issues, outcomes, infeasible_cutoff, return_outcomes, max_n_outcomes
        )

    @classmethod
    def random(
        cls, issues: List["Issue"], reserved_value=(0.0, 1.0), normalized=True, **kwargs
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
