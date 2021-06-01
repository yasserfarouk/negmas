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
    Type,
    Union,
)


from negmas.common import AgentMechanismInterface
from negmas.generics import GenericMapping, ienumerate, iget
from negmas.helpers import gmap, ikeys
from negmas.outcomes import (
    Issue,
    Outcome,
    outcome_as,
    outcome_as_tuple,
)
from .base import UtilityFunction, UtilityValue, ExactUtilityValue
from negmas.helpers import make_range
from negmas.serialization import serialize

__all__ = [
    "LinearUtilityAggregationFunction",
    "LinearUtilityFunction",
]


class LinearUtilityFunction(UtilityFunction):
    r"""A linear utility function for multi-issue negotiations.

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
        >>> f = LinearUtilityFunction({'price': 1.0, 'quality': 4.0})
        >>> float(f({'quality': 2, 'price': 14.0})
        ...       ) -  (14 + 8)
        0.0
        >>> f = LinearUtilityFunction([1.0, 2.0])
        >>> float(f((2, 14)) - (30))
        0.0

    Remarks:

        - The mapping need not use all the issues in the output as the first example shows.
        - If an outcome contains combinations of strings and numeric values that have corresponding weights, an
          exception will be raised when its utility is calculated


    """

    def __init__(
        self,
        weights: Optional[Union[Mapping[Any, float], Sequence[float]]] = None,
        biases: Optional[Union[Mapping[Any, float], Sequence[float]]] = None,
        missing_value: Optional[float] = None,
        name: Optional[str] = None,
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None,
        outcome_type: Optional[Type] = None,
        id: str = None,
    ) -> None:
        super().__init__(
            name=name,
            outcome_type=outcome_type,
            reserved_value=reserved_value,
            ami=ami,
            id=id,
        )
        self.weights = weights
        if biases is None:
            if isinstance(self.weights, dict):
                biases = dict(zip(weights.keys(), itertools.repeat(0.0)))
            else:
                biases = [0.0] * len(weights)
        self.biases = biases
        self.missing_value = missing_value

    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        if offer is None:
            return self.reserved_value
        # offer = outcome_for(offer, self.ami) if self.ami is not None else offer
        u = ExactUtilityValue(0.0)
        if isinstance(self.weights, dict):
            if isinstance(offer, dict):
                for k, w in self.weights.items():
                    u += w * (
                        iget(offer, k, self.missing_value) + self.biases.get(k, 0)
                    )
                return u
            else:
                if self.ami is not None:
                    newoffer = dict()
                    for i, v in enumerate(offer):
                        newoffer[self.ami.issues[i].name] = v
                elif self.issue_names is not None:
                    newoffer = dict()
                    for i, v in enumerate(offer):
                        newoffer[self.issue_names[i]] = v
                elif self.issues is not None:
                    newoffer = dict()
                    for i, v in enumerate(offer):
                        newoffer[self.issues[i].name] = v
                else:
                    raise ValueError(
                        f"Cannot find issue names but weights are given as a dict."
                    )
                for k, w in self.weights.items():
                    u += w * (
                        iget(offer, k, self.missing_value) + self.biases.get(k, 0)
                    )
                return u

        offer = outcome_as_tuple(offer)
        return sum(w * (v + b) for w, b, v in zip(self.weights, self.biases, offer))

    def xml(self, issues: List[Issue]) -> str:
        """Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> issues = [Issue(values=10, name='i1'), Issue(values=4, name='i2')]
            >>> f = LinearUtilityFunction(weights=[1.0, 4.0])
            >>> print(f.xml(issues))
            <issue index="1" etype="discrete" type="discrete" vtype="discrete" name="i1">
                <item index="1" value="0" evaluation="0" />
                <item index="2" value="1" evaluation="1" />
                <item index="3" value="2" evaluation="2" />
                <item index="4" value="3" evaluation="3" />
                <item index="5" value="4" evaluation="4" />
                <item index="6" value="5" evaluation="5" />
                <item index="7" value="6" evaluation="6" />
                <item index="8" value="7" evaluation="7" />
                <item index="9" value="8" evaluation="8" />
                <item index="10" value="9" evaluation="9" />
            </issue>
            <issue index="2" etype="discrete" type="discrete" vtype="discrete" name="i2">
                <item index="1" value="0" evaluation="0" />
                <item index="2" value="1" evaluation="1" />
                <item index="3" value="2" evaluation="2" />
                <item index="4" value="3" evaluation="3" />
            </issue>
            <weight index="1" value="1.0">
            </weight>
            <weight index="2" value="4.0">
            </weight>
            <BLANKLINE>

        """
        output = ""
        keys = list(ikeys(issues))
        for i, k in enumerate(keys):
            issue = iget(issues, k)
            issue_name = issue.name
            if issue.is_float():
                output += f'<issue index="{i + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
                output += f'<range lowerbound = {issue.min_value} upperbound = {issue.max_value} ></range>'
            # elif issue.is_integer():
            #     output += f'<issue index="{i + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            #     output += f'<range lowerbound = {issue.min_value} upperbound = {issue.max_value} ></range>'
            else:
                output += f'<issue index="{i+1}" etype="discrete" type="discrete" vtype="discrete" name="{issue_name}">\n'
            vals = iget(issues, k).all
            bias = iget(self.biases, k, 0.0)
            for indx, u in enumerate(vals):
                uu = issue.value_type(u + bias)
                output += f'    <item index="{indx+1}" value="{uu}" evaluation="{u}" />\n'

            output += "</issue>\n"
        if isinstance(issues, dict):
            if isinstance(self.weights, dict):
                weights = self.weights
            else:
                weights = {k: v for k, v in zip(ikeys(issues), self.weights)}
        else:
            if isinstance(self.weights, list) or isinstance(self.weights, tuple):
                weights = list(self.weights)
            else:
                weights = list(self.weights.get(i.name, 1.0) for i in issues)

        for i, k in enumerate(keys):
            output += f'<weight index="{i+1}" value="{iget(weights, k)}">\n</weight>\n'
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
        if issues is not None:
            ranges = [(i.min_value, i.max_value) for i in issues]
            u = sorted(
                [
                    (
                        self(
                            outcome_as(
                                outcome, self.outcome_type, [_.name for _ in issues]
                            )
                        ),
                        outcome,
                    )
                    for outcome in itertools.product(*ranges)
                ]
            )
            if return_outcomes:
                return (
                    u[0][0],
                    u[-1][0],
                    outcome_as(u[0][1], self.outcome_type, [_.name for _ in issues]),
                    outcome_as(u[-1][1], self.outcome_type, [_.name for _ in issues]),
                )
            return u[0][0], u[-1][0]
        return super().utility_range(
            issues, outcomes, infeasible_cutoff, return_outcomes, max_n_outcomes
        )

    @classmethod
    def random(
        cls, issues: List["Issue"], reserved_value=(0.0, 1.0), normalized=True, **kwargs
    ):
        from negmas.utilities.ops import normalize

        reserved_value = make_range(reserved_value)
        n_issues = len(issues)
        r = reserved_value if reserved_value is not None else random.random()
        s = 0.0
        weights = [2 * (random.random() - 0.5) for _ in range(n_issues)]
        biases = [2 * (random.random() - 0.5) for _ in range(n_issues)]
        ufun = cls(
            weights=weights,
            biases=biases,
            reserved_value=random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0],
        )
        if normalized:
            return normalize(
                ufun,
                outcomes=Issue.discretize_and_enumerate(
                    issues, n_discretization=10, max_n_outcomes=10000
                ),
            )
        return ufun

    def to_dict(self):
        return dict(
            weights=self.weights,
            biases=self.biases,
            missing_value=self.missing_value,
            name=self.name,
            id=self.id,
            reserved_value=self.reserved_value,
        )


def _rand_mapping(x):
    return (random.random() - 0.5) * x


def random_mapping(issue: "Issue"):
    if issubclass(issue.value_type, int) or issubclass(issue.value_type, float):
        return _rand_mapping
    return zip(issue.values, [random.random() - 0.5 for _ in range(issue.cardinality)])


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

        >>> from negmas.utilities.nonlinear import MappingUtilityFunction
        >>> issues = [Issue((10.0, 20.0), 'price'), Issue(['delivered', 'not delivered'], 'delivery')
        ...           , Issue(5, 'quality')]
        >>> print(list(map(str, issues)))
        ['price: (10.0, 20.0)', "delivery: ['delivered', 'not delivered']", 'quality: (0, 4)']
        >>> f = LinearUtilityAggregationFunction({'price': lambda x: 2.0*x
        ...                          , 'delivery': {'delivered': 10, 'not delivered': -10}
        ...                          , 'quality': MappingUtilityFunction(lambda x: x-3)}
        ...         , weights={'price': 1.0, 'delivery': 2.0, 'quality': 4.0})
        >>> float(f({'quality': 2, 'price': 14.0, 'delivery': 'delivered'})
        ...       ) -  (1.0*(2.0*14)+2.0*10+4.0*(2.0-3.0))
        0.0
        >>> f = LinearUtilityAggregationFunction({'price': lambda x: 2.0*x
        ...                          , 'delivery': {'delivered': 10, 'not delivered': -10}}
        ...         , weights={'price': 1.0, 'delivery': 2.0})
        >>> float(f({'quality': 2, 'price': 14.0, 'delivery': 'delivered'})) - (1.0*(2.0*14)+2.0*10)
        0.0

        You can use lists instead of dictionaries for defining outcomes, weights
        but that is less readable

        >>> f = LinearUtilityAggregationFunction([lambda x: 2.0*x
        ...                          , {'delivered': 10, 'not delivered': -10}
        ...                          , MappingUtilityFunction(lambda x: x-3)]
        ...         , weights=[1.0, 2.0, 4.0])
        >>> float(f((14.0, 'delivered', 2))) - (1.0*(2.0*14)+2.0*10+4.0*(2.0-3.0))
        0.0

    Remarks:
        The mapping need not use all the issues in the output as the last example show.

    """

    def __init__(
        self,
        issue_utilities: Union[
            MutableMapping[Any, GenericMapping], Sequence[GenericMapping]
        ],
        weights: Optional[Union[Mapping[Any, float], Sequence[float]]] = None,
        name: Optional[str] = None,
        reserved_value: UtilityValue = float("-inf"),
        ami: AgentMechanismInterface = None,
        outcome_type: Optional[Type] = None,
        id: str = None,
    ) -> None:
        from negmas.utilities.nonlinear import MappingUtilityFunction

        super().__init__(
            name=name,
            outcome_type=outcome_type,
            reserved_value=reserved_value,
            ami=ami,
            id=id,
        )
        if weights is None:
            weights = (
                {i: 1.0 for i in ikeys(issue_utilities)}
                if isinstance(issue_utilities, dict)
                else [1.0] * len(issue_utilities)
            )
        if isinstance(weights, dict) and not isinstance(issue_utilities, dict):
            raise ValueError(
                f"Type of weights is {type(weights)} but type of issue_utilities is {type(issue_utilities)}"
            )
        if not isinstance(weights, dict) and isinstance(issue_utilities, dict):
            raise ValueError(
                f"Type of weights is {type(weights)} but type of issue_utilities is {type(issue_utilities)}"
            )
        self.issue_utilities = issue_utilities
        self.weights = weights
        for k, v in ienumerate(self.issue_utilities):
            self.issue_utilities[k] = (
                v if isinstance(v, UtilityFunction) else MappingUtilityFunction(v)
            )
        if isinstance(issue_utilities, dict):
            self.issue_indices = dict(
                zip(self.issue_utilities.keys(), range(len(self.issue_utilities)))
            )
        else:
            self.issue_indices = dict(
                zip(range(len(self.issue_utilities)), range(len(self.issue_utilities)))
            )

    # @UtilityFunction.outcome_type.setter
    # def outcome_type(self, value: AgentMechanismInterface):
    #     UtilityFunction.outcome_type.fset(self, value)
    #     if isinstance(self.issue_utilities, dict):
    #         for k, v in self.issue_utilities.items():
    #             if isinstance(v, UtilityFunction):
    #                 v.outcome_type = value
    #     else:
    #         for v in self.issue_utilities:
    #             if isinstance(v, UtilityFunction):
    #                 v.outcome_type = value
    #
    # @UtilityFunction.ami.setter
    # def ami(self, value: AgentMechanismInterface):
    #     UtilityFunction.ami.fset(self, value)
    #     if isinstance(self.issue_utilities, dict):
    #         for k, v in self.issue_utilities.items():
    #             if isinstance(v, UtilityFunction):
    #                 v.ami = value
    #     else:
    #         for v in self.issue_utilities:
    #             if isinstance(v, UtilityFunction):
    #                 v.ami = value
    #
    def eval(self, offer: Optional["Outcome"]) -> Optional[UtilityValue]:
        if offer is None:
            return self.reserved_value
        u = ExactUtilityValue(0.0)
        for k in ikeys(self.issue_utilities):
            if isinstance(offer, tuple):
                v = iget(offer, self.issue_indices[k])
            else:
                v = iget(offer, k)
            current_utility = gmap(iget(self.issue_utilities, k), v)
            if current_utility is None:
                return None

            w = iget(self.weights, k)  # type: ignore
            if w is None:
                return None
            try:
                u += w * current_utility
            except FloatingPointError:
                continue
        return u

    def xml(self, issues: List[Issue]) -> str:
        """Generates an XML string representing the utility function

        Args:
            issues:

        Examples:

            >>> from negmas.utilities.nonlinear import MappingUtilityFunction
            >>> issues = [Issue(values=10, name='i1'), Issue(values=['delivered', 'not delivered'], name='i2')
            ...     , Issue(values=4, name='i3')]
            >>> f = LinearUtilityAggregationFunction([lambda x: 2.0*x
            ...                          , {'delivered': 10, 'not delivered': -10}
            ...                          , MappingUtilityFunction(lambda x: x-3)]
            ...         , weights=[1.0, 2.0, 4.0])
            >>> print(f.xml(issues))
            <issue index="1" etype="discrete" type="discrete" vtype="discrete" name="i1">
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
            <issue index="3" etype="discrete" type="discrete" vtype="discrete" name="i3">
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
            >>> print(f.xml({i:_ for i, _ in enumerate(issues)}))
            <issue index="1" etype="discrete" type="discrete" vtype="discrete" name="i1">
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
            <issue index="3" etype="discrete" type="discrete" vtype="discrete" name="i3">
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
        keys = list(ikeys(issues))
        for i, k in enumerate(keys):
            issue = iget(issues, k)
            issue_name = issue.name
            if issue.is_float():
                output += f'<issue index="{i + 1}" etype="real" type="real" vtype="real" name="{issue_name}">\n'
                output += f'<range lowerbound = {issue.min_value} upperbound = {issue.max_value} ></range>'
            # elif issue.is_integer():
            #     output += f'<issue index="{i + 1}" etype="integer" type="integer" vtype="integer" name="{issue_name}">\n'
            #     output += f'<range lowerbound = {issue.min_value} upperbound = {issue.max_value} ></range>'
            else:
                output += f'<issue index="{i+1}" etype="discrete" type="discrete" vtype="discrete" name="{issue_name}">\n'
                vals = iget(issues, k).all
                for indx, v in enumerate(vals):
                    try:
                        u = gmap(iget(self.issue_utilities, issue_name), v)
                    except:
                        u = gmap(iget(self.issue_utilities, k), v)
                    v_ = (
                        v
                        if not (isinstance(v, tuple) or isinstance(v, list))
                        else "-".join([str(_) for _ in v])
                    )
                    output += (
                        f'    <item index="{indx+1}" value="{v_}" evaluation="{u}" />\n'
                    )
            output += "</issue>\n"
        if isinstance(issues, dict):
            if isinstance(self.weights, dict):
                weights = self.weights
            else:
                weights = {k: v for k, v in zip(ikeys(issues), self.weights)}
        else:
            if isinstance(self.weights, list):
                weights = self.weights
            else:
                weights = list(self.weights.get(i.name, 1.0) for i in issues)

        for i, k in enumerate(keys):
            output += f'<weight index="{i+1}" value="{iget(weights, k)}">\n</weight>\n'
        return output

    def __str__(self):
        return f"u: {self.issue_utilities}\n w: {self.weights}"

    def to_dict(self):
        return dict(
            weights=self.weights,
            issue_utilities=serialize(self.issue_utilities),
            name=self.name,
            id=self.id,
            reserved_value=self.reserved_value,
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
        if issues is not None:
            uranges, vranges = [], []
            for issue in issues:
                u = self.issue_utilities[issue.name]
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
            for w, urng, vrng in zip(self.weights.values(), uranges, vranges):
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
        from negmas.utilities.ops import normalize

        reserved_value = make_range(reserved_value)

        n_issues = len(issues)
        # r = reserved_value if reserved_value is not None else random.random()
        s = 0.0
        weights = dict(
            zip([_.name for _ in issues], [random.random() for _ in range(n_issues)])
        )
        s = sum(_ for _ in weights.values())
        if s:
            for k, v in weights.items():
                weights[k] = v / s
        issue_utilities = dict(
            zip([_.name for _ in issues], [random_mapping(issue) for issue in issues])
        )

        ufun = cls(
            weights=weights,
            issue_utilities=issue_utilities,
            reserved_value=random.random() * (reserved_value[1] - reserved_value[0])
            + reserved_value[0],
            **kwargs,
        )
        if normalized:
            return normalize(
                ufun,
                outcomes=Issue.discretize_and_enumerate(issues, max_n_outcomes=5000),
            )

    # def outcome_with_utility(
    #     self,
    #     rng: Tuple[Optional[float], Optional[float]],
    #     issues: List[Issue] = None,
    #     outcomes: List[Outcome] = None,
    #     n_trials: int = 100,
    # ) -> Optional["Outcome"]:
    #     """
    #     Gets one outcome within the given utility range or None on failure
    #
    #     Args:
    #         self: The utility function
    #         rng: The utility range
    #         issues: The issues the utility function is defined on
    #         outcomes: The outcomes to sample from
    #         n_trials: Not used
    #
    #     Returns:
    #
    #         - Either issues, or outcomes should be given but not both
    #
    #     """
    #
